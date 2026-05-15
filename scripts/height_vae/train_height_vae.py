"""
高度图 VAE 训练脚本（显存优化版）

Phase 3: 训练高度图专用 VAE
目标：解决高程数据压缩失真的问题，确保陡崖边缘清晰

=== 显存估算 (fp32, BATCH_SIZE=1, grad_checkpointing=ON) ===
  模型参数: 55.32M (encoder 22.36M + decoder 32.96M)
  模型权重:  221 MB
  优化器:    442 MB (AdamW fp32, momentum + variance)
  梯度:      221 MB
  激活值:   ~430 MB (检查点边界张量, 最大 128MB@[1,128,512,512])
            + ~350 MB (单块重计算峰值, grad_checkpointing 下逐块重算)
  CUDA开销:  ~500-1000 MB (cuDNN workspace + PyTorch context)
  ─────────────────────
  总计:     ~2.5-3.5 GB  (8 GB 显卡充裕, 4 GB 勉强可跑)

=== 关键参数设计原则 ===
  - BATCH_SIZE=1 + gradient_accumulation_steps=8 → 模拟有效 batch=8
  - block_out_channels=(128,256,512) → 3 层, 55M 参数
  - enable_grad_checkpointing=True → 省 50%+ 激活值显存, 零精度损失
  - USE_AMP=False → fp32 运行 (与 grad_checkpointing 冲突)
  - 目标显卡: 6-8 GB VRAM

用法：
    # 全新训练
    python scripts/height_vae/train_height_vae.py --epochs 100

    # 断点续训（训练中断后恢复）
    python scripts/height_vae/train_height_vae.py --epochs 100 --checkpoint ./outputs/height_vae/checkpoint.pt

    # 测试重建质量
    python scripts/height_vae/train_height_vae.py --mode test --checkpoint ./outputs/height_vae/checkpoint.pt
"""

import argparse
import glob
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 导入项目模块
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from models.vae.heightmap_vae import HeightMapVAE
from dataset.height_map_dataset import HeightMapDataset

# =============================================================================
# 训练配置（硬编码）
# =============================================================================

# 数据配置
DATA_ROOT = "./data/process/heightmaps_hf"
IMAGE_SIZE = 512
BATCH_SIZE = 1  # 显存充足时可调至 2-8
NUM_WORKERS = 4
VAL_SPLIT = 0.1  # 验证集比例
SEED = 42  # 数据集划分随机种子

# 模型配置
H_MAX = 3000.0  # 全局最大高程（仅用于 raw-elevation 场景，log 归一化数据不使用）
BLOCK_OUT_CHANNELS = (128, 256, 512)  # 显存不足时可降为 (64, 128, 256)
ENABLE_GRAD_CHECKPOINTING = True  # 零精度损失，节省 ~50% 显存；不能与 AMP 同时开启

# 优化器配置
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

# 训练配置
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 8  # 模拟有效 batch=8，显存充足时可调至 1-4
USE_AMP = False  # 混合精度（fp16）会与梯度检查点冲突；禁用后以 fp32 运行

# 损失权重（loss_kl 已归一化为 per-dim 均值，范围为 0.05-10 nats/dim）
LOSS_WEIGHT_MSE = 1.0
LOSS_WEIGHT_KL = (
    0.01  # per-dim KL 权重（当 kl≈0.1 时贡献 ≈ 0.001，远小于 MSE 0.02-0.08）
)
LOSS_WEIGHT_GEO = 0.8
KL_FREE_BITS_WEIGHT = 0.5  # per-dim free bits 惩罚权重
KL_FREE_BITS_PER_DIM = 0.1  # 每维最低 0.1 nat，KL 低于此值会被强力上推
KL_ANNEALING_EPOCHS = 50  # 延长退火让 autoencoder 先学好重建再引入 KL 正则
USE_HUBER_LOSS = False  # SmoothL1 对高程跳变更鲁棒，beta 在计算时动态设为 0.01

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出配置
OUTPUT_DIR = "./outputs/height_vae"
CHECKPOINT_STEPS = 1000
LOG_STEPS = 10
VIZ_INTERVAL = 1  # 每隔 N 个 epoch 生成一张效果图
VIZ_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "visualizations")


class Trainer:
    """高度图 VAE 训练器"""

    def __init__(
        self,
        vae: HeightMapVAE,
        train_dataloader: DataLoader,
        output_dir: str,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        kl_annealing_epochs: int = 0,
        use_huber_loss: bool = False,
        kl_free_bits_per_dim: float = 0.0,
        val_dataloader: DataLoader | None = None,
    ):
        """
        初始化训练器

        Args:
            vae: 高度图 VAE 模型
            train_dataloader: 训练数据加载器
            output_dir: 输出目录
            device: 运行设备
            gradient_accumulation_steps: 梯度累积步数
            kl_annealing_epochs: KL 退火 epoch 数
            use_huber_loss: 是否用 SmoothL1 替代 MSE
            kl_free_bits_per_dim: 每维 KL free bits 阈值（0 禁用），loss_kl 低于此值会被罚
            val_dataloader: 验证数据加载器（用于可视化，不参与训练）
        """
        self.vae = vae.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = Path(output_dir)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.kl_annealing_epochs = kl_annealing_epochs
        self.use_huber_loss = use_huber_loss

        # Free bits: loss_kl 已归一化为 per-dim 均值，target 直接取 per-dim 阈值
        self.kl_free_bits_target = (
            kl_free_bits_per_dim if kl_free_bits_per_dim > 0 else 0.0
        )

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=WEIGHT_DECAY,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=NUM_EPOCHS - WARMUP_EPOCHS,
        )

        # 训练统计
        self.global_step = 0
        self.best_loss = float("inf")

        # 可视化：loss 历史 + 固定验证样本
        self.loss_history = []  # [(total, mse, kl, geo), ...]
        self.val_samples = self._init_val_samples()

        # 创建可视化输出目录
        self.viz_output_dir = Path(VIZ_OUTPUT_DIR)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

    def _init_val_samples(self) -> torch.Tensor | None:
        """从验证集抽取一组固定样本用于可视化"""
        if self.val_dataloader is None:
            val_iter = iter(self.train_dataloader)
            val_sample, _ = next(val_iter)
            return val_sample[:1]
        val_iter = iter(self.val_dataloader)
        val_sample, _ = next(val_iter)
        return val_sample[:1]

    def get_kl_weight(self, epoch: int) -> float:
        """获取当前 epoch 的 KL 权重（支持 annealing）"""
        if self.kl_annealing_epochs <= 0:
            return LOSS_WEIGHT_KL
        return min(1.0, epoch / max(1, self.kl_annealing_epochs)) * LOSS_WEIGHT_KL

    def train_epoch(self, epoch: int) -> dict:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号
        Returns:
            avg_loss_dict: 平均损失字典
        """
        self.vae.train()

        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_kl = 0.0
        total_loss_geo = 0.0
        num_batches = 0
        kl_weight = self.get_kl_weight(epoch)

        # 进度条
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (height_maps, _) in enumerate(pbar):
            # 数据移到设备
            height_maps = height_maps.to(self.device)

            # 前向传播
            recon, loss_dict = self.vae(height_maps, return_recon_only=False)
            loss_recon = (
                F.smooth_l1_loss(recon, height_maps, beta=0.01)
                if self.use_huber_loss
                else loss_dict["loss_mse"]
            )
            loss_kl = loss_dict["loss_kl"]
            loss_geo = loss_dict["loss_geo"]
            # Free bits: 当 per-dim KL 低于阈值时施加向上推力，防止后验坍塌
            kl_free_bits_penalty = F.relu(self.kl_free_bits_target - loss_kl)
            loss_total = (
                LOSS_WEIGHT_MSE * loss_recon
                + kl_weight * loss_kl
                + KL_FREE_BITS_WEIGHT * kl_free_bits_penalty
                + LOSS_WEIGHT_GEO * loss_geo
            )

            # 梯度累积
            loss_total = loss_total / self.gradient_accumulation_steps

            # 反向传播
            loss_total.backward()

            # 每 accumulation_steps 步更新一次权重
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 更新统计
            total_loss += loss_total.item() * self.gradient_accumulation_steps
            total_loss_recon += loss_recon.item()
            total_loss_kl += loss_kl.item()
            total_loss_geo += loss_geo.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / num_batches:.4f}",
                    "recon": f"{total_loss_recon / num_batches:.4f}",
                    "geo": f"{total_loss_geo / num_batches:.4f}",
                }
            )

            # 定期保存检查点
            if self.global_step % CHECKPOINT_STEPS == 0:
                self.save_checkpoint(epoch, is_lastest=False)

        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_loss_recon = total_loss_recon / num_batches
        avg_loss_kl = total_loss_kl / num_batches
        avg_loss_geo = total_loss_geo / num_batches

        return {
            "epoch": epoch,
            "loss": avg_loss,
            "loss_mse": avg_loss_recon,
            "loss_kl": avg_loss_kl,
            "loss_geo": avg_loss_geo,
        }

    @torch.no_grad()
    def visualize_epoch(self, epoch: int):
        """
        生成训练效果图（单样本重建对比 + loss 曲线）

        Args:
            epoch: 当前 epoch 编号
        """
        self.vae.eval()

        val = self.val_samples.to(self.device)
        recon = self.vae(val, return_recon_only=True)

        # 转为 numpy
        orig_np = val.squeeze().cpu().numpy()
        recon_np = recon.squeeze().cpu().numpy()
        error_np = np.abs(orig_np - recon_np)

        # 历史数据
        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_mse = [h[2] for h in self.loss_history]
        losses_kl = [h[3] for h in self.loss_history]
        losses_geo = [h[4] for h in self.loss_history]

        # ---- 绘图 ----
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"HeightMapVAE Training — Epoch {epoch}", fontsize=14, fontweight="bold"
        )

        # 第 1 行：4 个 Loss 曲线（1×4）
        ax1 = fig.add_subplot(3, 4, 1)
        ax1.plot(epochs, losses_total, "b-", linewidth=0.8)
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(3, 4, 2)
        ax2.plot(epochs, losses_mse, "g-", linewidth=0.8)
        ax2.set_title("MSE Loss")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(epochs, losses_kl, "m-", linewidth=0.8)
        ax3.set_title("KL Loss")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(epochs, losses_geo, "r-", linewidth=0.8)
        ax4.set_title("Geo Loss")
        ax4.set_xlabel("Epoch")
        ax4.grid(True, alpha=0.3)

        # 第 2 行：原始 / 重建 / 误差热力图（1×3，位置 5-7）
        ax5 = fig.add_subplot(3, 4, (5, 6))
        im5 = ax5.imshow(orig_np, cmap="terrain")
        ax5.set_title("Original")
        plt.colorbar(im5, ax=ax5, fraction=0.046)

        ax6 = fig.add_subplot(3, 4, (7, 8))
        im6 = ax6.imshow(recon_np, cmap="terrain")
        ax6.set_title("Reconstructed")
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        ax7 = fig.add_subplot(3, 4, (9, 10))
        im7 = ax7.imshow(error_np, cmap="hot")
        ax7.set_title(f"|Error|  (max={error_np.max():.3f})")
        plt.colorbar(im7, ax=ax7, fraction=0.046)

        # 第 2 行右侧：高程剖线 + 误差统计
        ax8 = fig.add_subplot(3, 4, (11, 12))
        center_row = orig_np.shape[0] // 2
        x = np.arange(orig_np.shape[1])
        ax8.plot(
            x, orig_np[center_row, :], "b-", linewidth=0.6, label="Original", alpha=0.8
        )
        ax8.plot(
            x,
            recon_np[center_row, :],
            "orange",
            linewidth=0.6,
            label="Reconstructed",
            alpha=0.8,
        )
        ax8.set_title(f"Elevation Profile (row {center_row})")
        ax8.set_xlabel("Pixel")
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

        # 误差统计文字（剖线图上叠加）
        mae = error_np.mean()
        max_err = error_np.max()
        ax8.text(
            0.02,
            0.98,
            f"MAE: {mae:.5f}\nMax Error: {max_err:.5f}",
            transform=ax8.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        save_path = self.viz_output_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(save_path, dpi=100)
        plt.close(fig)

        self.vae.train()

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        完整训练循环

        Args:
            num_epochs: 目标训练轮数
            start_epoch: 起始 epoch 编号（用于断点续训）
        """
        if start_epoch > 0:
            print(f"断点续训：从 Epoch {start_epoch} 开始，目标 {num_epochs} epochs")
        else:
            print(f"开始训练：{num_epochs} epochs")
        print(f"设备：{self.device}")
        print(
            f"批次大小：{BATCH_SIZE}"
            + (
                f" (有效: {BATCH_SIZE * self.gradient_accumulation_steps})"
                if self.gradient_accumulation_steps > 1
                else ""
            )
        )
        print(f"学习率：{LEARNING_RATE}")
        print(f"梯度累积步数：{self.gradient_accumulation_steps}")
        print(
            f"损失权重：MSE={LOSS_WEIGHT_MSE}, KL={LOSS_WEIGHT_KL}, GEO={LOSS_WEIGHT_GEO}"
        )
        if self.kl_annealing_epochs > 0:
            print(f"KL 退火：{self.kl_annealing_epochs} epochs 内从 0 线性增长")
        print("-" * 60)

        for epoch in range(start_epoch, num_epochs):
            # 训练一个 epoch
            loss_dict = self.train_epoch(epoch)

            # 学习率调度（warmup 后开始）
            if epoch >= WARMUP_EPOCHS:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # 打印日志
            if epoch % LOG_STEPS == 0 or epoch == num_epochs - 1:
                print(
                    f"\nEpoch {epoch:3d} | "
                    f"Loss: {loss_dict['loss']:.4f} | "
                    f"MSE: {loss_dict['loss_mse']:.4f} | "
                    f"KL: {loss_dict['loss_kl']:.4f} | "
                    f"Geo: {loss_dict['loss_geo']:.4f} | "
                    f"LR: {current_lr:.6f}"
                    + (
                        f" | KL_w: {self.get_kl_weight(epoch):.2e}"
                        if self.kl_annealing_epochs > 0
                        else ""
                    )
                )

            # 保存最佳模型
            if loss_dict["loss"] < self.best_loss:
                self.best_loss = loss_dict["loss"]
                self.save_checkpoint(epoch, is_best=True)

            # 记录 loss 历史并生成效果图
            self.loss_history.append(
                (
                    epoch,
                    loss_dict["loss"],
                    loss_dict["loss_mse"],
                    loss_dict["loss_kl"],
                    loss_dict["loss_geo"],
                )
            )
            if VIZ_INTERVAL > 0 and (
                epoch % VIZ_INTERVAL == 0 or epoch == num_epochs - 1
            ):
                self.visualize_epoch(epoch)

        # 保存最终检查点
        self.save_checkpoint(num_epochs - 1, is_final=True)
        print(f"\n训练完成！最佳 Loss: {self.best_loss:.4f}")

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_final: bool = False,
        is_lastest: bool = True,
    ):
        """
        保存检查点

        Args:
            epoch: 当前 epoch
            is_best: 是否是最佳模型
            is_final: 是否是最终检查点
            is_lastest: 是否是最新检查点
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.vae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "loss_history": self.loss_history,
        }

        # 保存文件名
        if is_final:
            filename = "final_checkpoint.pt"
        elif is_lastest:
            filename = "checkpoint.pt"
        else:
            filename = f"checkpoint_step{self.global_step}.pt"

        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
        Returns:
            start_epoch: 应从该 epoch 继续训练（已加载 epoch 的下一个）
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.vae.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        if "loss_history" in checkpoint:
            self.loss_history = checkpoint["loss_history"]

        saved_epoch = checkpoint["epoch"]
        start_epoch = saved_epoch + 1

        print(f"加载检查点：{checkpoint_path}")
        print(f"  已完成 Epoch: {saved_epoch}")
        print(f"  将从 Epoch {start_epoch} 继续训练")
        print(f"  Global Step: {checkpoint['global_step']}")
        print(f"  Best Loss: {checkpoint['best_loss']:.4f}")

        return start_epoch


def load_norm_params(data_root: str) -> dict | None:
    """加载预处理阶段的归一化参数（log 变换参数）"""
    params_path = os.path.join(data_root, "norm_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            return json.load(f)
    return None


def denormalize_to_elevation(norm_arr: np.ndarray, params: dict) -> np.ndarray:
    """
    将对数变换归一化后的值反归一化到物理高程（uint16 米）
    与 scripts/data_process/preprocess/preprocess_heightmaps.py 的 denormalize() 一致

    h_log = norm_val * (max_log - min_log) + min_log
    h = exp(h_log) + p_low - 1
    """
    log_h = norm_arr * (params["max_log"] - params["min_log"]) + params["min_log"]
    h = np.exp(log_h) + params["p_low"] - 1
    return np.round(np.clip(h, 0, 65535)).astype(np.uint16)


@torch.no_grad()
def test_reconstruction(
    vae: HeightMapVAE,
    data_root: str,
    output_dir: str,
    device: str = "cuda",
    num_samples: int = 5,
):
    """
    测试重建质量

    Args:
        vae: VAE 模型
        data_root: 数据目录
        output_dir: 输出目录
        device: 运行设备
        num_samples: 测试样本数
    """
    print("\n=== 测试重建质量 ===")

    vae.eval()
    vae.to(device)

    # 加载归一化参数（用于反归一化到物理高程）
    norm_params = load_norm_params(data_root)
    if norm_params:
        print(
            f"已加载归一化参数: p_low={norm_params['p_low']:.1f}, p_high={norm_params['p_high']:.1f}"
        )
    else:
        print(
            "警告：未找到 norm_params.json，将使用原始 VAE denormalize_height（×3000）"
        )

    # 创建测试数据集（不使用增强）
    dataset = HeightMapDataset(
        data_root=data_root,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 测试前 num_samples 个样本
    for i in range(min(num_samples, len(dataset))):
        height_map, info = dataset[i]
        height_map = height_map.unsqueeze(0).to(device)  # 添加 batch 维度

        # 重建
        recon = vae(height_map, return_recon_only=True)

        # 反归一化到真实高程
        if norm_params:
            height_map_np = height_map.squeeze().cpu().numpy()
            recon_np = recon.squeeze().cpu().numpy()
            height_map_real = denormalize_to_elevation(height_map_np, norm_params)
            recon_real = denormalize_to_elevation(recon_np, norm_params)
            # 转回 tensor 用于计算误差
            height_map_real_t = torch.from_numpy(height_map_real.astype(np.float32)).to(
                device
            )
            recon_real_t = torch.from_numpy(recon_real.astype(np.float32)).to(device)
        else:
            height_map_real_t = HeightMapVAE.denormalize_height(height_map)
            recon_real_t = HeightMapVAE.denormalize_height(recon)
            height_map_real = height_map_real_t.squeeze().cpu().numpy()
            recon_real = recon_real_t.squeeze().cpu().numpy()

        # 计算误差
        mae = torch.abs(height_map_real_t - recon_real_t).mean().item()
        max_error = torch.abs(height_map_real_t - recon_real_t).max().item()

        print(f"\n样本 {i + 1}: {info['file_path']}")
        print(f"  MAE: {mae:.2f} m")
        print(f"  Max Error: {max_error:.2f} m")

        # 保存结果（.npz 格式）
        output_file = output_path / f"test_{i:03d}.npz"
        np.savez(
            output_file,
            original=height_map_real,
            reconstruction=recon_real,
            error=np.abs(
                height_map_real.astype(np.float32) - recon_real.astype(np.float32)
            ),
        )
        print(f"  保存到：{output_file}")

    print("\n测试完成！")


def main():
    parser = argparse.ArgumentParser(description="高度图 VAE 训练")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="运行模式：train 或 test",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
        help="数据根目录",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="训练轮数",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="批次大小",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="检查点路径（用于恢复训练或测试）",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="输出目录",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="测试样本数",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # ==================== 训练模式 ====================
        print("=== 高度图 VAE 训练 ===")

        # 创建数据集（先获取完整文件列表以支持确定性拆分）
        full_file_list = sorted(
            glob.glob(os.path.join(args.data_root, "hmap_*.npy"))
        )
        generator = torch.Generator().manual_seed(SEED)
        val_size = max(1, int(len(full_file_list) * VAL_SPLIT))
        train_size = len(full_file_list) - val_size
        train_indices, val_indices = random_split(
            range(len(full_file_list)),
            [train_size, val_size],
            generator=generator,
        )
        train_file_list = [full_file_list[i] for i in train_indices.indices]
        val_file_list = [full_file_list[i] for i in val_indices.indices]
        print(
            f"数据集划分：训练集 {len(train_file_list)} 样本, 验证集 {len(val_file_list)} 样本 (split={VAL_SPLIT})"
        )

        # 创建训练数据加载器
        train_dataset = HeightMapDataset(
            data_root=args.data_root,
            image_size=IMAGE_SIZE,
            augment=True,
            file_list=train_file_list,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # 创建验证数据加载器
        val_dataset = HeightMapDataset(
            data_root=args.data_root,
            image_size=IMAGE_SIZE,
            augment=False,
            file_list=val_file_list,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # 创建模型（支持梯度检查点）
        vae = HeightMapVAE(
            block_out_channels=BLOCK_OUT_CHANNELS,
            enable_grad_checkpointing=ENABLE_GRAD_CHECKPOINTING,
        )

        # 创建训练器
        trainer = Trainer(
            vae=vae,
            train_dataloader=train_dataloader,
            output_dir=args.output_dir,
            device=DEVICE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            kl_annealing_epochs=KL_ANNEALING_EPOCHS,
            use_huber_loss=USE_HUBER_LOSS,
            kl_free_bits_per_dim=KL_FREE_BITS_PER_DIM,
            val_dataloader=val_dataloader,
        )

        # 恢复训练（如果有检查点）
        start_epoch = 0
        if args.checkpoint:
            start_epoch = trainer.load_checkpoint(args.checkpoint)

        # 开始训练
        trainer.train(args.epochs, start_epoch=start_epoch)

    elif args.mode == "test":
        # ==================== 测试模式 ====================
        print("=== 高度图 VAE 测试 ===")

        if not args.checkpoint:
            print("错误：测试模式需要指定 --checkpoint 参数")
            return

        # 创建模型
        vae = HeightMapVAE()

        # 加载检查点
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        vae.load_state_dict(checkpoint["model_state_dict"])

        # 测试重建质量
        test_reconstruction(
            vae=vae,
            data_root=args.data_root,
            output_dir=os.path.join(args.output_dir, "test_results"),
            device=DEVICE,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
