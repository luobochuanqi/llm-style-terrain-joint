"""
高度图 VAE 训练脚本

Phase 3: 训练高度图专用 VAE
目标：解决高程数据压缩失真的问题，确保陡崖边缘清晰

用法：
    # 训练
    python scripts/train_height_vae.py --data_root ./data/height_maps --epochs 100

    # 测试重建质量
    python scripts/train_height_vae.py --mode test --checkpoint ./outputs/height_vae/checkpoint.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 导入项目模块
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae.heightmap_vae import HeightMapVAE
from data.height_dataset import HeightMapDataset


# =============================================================================
# 训练配置（硬编码）
# =============================================================================

# 数据配置
DATA_ROOT = "./data/height_maps"
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 4

# 模型配置
H_MAX = 3000.0  # 全局最大高程

# 优化器配置
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

# 训练配置
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

# 损失权重
LOSS_WEIGHT_MSE = 1.0
LOSS_WEIGHT_KL = 1e-6
LOSS_WEIGHT_GEO = 0.8

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出配置
OUTPUT_DIR = "./outputs/height_vae"
CHECKPOINT_STEPS = 1000
LOG_STEPS = 10


class Trainer:
    """高度图 VAE 训练器"""

    def __init__(
        self,
        vae: HeightMapVAE,
        dataloader: DataLoader,
        output_dir: str,
        device: str = "cuda",
    ):
        """
        初始化训练器

        Args:
            vae: 高度图 VAE 模型
            dataloader: 数据加载器
            output_dir: 输出目录
            device: 运行设备
        """
        self.vae = vae.to(device)
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.device = device

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

    def compute_loss(
        self,
        height_map: torch.Tensor,
        recon: torch.Tensor,
        posterior,
    ) -> dict:
        """
        计算综合损失

        L = L_mse + 1e-6 * L_kl + 0.8 * L_geo

        Args:
            height_map: 原始高度图
            recon: 重建高度图
            posterior: VAE 编码器输出的后验分布
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # MSE 损失（像素级重建损失）
        loss_mse = F.mse_loss(recon, height_map)

        # KL 散度损失（正则化隐空间）
        loss_kl = posterior.kl().mean()

        # 地形几何约束损失（保持陡崖边缘）
        loss_geo = self.vae.compute_geo_loss(recon, height_map)

        # 组合损失
        loss_total = (
            LOSS_WEIGHT_MSE * loss_mse
            + LOSS_WEIGHT_KL * loss_kl
            + LOSS_WEIGHT_GEO * loss_geo
        )

        return {
            "loss": loss_total.item(),
            "loss_mse": loss_mse.item(),
            "loss_kl": loss_kl.item(),
            "loss_geo": loss_geo.item(),
        }

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
        total_loss_mse = 0.0
        total_loss_kl = 0.0
        total_loss_geo = 0.0
        num_batches = 0

        # 进度条
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (height_maps, _) in enumerate(pbar):
            # 数据移到设备
            height_maps = height_maps.to(self.device)

            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播
            recon, loss_dict = self.vae(height_maps, return_recon_only=False)

            # 提取损失
            loss = loss_dict["loss"]

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.vae.parameters(),
                GRAD_CLIP,
            )

            # 更新权重
            self.optimizer.step()

            # 更新统计
            total_loss += loss_dict["loss"].item()
            total_loss_mse += loss_dict["loss_mse"].item()
            total_loss_kl += loss_dict["loss_kl"].item()
            total_loss_geo += loss_dict["loss_geo"].item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['loss'].item():.4f}",
                    "mse": f"{loss_dict['loss_mse'].item():.4f}",
                    "geo": f"{loss_dict['loss_geo'].item():.4f}",
                }
            )

            # 定期保存检查点
            if self.global_step % CHECKPOINT_STEPS == 0:
                self.save_checkpoint(epoch, is_lastest=False)

        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_loss_mse = total_loss_mse / num_batches
        avg_loss_kl = total_loss_kl / num_batches
        avg_loss_geo = total_loss_geo / num_batches

        return {
            "epoch": epoch,
            "loss": avg_loss,
            "loss_mse": avg_loss_mse,
            "loss_kl": avg_loss_kl,
            "loss_geo": avg_loss_geo,
        }

    def train(self, num_epochs: int):
        """
        完整训练循环

        Args:
            num_epochs: 训练轮数
        """
        print(f"开始训练：{num_epochs} epochs")
        print(f"设备：{self.device}")
        print(f"批次大小：{BATCH_SIZE}")
        print(f"学习率：{LEARNING_RATE}")
        print(
            f"损失权重：MSE={LOSS_WEIGHT_MSE}, KL={LOSS_WEIGHT_KL}, GEO={LOSS_WEIGHT_GEO}"
        )
        print("-" * 60)

        for epoch in range(num_epochs):
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
                )

            # 保存最佳模型
            if loss_dict["loss"] < self.best_loss:
                self.best_loss = loss_dict["loss"]
                self.save_checkpoint(epoch, is_best=True)

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

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.vae.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"加载检查点：{checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Global Step: {checkpoint['global_step']}")
        print(f"  Best Loss: {checkpoint['best_loss']:.4f}")


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
        height_map_real = HeightMapVAE.denormalize_height(height_map)
        recon_real = HeightMapVAE.denormalize_height(recon)

        # 计算误差
        mae = torch.abs(height_map_real - recon_real).mean().item()
        max_error = torch.abs(height_map_real - recon_real).max().item()

        print(f"\n样本 {i + 1}: {info['file_path']}")
        print(f"  MAE: {mae:.2f} m")
        print(f"  Max Error: {max_error:.2f} m")

        # 保存结果（.npy 格式）
        output_file = output_path / f"test_{i:03d}.npz"
        np.savez(
            output_file,
            original=height_map_real.squeeze().cpu().numpy(),
            reconstruction=recon_real.squeeze().cpu().numpy(),
            error=torch.abs(height_map_real - recon_real).squeeze().cpu().numpy(),
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

        # 创建数据加载器
        dataloader = DataLoader(
            HeightMapDataset(
                data_root=args.data_root,
                image_size=IMAGE_SIZE,
                augment=True,  # 训练时使用增强
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # 创建模型
        vae = HeightMapVAE()

        # 创建训练器
        trainer = Trainer(
            vae=vae,
            dataloader=dataloader,
            output_dir=args.output_dir,
            device=DEVICE,
        )

        # 恢复训练（如果有检查点）
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # 开始训练
        trainer.train(args.epochs)

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
