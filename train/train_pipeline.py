"""
8 通道去噪模型训练流水线，支持 UNet 和 DiT 两种架构（通过 --model_type 切换）。

=== 使用方式 ===

通过训练入口脚本调用（推荐）：
    # UNet 训练
    python scripts/unet/train_unet_full.py --epochs 50 --data_root ./data/unet_training

    # DiT 训练
    python scripts/dit/train_dit_full.py --epochs 50 --data_root ./data/unet_training

也可通过代码直接调用：
    import argparse
    from train.train_pipeline import UNetTrainingPipeline

    args = argparse.Namespace(
        data_root="./data/unet_training",
        output_dir="./outputs/unet_8ch",
        epochs=50, batch_size=4, learning_rate=1e-4, weight_decay=1e-4,
        num_workers=4, save_steps=1000, viz_interval=1, warmup_epochs=5,
        use_amp=True, dem_vae_ckpt="", model_type="unet",  # model_type="dit" 切换 DiT
    )
    pipeline = UNetTrainingPipeline(args)
    pipeline.train()

=== 关键参数 ===

--model_type str      去噪模型类型，"unet" (默认) 或 "dit"
--data_root str       数据目录，含 rgb/、dem/、txt/ 三个子目录
--epochs int          训练 epoch 数
--batch_size int      批次大小 (UNet 默认 4, DiT 默认 2)
--learning_rate float 学习率 (默认 1e-4)
--use_amp             启用 AMP fp16 混合精度 (DiT 默认开启)
--dem_vae_ckpt str    高度图 VAE checkpoint 路径 (可选)
--checkpoint str      断点续训或测试用的 checkpoint 路径
--mode train|test     运行模式

DiT 专属参数 (通过 scripts/dit/train_dit_full.py 传入):
--pretrained_path str PixArt-alpha 预训练权重路径 (默认 HuggingFace)
--stage1_epochs int   Stage 1 burn-in epoch 数 (默认 10, 仅训练适配层)

=== 训练流程 ===

1. 构建模型：CLIP 文本编码器 (冻结) + SD VAE (冻结) + 高度图 VAE (可选, 冻结) + 去噪模型
2. 编码阶段：RGB/DEM 像素 → VAE → 8ch 联合隐向量 [B,8,64,64]；文本 → CLIP → 全局+局部特征
3. 去噪训练：DDPM 前向加噪 → 去噪模型预测噪声 → 计算 MSE 损失 → 反向传播
4. 检查点：每 epoch 保存 checkpoint.pt (最新) 和 best_checkpoint.pt (最优)
5. 日志：CSV (training_log.csv) + loss 曲线图 (visualizations/)

=== 分阶段训练 (DiT) ===

Stage 1 (burn-in): 冻结 DiT backbone，仅训练 CLIP 适配层 (global_text_proj, local_text_proj)
Stage 2 (全量微调): 解冻所有参数，适配层学习率 10×
Stage 切换由 scripts/dit/train_dit_full.py 自动管理，本 Pipeline 通过 end_epoch 参数配合。

=== 注意事项 ===

- 属性名 self.unet 同时用于 UNet 和 DiT，确保训练循环代码不改动
- AMP + grad_checkpointing 互斥，二者只能选一
- 梯度包含 NaN/Inf 时自动跳过该 batch 的更新
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from PIL import Image

from diffusers import DDPMScheduler, AutoencoderKL, DDIMScheduler
from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging
import torchvision.utils as vutils

from dataset.unet_dataset import UNetDataset
from models.clip.text_encoder import build_text_encoder
from models.unet.unet_8ch import build_unet
from models.vae.heightmap_vae import HeightMapVAE

diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


class UNetTrainingPipeline:
    """
    8 通道 U-Net 训练流水线。

    负责构建所有子模块、管理训练循环、checkpoint 持久化和可视化。

    参数
    ----------
    args : argparse.Namespace
        命令行参数，需包含 data_root, dem_vae_ckpt, output_dir, epochs,
        batch_size, learning_rate, weight_decay, num_workers, save_steps,
        viz_interval, use_amp 等字段。
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.viz_output_dir = self.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.loss_history: list[tuple] = []       # [(epoch, total, img, dem), ...]
        self.global_step = 0
        self.best_loss = float("inf")

        # 构建各组件
        self._build_models()
        self._build_optimizer_and_scheduler()
        self._build_scaler()
        self._build_dataloader()

        params_file = os.path.join("./data/process/heightmaps_hf", "norm_params.json")
        with open(params_file, "r") as f:
            self.norm_params = json.load(f)

        print(f"设备: {self.device} | 混合精度: {args.use_amp}")
        print(f"数据集: {args.data_root} | 批次: {args.batch_size} | 目标轮数: {args.epochs}")
        print(f"学习率: {args.learning_rate} | 权重衰减: {args.weight_decay}")
        print("-" * 50)

    # -------------------------------------------------------------------------
    # 构建组件
    # -------------------------------------------------------------------------

    def _build_models(self) -> None:
        """构建 U-Net、CLIP 编码器和 VAE 编码器。"""
        # 去噪模型（U-Net 或 DiT）
        model_type = getattr(self.args, "model_type", "unet")
        if model_type == "dit":
            from models.dit.dit_8ch import build_dit
            pretrained_path = getattr(self.args, "dit_pretrained_path", None)
            print("构建 DiT 8 通道去噪模型...")
            self.unet = build_dit(
                pretrained_path=pretrained_path,
                in_channels=8,
                out_channels=8,
                cross_attention_dim=768,
            ).to(self.device)
        else:
            print("构建 8 通道 U-Net...")
            self.unet = build_unet(
                in_channels=8, out_channels=8, cross_attention_dim=768
            ).to(self.device)

        # CLIP 编码器（冻结）
        print("加载 CLIP 文本编码器...")
        self.text_encoder = build_text_encoder(
            "openai/clip-vit-large-patch14"
        ).to(self.device)

        # RGB VAE（冻结）
        print("加载 SD VAE...")
        self.rgb_vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        ).to(self.device)
        self.rgb_vae.eval()
        self.rgb_vae.requires_grad_(False)

        # 高度图 VAE（可选，冻结）
        if self.args.dem_vae_ckpt:
            print("加载高度图 VAE...")
            self.dem_vae = HeightMapVAE(
                block_out_channels=(128, 256, 512, 512)
            ).to(self.device)
            ckpt = torch.load(self.args.dem_vae_ckpt, map_location=self.device)
            self.dem_vae.load_state_dict(ckpt["model_state_dict"])
            self.dem_vae.eval()
            self.dem_vae.requires_grad_(False)
        else:
            self.dem_vae = None
            print("未提供高度图 VAE 权重，DEM 编码将使用 RGB VAE 降级方案。")

    def _build_optimizer_and_scheduler(self) -> None:
        """构建优化器和学习率调度器。"""
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.args.epochs - getattr(self.args, "warmup_epochs", 5)),
            eta_min=self.args.learning_rate * 1e-3,
        )

    def _build_scaler(self) -> None:
        """构建 AMP GradScaler（仅 CUDA 且 use_amp=True 时）。"""
        use_cuda_amp = self.args.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None

    def _build_dataloader(self) -> None:
        """构建数据集/DataLoader，固定验证批次。"""
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        dataset = UNetDataset(data_root=self.args.data_root, augment=True)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # 固定验证批次
        val_batch = next(iter(self.dataloader))
        self.val_rgb = val_batch["rgb"][:1].to(self.device)
        self.val_dem = val_batch["dem"][:1].to(self.device)
        self.val_prompt = val_batch["prompt"][:1]

    # -------------------------------------------------------------------------
    # 编码工具
    # -------------------------------------------------------------------------

    def encode_to_latent(
        self, rgb_pixels: torch.Tensor, dem_pixels: torch.Tensor
    ) -> torch.Tensor:
        """
        将 RGB 与 DEM 像素编码为 8 通道联合隐向量。

        RGB [B,3,512,512] → VAE → [B,4,64,64]  ×0.18215
        DEM [B,1,512,512] → VAE → [B,4,64,64]  ×0.18215
        → concat → [B,8,64,64]
        """
        rgb_latent = self.rgb_vae.encode(rgb_pixels).latent_dist.sample() * 0.18215

        if self.dem_vae is not None:
            dem_latent = self.dem_vae.encode(dem_pixels).latent_dist.sample()
        else:
            dem_pixels_3ch = dem_pixels.repeat(1, 3, 1, 1)
            dem_latent = self.rgb_vae.encode(dem_pixels_3ch).latent_dist.sample() * 0.18215

        return torch.cat([rgb_latent, dem_latent], dim=1)

    def encode_text(
        self, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        调用 CLIP 编码器获取全局特征和序列特征。
        返回 (pooled [B,768], hidden_states [B,77,768])。
        """
        return self.text_encoder(prompts)

    # -------------------------------------------------------------------------
    # 验证
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def compute_validation_loss(self) -> dict:
        """在固定验证批次上计算 loss，不更新参数。"""
        self.unet.eval()

        with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
            pooled, hidden = self.encode_text(self.val_prompt)
            latents = self.encode_to_latent(self.val_rgb, self.val_dem)
            noise = torch.randn_like(latents)
            b = latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (b,), device=self.device
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = self.unet(
                noisy_latent=noisy_latents,
                timestep=timesteps,
                global_features=pooled,
                local_features=hidden,
            )
            loss_dict = self.unet.loss(noise_pred, noise)

        self.unet.train()
        return loss_dict

    # -------------------------------------------------------------------------
    # 训练一个 epoch
    # -------------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        """执行一个 epoch 的训练，返回平均 loss 字典。"""
        self.unet.train()

        total_loss, total_img_loss, total_dem_loss = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch:3d}", leave=False)

        cfg_drop_rate = getattr(self.args, "cfg_drop_rate", 0.1)
        min_snr_gamma = getattr(self.args, "min_snr_gamma", 5.0)

        for batch in pbar:
            rgb_pixels = batch["rgb"].to(self.device)
            dem_pixels = batch["dem"].to(self.device)
            prompts = batch["prompt"]
            b = rgb_pixels.shape[0]

            cfged_prompts = []
            import random
            for p in prompts:
                if random.random() < cfg_drop_rate:
                    cfged_prompts.append("")
                else:
                    cfged_prompts.append(p)

            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                with torch.no_grad():
                    pooled, hidden = self.encode_text(cfged_prompts)
                    latents = self.encode_to_latent(rgb_pixels, dem_pixels)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (b,), device=self.device
                ).long()
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                noise_pred = self.unet(
                    noisy_latent=noisy_latents,
                    timestep=timesteps,
                    global_features=pooled,
                    local_features=hidden,
                )
                
                alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
                alpha_prod_t = alphas_cumprod[timesteps]
                beta_prod_t = 1 - alpha_prod_t
                snr = alpha_prod_t / beta_prod_t
                
                # 计算 Min-SNR 权重 (如果设为 0 则退化为普通的 1.0 权重)
                if min_snr_gamma > 0:
                    min_snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr
                else:
                    min_snr_weight = torch.ones_like(snr)

                # 分别计算高低频特征的 Loss
                loss_img_none = F.mse_loss(noise_pred[:, :4, :, :], noise[:, :4, :, :], reduction="none")
                loss_dem_none = F.mse_loss(noise_pred[:, 4:, :, :], noise[:, 4:, :, :], reduction="none")
                
                # [B] 维度的平均
                loss_img_batch = loss_img_none.mean(dim=[1, 2, 3])
                loss_dem_batch = loss_dem_none.mean(dim=[1, 2, 3])
                
                # 按 1.5 倍权重组合，并施加 Min-SNR 抑制过大梯度
                loss_batch = loss_img_batch + loss_dem_batch * 1.5
                loss = (loss_batch * min_snr_weight).mean()

                # 打包供日志记录
                loss_dict = {
                    "loss": loss,
                    "loss_img": loss_img_batch.mean(),
                    "loss_dem": loss_dem_batch.mean()
                }

            # NaN 检测
            if not torch.isfinite(loss_dict["loss"]):
                tqdm.write(
                    f"[警告] Epoch {epoch}, batch {num_batches}: "
                    f"loss={loss_dict['loss'].item():.4f}，跳过该 batch"
                )
                self.optimizer.zero_grad()
                continue

            # 反向传播
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss_dict["loss"]).backward()
                self.scaler.unscale_(self.optimizer)

                grad_nan = any(
                    not torch.isfinite(p.grad).all()
                    for p in self.unet.parameters()
                    if p.grad is not None
                )
                if grad_nan:
                    tqdm.write(
                        f"[警告] Epoch {epoch}, batch {num_batches}: "
                        f"梯度包含 NaN/Inf，跳过本次更新"
                    )
                    self.optimizer.zero_grad()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    continue

                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict["loss"].backward()

                grad_nan = any(
                    not torch.isfinite(p.grad).all()
                    for p in self.unet.parameters()
                    if p.grad is not None
                )
                if grad_nan:
                    tqdm.write(
                        f"[警告] Epoch {epoch}, batch {num_batches}: "
                        f"梯度包含 NaN/Inf，跳过本次更新"
                    )
                    self.optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_img_loss += loss_dict["loss_img"].item()
            total_dem_loss += loss_dict["loss_dem"].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "img":  f"{total_img_loss / num_batches:.4f}",
                "dem":  f"{total_dem_loss / num_batches:.4f}",
            })

            if self.global_step % self.args.save_steps == 0:
                self.save_checkpoint(epoch, is_latest=True)

        return {
            "loss": total_loss / num_batches,
            "img":  total_img_loss / num_batches,
            "dem":  total_dem_loss / num_batches,
        }

    # -------------------------------------------------------------------------
    # Checkpoint 管理（仅保留 latest + best）
    # -------------------------------------------------------------------------

    def save_checkpoint(
        self, epoch: int, is_best: bool = False, is_latest: bool = False
    ) -> None:
        """保存 checkpoint，仅保留 latest (checkpoint.pt) 和 best (best_checkpoint.pt)。"""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.unet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "loss_history": self.loss_history,
        }
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        if is_latest:
            torch.save(ckpt, self.output_dir / "checkpoint.pt")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_checkpoint.pt")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载 checkpoint 并恢复训练状态。

        返回
        -------
        int
            下一个应训练的 epoch 编号。
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.unet.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_loss = ckpt["best_loss"]
        if "loss_history" in ckpt:
            self.loss_history = ckpt["loss_history"]
        if "scaler_state_dict" in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        saved_epoch = ckpt["epoch"]
        start_epoch = saved_epoch + 1

        print(f"加载 checkpoint: {checkpoint_path}")
        print(f"  已完成 Epoch: {saved_epoch}")
        print(f"  将从 Epoch {start_epoch} 继续训练")
        print(f"  Global Step: {ckpt['global_step']}")
        print(f"  Best Loss: {ckpt['best_loss']:.4f}")

        return start_epoch

    # -------------------------------------------------------------------------
    # 可视化
    # -------------------------------------------------------------------------

    def visualize_epoch(self, epoch: int) -> None:
        """绘制 loss 曲线（总损失 / RGB / DEM）。"""
        if not self.loss_history:
            return

        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_img = [h[2] for h in self.loss_history]
        losses_dem = [h[3] for h in self.loss_history]
        lr_vals = [
            self.scheduler.get_last_lr()[0]
            for _ in range(len(self.loss_history))
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"UNet 8-Channel Training — Epoch {epoch}",
            fontsize=14, fontweight="bold",
        )

        ax0 = axes[0, 0]
        ax0.plot(epochs, losses_total, "b-", linewidth=1.0)
        ax0.set_title("Total Loss")
        ax0.set_xlabel("Epoch")
        ax0.grid(True, alpha=0.3)

        ax1 = axes[0, 1]
        ax1.plot(epochs, losses_img, "g-", linewidth=1.0)
        ax1.set_title("RGB Texture Loss")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1, 0]
        ax2.plot(epochs, losses_dem, "r-", linewidth=1.0)
        ax2.set_title("DEM Height Loss")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 1]
        ax3.plot(epochs[-len(lr_vals):], lr_vals, "m-", linewidth=1.0)
        ax3.set_title("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)

        # 统计标注
        if losses_total:
            ax0.text(
                0.02, 0.98,
                f"Best: {min(losses_total):.4f}\nLast: {losses_total[-1]:.4f}",
                transform=ax0.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

        plt.tight_layout()
        fig.savefig(self.viz_output_dir / f"loss_epoch_{epoch:04d}.png", dpi=100)
        plt.close(fig)

    @torch.no_grad()
    def generate_validation_samples(self, epoch_or_name: str | int = "test_mode", guidance_scale: float = 5.5, num_inference_steps: int = 50) -> None:
        """执行 50 步 DDIM 反向去噪，生成并保存验证图片。"""
        self.unet.eval()
        tqdm.write(f"\n正在生成视觉验证图 (DDIM {num_inference_steps}步, CFG={guidance_scale})...")

        # 1. 准备条件与无条件文本特征 (CFG)
        prompts = self.val_prompt
        b = len(prompts)
        cond_pooled, cond_hidden = self.encode_text(prompts)
        uncond_pooled, uncond_hidden = self.encode_text([""] * b) # 空字符串用于无条件引导

        # 2. 初始化纯高斯噪声 [B, 8, 64, 64]
        # 注意尺寸是原图 512 的 1/8
        latents = torch.randn(
            (b, 8, self.val_rgb.shape[2] // 8, self.val_rgb.shape[3] // 8),
            device=self.device
        )

        # 3. 设置 DDIM 调度器
        self.infer_scheduler.set_timesteps(num_inference_steps, device=self.device)

        p_low, min_log, max_log = self.norm_params["p_low"], self.norm_params["min_log"], self.norm_params["max_log"]

        # 4. 反向去噪循环
        for t in tqdm(self.infer_scheduler.timesteps, desc="Sampling", leave=False):
            # 将 latent 复制一份，一半输入给无条件，一半输入给有条件
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.infer_scheduler.scale_model_input(latent_model_input, t)

            t_input = torch.tensor([t] * (b * 2), device=self.device)
            pooled_input = torch.cat([uncond_pooled, cond_pooled])
            hidden_input = torch.cat([uncond_hidden, cond_hidden])

            # 预测噪声
            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                noise_pred = self.unet(
                    noisy_latent=latent_model_input,
                    timestep=t_input,
                    global_features=pooled_input,
                    local_features=hidden_input
                )

            # 提取无条件和有条件的预测结果，应用 CFG 公式
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 计算上一步的 latent
            latents = self.infer_scheduler.step(noise_pred, t, latents).prev_sample

        # 5. VAE 解码回像素空间
        # 必须除以 SD 固有的缩放因子
        rgb_latent = 1 / 0.18215 * latents[:, :4, :, :]
        dem_latent = latents[:, 4:, :, :]

        # latents = 1 / 0.18215 * latents
        # rgb_latent, dem_latent = latents[:, :4, :, :], latents[:, 4:, :, :]

        with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
            rgb_imgs = self.rgb_vae.decode(rgb_latent).sample
            
            if self.dem_vae is not None:
                dem_imgs = self.dem_vae.decode(dem_latent).sample
            else:
                # 降级方案：用 RGB VAE 解码再转单通道灰度
                dem_imgs = self.rgb_vae.decode(dem_latent).sample
                dem_imgs = dem_imgs.mean(dim=1, keepdim=True)

        # SD VAE 的输出在 [-1, 1] 之间，需要映射到 [0, 1] 以便保存为普通图片
        # 6. 归一化并保存
        rgb_imgs = (rgb_imgs / 2 + 0.5).clamp(0, 1)
        dem_imgs = dem_imgs.clamp(0, 1)

        # 动态处理文件名（如果是数字就加前缀，如果是字符串就直接用）
        if isinstance(epoch_or_name, int):
            prefix = f"epoch_{epoch_or_name:04d}"
        else:
            prefix = str(epoch_or_name)

        out_rgb_path = self.viz_output_dir / f"{prefix}_gen_rgb.png"
        out_dem_path = self.viz_output_dir / f"{prefix}_gen_dem.png"

        gen_rgb_np = (rgb_imgs / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        gen_dem_np = dem_imgs[0][0].cpu().numpy()

        h_real = np.exp(gen_dem_np * (max_log - min_log) + min_log) + p_low - 1
        dem_save_array = np.round(np.clip(h_real, 0, 65535)).astype(np.uint16)
        rgb_save_array = (gen_rgb_np * 255).astype(np.uint8)

        Image.fromarray(dem_save_array).save(out_dem_path)
        Image.fromarray(rgb_save_array).save(out_rgb_path)

        # vutils.save_image(rgb_imgs, out_rgb_path, nrow=b)
        # vutils.save_image(dem_imgs, out_dem_path, nrow=b)
        
        tqdm.write(f"生成完毕！图片已保存至: {self.viz_output_dir}")
        self.unet.train()

    # -------------------------------------------------------------------------
    # 训练主循环
    # -------------------------------------------------------------------------

    def train(self, start_epoch: int = 0, end_epoch: int | None = None) -> None:
        """
        运行完整训练循环。

        参数
        ----------
        start_epoch : int
            起始 epoch 编号（用于断点续训）。
        end_epoch : int | None
            结束 epoch 编号（不包含），None 表示使用 args.epochs。
        """
        if start_epoch > 0:
            print(f"断点续训：从 Epoch {start_epoch} 开始，目标 {self.args.epochs} epochs")
        else:
            print(f"开始训练：{self.args.epochs} epochs")

        warmup_epochs = getattr(self.args, "warmup_epochs", 5)

        # CSV 日志
        log_path = self.output_dir / "training_log.csv"
        is_new_log = not log_path.exists() or log_path.stat().st_size == 0
        log_f = open(log_path, "a")
        log_writer = csv.writer(log_f)
        if is_new_log:
            header = ["epoch", "loss", "loss_img", "loss_dem", "lr"]
            if self.scaler is not None:
                header.append("amp_scale")
            log_writer.writerow(header)
            log_f.flush()

        end = end_epoch if end_epoch is not None else self.args.epochs
        for epoch in range(start_epoch, end):
            avg_loss = self.train_epoch(epoch)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # 学习率调度（warmup 后启动）
            if epoch >= warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # CSV 写入
            row = [
                epoch,
                f"{avg_loss['loss']:.6f}",
                f"{avg_loss['img']:.6f}",
                f"{avg_loss['dem']:.6f}",
                f"{current_lr:.8f}",
            ]
            if self.scaler is not None:
                row.append(f"{self.scaler.get_scale():.0f}")
            log_writer.writerow(row)
            log_f.flush()

            # 控制台输出
            tqdm.write(
                f"Epoch {epoch:3d} | "
                f"Total: {avg_loss['loss']:.4f} | "
                f"Img: {avg_loss['img']:.4f} | "
                f"Dem: {avg_loss['dem']:.4f} | "
                f"LR: {current_lr:.6f}"
                + (
                    f" | Scale: {self.scaler.get_scale():.0f}"
                    if self.scaler is not None
                    else ""
                )
            )

            # 验证 loss（不参与训练，仅观察趋势）
            val_loss = self.compute_validation_loss()
            tqdm.write(
                f"       验证 | "
                f"Total: {val_loss['loss']:.4f} | "
                f"Img: {val_loss['loss_img']:.4f} | "
                f"Dem: {val_loss['loss_dem']:.4f}"
            )

            # 保存最佳模型
            if avg_loss["loss"] < self.best_loss:
                tqdm.write(
                    f"       新的最佳 Loss "
                    f"({self.best_loss:.4f} -> {avg_loss['loss']:.4f})，"
                    f"保存 best_checkpoint.pt"
                )
                self.best_loss = avg_loss["loss"]
                self.save_checkpoint(epoch, is_best=True)

            # 保存最新 checkpoint
            self.save_checkpoint(epoch, is_latest=True)

            # 记录并绘图
            self.loss_history.append(
                (epoch, avg_loss["loss"], avg_loss["img"], avg_loss["dem"])
            )
            if (
                self.args.viz_interval > 0
                and (epoch % self.args.viz_interval == 0 or epoch == self.args.epochs - 1)
            ):
                self.visualize_epoch(epoch)

        log_f.close()
        tqdm.write(f"训练完成！历史最佳 Loss: {self.best_loss:.4f}")


# -------------------------------------------------------------------------
# 测试模式
# -------------------------------------------------------------------------

@torch.no_grad()
def test_noise_prediction(
    pipeline: UNetTrainingPipeline,
    num_samples: int = 10,
) -> None:
    """
    测试 UNet 的噪声预测精度：在验证集上计算平均 MSE。

    适用于训练完成后的快速质量评估，无需完整 DDIM 推理。
    """
    print("\n=== 测试噪声预测精度 ===")
    pipeline.unet.eval()

    total_loss, total_img, total_dem = 0.0, 0.0, 0.0
    count = 0

    for batch in pipeline.dataloader:
        if count >= num_samples:
            break

        rgb_pixels = batch["rgb"].to(pipeline.device)
        dem_pixels = batch["dem"].to(pipeline.device)
        prompts = batch["prompt"]
        b = rgb_pixels.shape[0]

        with torch.autocast(device_type="cuda", enabled=pipeline.args.use_amp):
            pooled, hidden = pipeline.encode_text(prompts)
            latents = pipeline.encode_to_latent(rgb_pixels, dem_pixels)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipeline.noise_scheduler.config.num_train_timesteps,
                (b,), device=pipeline.device
            ).long()
            noisy = pipeline.noise_scheduler.add_noise(latents, noise, timesteps)
            pred = pipeline.unet(
                noisy_latent=noisy,
                timestep=timesteps,
                global_features=pooled,
                local_features=hidden,
            )
            ld = pipeline.unet.loss(pred, noise)

        total_loss += ld["loss"].item()
        total_img  += ld["loss_img"].item()
        total_dem  += ld["loss_dem"].item()
        count += 1

    print(f"测试样本数: {count}")
    print(f"平均 Total Loss: {total_loss / count:.4f}")
    print(f"平均 Img  Loss: {total_img / count:.4f}")
    print(f"平均 Dem  Loss: {total_dem / count:.4f}")
    print("测试完成！")
