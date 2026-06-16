"""
DiT 8 通道联合生成训练与测试脚本

Facebook DiT (adaLN-Zero) + Flow Matching (velocity prediction)。
独立训练器，遵循 UNetTrainer 模式。

数据需求:
    data_root/{rgb/, dem/, txt/} 三元组——
    rgb/:  512×512 RGB 纹理图 (PNG/JPG/NPY)
    dem/:  512×512 DEM 高程图 (PNG/NPY/TIF, [0,1] or raw)
    txt/:  对应文本 prompt

    数据预处理:
    - DEM 需经 scripts/data_process/preprocess/preprocess_heightmaps.py 处理
    - HeightMapVAE 需提前训练: scripts/height_vae/train_height_vae_full.py

数据流:
    输入: (RGB 512×512, DEM 512×512, 文本 prompt)
      → VAE 编码: RGB → latent [4,64,64] (x0.18215)
                   DEM → latent [4,64,64]
      → 拼接 8 通道联合隐向量 [8,64,64] (ch 0-3 RGB, ch 4-7 DEM)
      → CLIP 编码文本 → pooler [768] (adaLN) + hidden [77,768] (cross-attn)
      → Flow Matching 加噪 → DiT 预测速度场 → MSE (DEM x1.5)
      → Euler ODE 去噪推理 (CFG) → VAE 解码 → 纹理图 + 高程图

用法:
    # 训练 (从零开始)
    python scripts/dit/train_dit_full.py --mode train --epochs 50

    # 续训 (从 checkpoint 恢复)
    python scripts/dit/train_dit_full.py --mode train --checkpoint outputs/dit/latest_checkpoint.pt

    # 微调 (仅加载权重，重置优化器)
    python scripts/dit/train_dit_full.py --mode train --checkpoint <path> --finetune

    # 测试 (在验证集上推理)
    python scripts/dit/train_dit_full.py --mode test --checkpoint outputs/dit/best_dit.pt

参数:
    --mode train|test        运行模式 (默认: train)
    --data_root PATH         数据根目录 (默认: ./data/dit_training)
    --dem_vae_ckpt PATH      HeightMapVAE checkpoint (默认: ./data/vae_model_data/best_checkpoint.pt)
    --output_dir PATH        输出目录 (默认: ./outputs/dit)
    --checkpoint PATH        用于续训或测试的 checkpoint 路径
    --epochs N               训练轮数 (默认: 50)
    --batch_size N           批次大小 (默认: 4)
    --learning_rate LR       学习率 (默认: 2e-4)
    --num_workers N          DataLoader 进程数 (默认: 4)
    --num_samples N          测试时生成样本数 (默认: 5)
    --use_amp yes|no         AMP 混合精度 (默认: yes)
    --viz_interval N         可视化间隔 epoch 数 (默认: 1)
    --finetune               微调模式: 仅加载权重，重置优化器和调度器

输出:
    outputs/dit/
      ├── latest_checkpoint.pt    每 epoch 保存
      ├── best_dit.pt             最佳 loss checkpoint
      ├── checkpoint_epoch_XXXX.pt 每 5 epoch 周期保存
      ├── training_log.csv        训练 loss 日志 (epoch,loss,rgb_loss,dem_loss)
      ├── visualizations/
      │   ├── dashboard_epoch_XXXX.png  2x4 仪表盘
      │   ├── latest_output/           最新生成纹理图+高程图
      │   └── prompt.txt
      └── test_results_dit/      测试模式生成结果
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import random
import json
import csv

import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from diffusers import AutoencoderKL
from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging

diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

from models.dit.dit import DiT
from dataset.dit_dataset import DiTDataset
from models.vae.heightmap_vae import HeightMapVAE
from models.clip.text_encoder import build_text_encoder

# 默认配置
DATA_ROOT = "./data/dit_training"
DEM_VAE_CKPT = "./data/vae_model_data/best_checkpoint.pt"
OUTPUT_DIR = "./outputs/dit"
MODELS_DIR = "./data/models"
USE_LOCAL_MODELS = True

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
USE_AMP = True
VAL_SPLIT = 0.05
SEED = 45
CFG_DROP_RATE = 0.1
GUIDANCE_SCALE = 4.0
VIZ_INTERVAL = 1
NUM_INFERENCE_STEPS = 50


# Flow Matching Scheduler
class FlowMatchScheduler:
    """轻量 Flow Matching 调度器: 训练加噪 + 推理 Euler 步进"""

    def __init__(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps

    @staticmethod
    def add_noise(
        clean_latent: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """x_t = (1-t)·x_0 + t·ε"""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * clean_latent + t * noise

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device
    ) -> torch.Tensor:
        """返回从 1 → 0 的均匀时间步序列"""
        self.num_inference_steps = num_inference_steps
        return torch.linspace(1, 0, num_inference_steps + 1, device=device)

    @staticmethod
    def euler_step(
        model_output: torch.Tensor, sample: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """x_{t-Δt} = x_t - v × Δt"""
        return sample - model_output * dt


# DiT Trainer
class DiTTrainer:
    """DiT 训练器，管理训练循环、可视化、checkpoint 和推理生成"""

    def __init__(self, dit, train_dataloader, val_dataloader, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dit = dit
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # 目录
        self.output_dir = Path(args.output_dir)
        self.viz_output_dir = self.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

        # 对数归一化参数
        params_file = os.path.join("./data/process/heightmaps_hf", "norm_params.json")
        with open(params_file, "r") as f:
            self.norm_params = json.load(f)

        # 状态
        self.loss_history = []
        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.use_amp = getattr(args, "use_amp", USE_AMP) and torch.cuda.is_available()

        # 优化器 (单参数组，从零训练无需分层 lr)
        self.optimizer = torch.optim.AdamW(
            dit.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY
        )

        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # LR scheduler
        from diffusers.optimization import get_scheduler

        if self.train_dataloader is not None:
            self.total_steps = self.args.epochs * len(self.train_dataloader)
            self.warmup_steps = int(self.total_steps * 0.05)
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )

        # 加载环境模型
        use_local = getattr(args, "use_local_models", USE_LOCAL_MODELS)
        models_dir = getattr(args, "models_dir", MODELS_DIR)

        if use_local:
            clip_path = os.path.join(models_dir, "clip-vit-large-patch14")
            vae_path = os.path.join(models_dir, "sd-vae-ft-mse")
            print(f"正在从本地加载 CLIP 和 VAE 模型... (models_dir={models_dir})")
        else:
            clip_path = "openai/clip-vit-large-patch14"
            vae_path = "stabilityai/sd-vae-ft-mse"
            print("正在从 HuggingFace Hub 下载/加载 CLIP 和 VAE 模型...")

        self.text_encoder = build_text_encoder(
            model_name=clip_path
        ).to(self.device)
        self.text_encoder.eval()

        self.rgb_vae = AutoencoderKL.from_pretrained(vae_path).to(
            self.device
        )
        self.rgb_vae.eval().requires_grad_(False)

        if args.dem_vae_ckpt:
            self.dem_vae = HeightMapVAE(block_out_channels=(128, 256, 512, 512)).to(
                self.device
            )
            self.dem_vae.load_state_dict(
                torch.load(args.dem_vae_ckpt, map_location=self.device)[
                    "model_state_dict"
                ]
            )
            self.dem_vae.eval().requires_grad_(False)
        else:
            self.dem_vae = None

        self.scheduler = FlowMatchScheduler(NUM_INFERENCE_STEPS)

        # 固定验证样本
        if self.val_dataloader is not None:
            val_batch = next(iter(self.val_dataloader))
            self.val_prompt = val_batch["prompt"][0]
            self.gt_name = val_batch["basename"][0]
            self.val_rgb = val_batch["rgb"][:1].to(self.device)
            self.val_dem = val_batch["dem"][:1].to(self.device)

            prompt_path = self.viz_output_dir / "prompt.txt"
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(self.val_prompt)

    def encode_to_latent(self, rgb_pixels, dem_pixels):
        """VAE 编码: RGB [B,3,512,512] + DEM [B,1,512,512] → joint [B,8,64,64]"""
        rgb_latent = self.rgb_vae.encode(rgb_pixels).latent_dist.sample() * 0.18215
        if self.dem_vae is not None:
            dem_latent = self.dem_vae.encode(dem_pixels).latent_dist.sample()
        else:
            dem_pixels_3ch = dem_pixels.repeat(1, 3, 1, 1)
            dem_latent = (
                self.rgb_vae.encode(dem_pixels_3ch).latent_dist.sample() * 0.18215
            )

        if dem_latent.shape[2:] != rgb_latent.shape[2:]:
            dem_latent = F.interpolate(
                dem_latent,
                size=rgb_latent.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return torch.cat([rgb_latent, dem_latent], dim=1)

    def train_epoch(self, epoch):
        self.dit.train()
        total_loss_epoch, total_rgb_loss, total_dem_loss, num_batches = 0.0, 0.0, 0.0, 0
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            rgb_pixels = batch["rgb"].to(self.device)
            dem_pixels = batch["dem"].to(self.device)
            prompts = batch["prompt"]
            batch_size = rgb_pixels.shape[0]

            # 尺寸对齐
            TARGET_SIZE = (512, 512)
            if rgb_pixels.shape[2:] != TARGET_SIZE:
                rgb_pixels = F.interpolate(
                    rgb_pixels, size=TARGET_SIZE, mode="bilinear", align_corners=False
                )
            if dem_pixels.shape[2:] != TARGET_SIZE:
                dem_pixels = F.interpolate(
                    dem_pixels, size=TARGET_SIZE, mode="bilinear", align_corners=False
                )

            # CFG dropout: 10% 替换为空文本
            cfged_prompts = []
            for p in prompts:
                cfged_prompts.append("" if random.random() < CFG_DROP_RATE else p)

            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                with torch.no_grad():
                    global_features, local_features = self.text_encoder(cfged_prompts)
                    x_0 = self.encode_to_latent(rgb_pixels, dem_pixels)

                # Flow Matching
                t = torch.rand(batch_size, device=self.device)  # [B] ∈ [0,1]
                noise = torch.randn_like(x_0)  # ε ~ N(0,I)
                x_t = self.scheduler.add_noise(x_0, noise, t)  # 加噪
                v_target = noise - x_0  # 目标速度场

                # DiT 预测
                output = self.dit(
                    sample=x_t,
                    timestep=t,
                    encoder_hidden_states=local_features,
                    pooler_output=global_features,
                )
                v_pred = output.sample

                # Loss: RGB + 1.5×DEM (delegates to DiT.loss())
                loss, loss_rgb, loss_dem = self.dit.loss(v_pred, v_target)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.dit.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dit.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.lr_scheduler.step()

            total_loss_epoch += loss.item()
            total_rgb_loss += loss_rgb.item()
            total_dem_loss += loss_dem.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                {
                    "Loss": f"{total_loss_epoch / num_batches:.4f}",
                    "Dem": f"{total_dem_loss / num_batches:.4f}",
                }
            )

        return {
            "loss": total_loss_epoch / num_batches,
            "rgb": total_rgb_loss / num_batches,
            "dem": total_dem_loss / num_batches,
        }

    @torch.no_grad()
    def visualize_epoch(self, epoch: int):
        """2x4 仪表盘: loss 曲线 + GT vs 生成纹理/高程"""
        if len(self.loss_history) == 0:
            return
        self.dit.eval()

        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_rgb = [h[2] for h in self.loss_history]
        losses_dem = [h[3] for h in self.loss_history]

        gt_rgb_np = (
            (self.val_rgb[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        )
        gt_dem_np = self.val_dem[0].clamp(0, 1).cpu()[0].numpy()

        # CLIP encode (一次调用同时获取 pooler 和 local)
        uncond_pooler, uncond_local = self.text_encoder([""])
        cond_pooler, cond_local = self.text_encoder([self.val_prompt])

        # Batching for CFG: [uncond, cond]
        local_batch = torch.cat([uncond_local, cond_local])  # [2, 77, 768]
        pooler_batch = torch.cat([uncond_pooler, cond_pooler])  # [2, 768]

        # Flow Matching Euler 推理
        latents = torch.randn((1, 8, 64, 64), device=self.device)
        timesteps = self.scheduler.set_timesteps(NUM_INFERENCE_STEPS, self.device)

        for i in tqdm(range(NUM_INFERENCE_STEPS), desc="Sampling Image", leave=False):
            t_i = timesteps[i]
            dt = 1.0 / NUM_INFERENCE_STEPS
            t_batch = torch.full((2,), t_i, device=self.device)
            latents_batch = torch.cat([latents] * 2)

            v_pred = self.dit(
                sample=latents_batch,
                timestep=t_batch,
                encoder_hidden_states=local_batch,
                pooler_output=pooler_batch,
            ).sample

            v_uncond, v_cond = v_pred.chunk(2)
            v_cfg = v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)

            latents = self.scheduler.euler_step(v_cfg, latents, dt)

        # VAE 解码
        rgb_latent = latents[:, :4] / 0.18215
        dem_latent = latents[:, 4:]

        rgb_image = self.rgb_vae.decode(rgb_latent).sample
        dem_image = (
            self.dem_vae.decode(dem_latent).sample.clamp(0, 1)
            if self.dem_vae
            else self.rgb_vae.decode(dem_latent).sample.clamp(-1, 1) / 2 + 0.5
        )
        rgb_image = (rgb_image / 2 + 0.5).clamp(0, 1)

        gen_rgb_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
        gen_dem_np = dem_image[0][0].cpu().numpy()

        # 保存最新输出
        p_low, min_log, max_log = (
            self.norm_params["p_low"],
            self.norm_params["min_log"],
            self.norm_params["max_log"],
        )
        h_real = np.exp(gen_dem_np * (max_log - min_log) + min_log) + p_low - 1
        dem_save_array = np.round(np.clip(h_real, 0, 65535)).astype(np.uint16)

        latest_output_dir = self.viz_output_dir / "latest_output"
        latest_output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((gen_rgb_np * 255).astype(np.uint8)).save(
            latest_output_dir / f"latest_texture_{epoch:04d}.png"
        )
        Image.fromarray(dem_save_array, mode="I;16").save(
            latest_output_dir / f"latest_heightmap_{epoch:04d}.png"
        )

        # 仪表盘绘图
        fig = plt.figure(figsize=(22, 10))
        fig.suptitle(
            f"DiT 8-Ch Dashboard — Epoch {epoch}\nPrompt: '{self.val_prompt}'",
            fontsize=16,
            fontweight="bold",
        )

        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(epochs, losses_total, "b-", linewidth=1.5, marker="o")
        ax1.set_title("Total Loss")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(epochs, losses_rgb, "g-", linewidth=1.5, marker="o", markersize=4)
        ax2.set_title("RGB Texture Loss")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(epochs, losses_dem, "r-", linewidth=1.5, marker="o", markersize=4)
        ax3.set_title("DEM Height Loss")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(2, 4, 4)
        ax4.axis("off")

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.imshow(gt_rgb_np)
        ax5.set_title(f"[GT] {self.gt_name} Texture", color="blue", fontweight="bold")
        ax5.axis("off")

        ax6 = fig.add_subplot(2, 4, 6)
        ax6.imshow(gen_rgb_np)
        ax6.set_title("[AI] Generated Texture", color="darkgreen", fontweight="bold")
        ax6.axis("off")

        ax7 = fig.add_subplot(2, 4, 7)
        ax7.imshow(gt_dem_np, cmap="gray")
        ax7.set_title(f"[GT] {self.gt_name} DEM", color="blue", fontweight="bold")
        ax7.axis("off")

        ax8 = fig.add_subplot(2, 4, 8)
        ax8.imshow(gen_dem_np, cmap="gray")
        ax8.set_title("[AI] Generated DEM", color="darkgreen", fontweight="bold")
        ax8.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(self.viz_output_dir / f"dashboard_epoch_{epoch:04d}.png", dpi=120)
        plt.close(fig)

    def train(self):
        print(f"=== 训练 8 通道 DiT (Flow Matching) ===")
        print(f"Batch Size: {self.args.batch_size} | 目标 Epochs: {self.args.epochs}")

        # CSV 日志记录
        csv_path = self.output_dir / "training_log.csv"
        csv_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(["epoch", "loss", "rgb_loss", "dem_loss"])

        for epoch in range(self.start_epoch, self.args.epochs):
            avg_loss = self.train_epoch(epoch)
            print(
                f"\nEpoch {epoch:3d} | Loss: {avg_loss['loss']:.4f} | "
                f"RGB: {avg_loss['rgb']:.4f} | DEM: {avg_loss['dem']:.4f}"
            )

            # CSV 记录
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, avg_loss["loss"], avg_loss["rgb"], avg_loss["dem"]]
                )

            trainable_state_dict = self.dit.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model_state_dict": trainable_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict()
                    if self.scaler
                    else None,
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "best_loss": self.best_loss,
                    "loss_history": self.loss_history,
                },
                self.output_dir / "latest_checkpoint.pt",
            )

            if avg_loss["loss"] < self.best_loss:
                self.best_loss = avg_loss["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "model_state_dict": trainable_state_dict,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict()
                        if self.scaler
                        else None,
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "best_loss": self.best_loss,
                        "loss_history": self.loss_history,
                    },
                    self.output_dir / "best_dit.pt",
                )

            if epoch % 5 == 0:
                torch.save(
                    {"epoch": epoch, "model_state_dict": trainable_state_dict},
                    self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                )

            self.loss_history.append(
                (epoch, avg_loss["loss"], avg_loss["rgb"], avg_loss["dem"])
            )
            if (
                epoch % getattr(self.args, "viz_interval", VIZ_INTERVAL) == 0
                or epoch == self.args.epochs - 1
            ):
                self.visualize_epoch(epoch)

    @torch.no_grad()
    def test(self, num_samples: int):
        print(f"\n=== 推理测试 (采样 {num_samples} 组) ===")
        self.dit.eval()
        test_dir = self.output_dir / "test_results_dit"
        test_dir.mkdir(parents=True, exist_ok=True)

        p_low, min_log, max_log = (
            self.norm_params["p_low"],
            self.norm_params["min_log"],
            self.norm_params["max_log"],
        )

        timesteps = self.scheduler.set_timesteps(NUM_INFERENCE_STEPS, self.device)

        count = 0
        for batch in self.val_dataloader:
            if count >= num_samples:
                break

            prompt = batch["prompt"][0]
            basename = batch["basename"][0]
            print(f"正在生成 {count + 1}/{num_samples}: {basename} | Prompt: {prompt}")

            # CLIP encode (一次调用同时获取 pooler 和 local)
            uncond_pooler, uncond_local = self.text_encoder([""])
            cond_pooler, cond_local = self.text_encoder([prompt])

            local_batch = torch.cat([uncond_local, cond_local])
            pooler_batch = torch.cat([uncond_pooler, cond_pooler])

            latents = torch.randn((1, 8, 64, 64), device=self.device)

            for i in tqdm(range(NUM_INFERENCE_STEPS), desc="Sampling", leave=False):
                t_i = timesteps[i]
                dt = 1.0 / NUM_INFERENCE_STEPS
                t_batch = torch.full((2,), t_i, device=self.device)
                latents_batch = torch.cat([latents] * 2)

                v_pred = self.dit(
                    sample=latents_batch,
                    timestep=t_batch,
                    encoder_hidden_states=local_batch,
                    pooler_output=pooler_batch,
                ).sample

                v_uncond, v_cond = v_pred.chunk(2)
                v_cfg = v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)
                latents = self.scheduler.euler_step(v_cfg, latents, dt)

            # 解码
            rgb_latent = latents[:, :4] / 0.18215
            dem_latent = latents[:, 4:]

            rgb_image = self.rgb_vae.decode(rgb_latent).sample
            dem_image = (
                self.dem_vae.decode(dem_latent).sample.clamp(0, 1)
                if self.dem_vae
                else self.rgb_vae.decode(dem_latent).sample.clamp(-1, 1) / 2 + 0.5
            )

            gen_rgb_np = (
                (rgb_image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            )
            gen_dem_np = dem_image[0][0].cpu().numpy()

            h_real = np.exp(gen_dem_np * (max_log - min_log) + min_log) + p_low - 1
            dem_save_array = np.round(np.clip(h_real, 0, 65535)).astype(np.uint16)
            rgb_save_array = (gen_rgb_np * 255).astype(np.uint8)

            Image.fromarray(rgb_save_array).save(
                test_dir / f"{basename}_gen_texture.png"
            )
            Image.fromarray(dem_save_array, mode="I;16").save(
                test_dir / f"{basename}_gen_heightmap.png"
            )
            count += 1

        print(f"\n测试完成! 生成的模型资产已保存至: {test_dir}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.dit.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and self.args.mode == "train":
            if getattr(self.args, "finetune", False):
                print(
                    f"\n[微调模式] 已加载第 {checkpoint['epoch']} 轮权重，重置优化器和调度器。"
                )
                self.start_epoch = 0
            else:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if self.scaler and "scaler_state_dict" in checkpoint:
                        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except ValueError as e:
                    print(f"\n[警告] 优化器参数组发生变化，跳过旧状态加载: {e}")
                if "scheduler_state_dict" in checkpoint and hasattr(
                    self, "lr_scheduler"
                ):
                    self.lr_scheduler.load_state_dict(
                        checkpoint["scheduler_state_dict"]
                    )
                self.start_epoch = checkpoint["epoch"] + 1
                self.global_step = checkpoint["global_step"]
                self.best_loss = checkpoint.get("best_loss", float("inf"))
                self.loss_history = checkpoint.get("loss_history", [])
                print(f"成功恢复训练状态！将从 Epoch {self.start_epoch} 继续。")
        else:
            print(f"已加载 epoch {checkpoint['epoch']} 的模型权重")


# CLI Entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--dem_vae_ckpt", type=str, default=DEM_VAE_CKPT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--num_samples", type=int, default=5)

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser.add_argument("--use_amp", type=str2bool, default=USE_AMP)
    parser.add_argument("--viz_interval", type=int, default=VIZ_INTERVAL)
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="微调模式：仅加载权重，重置优化器和调度器",
    )
    parser.add_argument(
        "--use_local_models",
        type=str2bool,
        default=USE_LOCAL_MODELS,
        help=f"使用本地已下载的 CLIP/VAE 模型 (默认: {USE_LOCAL_MODELS})",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=MODELS_DIR,
        help=f"本地模型目录 (默认: {MODELS_DIR})",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据集切分
    print(f"=== 初始化数据集 (基于固定 SEED={SEED} 切分 {VAL_SPLIT * 100}% 验证集) ===")
    full_dataset = DiTDataset(data_root=args.data_root, augment=False)
    full_metadata = full_dataset.metadata
    total_size = len(full_metadata)

    val_size = max(1, int(total_size * VAL_SPLIT))
    train_size = total_size - val_size

    import torch.utils.data as data

    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = data.random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    print(f"数据总数: {total_size} | 训练集: {train_size} | 验证集: {val_size}")

    if args.mode == "train":
        train_dataset = DiTDataset(
            data_root=args.data_root,
            augment=True,
            metadata=[full_metadata[i] for i in train_indices.indices],
        )
        val_dataset = DiTDataset(
            data_root=args.data_root,
            augment=False,
            metadata=[full_metadata[i] for i in val_indices.indices],
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        dit = DiT(
            in_channels=8, out_channels=8, depth=18, hidden_size=1024, num_heads=16
        ).to(device)
        print(f"DiT 参数量: {sum(p.numel() for p in dit.parameters()) / 1e6:.1f}M")

        trainer = DiTTrainer(dit, train_dataloader, val_dataloader, args)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        trainer.train()

    elif args.mode == "test":
        if not args.checkpoint:
            raise ValueError("测试模式必须提供 --checkpoint 参数！")

        test_dataset = DiTDataset(
            data_root=args.data_root,
            augment=False,
            metadata=[full_metadata[i] for i in val_indices.indices],
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        dit = DiT(
            in_channels=8, out_channels=8, depth=18, hidden_size=1024, num_heads=16
        ).to(device)
        trainer = DiTTrainer(dit, None, test_dataloader, args)
        trainer.load_checkpoint(args.checkpoint)
        trainer.test(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
