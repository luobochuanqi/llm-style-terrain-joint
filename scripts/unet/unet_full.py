"""
8通道 U-Net 训练与测试脚本 (架构统一版)

用法：
    # 正常全新训练
    python ./scripts/unet/unet_full.py --mode train --epochs 50
    
    # 自动寻找最新断点继续训练
    python ./scripts/unet/unet_full.py --mode train --checkpoint ./outputs/unet_8ch/latest_checkpoint.pt
    
    # 测试模式：批量生成地型并保存引擎可用的 PNG
    python ./scripts/unet/unet_full.py --mode test --checkpoint ./outputs/unet_8ch/latest_checkpoint.pt --num_samples 5
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
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

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from diffusers import logging as diffusers_logging
from diffusers import UNet2DConditionModel
from transformers import logging as transformers_logging
diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

from dataset.unet_dataset import UNetDataset
from models.vae.heightmap_vae import HeightMapVAE
from models.clip.text_encoder import build_text_encoder

# =============================================================================
# 训练配置
# =============================================================================

DATA_ROOT = "./data/unet_training"
DEM_VAE_CKPT = "./data/vae_model_data/best_checkpoint.pt" 
OUTPUT_DIR = "/root/autodl-fs/unet_outputs"

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
USE_AMP = True
VAL_SPLIT = 0.05  # 验证集比例
SEED = 65
USE_CFGE = 0.1  # 使用无分类引导的概率
GUIDANCE_SCALE = 4 # 1代表完全原始的输出

SAVE_STEPS = 12000
LOG_STEPS = 10
VIZ_INTERVAL = 1


def build_8ch_unet_from_sd(device="cuda"):
    """从 SD1.5 UNet 构建 8 通道联合去噪模型。

    前 4 通道载入 SD 预训练权重，后 4 通道（DEM）用 randn 小系数初始化。
    采用阶梯式解冻：conv_in/conv_out + 注意力层 + 浅层 encoder/decoder 解冻。

    Forward 接口：
        model(sample=noisy_latent, timestep=timesteps,
              encoder_hidden_states=local_features) → UNet2DConditionOutput
        噪声预测通过 output.sample 属性获取。

    注意：此接口与 models/unet/unet_8ch.py 的 build_unet 不同
    （build_unet 额外接受 global_features 参数，且直接返回 Tensor）。
    """
    print("正在加载 SD 的 U-Net 模型...")
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

    print("正在将 4 通道改为 8 通道...")
    old_conv_in = unet.conv_in
    new_conv_in = nn.Conv2d(
        in_channels=8, out_channels=old_conv_in.out_channels, 
        kernel_size=old_conv_in.kernel_size, stride=old_conv_in.stride, padding=old_conv_in.padding
    )
    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight.clone()
        new_conv_in.weight[:, 4:, :, :] = torch.zeros_like(old_conv_in.weight)
        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())
    unet.conv_in = new_conv_in

    old_conv_out = unet.conv_out
    new_conv_out = nn.Conv2d(
        in_channels=old_conv_out.in_channels, out_channels=8, 
        kernel_size=old_conv_out.kernel_size, stride=old_conv_out.stride, padding=old_conv_out.padding
    )
    with torch.no_grad():
        new_conv_out.weight[:4, :, :, :] = old_conv_out.weight.clone()
        # [核心修复 1] 换成 randn_like 正态分布，系数缩小至 0.01 避免正向偏移
        new_conv_out.weight[4:, :, :, :] = torch.randn_like(old_conv_out.weight) * 0.01
        new_conv_out.bias[:4] = old_conv_out.bias.clone()
        new_conv_out.bias[4:] = torch.zeros_like(old_conv_out.bias)
    unet.conv_out = new_conv_out

    unet.config.in_channels = 8
    unet.config.out_channels = 8

    print("正在执行精准阶梯式解冻...")
    unet.requires_grad_(False)
    unet.conv_in.requires_grad_(True)
    unet.conv_out.requires_grad_(True)

    for name, param in unet.named_parameters():
        if "attn1" in name or "attn2" in name: 
            param.requires_grad = True
        elif "up_blocks.2" in name or "up_blocks.3" in name:
            param.requires_grad = True
        elif "down_blocks.0" in name or "down_blocks.1" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"完成! 可训练参数比例: {trainable_params/total_params*100:.2f}% ({trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M)")

    return unet.to(device)


class UNetTrainer:
    def __init__(self, unet, train_dataloader, val_dataloader, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet = unet
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 目录设置
        self.output_dir = Path(args.output_dir)
        self.viz_output_dir = self.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        self.dem_data = self.viz_output_dir / "dem_data.txt"

        # 加载对数归一化参数
        import json
        params_file = os.path.join("./data/process/heightmaps_hf", "norm_params.json")
        with open(params_file, "r") as f:
            self.norm_params = json.load(f)

        # 状态记录
        self.loss_history = []  
        self.global_step = 0
        self.start_epoch = 0           
        self.best_loss = float("inf")

        # 优化器
        conv_params = []
        core_params = []
        
        for name, param in self.unet.named_parameters():
            if not param.requires_grad:
                continue
            # 将首尾卷积层（包含我们刚初始化的 4 个新通道）单独拎出来
            if "conv_in" in name or "conv_out" in name:
                conv_params.append(param)
            else:
                core_params.append(param)

        # 核心网络使用传入的保守学习率 (如 2e-5)
        # 首尾卷积层使用 5 倍学习率 (如 1e-4) 加速 DEM 通道成型
        self.optimizer = torch.optim.AdamW([
            {"params": core_params, "lr": args.learning_rate},
            {"params": conv_params, "lr": args.learning_rate * 5.0} 
        ], weight_decay=WEIGHT_DECAY)

        self.scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

        from diffusers.optimization import get_scheduler

        if self.train_dataloader is not None:

            self.total_steps = self.args.epochs * len(self.train_dataloader)

            self.warmup_steps = int(self.total_steps * 0.05)

            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )

        # 加载环境模型 (CLIP, VAEs, Scheduler)
        print("正在加载 CLIP 和 VAE 环境...")
        self.text_encoder = build_text_encoder(model_name="openai/clip-vit-large-patch14").to(self.device)
        self.text_encoder.eval()

        self.rgb_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        self.rgb_vae.eval().requires_grad_(False)

        if args.dem_vae_ckpt:
            self.dem_vae = HeightMapVAE(block_out_channels=(128, 256, 512, 512)).to(self.device)
            self.dem_vae.load_state_dict(torch.load(args.dem_vae_ckpt, map_location=self.device)["model_state_dict"])
            self.dem_vae.eval().requires_grad_(False)
        else:
            self.dem_vae = None

        self.train_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.infer_scheduler = DDIMScheduler(num_train_timesteps=1000)

        # 计算并保存 DEM 隐空间分布用于正态缩放
        val_batch = next(iter(self.val_dataloader))
        # [核心修复 2]：去掉了 [0] 切片引发的单字母 Bug
        self.val_prompt = val_batch["prompt"][0]
        self.gt_name = val_batch["basename"][0] 
        self.val_rgb = val_batch["rgb"][:1].to(self.device)
        self.val_dem = val_batch["dem"][:1].to(self.device)

        prompt_path = self.viz_output_dir / "prompt.txt"

        with open(prompt_path, "w", encoding="utf-8") as f:

            f.write(self.val_prompt)

    def encode_to_latent(self, rgb_pixels, dem_pixels):
        rgb_latent = self.rgb_vae.encode(rgb_pixels).latent_dist.sample() * 0.18215
        if self.dem_vae is not None:
            dem_latent = self.dem_vae.encode(dem_pixels).latent_dist.sample()
        else:
            dem_pixels_3ch = dem_pixels.repeat(1, 3, 1, 1)
            dem_latent = self.rgb_vae.encode(dem_pixels_3ch).latent_dist.sample() * 0.18215
            
        if dem_latent.shape[2:] != rgb_latent.shape[2:]:
            dem_latent = F.interpolate(dem_latent, size=rgb_latent.shape[2:], mode="bilinear", align_corners=False)
            
        return torch.cat([rgb_latent, dem_latent], dim=1)

    def train_epoch(self, epoch):
        self.unet.train()
        total_loss_epoch, total_img_loss, total_dem_loss, num_batches = 0.0, 0.0, 0.0, 0
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            rgb_pixels, dem_pixels, prompts = batch["rgb"].to(self.device), batch["dem"].to(self.device), batch["prompt"]
            batch_size = rgb_pixels.shape[0]

            TARGET_SIZE = (512, 512)
            if rgb_pixels.shape[2:] != TARGET_SIZE:
                rgb_pixels = F.interpolate(rgb_pixels, size=TARGET_SIZE, mode="bilinear", align_corners=False)
            if dem_pixels.shape[2:] != TARGET_SIZE:
                dem_pixels = F.interpolate(dem_pixels, size=TARGET_SIZE, mode="bilinear", align_corners=False)

            # 加入无分类引导

            cfged_prompts = []
            for p in prompts:
                if random.random() < USE_CFGE: 
                    cfged_prompts.append("")
                else:
                    cfged_prompts.append(p)

            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                with torch.no_grad():
                    global_features, local_features = self.text_encoder(cfged_prompts)
                    latents = self.encode_to_latent(rgb_pixels, dem_pixels)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
                noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)

                output = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=local_features)
                noise_pred = output.sample
                
                # Min SNR机制

                gamma = 5.0  # 截断阈值，5.0 是业界标准推荐值
                alphas_cumprod = self.train_scheduler.alphas_cumprod.to(self.device)
                
                # 获取当前 Batch 时间步对应的 alpha 累乘值
                alpha_prod_t = alphas_cumprod[timesteps]
                beta_prod_t = 1 - alpha_prod_t

                snr = alpha_prod_t / beta_prod_t
                
                # 计算 Min-SNR 权重：min(snr, gamma) / snr
                min_snr_weight = torch.clamp(snr, max=gamma) / snr

                # ==========================================================
                # [修改] 2. 计算未求总均值的多任务 Loss
                # ==========================================================
                # reduction="none" 保证输出形状与输入一致，不急着求均值
                loss_img_none = F.mse_loss(noise_pred[:, :4, :, :], noise[:, :4, :, :], reduction="none")
                loss_dem_none = F.mse_loss(noise_pred[:, 4:, :, :], noise[:, 4:, :, :], reduction="none")
                
                # 把每张图的空间和通道维度(dim=1,2,3)拉平求均值，保留 batch 维度 (dim=0)
                # 现在的尺寸是 [batch_size]
                loss_img_batch = loss_img_none.mean(dim=[1, 2, 3])
                loss_dem_batch = loss_dem_none.mean(dim=[1, 2, 3])
                
                
                loss_batch = loss_img_batch + loss_dem_batch
                
                # ==========================================================
                # [新增] 3. 应用权重并得出最终 Loss
                # ==========================================================
                # 将每张图的 loss 乘以它自己的 Min-SNR 权重，最后再对 batch 求均值
                loss = (loss_batch * min_snr_weight).mean()
                
                # 为了不影响下方进度条的打印，将独立 loss 也转为标量
                loss_img = loss_img_batch.mean()
                loss_dem = loss_dem_batch.mean()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # --- 新增：解缩放梯度并进行裁剪 ---
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                # ---------------------------------
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # --- 新增：常规梯度裁剪 ---
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                # ---------------------------------
                self.optimizer.step()
                
            self.lr_scheduler.step()

            self.optimizer.zero_grad()
            
            total_loss_epoch += loss.item()
            total_img_loss += loss_img.item()
            total_dem_loss += loss_dem.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"Loss": f"{total_loss_epoch/num_batches:.4f}", "Dem": f"{total_dem_loss/num_batches:.4f}"})

        return {"loss": total_loss_epoch / num_batches, 
                "img": total_img_loss / num_batches, 
                "dem": total_dem_loss / num_batches
            }

    @torch.no_grad()
    def visualize_epoch(self, epoch: int):
        if len(self.loss_history) == 0: return
        self.unet.eval()
        
        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_img = [h[2] for h in self.loss_history]
        losses_dem = [h[3] for h in self.loss_history]
        
        gt_rgb_np = (self.val_rgb[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        gt_dem_np = self.val_dem[0].clamp(0, 1).cpu()[0].numpy()

        prompt = self.val_prompt 
        guidance_scale = GUIDANCE_SCALE  # 标准 CFG 引导系数
        
        # --- 新增：分别获取无条件和有条件的特征 ---
        _, uncond_local_features = self.text_encoder([""])
        _, cond_local_features = self.text_encoder([prompt])
        
        # 拼接在一起，尺寸变为 [2, seq_len, dim]
        local_features = torch.cat([uncond_local_features, cond_local_features])

        latents = torch.randn((1, 8, 64, 64), device=self.device)
        self.infer_scheduler.set_timesteps(50)
        
        for t in tqdm(self.infer_scheduler.timesteps, desc="Sampling Image", leave=False):
            # 将隐变量翻倍，以匹配 local_features 的 batch size
            latent_model_input = torch.cat([latents] * 2)
            
            # 由于部分 scheduler 需要缩放，DDIM 通常不需要，但保持规范
            # latent_model_input = self.infer_scheduler.scale_model_input(latent_model_input, t)
            
            output = self.unet(
                sample=latent_model_input, 
                timestep=t.unsqueeze(0).to(self.device), 
                encoder_hidden_states=local_features
            )
            
            # --- 新增：计算 CFG ---
            noise_pred_uncond, noise_pred_text = output.sample.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.infer_scheduler.step(noise_pred, t, latents).prev_sample
            
        rgb_latent, dem_latent = torch.chunk(latents, 2, dim=1)
        rgb_latent = rgb_latent / 0.18215
        
        rgb_image = self.rgb_vae.decode(rgb_latent).sample
        dem_image = self.dem_vae.decode(dem_latent).sample.clamp(0, 1) if self.dem_vae else self.rgb_vae.decode(dem_latent).sample.clamp(-1, 1) / 2 + 0.5
        rgb_image = (rgb_image / 2 + 0.5).clamp(0, 1)
        
        gen_rgb_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
        gen_dem_np = dem_image[0][0].cpu().numpy()

        # [核心修复 4]：逆向对数还原为 16 位物理高差数据
        p_low, min_log, max_log = self.norm_params["p_low"], self.norm_params["min_log"], self.norm_params["max_log"]
        h_real = np.exp(gen_dem_np * (max_log - min_log) + min_log) + p_low - 1
        dem_save_array = np.round(np.clip(h_real, 0, 65535)).astype(np.uint16)
        
        latest_output_dir = self.viz_output_dir / "latest_output"
        latest_output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((gen_rgb_np * 255).astype(np.uint8)).save(latest_output_dir / f"latest_texture_{epoch:04d}.png")
        Image.fromarray(dem_save_array, mode='I;16').save(latest_output_dir / f"latest_heightmap_{epoch:04d}.png")

        fig = plt.figure(figsize=(22, 10))
        fig.suptitle(f"U-Net 8-Ch Dashboard — Epoch {epoch}\nPrompt: '{prompt}'", fontsize=16, fontweight="bold")

        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(epochs, losses_total, "b-", linewidth=1.5, marker='o')
        ax1.set_title("Total Loss")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(epochs, losses_img, "g-", linewidth=1.5, marker='o', markersize=4)
        ax2.set_title("RGB Texture Loss")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(epochs, losses_dem, "r-", linewidth=1.5, marker='o', markersize=4)
        ax3.set_title("DEM Height Loss")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(2, 4, 4)
        ax4.axis('off')
        latest_total = losses_total[-1]
        ax4.text(0.5, 0.5, f"Current Total Loss:\n{latest_total:.4f}\n\nTop: Ground Truth\nBottom: Generated", 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=18, fontweight="bold", color='#333333')

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.imshow(gt_rgb_np)
        ax5.set_title(f"[GT] {self.gt_name} Texture", color='blue', fontweight='bold')
        ax5.axis('off')

        ax6 = fig.add_subplot(2, 4, 6)
        ax6.imshow(gen_rgb_np)
        ax6.set_title("[AI] Generated Texture", color='darkgreen', fontweight='bold')
        ax6.axis('off')

        ax7 = fig.add_subplot(2, 4, 7)
        
        ax7.imshow(gt_dem_np, cmap='gray')
        ax7.set_title(f"[GT] {self.gt_name} DEM", color='blue', fontweight='bold')
        ax7.axis('off')

        ax8 = fig.add_subplot(2, 4, 8)
        ax8.imshow(gen_dem_np, cmap='gray')
        ax8.set_title("[AI] Generated DEM", color='darkgreen', fontweight='bold')
        ax8.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(self.viz_output_dir / f"dashboard_epoch_{epoch:04d}.png", dpi=120)
        plt.close(fig)



    def train(self):
        print(f"=== 开始训练 8 通道 U-Net ===")
        print(f"Batch Size: {self.args.batch_size} | 目标 Epochs: {self.args.epochs}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch:3d} | Loss: {avg_loss['loss']:.4f} | Img: {avg_loss['img']:.4f} | Dem: {avg_loss['dem']:.4f}")
            
            trainable_state_dict = self.unet.state_dict()

            torch.save({
                "epoch": epoch, 
                "global_step": self.global_step, 
                "model_state_dict": trainable_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(), 
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "best_loss": self.best_loss, 
                "loss_history": self.loss_history
            }, self.output_dir / "latest_checkpoint.pt")
            
            if avg_loss["loss"] < self.best_loss:
                self.best_loss = avg_loss["loss"]
                torch.save({
                    "epoch": epoch, 
                    "global_step": self.global_step, 
                    "model_state_dict": trainable_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(), 
                    "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "best_loss": self.best_loss, 
                    "loss_history": self.loss_history
                }, self.output_dir / "best_unet.pt")
            
            if epoch % 5 == 0:

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": trainable_state_dict
                }, self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt")

            self.loss_history.append((epoch, avg_loss["loss"], avg_loss["img"], avg_loss["dem"]))
            if epoch % self.args.viz_interval == 0 or epoch == self.args.epochs - 1:
                self.visualize_epoch(epoch)

    @torch.no_grad()
    def test(self, num_samples: int):
        """测试生成模式：批量生成地型并保存引擎可用的资产"""
        print(f"\n=== 开始生成测试 (采样 {num_samples} 组) ===")
        self.unet.eval()
        test_dir = self.output_dir / "test_results_unet"
        test_dir.mkdir(parents=True, exist_ok=True)

        p_low, min_log, max_log = self.norm_params["p_low"], self.norm_params["min_log"], self.norm_params["max_log"]

        count = 0
        for batch in self.val_dataloader:
            if count >= num_samples:
                break

            prompt = batch["prompt"][0]
            basename = batch["basename"][0]
            print(f"正在生成 {count+1}/{num_samples}: {basename} | Prompt: {prompt}")

            guidance_scale = GUIDANCE_SCALE

            _, uncond_local_features = self.text_encoder([""])
            _, cond_local_features = self.text_encoder([prompt])
            local_features = torch.cat([uncond_local_features, cond_local_features])

            latents = torch.randn((1, 8, 64, 64), device=self.device)

            self.infer_scheduler.set_timesteps(50)

            for t in tqdm(self.infer_scheduler.timesteps, desc="Sampling", leave=False):
                latent_model_input = torch.cat([latents] * 2)

                output = self.unet(
                    sample=latent_model_input, 
                    timestep=t.unsqueeze(0).to(self.device), 
                    encoder_hidden_states=local_features
                )

                noise_pred_uncond, noise_pred_text = output.sample.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.infer_scheduler.step(noise_pred, t, latents).prev_sample

            rgb_latent, dem_latent = torch.chunk(latents, 2, dim=1)
            rgb_latent = rgb_latent / 0.18215

            rgb_image = self.rgb_vae.decode(rgb_latent).sample
            dem_image = self.dem_vae.decode(dem_latent).sample.clamp(0, 1) if self.dem_vae else self.rgb_vae.decode(dem_latent).sample.clamp(-1, 1) / 2 + 0.5

            gen_rgb_np = (rgb_image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            gen_dem_np = dem_image[0][0].cpu().numpy()

            h_real = np.exp(gen_dem_np * (max_log - min_log) + min_log) + p_low - 1
            dem_save_array = np.round(np.clip(h_real, 0, 65535)).astype(np.uint16)
            rgb_save_array = (gen_rgb_np * 255).astype(np.uint8)

            Image.fromarray(rgb_save_array).save(test_dir / f"{basename}_gen_texture.png")
            Image.fromarray(dem_save_array, mode='I;16').save(test_dir / f"{basename}_gen_heightmap.png")
            count += 1
            
        print(f"\n测试完成! 生成的模型资产已保存至: {test_dir}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.unet.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # print()

        if "optimizer_state_dict" in checkpoint and self.args.mode == "train":
            
            if getattr(self.args, "finetune", False):

                print(f"\n[微调模式启动] 已成功加载第 {checkpoint['epoch']} 轮的基础权重！")
                print(f"安全丢弃旧优化器和调度器。将使用全新的学习率 {self.args.learning_rate} 重新积累动量。")
                self.start_epoch = 0 # 微调时 epoch 重新从 0 算起
            
            else:
                
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if self.scaler and "scaler_state_dict" in checkpoint:
                        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except ValueError as e:
                    print(f"\n[优化器警告] 检测到优化器参数组发生变化 (通常是因为修改了分层学习率)。")
                    print(f"安全跳过加载旧的优化器状态，动量将重新积累，不会影响模型权重！")
                    print(f"({e})\n")

                # <-- 新增：恢复调度器状态
                if "scheduler_state_dict" in checkpoint and hasattr(self, 'lr_scheduler'):
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                self.start_epoch = checkpoint["epoch"] + 1 
                self.global_step = checkpoint["global_step"]
                self.best_loss = checkpoint.get("best_loss", float("inf"))
                self.loss_history = checkpoint.get("loss_history", [])
                print(f"成功恢复训练状态！将从 Epoch {self.start_epoch} 继续。")
        else:
            print(f"这是第{checkpoint['epoch']}轮的数据\n")
            print("成功加载权重！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "test"]
    )

    parser.add_argument(
        "--data_root", 
        type=str, 
        default=DATA_ROOT
    )

    parser.add_argument(
        "--dem_vae_ckpt",
        type=str, 
        default=DEM_VAE_CKPT
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=OUTPUT_DIR
    )

    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None
    )

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=EPOCHS
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=LEARNING_RATE
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=NUM_WORKERS
    )    
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5, 
        help="测试时生成的数量"
    )
    
    def str2bool(v): return v.lower() in ("yes", "true", "t", "1")

    parser.add_argument(
        "--use_amp", 
        type=str2bool, 
        default=USE_AMP
    )

    parser.add_argument(
        "--viz_interval", 
        type=int, 
        default=VIZ_INTERVAL
    )

    parser.add_argument(
        "--finetune", 
        action="store_true", 
        help="开启精细微调模式（仅加载模型权重，重置优化器和调度器）"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== 初始化数据集 (基于固定 SEED={SEED} 切分 {VAL_SPLIT * 100}% 验证集) ===")
    full_dataset = UNetDataset(data_root=args.data_root, augment=False)
    full_metadata = full_dataset.metadata
    total_size = len(full_metadata)
    
    val_size = max(1, int(total_size * VAL_SPLIT))
    train_size = total_size - val_size
    
    import torch.utils.data as data
    generator = torch.Generator().manual_seed(SEED)
    # 因为 SEED 是固定的 42，每次运行脚本，切分出的索引列表是绝对一致的
    train_indices, val_indices = data.random_split(range(total_size), [train_size, val_size], generator=generator)
    
    print(f"数据总数: {total_size} | 训练集分配: {train_size} | 验证/测试集分配: {val_size}")

    # =========================================================================
    # 训练模式
    # =========================================================================
    if args.mode == "train":
        train_dataset = UNetDataset(data_root=args.data_root, augment=True, metadata_list=[full_metadata[i] for i in train_indices.indices])
        val_dataset = UNetDataset(data_root=args.data_root, augment=False, metadata_list=[full_metadata[i] for i in val_indices.indices])
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        unet = build_8ch_unet_from_sd(device)
        trainer = UNetTrainer(unet, train_dataloader, val_dataloader, args)

        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        trainer.train()

    # =========================================================================
    # 测试模式
    # =========================================================================
    elif args.mode == "test":
        print("=== 初始化测试环境 ===")
        if not args.checkpoint:
            raise ValueError("测试模式必须提供 --checkpoint 参数！")
        
        # [核心修复] 测试集严格使用 validation 的切分索引，杜绝训练数据泄露
        test_dataset = UNetDataset(data_root=args.data_root, augment=False, metadata_list=[full_metadata[i] for i in val_indices.indices])
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        unet = build_8ch_unet_from_sd(device)
        # 将 test_dataloader 作为 val_dataloader 传给 Trainer 初始化
        trainer = UNetTrainer(unet, None, test_dataloader, args)
        trainer.load_checkpoint(args.checkpoint)
        trainer.test(num_samples=args.num_samples)

if __name__ == "__main__":
    main()