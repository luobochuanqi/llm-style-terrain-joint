"""
8通道 U-Net 训练脚本

用法：
    # 正常全新训练
    python ./scripts/unet/unet_full.py --epoch 3
    
    # 自动寻找最新断点继续训练
    python ./scripts/unet/unet_full.py --resume True
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
matplotlib.use("Agg") # 极其重要：防止在无 GUI 的 AutoDL 服务器上画图报错
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

import sys
# 退三层：train_unet_full.py -> unet -> scripts -> 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# === 开源组件 ===
from diffusers import DDPMScheduler, AutoencoderKL

from diffusers import logging as diffusers_logging
from diffusers import UNet2DConditionModel
from transformers import logging as transformers_logging
diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# === 自己写的模块 ===
from dataset.unet_dataset import UNetDataset
from models.unet.unet_8ch import build_unet
from models.vae.heightmap_vae import HeightMapVAE
from models.clip.text_encoder import build_text_encoder

# =============================================================================
# 训练配置（软编码区：像仪表盘一样统一管理参数）
# =============================================================================

DATA_ROOT = "./data/unet_training"  # 你的数据集根目录
DEM_VAE_CKPT = "./data/vae_model_data/best_checkpoint.pt" # 高程 VAE 权重路径 (为空则临时用 RGB_VAE 代替)
OUTPUT_DIR = "./outputs/unet_8ch"   # 模型和图片输出的根目录

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
USE_AMP = True                      # 是否开启 fp16 混合精度加速

SAVE_STEPS = 12000                   # 每隔多少步保存一次常规快照
LOG_STEPS = 10                      # 每隔多少步在控制台打印一次日志
VIZ_INTERVAL = 1                    # 每隔几个 epoch 画一次 Loss 曲线图
RESUME = False                      # 默认不自动开启续训，防止覆盖意图


def build_8ch_unet_from_sd(device = "cuda"):
    
    print("正在加载sd的unet模型...")

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="unet"
    )

    print("正在将4通道改为8通道...")

    old_conv_in = unet.conv_in

    new_conv_in = nn.Conv2d(
        in_channels=8, 
        out_channels=old_conv_in.out_channels, 
        kernel_size=old_conv_in.kernel_size, 
        stride=old_conv_in.stride,     # 把原来的步长也安全继承过来
        padding=old_conv_in.padding    # 指名道姓这是 padding
    )

    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight.clone()
        new_conv_in.weight[:, 4:, :, :] = torch.zeros_like(old_conv_in.weight)
        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())

    unet.conv_in = new_conv_in

    old_conv_out = unet.conv_out

    new_conv_out = nn.Conv2d(
        in_channels=old_conv_out.in_channels, 
        out_channels=8, 
        kernel_size=old_conv_out.kernel_size, 
        stride=old_conv_out.stride, 
        padding=old_conv_out.padding
    )

    with torch.no_grad():
        new_conv_out.weight[:4, :, :, :] = old_conv_out.weight.clone()
        new_conv_out.weight[4:, :, :, :] = torch.zeros_like(old_conv_out.weight)
        new_conv_out.bias[:4] = old_conv_out.bias.clone()
        new_conv_out.bias[4:] = torch.zeros_like(old_conv_out.bias)
    unet.conv_out = new_conv_out

    unet.config.in_channels = 8
    unet.config.out_channels = 8

    print("正在执行阶梯式解冻")

    unet.requires_grad_(False)

    unet.conv_in.requires_grad_(True)
    unet.conv_out.requires_grad_(True)

    for name, param in unet.named_parameters():
        if "attn2" in name: 
            param.requires_grad = True

        elif "attn1" in name:
            param.requires_grad = True
        
        elif "down_blocks.0" in name or "down_blocks.1" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"完成! 可训练参数比例: {trainable_params/total_params*100:.2f}% ({trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M)")

    return unet.to(device)

class UNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建输出目录和可视化目录
        self.output_dir = Path(args.output_dir)
        self.viz_output_dir = self.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于记录画图的数据与进度
        self.loss_history = []  
        self.global_step = 0
        self.start_epoch = 0           # 新增：记录起始 Epoch
        self.best_loss = float("inf")

        # ==========================================
        # 1. 组建做题家 (U-Net) 与优化器
        # ==========================================
        print("正在组装 8 通道 U-Net...")
        
        self.unet = build_8ch_unet_from_sd(self.device)

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(), 
            lr=args.learning_rate, 
            weight_decay=WEIGHT_DECAY
        )
        self.scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

        # ==========================================
        # 1.5 核心逻辑：断点恢复 (Checkpoint Loading)
        # ==========================================
        if self.args.resume:
            ckpt_path = self.output_dir / "latest_checkpoint.pt"
            if ckpt_path.exists():
                print(f"\n发现断点文件: {ckpt_path}")
                print("正在恢复模型、优化器及训练状态...")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                
                self.unet.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scaler is not None and "scaler_state_dict" in checkpoint:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                
                self.start_epoch = checkpoint["epoch"] + 1 # 从下一个 epoch 开始
                self.global_step = checkpoint["global_step"]
                self.best_loss = checkpoint.get("best_loss", float("inf"))
                self.loss_history = checkpoint.get("loss_history", [])
                
                print(f"成功恢复！将从 Epoch {self.start_epoch} 继续训练，当前历史最佳 Loss: {self.best_loss:.4f}\n")
            else:
                print(f"\n未找到断点文件 ({ckpt_path})，将从头开始训练。\n")

        # ==========================================
        # 2. 挂载黑盒翻译官 (CLIP)
        # ==========================================
        print("正在加载 CLIP 文本编码器...")
        self.text_encoder = build_text_encoder(model_name="openai/clip-vit-large-patch14").to(self.device)
        self.text_encoder.eval()

        # ==========================================
        # 3. 挂载空间压缩器 (两个 VAE)
        # ==========================================
        print("正在加载 SD 标准 VAE...")
        self.rgb_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        self.rgb_vae.eval()
        self.rgb_vae.requires_grad_(False)

        if args.dem_vae_ckpt:
            print("正在加载 高程图专属 VAE...")
            self.dem_vae = HeightMapVAE(block_out_channels=(128, 256, 512, 512)).to(self.device)
            self.dem_vae.load_state_dict(torch.load(args.dem_vae_ckpt, map_location=self.device)["model_state_dict"])
            self.dem_vae.eval()
            self.dem_vae.requires_grad_(False)
        else:
            self.dem_vae = None
            print("未提供 DEM VAE 权重，将临时使用 RGB VAE 处理高程图以跑通测试。")

        # ==========================================
        # 4. 挂载造题机器 (Scheduler) & 数据集
        # ==========================================
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        dataset = UNetDataset(data_root=args.data_root, augment=True)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        val_batch = next(iter(self.dataloader))
        self.val_prompt = val_batch["prompt"][:1]

        self.val_rgb = val_batch["rgb"][:1].to(self.device)
        self.val_dem = val_batch["dem"][:1].to(self.device)

        print(f"验证样本准备就绪，提示词: {self.val_prompt[0]}")

        prompt_output = self.viz_output_dir / "prompt.txt"

        with open(prompt_output, "w", encoding="utf-8") as f:

            f.write(self.val_prompt[0])

    def encode_to_latent(self, rgb_pixels, dem_pixels):
        """核心魔法：把像素变成隐向量，并设立 64x64 绝对屏障"""
        # 1. 官方 VAE，绝对标准的 64x64
        rgb_latent = self.rgb_vae.encode(rgb_pixels).latent_dist.sample() * 0.18215
        
        # 2. 你们的 VAE，可能会吐出残疾的 62x62
        if self.dem_vae is not None:
            dem_latent = self.dem_vae.encode(dem_pixels).latent_dist.sample() * 0.993099
        else:
            dem_pixels_3ch = dem_pixels.repeat(1, 3, 1, 1)
            dem_latent = self.rgb_vae.encode(dem_pixels_3ch).latent_dist.sample() * 0.18215
            
        # 3. 终极防御：如果 DEM 掉链子（比如变成了 62x62），强行把它拉伸到 RGB 的尺寸（64x64）！
        # 千万不能反向把 RGB 缩小去迎合 DEM！
        if dem_latent.shape[2:] != rgb_latent.shape[2:]:
            dem_latent = F.interpolate(
                dem_latent, 
                size=rgb_latent.shape[2:], # 永远对齐官方的 64x64
                mode="bilinear", 
                align_corners=False
            )
            
        latent_8ch = torch.cat([rgb_latent, dem_latent], dim=1)
        return latent_8ch

    def train_epoch(self, epoch):
        """只负责一个 Epoch 内的所有运算，返回平均 Loss"""
        self.unet.train()
        
        total_loss_epoch = 0.0
        total_img_loss = 0.0
        total_dem_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            rgb_pixels = batch["rgb"].to(self.device)
            dem_pixels = batch["dem"].to(self.device)
            prompts = batch["prompt"]
            batch_size = rgb_pixels.shape[0]

            TARGET_SIZE = (512, 512)
            if rgb_pixels.shape[2:] != TARGET_SIZE:
                rgb_pixels = F.interpolate(rgb_pixels, size=TARGET_SIZE, mode="bilinear", align_corners=False)
            if dem_pixels.shape[2:] != TARGET_SIZE:
                dem_pixels = F.interpolate(dem_pixels, size=TARGET_SIZE, mode="bilinear", align_corners=False)

            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                with torch.no_grad():
                    global_features, local_features = self.text_encoder(prompts)
                    latents = self.encode_to_latent(rgb_pixels, dem_pixels)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # [核心修复 1]：对接官方 U-Net 前向传播 API
                output = self.unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=local_features
                )
                noise_pred = output.sample
                
                # [核心修复 2]：手动计算通道均方误差 (MSE Loss)
                loss_img = F.mse_loss(noise_pred[:, :4, :, :], noise[:, :4, :, :])
                loss_dem = F.mse_loss(noise_pred[:, 4:, :, :], noise[:, 4:, :, :])
                loss = loss_img + loss_dem

            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            # 统计数据
            total_loss_epoch += loss.item()
            total_img_loss += loss_img.item()
            total_dem_loss += loss_dem.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "Loss": f"{total_loss_epoch/num_batches:.4f}",
                "Img": f"{total_img_loss/num_batches:.4f}",
                "Dem": f"{total_dem_loss/num_batches:.4f}"
            })

        return {
            "loss": total_loss_epoch / num_batches,
            "img": total_img_loss / num_batches,
            "dem": total_dem_loss / num_batches
        }

    @torch.no_grad()
    def visualize_epoch(self, epoch: int):
        """生成【Loss 曲线 + 原图 + 验证图像】的终极全景看板"""
        if len(self.loss_history) == 0:
            return

        self.unet.eval()
        
        # ==========================================
        # 1. 准备 Loss 数据
        # ==========================================
        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_img = [h[2] for h in self.loss_history]
        losses_dem = [h[3] for h in self.loss_history]

        # ==========================================
        # 2. 准备真实的 Ground Truth (原图)
        # ==========================================
        # 原图在 dataloader 里被归一化到了 [-1, 1]，我们需要把它拉回 [0, 1] 才能画出来
        gt_rgb_np = (self.val_rgb[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        gt_dem_np = (self.val_dem[0] / 2 + 0.5).clamp(0, 1).cpu()[0].numpy() # [C, H, W] -> 取第一个通道

        # ==========================================
        # 3. 运行逆向扩散，生成验证图像 (AI画的图)
        # ==========================================
        prompt = self.val_prompt[0] 
        global_features, local_features = self.text_encoder([prompt])
        # 从纯噪声开始
        latents = torch.randn((1, 8, 64, 64), device=self.device)
        self.noise_scheduler.set_timesteps(50)
        
        for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling Image"):
            # [核心修复 3]：可视化时的逆向扩散推断 API
            output = self.unet(
                sample=latents,
                timestep=t.unsqueeze(0).to(self.device),
                encoder_hidden_states=local_features
            )
            noise_pred = output.sample
            
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        rgb_latent, dem_latent = torch.chunk(latents, 2, dim=1)
        rgb_latent = rgb_latent / 0.18215
        dem_latent = dem_latent / 0.993099
        
        rgb_image = self.rgb_vae.decode(rgb_latent).sample
        if self.dem_vae:
            dem_image = self.dem_vae.decode(dem_latent).sample
        else:
            dem_image = self.rgb_vae.decode(dem_latent).sample
            
        rgb_image = (rgb_image / 2 + 0.5).clamp(0, 1)
        dem_image = (dem_image / 2 + 0.5).clamp(0, 1)
        
        gen_rgb_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
        gen_dem_np = dem_image[0][0].cpu().numpy()

        latest_output_dir = self.viz_output_dir / "latest_output"
        latest_output_dir.mkdir(parents=True, exist_ok=True)

        rgb_save_array = (gen_rgb_np * 255).astype(np.uint8)
        dem_save_array = (gen_dem_np * 65535.0).astype(np.uint16)

        Image.fromarray(rgb_save_array).save(latest_output_dir / "latest_texture.png")
        Image.fromarray(dem_save_array, mode='I;16').save(latest_output_dir / "latest_heightmap.png")

        # ==========================================
        # 4. 绘制 2行4列 的终极大图
        # ==========================================
        fig = plt.figure(figsize=(22, 10)) # 加宽画布
        fig.suptitle(f"U-Net 8-Channel Training Dashboard — Epoch {epoch}\nPrompt: '{prompt}'", fontsize=16, fontweight="bold")

        # --- 第一行：Loss 曲线 (占前3个位置) ---
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(epochs, losses_total, "b-", linewidth=1.5, marker='o', markersize=4)
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Epoch")
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

        # --- 第一行第4个位置：文本信息区 ---
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.axis('off')
        latest_total = losses_total[-1]
        ax4.text(0.5, 0.5, f"Current Total Loss:\n{latest_total:.4f}\n\nTop: Ground Truth\nBottom: Generated", 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=18, fontweight="bold", color='#333333')

        # --- 第二行：原图 vs 生成图 的对比 ---
        # 1. 真实 RGB
        ax5 = fig.add_subplot(2, 4, 5)
        ax5.imshow(gt_rgb_np)
        ax5.set_title("[GT] Original RGB Texture", color='blue', fontweight='bold')
        ax5.axis('off')

        # 2. 生成 RGB
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.imshow(gen_rgb_np)
        ax6.set_title("[AI] Generated RGB Texture", color='darkgreen', fontweight='bold')
        ax6.axis('off')

        # 3. 真实 DEM
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.imshow(gt_dem_np, cmap='gray')
        ax7.set_title("[GT] Original DEM Heightmap", color='blue', fontweight='bold')
        ax7.axis('off')

        # 4. 生成 DEM
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.imshow(gen_dem_np, cmap='gray')
        ax8.set_title("[AI] Generated DEM Heightmap", color='darkgreen', fontweight='bold')
        ax8.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9) 
        
        save_path = self.viz_output_dir / f"dashboard_epoch_{epoch:04d}.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

        self.unet.train()

    def train(self):
        """控制完整的训练循环，负责保存最佳状态"""
        print(f"=== 开始训练 8 通道 U-Net ===")
        print(f"数据集: {self.args.data_root}")
        print(f"Batch Size: {self.args.batch_size} | 目标 Epochs: {self.args.epochs}")
        print("-" * 50)
        
        # 修改处：从 start_epoch 开始循环，而不是永远从 0 开始
        for epoch in range(self.start_epoch, self.args.epochs):
            # 执行一个 Epoch 的训练
            avg_loss = self.train_epoch(epoch)
            
            print(
                f"\nEpoch {epoch:3d} | "
                f"Total Loss: {avg_loss['loss']:.4f} | "
                f"Img Loss: {avg_loss['img']:.4f} | "
                f"Dem Loss: {avg_loss['dem']:.4f}"
            )
            
            # 核心逻辑：每一轮结束，都无条件更新 latest_checkpoint.pt
            latest_path = self.output_dir / "latest_checkpoint.pt"
            torch.save({
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.unet.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "best_loss": self.best_loss,
                "loss_history": self.loss_history
            }, latest_path)
            
            # 拦截并保存最佳模型
            if avg_loss["loss"] < self.best_loss:
                print(f"发现新的最佳 Loss ({self.best_loss:.4f} -> {avg_loss['loss']:.4f})，已保存为 best_unet.pt")
                self.best_loss = avg_loss["loss"]
                best_path = self.output_dir / "best_unet.pt"
                # 最佳模型只需要存权重就够了，不用存优化器状态，省硬盘
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.unet.state_dict(),
                    "best_loss": self.best_loss,
                }, best_path)
            
            # 记录历史数据用于画图
            self.loss_history.append((
                epoch, 
                avg_loss["loss"], 
                avg_loss["img"], 
                avg_loss["dem"]
            ))
            
            # 触发画图
            if epoch % self.args.viz_interval == 0 or epoch == self.args.epochs - 1:
                self.visualize_epoch(epoch)
            
        print(f"\n训练圆满完成！历史最佳 Loss 为: {self.best_loss:.4f}")
        torch.save({
            "epoch": self.args.epochs - 1,
            "model_state_dict": self.unet.state_dict(),
        }, self.output_dir / "unet_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--save_steps", 
        type=int, 
        default=SAVE_STEPS
    )
    
    # 只需要传字符串 "True" 或 "False"
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
        
    parser.add_argument(
        "--use_amp", 
        type=str2bool, 
        default=USE_AMP
    )
    
    parser.add_argument(
        "--resume", 
        type=str2bool, 
        default=RESUME, 
        help="是否从最新的检查点恢复训练"
    )
    
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=NUM_WORKERS
    )

    parser.add_argument(
        "--viz_interval", 
        type=int, 
        default=VIZ_INTERVAL
    )
    
    args = parser.parse_args()
    
    trainer = UNetTrainer(args)
    trainer.train()