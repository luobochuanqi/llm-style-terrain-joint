"""
8通道 U-Net 训练脚本（规范化与可视化版）

用法：
    python train_unet_full.py
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
matplotlib.use("Agg") # 极其重要：防止在无 GUI 的 AutoDL 服务器上画图报错
import matplotlib.pyplot as plt
from pathlib import Path

import sys
# 退三层：train_unet_full.py -> unet -> scripts -> 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# === 白嫖的开源组件 ===
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging
diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# === 自己写的模块 ===
from dataset.unet_dataset import UNetDataset
from models.unet.unet_8ch import build_unet
from models.vae.heightmap_vae import HeightMapVAE

# =============================================================================
# 训练配置（软编码区：像仪表盘一样统一管理参数）
# =============================================================================

DATA_ROOT = "./data/unet_training"  # 你的数据集根目录
DEM_VAE_CKPT = ""                   # 高程 VAE 权重路径 (为空则临时用 RGB_VAE 代替)
OUTPUT_DIR = "./outputs/unet_8ch"   # 模型和图片输出的根目录

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
USE_AMP = True                      # 是否开启 fp16 混合精度加速

SAVE_STEPS = 1000                   # 每隔多少步保存一次常规快照
LOG_STEPS = 10                      # 每隔多少步在控制台打印一次日志
VIZ_INTERVAL = 1                    # 每隔几个 epoch 画一次 Loss 曲线图


class UNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建输出目录和可视化目录
        self.output_dir = Path(args.output_dir)
        self.viz_output_dir = self.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于记录画图的数据
        self.loss_history = []  # 格式: [(epoch, total_loss, img_loss, dem_loss), ...]
        self.global_step = 0
        
        # 新增：记录最佳 Loss 的变量
        self.best_loss = float("inf")

        # ==========================================
        # 1. 组建做题家 (U-Net) 与优化器
        # ==========================================
        print("正在组装 8 通道 U-Net...")
        self.unet = build_unet(in_channels=8, out_channels=8, cross_attention_dim=768).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(), 
            lr=args.learning_rate, 
            weight_decay=WEIGHT_DECAY
        )
        
        # ==========================================
        # 2. 挂载黑盒翻译官 (CLIP)
        # ==========================================
        print("正在加载 CLIP 文本编码器...")
        model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(self.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        # ==========================================
        # 3. 挂载空间压缩器 (两个 VAE)
        # ==========================================
        print("正在加载 SD 标准 VAE...")
        self.rgb_vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        self.rgb_vae.eval()
        self.rgb_vae.requires_grad_(False)

        # 临时脚手架：如果没传 DEM VAE，就暂时不加载，交由 encode_to_latent 处理
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

        self.scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
        
        val_batch = next(iter(self.dataloader))
        self.val_rgb = val_batch["rgb"][:1].to(self.device)
        self.val_dem = val_batch["dem"][:1].to(self.device)
        self.val_prompt = val_batch["prompt"][:1]
        
        print(f"验证样本准备就绪，提示词: {self.val_prompt[0]}")

    def encode_to_latent(self, rgb_pixels, dem_pixels):
        """核心魔法：把 512x512 的像素变成 64x64 的隐向量"""
        rgb_latent = self.rgb_vae.encode(rgb_pixels).latent_dist.sample() * 0.18215
        
        if self.dem_vae is not None:
            dem_latent = self.dem_vae.encode(dem_pixels).latent_dist.sample() * 0.18215
        else:
            # 临时脚手架：强行把单通道 DEM 变 3 通道喂给 RGB VAE
            dem_pixels_3ch = dem_pixels.repeat(1, 3, 1, 1)
            dem_latent = self.rgb_vae.encode(dem_pixels_3ch).latent_dist.sample() * 0.18215
            
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

            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                with torch.no_grad():
                    text_inputs = self.tokenizer(
                        prompts, padding="max_length", max_length=self.tokenizer.model_max_length, 
                        truncation=True, return_tensors="pt"
                    ).to(self.device)

                    clip_output = self.text_encoder(text_inputs.input_ids)
                    encoder_hidden_states = clip_output[0]
                    pooled_features = clip_output[1]

                    latents = self.encode_to_latent(rgb_pixels, dem_pixels)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = self.unet(
                    noisy_latent=noisy_latents,
                    timestep=timesteps,
                    local_features=encoder_hidden_states,
                    global_features=pooled_features,
                )
                
                loss_dict = self.unet.loss(noise_pred, noise)
                loss = loss_dict["loss"]
                loss_img = loss_dict["loss_img"]
                loss_dem = loss_dict["loss_dem"]

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

            # 常规步数保存（应对突然断电）
            if self.global_step % self.args.save_steps == 0:
                save_path = self.output_dir / f"unet_step_{self.global_step}.pt"
                torch.save(self.unet.state_dict(), save_path)

        # 返回当前 epoch 的平均 loss
        return {
            "loss": total_loss_epoch / num_batches,
            "img": total_img_loss / num_batches,
            "dem": total_dem_loss / num_batches
        }

    @torch.no_grad()
    def visualize_epoch(self, epoch: int):
        """利用 matplotlib 绘制并保存 Loss 曲线图"""
        if len(self.loss_history) == 0:
            return

        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_img = [h[2] for h in self.loss_history]
        losses_dem = [h[3] for h in self.loss_history]

        fig = plt.figure(figsize=(18, 5))
        fig.suptitle(f"U-Net 8-Channel Training — Epoch {epoch}", fontsize=14, fontweight="bold")

        # 1. 绘制总 Loss
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(epochs, losses_total, "b-", linewidth=1.5, marker='o', markersize=4)
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)

        # 2. 绘制 RGB 图像 Loss
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(epochs, losses_img, "g-", linewidth=1.5, marker='o', markersize=4)
        ax2.set_title("RGB Texture Loss")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        # 3. 绘制 DEM 高程 Loss
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(epochs, losses_dem, "r-", linewidth=1.5, marker='o', markersize=4)
        ax3.set_title("DEM Height Loss")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_output_dir / f"loss_curve_epoch_{epoch:04d}.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def train(self):
        """控制完整的训练循环，负责保存最佳状态"""
        print(f"=== 开始训练 8 通道 U-Net ===")
        print(f"数据集: {self.args.data_root}")
        print(f"Batch Size: {self.args.batch_size} | Epochs: {self.args.epochs}")
        print("-" * 50)
        
        for epoch in range(self.args.epochs):
            # 执行一个 Epoch 的训练
            avg_loss = self.train_epoch(epoch)
            
            # 打印 Epoch 级日志
            print(
                f"\nEpoch {epoch:3d} | "
                f"Total Loss: {avg_loss['loss']:.4f} | "
                f"Img Loss: {avg_loss['img']:.4f} | "
                f"Dem Loss: {avg_loss['dem']:.4f}"
            )
            
            # ⚠️ 拦截并保存最佳模型
            if avg_loss["loss"] < self.best_loss:
                print(f"🌟 发现新的最佳 Loss ({self.best_loss:.4f} -> {avg_loss['loss']:.4f})，正在保存...")
                self.best_loss = avg_loss["loss"]
                best_path = self.output_dir / "best_unet.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.unet.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
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
        # 训练结束后保存最后一轮的权重作为 final
        torch.save({
            "epoch": self.args.epochs - 1,
            "model_state_dict": self.unet.state_dict(),
        }, self.output_dir / "unet_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这里的 default 优先读取文件顶部的软编码大写变量
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
    
    parser.add_argument(
        "--use_amp", 
        type=bool, 
        default=USE_AMP
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

    