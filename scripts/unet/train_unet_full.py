"""
8 通道 U-Net 训练脚本

基于 DDPM 的联合生成训练：将 RGB 纹理图与 DEM 高度图分别编码为隐向量，
拼接为 8 通道联合隐变量（通道 0-3: RGB 纹理, 通道 4-7: DEM 高程），
由 U-Net 在 CLIP 文本特征的指导下预测所添加的噪声。

用法：
    python scripts/unet/train_unet_full.py
    python scripts/unet/train_unet_full.py --epochs 100 --batch_size 8 --no-amp
    python scripts/unet/train_unet_full.py --data_root ./data/my_dataset --dem_vae_ckpt ./outputs/vae/best.pt

数据目录结构：
    data_root/
      ├── rgb/      # RGB 纹理图 (.png / .jpg)
      ├── dem/      # DEM 高度图 (.npy / .png / .tif)
      └── txt/      # 文本提示词 (.txt)，与图片同名
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from diffusers import DDPMScheduler, AutoencoderKL
from diffusers import logging as diffusers_logging
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging

from dataset.unet_dataset import UNetDataset
from models.unet.unet_8ch import build_unet
from models.vae.heightmap_vae import HeightMapVAE

diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


# =============================================================================
# 默认超参数（可通过命令行参数覆盖）
# =============================================================================

_DEFAULT_DATA_ROOT = "./data/unet_training"
_DEFAULT_DEM_VAE_CKPT = ""
_DEFAULT_OUTPUT_DIR = "./outputs/unet_8ch"
_DEFAULT_EPOCHS = 50
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_LEARNING_RATE = 1e-4
_DEFAULT_WEIGHT_DECAY = 1e-4
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_USE_AMP = True
_DEFAULT_SAVE_STEPS = 1000
_DEFAULT_VIZ_INTERVAL = 1


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="8 通道 U-Net 训练脚本")

    parser.add_argument(
        "--data_root", type=str, default=_DEFAULT_DATA_ROOT, help="数据集根目录"
    )
    parser.add_argument(
        "--dem_vae_ckpt",
        type=str,
        default=_DEFAULT_DEM_VAE_CKPT,
        help="高度图 VAE 权重路径（为空则用 RGB VAE 替代）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR, help="模型输出根目录"
    )
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS, help="训练轮数")
    parser.add_argument(
        "--batch_size", type=int, default=_DEFAULT_BATCH_SIZE, help="批次大小"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=_DEFAULT_LEARNING_RATE, help="学习率"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=_DEFAULT_WEIGHT_DECAY, help="权重衰减系数"
    )
    parser.add_argument(
        "--num_workers", type=int, default=_DEFAULT_NUM_WORKERS, help="数据加载进程数"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=_DEFAULT_SAVE_STEPS,
        help="每隔多少步保存一次快照",
    )
    parser.add_argument(
        "--viz_interval",
        type=int,
        default=_DEFAULT_VIZ_INTERVAL,
        help="每隔几个 epoch 绘制一次 loss 曲线（0 表示不绘制）",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=_DEFAULT_USE_AMP,
        help="启用 fp16 混合精度训练",
    )
    parser.add_argument(
        "--no-amp", dest="use_amp", action="store_false", help="禁用混合精度训练"
    )

    return parser


class UNetTrainer:
    """8 通道 U-Net 训练器，管理模型构建、训练循环、checkpoint 和可视化。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 输出目录与可视化子目录
        self.output_dir = Path(args.output_dir)
        self.viz_output_dir = self.output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_output_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.loss_history: list[
            tuple
        ] = []  # [(epoch, total_loss, img_loss, dem_loss), ...]
        self.global_step = 0
        self.best_loss = float("inf")

        self._build_model()
        self._build_condition_encoders()
        self._build_vae_encoders()
        self._build_dataloader()
        self._build_scaler()

        print(f"设备: {self.device} | 混合精度: {args.use_amp}")
        print(
            f"数据集: {args.data_root} | 批次: {args.batch_size} | 轮数: {args.epochs}"
        )
        print("-" * 50)

    def _build_model(self) -> None:
        """构建 8 通道 U-Net 和优化器。"""
        print("构建 8 通道 U-Net...")
        self.unet = build_unet(
            in_channels=8, out_channels=8, cross_attention_dim=768
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _build_condition_encoders(self) -> None:
        """加载 CLIP 文本编码器并冻结参数。"""
        print("加载 CLIP 文本编码器 (ViT-L/14)...")
        model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(self.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

    def _build_vae_encoders(self) -> None:
        """加载 RGB VAE（固定）与高度图 VAE（可选）。"""
        print("加载 SD VAE...")
        self.rgb_vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        ).to(self.device)
        self.rgb_vae.eval()
        self.rgb_vae.requires_grad_(False)

        if self.args.dem_vae_ckpt:
            print("加载高度图 VAE...")
            self.dem_vae = HeightMapVAE(block_out_channels=(128, 256, 512, 512)).to(
                self.device
            )
            ckpt = torch.load(self.args.dem_vae_ckpt, map_location=self.device)
            self.dem_vae.load_state_dict(ckpt["model_state_dict"])
            self.dem_vae.eval()
            self.dem_vae.requires_grad_(False)
        else:
            self.dem_vae = None
            print("未提供高度图 VAE 权重，将使用 RGB VAE 对 DEM 编码（降级方案）。")

    def _build_dataloader(self) -> None:
        """构建数据集与 DataLoader，并固定一个验证批次。"""
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        dataset = UNetDataset(data_root=self.args.data_root, augment=True)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        # 固定一个验证批次，用于周期性验证 loss 计算
        val_batch = next(iter(self.dataloader))
        self.val_rgb = val_batch["rgb"][:1].to(self.device)
        self.val_dem = val_batch["dem"][:1].to(self.device)
        self.val_prompt = val_batch["prompt"][:1]

    def _build_scaler(self) -> None:
        """构建 GradScaler（仅在 CUDA + AMP 时启用）。"""
        use_cuda_amp = self.args.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device="cuda") if use_cuda_amp else None

    # -------------------------------------------------------------------------
    # 核心编码逻辑
    # -------------------------------------------------------------------------

    def encode_to_latent(
        self, rgb_pixels: torch.Tensor, dem_pixels: torch.Tensor
    ) -> torch.Tensor:
        """
        将 RGB 与 DEM 像素编码为 8 通道联合隐向量。

        流程：
        1. RGB [B, 3, 512, 512] → RGB VAE → latent × 0.18215 → [B, 4, 64, 64]
        2. DEM [B, 1, 512, 512] → DEM VAE（或 3 通道 RGB VAE 降级方案）
           → latent × 0.18215 → [B, 4, 64, 64]
        3. 通道拼接 → [B, 8, 64, 64]
        """
        rgb_latent = self.rgb_vae.encode(rgb_pixels).latent_dist.sample() * 0.18215

        if self.dem_vae is not None:
            dem_latent = self.dem_vae.encode(dem_pixels).latent_dist.sample() * 0.18215
        else:
            # 降级方案：单通道 DEM 复制为 3 通道后，用 RGB VAE 编码
            dem_pixels_3ch = dem_pixels.repeat(1, 3, 1, 1)
            dem_latent = (
                self.rgb_vae.encode(dem_pixels_3ch).latent_dist.sample() * 0.18215
            )

        return torch.cat([rgb_latent, dem_latent], dim=1)

    # -------------------------------------------------------------------------
    # 训练与验证
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def compute_validation_loss(self) -> dict:
        """
        在固定验证批次上计算一次 loss（不更新参数），用于监控训练趋势。
        """
        self.unet.eval()

        with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
            text_inputs = self.tokenizer(
                self.val_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            clip_output = self.text_encoder(text_inputs.input_ids)
            encoder_hidden_states = clip_output[0]
            pooled_features = clip_output[1]

            latents = self.encode_to_latent(self.val_rgb, self.val_dem)
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = self.unet(
                noisy_latent=noisy_latents,
                timestep=timesteps,
                global_features=pooled_features,
                local_features=encoder_hidden_states,
            )
            loss_dict = self.unet.loss(noise_pred, noise)

        self.unet.train()
        return loss_dict

    def train_epoch(self, epoch: int) -> dict:
        """
        执行一个 epoch 的训练，返回平均 loss。
        """
        self.unet.train()

        total_loss, total_img_loss, total_dem_loss = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch:3d}", leave=False)

        for batch in pbar:
            rgb_pixels = batch["rgb"].to(self.device)
            dem_pixels = batch["dem"].to(self.device)
            prompts = batch["prompt"]
            batch_size = rgb_pixels.shape[0]

            with torch.autocast(device_type="cuda", enabled=self.args.use_amp):
                # 文本编码（冻结的 CLIP）
                with torch.no_grad():
                    text_inputs = self.tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)

                    clip_output = self.text_encoder(text_inputs.input_ids)
                    encoder_hidden_states = clip_output[0]
                    pooled_features = clip_output[1]

                    latents = self.encode_to_latent(rgb_pixels, dem_pixels)

                # 加噪
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=self.device,
                ).long()
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # U-Net 前向
                noise_pred = self.unet(
                    noisy_latent=noisy_latents,
                    timestep=timesteps,
                    global_features=pooled_features,
                    local_features=encoder_hidden_states,
                )

                loss_dict = self.unet.loss(noise_pred, noise)

            # 反向传播（AMP 或普通模式）
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss_dict["loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict["loss"].backward()
                self.optimizer.step()

            # 统计
            total_loss += loss_dict["loss"].item()
            total_img_loss += loss_dict["loss_img"].item()
            total_dem_loss += loss_dict["loss_dem"].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / num_batches:.4f}",
                    "img": f"{total_img_loss / num_batches:.4f}",
                    "dem": f"{total_dem_loss / num_batches:.4f}",
                }
            )

            if self.global_step % self.args.save_steps == 0:
                ckpt_path = self.output_dir / f"unet_step_{self.global_step}.pt"
                torch.save(self.unet.state_dict(), ckpt_path)

        return {
            "loss": total_loss / num_batches,
            "img": total_img_loss / num_batches,
            "dem": total_dem_loss / num_batches,
        }

    # -------------------------------------------------------------------------
    # 可视化
    # -------------------------------------------------------------------------

    def visualize_loss_curve(self, epoch: int) -> None:
        """绘制并保存三条 Loss 曲线（总损失 / RGB / DEM）。"""
        if not self.loss_history:
            return

        epochs = [h[0] for h in self.loss_history]
        losses_total = [h[1] for h in self.loss_history]
        losses_img = [h[2] for h in self.loss_history]
        losses_dem = [h[3] for h in self.loss_history]

        fig = plt.figure(figsize=(18, 5))
        fig.suptitle(
            f"U-Net 8-Channel Training — Epoch {epoch}", fontsize=14, fontweight="bold"
        )

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(epochs, losses_total, "b-", linewidth=1.5, marker="o", markersize=4)
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(epochs, losses_img, "g-", linewidth=1.5, marker="o", markersize=4)
        ax2.set_title("RGB Texture Loss")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(epochs, losses_dem, "r-", linewidth=1.5, marker="o", markersize=4)
        ax3.set_title("DEM Height Loss")
        ax3.set_xlabel("Epoch")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.viz_output_dir / f"loss_curve_epoch_{epoch:04d}.png", dpi=120)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 训练主循环
    # -------------------------------------------------------------------------

    def train(self) -> None:
        """运行完整训练循环。"""
        for epoch in range(self.args.epochs):
            avg_loss = self.train_epoch(epoch)

            tqdm.write(
                f"Epoch {epoch:3d} | "
                f"Total: {avg_loss['loss']:.4f} | "
                f"Img: {avg_loss['img']:.4f} | "
                f"Dem: {avg_loss['dem']:.4f}"
            )

            # 周期性验证 loss（不参与训练，仅观察）
            if epoch % max(1, self.args.viz_interval) == 0:
                val_loss_dict = self.compute_validation_loss()
                tqdm.write(
                    f"      验证 | "
                    f"Total: {val_loss_dict['loss']:.4f} | "
                    f"Img: {val_loss_dict['loss_img']:.4f} | "
                    f"Dem: {val_loss_dict['loss_dem']:.4f}"
                )

            # 保存最佳模型（基于训练 loss）
            if avg_loss["loss"] < self.best_loss:
                tqdm.write(
                    f"      新的最佳 Loss ({self.best_loss:.4f} -> {avg_loss['loss']:.4f})，保存 checkpoint..."
                )
                self.best_loss = avg_loss["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.unet.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_loss": self.best_loss,
                    },
                    self.output_dir / "best_unet.pt",
                )

            # 记录并绘制 loss 曲线
            self.loss_history.append(
                (epoch, avg_loss["loss"], avg_loss["img"], avg_loss["dem"])
            )
            if self.args.viz_interval > 0 and (
                epoch % self.args.viz_interval == 0 or epoch == self.args.epochs - 1
            ):
                self.visualize_loss_curve(epoch)

        tqdm.write(f"训练完成！历史最佳 Loss: {self.best_loss:.4f}")
        torch.save(
            {
                "epoch": self.args.epochs - 1,
                "model_state_dict": self.unet.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.output_dir / "unet_final.pt",
        )


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    trainer = UNetTrainer(args)
    trainer.train()
