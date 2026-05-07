import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL


class HeightMapVAE(AutoencoderKL):
    """
    魔改的高程图 VAE 模型
    输入：1 通道 512x512 高程图（归一化到 [-1, 1]，使用全局最大高程 H_max=3000m）
    输出：1 通道 512x512 重建高程图
    隐空间：固定为 [4, 64, 64]，与纹理 VAE 对齐
    """

    H_MAX = 3000.0  # 全局最大高程用于归一化

    """
    参数 AutoencoderKL
    in_channels: int，可选，默认值为 3 输入图像的通道数。 
    out_channels: int，可选，默认值为 3 输出通道数。 
    down_block_types: ，可选，默认值为 下采样块类型的元组。 
    up_block_types: ，可选，默认值为 上采样块类型的元组。 
    block_out_channels: ，可选，默认值为 块输出通道的元组。 
    act_fn: ，可选，默认值为 使用的激活函数。 
    latent_channels: ，可选，默认值为 4 潜在空间的通道数。 
    sample_size: ，可选，默认值为 样本输入大小。 
    scaling_factor: ，可选，默认值为 0.18215
    """

    def __init__(self, block_out_channels=(128, 256, 512), enable_grad_checkpointing=False):
        num_blocks = len(block_out_channels)
        super().__init__(
            in_channels=1,
            out_channels=1,
            down_block_types=["DownEncoderBlock2D"] * num_blocks,
            up_block_types=["UpDecoderBlock2D"] * num_blocks,
            block_out_channels=block_out_channels,
            layers_per_block=2,
            act_fn="silu",
            latent_channels=4,
            sample_size=512,
            scaling_factor=1.0,
        )
        if enable_grad_checkpointing:
            self.enable_gradient_checkpointing()

    @staticmethod
    def normalize_height(height_map: torch.Tensor) -> torch.Tensor:
        """
        使用全局最大高程归一化到 [-1, 1]
        Args:
            height_map: 原始高程图，单位米
        Returns:
            归一化后的高程图，范围 [-1, 1]
        """
        return height_map / HeightMapVAE.H_MAX

    @staticmethod
    def denormalize_height(normalized_map: torch.Tensor) -> torch.Tensor:
        """
        反归一化到真实高程
        Args:
            normalized_map: 归一化后的高程图，范围 [-1, 1]
        Returns:
            原始高程图，单位米
        """
        return normalized_map * HeightMapVAE.H_MAX

    def compute_slope(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        计算坡度（梯度幅值）
        使用 3×3 Sobel 算子计算 x 和 y 方向的梯度，输出与输入等高宽

        内部强制 fp32 计算，避免 AMP fp16 下 sqrt(eps==0) 的无穷梯度。
        torch.autocast 会拦截 F.conv2d 并重铸为 fp16，因此必须显式禁用。
        """
        with torch.autocast(device_type=height_map.device.type, enabled=False):
            hmap_f32 = height_map.float()
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32, device=height_map.device,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32, device=height_map.device,
            ).view(1, 1, 3, 3)
            grad_x = F.conv2d(hmap_f32, sobel_x, padding=1)  # [B,1,H,W]
            grad_y = F.conv2d(hmap_f32, sobel_y, padding=1)  # [B,1,H,W]
            slope = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            return slope

    def compute_curvature(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        计算曲率（二阶导数）
        使用拉普拉斯算子近似，输出与输入等高宽

        内部强制 fp32 计算，避免 AMP fp16 下卷积精度损失。
        torch.autocast 会拦截 F.conv2d 并重铸为 fp16，因此必须显式禁用。
        """
        with torch.autocast(device_type=height_map.device.type, enabled=False):
            hmap_f32 = height_map.float()
            kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=torch.float32, device=height_map.device,
            ).view(1, 1, 3, 3)
            curvature = F.conv2d(hmap_f32, kernel, padding=1)  # [B,1,H,W]
            return curvature

    def compute_geo_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        地形几何约束损失
        强制解码输出的高程图在陡崖边缘的梯度变化与真实数据一致
        """
        # 坡度损失（一阶梯度）
        slope_pred = self.compute_slope(pred)
        slope_target = self.compute_slope(target)
        loss_slope = F.mse_loss(slope_pred, slope_target)

        # 曲率损失（二阶梯度）
        curvature_pred = self.compute_curvature(pred)
        curvature_target = self.compute_curvature(target)
        loss_curvature = F.mse_loss(curvature_pred, curvature_target)

        # 组合几何损失
        loss_geo = loss_slope + 0.5 * loss_curvature
        return loss_geo

    def forward(
        self,
        height_map: torch.Tensor,
        return_recon_only: bool = False,
    ):
        """
        前向传播
        Args:
            height_map: 输入高程图 [B, 1, 512, 512]，已归一化到 [-1, 1]
            return_recon_only: 是否只返回重建结果
        Returns:
            如果 return_recon_only=True: 返回 recon_height_map [B, 1, 512, 512]
            否则返回 (recon_height_map, loss_dict)
        """
        # 使用父类的编码器 - 解码器结构
        posterior = self.encode(height_map).latent_dist
        latent = posterior.sample()
        recon = self.decode(latent).sample

        if return_recon_only:
            return recon

        # 计算各项损失
        loss_mse = F.mse_loss(recon, height_map)
        loss_kl = posterior.kl().mean()
        loss_geo = self.compute_geo_loss(recon, height_map)

        # 组合损失
        loss_vae = loss_mse + 1e-6 * loss_kl + 0.8 * loss_geo

        return recon, {
            "loss": loss_vae,
            "loss_mse": loss_mse,
            "loss_kl": loss_kl,
            "loss_geo": loss_geo,
        }
