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

    def __init__(self):
        super().__init__(
            in_channels=1,
            out_channels=1,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            block_out_channels=(64, 128, 256, 512),
            layers_per_block=2,
            act_fn="silu",
            # 焊死 4 通道，与纹理 VAE 对齐
            latent_channels=4,
            sample_size=512,
            scaling_factor=0.18215,
        )

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
        使用中心差分法计算 x 和 y 方向的梯度
        """
        # x 方向梯度（左右错位相减）
        grad_x = height_map[:, :, :, 1:] - height_map[:, :, :, :-1]
        # y 方向梯度（上下错位相减）
        grad_y = height_map[:, :, 1:, :] - height_map[:, :, :-1, :]

        # 梯度幅值
        slope = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return slope

    def compute_curvature(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        计算曲率（二阶导数）
        使用拉普拉斯算子近似
        """
        # 二阶中心差分
        dxx = (
            height_map[:, :, :, 2:]
            - 2 * height_map[:, :, :, 1:-1]
            + height_map[:, :, :, :-2]
        )
        dyy = (
            height_map[:, :, 2:, :]
            - 2 * height_map[:, :, 1:-1, :]
            + height_map[:, :, :-2, :]
        )

        # 平均曲率近似
        curvature = (dxx + dyy) / 2.0
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
        # 对齐形状
        min_h = min(slope_pred.shape[2], slope_target.shape[2])
        min_w = min(slope_pred.shape[3], slope_target.shape[3])
        slope_pred = slope_pred[:, :, :min_h, :min_w]
        slope_target = slope_target[:, :, :min_h, :min_w]
        loss_slope = F.mse_loss(slope_pred, slope_target)

        # 曲率损失（二阶梯度）
        curvature_pred = self.compute_curvature(pred)
        curvature_target = self.compute_curvature(target)
        # 对齐形状
        min_h_c = min(curvature_pred.shape[2], curvature_target.shape[2])
        min_w_c = min(curvature_pred.shape[3], curvature_target.shape[3])
        curvature_pred = curvature_pred[:, :, :min_h_c, :min_w_c]
        curvature_target = curvature_target[:, :, :min_h_c, :min_w_c]
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
        posterior = self.encode(height_map)
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
