"""
联合隐空间管理工具

根据 roadmap 描述：
- 纹理隐向量：4x64x64（SD VAE 压缩）
- 高度隐向量：4x64x64（魔改 VAE 压缩）
- 联合隐向量 z_0：8x64x64（torch.cat 拼接）

本模块提供隐向量的拼接、拆分、加噪、去噪等工具函数
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def concatenate_latents(
    height_latent: torch.Tensor,
    texture_latent: torch.Tensor,
) -> torch.Tensor:
    """
    拼接高度隐向量和纹理隐向量

    Args:
        height_latent: 高度隐向量 [B, 4, 64, 64]
        texture_latent: 纹理隐向量 [B, 4, 64, 64]
    Returns:
        joint_latent: 联合隐向量 [B, 8, 64, 64]
    """
    # 按通道维度拼接
    joint_latent = torch.cat([height_latent, texture_latent], dim=1)
    return joint_latent


def split_joint_latent(
    joint_latent: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    拆分联合隐向量

    Args:
        joint_latent: 联合隐向量 [B, 8, 64, 64]
    Returns:
        height_latent: 高度隐向量 [B, 4, 64, 64]
        texture_latent: 纹理隐向量 [B, 4, 64, 64]
    """
    # 从中间劈开
    height_latent = joint_latent[:, :4, :, :]  # 前 4 通道
    texture_latent = joint_latent[:, 4:, :, :]  # 后 4 通道
    return height_latent, texture_latent


def add_noise(
    z_0: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    前向加噪过程（DDPM）

    把噪声按照时间步 t 的强度，混入联合隐向量 z_0 中

    Args:
        z_0: 原始联合隐向量 [B, 8, 64, 64]
        t: 时间步 [B]，范围 [0, T-1]
        alphas_cumprod: 累积 alpha 乘积 [T]，预定义调度

    Returns:
        z_t: 加噪后的脏图 [B, 8, 64, 64]
        noise: 添加的噪声 [B, 8, 64, 64]
    """
    # 生成随机高斯噪声
    noise = torch.randn_like(z_0)

    # 获取对应时间步的 alpha_cumprod
    # alphas_cumprod[t] 对应每个样本的 alpha_bar_t
    alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)

    # DDPM 加噪公式：
    # z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod_t)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod_t)

    z_t = sqrt_alpha_cumprod * z_0 + sqrt_one_minus_alpha_cumprod * noise

    return z_t, noise


def extract_into_tensor(
    a: torch.Tensor,
    t: torch.Tensor,
    x_shape: torch.Size,
) -> torch.Tensor:
    """
    从一维张量中提取对应时间步的值，并调整为适合广播的形状

    Args:
        a: 一维张量 [T]，如 alphas_cumprod
        t: 时间步 [B]
        x_shape: 目标形状，如 [B, 8, 64, 64]
    Returns:
        a_t: 调整后的张量 [B, 1, 1, 1]，便于广播
    """
    # 收集对应时间步的值
    res = a.gather(-1, t.cpu())

    # 调整形状以匹配 x_shape
    # [B] -> [B, 1, 1, 1]
    res = res.to(t.device).view(-1, 1, 1, 1)

    # 广播到目标形状
    while len(res.shape) < len(x_shape):
        res = res.unsqueeze(-1)

    return res


class DDIMScheduler:
    """
    DDIM 采样器调度类

    用于推理阶段的去噪过程
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        """
        初始化 DDIM 调度器

        Args:
            num_train_timesteps: 训练时总时间步数
            beta_start: beta 起始值
            beta_end: beta 结束值
        """
        self.num_train_timesteps = num_train_timesteps

        # 线性 beta 调度
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)

        # 计算 alpha 和累积 alpha
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 用于 DDIM 采样的额外计算
        # TODO: 初始化 DDIM 特定参数

    def set_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """
        设置推理时的时间步序列

        Args:
            num_inference_steps: 推理步数（如 50 步）
        Returns:
            timesteps: 时间步序列
        """
        # TODO: 实现 DDIM 时间步采样
        # 从 1000 步中均匀采样 50 步
        raise NotImplementedError("时间步设置功能待实现")

    def step(
        self,
        noise_pred: torch.Tensor,
        t: int,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM 单步去噪

        Args:
            noise_pred: U-Net 预测的噪声 [B, 8, 64, 64]
            t: 当前时间步
            z_t: 当前带噪隐向量
        Returns:
            z_t_minus_1: 去噪后的隐向量
        """
        # TODO: 实现 DDIM 去噪公式
        raise NotImplementedError("DDIM 去噪功能待实现")
