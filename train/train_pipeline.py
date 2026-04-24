"""
训练流水线模块

实现 roadmap 中描述的训练过程：
1. 文本编码（题目翻译）
2. 图像压缩（答案打包）
3. 前向加噪（破坏答案）
4. 模型预测（做题）
5. 计算误差与反向传播（对答案与自我修正）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional


class TrainingPipeline:
    """
    训练流水线

    输入：真实三元组数据 (Prompt 字符串，真实高度图矩阵，真实纹理图矩阵)
    目的：更新 8 通道 U-Net 和交叉注意力层的权重
    """

    def __init__(
        self,
        unet: nn.Module,
        text_encoder: nn.Module,
        height_vae: nn.Module,
        texture_vae: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
    ):
        """
        初始化训练流水线

        Args:
            unet: 8 通道 U-Net 模型
            text_encoder: 双分支 CLIP 文本编码器
            height_vae: 高程图 VAE 编码器/解码器
            texture_vae: 纹理图 VAE 编码器/解码器（SD VAE）
            optimizer: 优化器（如 AdamW）
            device: 运行设备
        """
        self.unet = unet
        self.text_encoder = text_encoder
        self.height_vae = height_vae
        self.texture_vae = texture_vae
        self.optimizer = optimizer
        self.device = device

        # 将所有模型移到指定设备
        self.unet.to(device)
        self.text_encoder.to(device)
        self.height_vae.to(device)
        if self.texture_vae is not None:
            self.texture_vae.to(device)

    def encode_text(self, prompts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        步骤 1: 文本编码（题目翻译）

        Args:
            prompts: Prompt 字符串列表
        Returns:
            global_features: 全局特征向量 [B, D_global]
            local_features: 细节特征向量 [B, N, D_local]
        """
        # TODO: 实现双分支 CLIP 编码
        # global_features, local_features = self.text_encoder(prompts)
        raise NotImplementedError("文本编码功能待实现")

    def encode_images(
        self,
        height_maps: torch.Tensor,
        texture_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        步骤 2: 图像压缩（答案打包）

        Args:
            height_maps: 真实高度图 [B, 1, 512, 512]
            texture_maps: 真实纹理图 [B, 3, 512, 512]
        Returns:
            joint_latent: 联合隐向量 z_0 [B, 8, 64, 64]
        """
        # 压缩纹理图
        texture_latent = self.texture_vae.encode(texture_maps)

        # 压缩高程图
        height_latent = self.height_vae.encode(height_maps)

        # 拼接：[B, 4, 64, 64] + [B, 4, 64, 64] -> [B, 8, 64, 64]
        joint_latent = torch.cat([height_latent, texture_latent], dim=1)

        return joint_latent

    def add_noise(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        步骤 3: 前向加噪（破坏答案）

        Args:
            z_0: 原始联合隐向量 [B, 8, 64, 64]
            t: 时间步 [B]
        Returns:
            z_t: 加噪后的脏图 [B, 8, 64, 64]
            noise: 实际添加的噪声 [B, 8, 64, 64]
        """
        # TODO: 实现 DDPM 加噪过程
        # noise = torch.randn_like(z_0)
        # z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
        raise NotImplementedError("加噪功能待实现")

    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        步骤 4: 模型预测（做题）

        Args:
            z_t: 脏图 [B, 8, 64, 64]
            t: 时间步 [B]
            global_features: 全局文本特征
            local_features: 细节文本特征
        Returns:
            noise_pred: 预测的噪声 [B, 8, 64, 64]
        """
        # TODO: U-Net 前向传播
        # noise_pred = self.unet(z_t, t, global_features, local_features)
        raise NotImplementedError("噪声预测功能待实现")

    def compute_loss(
        self,
        noise_true: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        步骤 5: 计算误差

        Args:
            noise_true: 真实噪声
            noise_pred: 预测噪声
        Returns:
            loss: MSE 损失
        """
        loss = nn.functional.mse_loss(noise_pred, noise_true)
        return loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        单个训练步骤

        Args:
            batch: 一个 batch 的数据，包含:
                - prompts: Prompt 字符串列表
                - height_maps: 真实高度图 [B, 1, 512, 512]
                - texture_maps: 真实纹理图 [B, 3, 512, 512]
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 设置训练模式
        self.unet.train()
        self.text_encoder.train()
        self.height_vae.train()
        self.texture_vae.train()

        # 清空梯度
        self.optimizer.zero_grad()

        prompts = batch["prompts"]
        height_maps = batch["height_maps"].to(self.device)
        texture_maps = batch["texture_maps"].to(self.device)

        # 步骤 1: 文本编码
        global_features, local_features = self.encode_text(prompts)

        # 步骤 2: 图像压缩
        z_0 = self.encode_images(height_maps, texture_maps)

        # 步骤 3: 前向加噪
        # 随机生成时间步 t
        batch_size = z_0.shape[0]
        t = torch.randint(0, 1000, (batch_size,), device=self.device)
        z_t, noise_true = self.add_noise(z_0, t)

        # 步骤 4: 模型预测
        noise_pred = self.predict_noise(z_t, t, global_features, local_features)

        # 步骤 5: 计算误差与反向传播
        loss = self.compute_loss(noise_true, noise_pred)
        loss.backward()

        # 更新权重
        self.optimizer.step()

        return {
            "loss": loss.item(),
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 编号
        Returns:
            avg_loss_dict: 平均损失字典
        """
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss_dict = self.train_step(batch)
            total_loss += loss_dict["loss"]
            num_batches += 1

        avg_loss = total_loss / num_batches

        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
        }
