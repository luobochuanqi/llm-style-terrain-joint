"""
8通道 U-Net 模型 —— 纹理与高程联合扩散

本模块实现文本条件的 8 通道 U-Net，用于纹理 + 高度图的隐空间联合生成。
文本条件通过两条路径注入：

    1. 全局特征（CLIP pooled 输出）经线性投影后叠加到时间步嵌入上，统一调制所有残差块。
       注意：当前 SD UNet2DConditionModel 训练中 global_features 路径暂未启用，
       仅使用 cross-attention（local_features）。本模块的 global_text_proj 仅保留接口。
    2. 序列级特征（CLIP hidden states）作为 encoder_hidden_states 传入交叉注意力层，
        实现空间感知的条件控制。

结构概览：
    输入  [B, 8, 64, 64]   (4 通道纹理 + 4 通道高度)
      conv_in  →  [B, 320, 64, 64]
      Down blocks (3×CrossAttn + 1×Down)  →  多层特征
      Mid block (CrossAttn)                →  瓶颈
      Up blocks (1×Up + 3×CrossAttn)       →  跳跃连接重建
      conv_out  →  [B, 8, 64, 64]
    输出: 预测噪声
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
)
from diffusers.models.activations import get_activation


class UNet8Channel(nn.Module):
    """
    文本条件的 8 通道 U-Net，用于纹理与高程联合去噪。

    参数
    ----------
    in_channels : int
        输入通道数，默认 8（ch 0-3 纹理 + ch 4-7 高度）。
    out_channels : int
        输出通道数，默认 8。
    down_block_types : tuple of str
        下采样路径中各 block 的类型序列。
    up_block_types : tuple of str
        上采样路径中各 block 的类型序列。
    block_out_channels : tuple of int
        每个 block 的输出通道数。
    layers_per_block : int
        每个下采样/中间 block 的 ResNet 层数。
        上采样 block 自动使用 ``layers_per_block + 1`` 层，
        多出的一层用于消化跳跃连接拼接后的特征。
    attention_head_dim : int
        交叉注意力中每个注意力头的维度。
    cross_attention_dim : int
        文本编码器输出的特征维度（如 CLIP ViT-L/14 为 768）。
    use_linear_projection : bool
        为 attention 投影方式预留的参数，当前未使用。
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        cross_attention_dim: int = 768,
        use_linear_projection: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block

        time_embed_dim = self.block_out_channels[0] * 4

        # ---- 时间步编码 ----
        # 将标量时间步 t 展开为正余弦向量，再映射为全局控制向量
        self.time_proj = Timesteps(
            num_channels=self.block_out_channels[0],
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=self.block_out_channels[0],
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # ---- 全局文本特征投影 ----
        # 将 CLIP 全局特征映射到时间嵌入维度，叠加到 t_emb 上
        self.global_text_proj = nn.Linear(cross_attention_dim, time_embed_dim)

        # ---- 输入卷积 ----
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # ---- 下采样模块 ----
        # 逐步降低空间分辨率，提取多尺度特征
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(down_block_types) - 1

            down_block = get_down_block(
                down_block_type=down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=32,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=attention_head_dim,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

        # ---- 中间模块 ----
        # 在最低分辨率下进行深层特征融合
        self.mid_block = get_mid_block(
            mid_block_type="UNetMidBlock2DCrossAttn",
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=1e-5,
            resnet_act_fn="silu",
            resnet_groups=32,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=attention_head_dim,
        )

        # ---- 上采样模块 ----
        # 逐步恢复空间分辨率，通过跳跃连接融合下采样路径的低层特征
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type=up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=32,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=attention_head_dim,
            )
            self.up_blocks.append(up_block)

        # ---- 输出投影 ----
        # 将特征映射回 8 通道噪声空间
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=32, eps=1e-5
        )
        self.conv_act = get_activation("silu")
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor
    ) -> dict:
        """
        分通道计算 MSE 损失，DEM 通道权重大于 RGB 通道。

        RGB（通道 0-3）与 DEM（通道 4-7）分别计算 MSE，最后加权求和。
        DEM 损失权重 1.5×，以优先保证地形结构精度。

        返回
        -------
        dict
            ``loss``: 加权总损失；
            ``loss_img``: RGB 通道 MSE；
            ``loss_dem``: DEM 通道 MSE。
        """
        img_noise_pred = noise_pred[:, :4, :, :]
        dem_noise_pred = noise_pred[:, 4:, :, :]
        img_noise = noise[:, :4, :, :]
        dem_noise = noise[:, 4:, :, :]

        loss_img = F.mse_loss(img_noise_pred, img_noise)
        loss_dem = F.mse_loss(dem_noise_pred, dem_noise)
        loss = loss_img + 1.5 * loss_dem

        return {
            "loss_img": loss_img,
            "loss_dem": loss_dem,
            "loss": loss,
        }

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：预测给定时间步下添加到隐变量上的噪声。

        参数
        ----------
        noisy_latent : Tensor
            带噪联合隐向量 [B, 8, H, W]，通常 H=W=64。
        timestep : Tensor
            扩散时间步索引 [B]。
        global_features : Tensor
            CLIP 全局文本特征 [B, D]（pooled 输出）。
        local_features : Tensor
            CLIP 序列文本特征 [B, N, D]（hidden states）。

        返回
        -------
        Tensor
            预测的噪声 [B, 8, H, W]。
        """
        # 1. 构建条件嵌入：时间 + 文本
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=noisy_latent.dtype)
        t_emb = self.time_embedding(t_emb)

        global_emb = self.global_text_proj(global_features)
        t_emb = t_emb + global_emb

        encoder_hidden_states = local_features

        # 2. 输入投影
        sample = self.conv_in(noisy_latent)
        down_block_res_samples = (sample,)

        # 3. 下采样路径：逐层降维，收集跳跃连接
        for ds_block in self.down_blocks:
            if (
                hasattr(ds_block, "has_cross_attention")
                and ds_block.has_cross_attention
            ):
                sample, res_samples = ds_block(
                    hidden_states=sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = ds_block(
                    hidden_states=sample, temb=t_emb
                )
            down_block_res_samples += res_samples

        # 4. 瓶颈层：最低分辨率下的深层特征处理
        if (
            hasattr(self.mid_block, "has_cross_attention")
            and self.mid_block.has_cross_attention
        ):
            sample = self.mid_block(
                sample,
                t_emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = self.mid_block(sample, t_emb)

        # 5. 上采样路径：逐步恢复分辨率，融入跳跃连接
        for us_block in self.up_blocks:
            res_samples = down_block_res_samples[
                -len(us_block.resnets) :
            ]
            down_block_res_samples = down_block_res_samples[
                : -len(us_block.resnets)
            ]

            if (
                hasattr(us_block, "has_cross_attention")
                and us_block.has_cross_attention
            ):
                sample = us_block(
                    hidden_states=sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = us_block(
                    hidden_states=sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples,
                )

        # 6. 输出投影
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        noise_pred = self.conv_out(sample)

        return noise_pred


def build_unet(
    in_channels: int = 8,
    out_channels: int = 8,
    cross_attention_dim: int = 768,
) -> UNet8Channel:
    """
    ``UNet8Channel`` 工厂函数。

    参数
    ----------
    in_channels : int
        输入通道数，默认 8。
    out_channels : int
        输出通道数，默认 8。
    cross_attention_dim : int
        文本编码器的特征维度（默认 768，与 CLIP ViT-L/14 一致）。

    返回
    -------
    UNet8Channel
    """
    return UNet8Channel(
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim,
    )
