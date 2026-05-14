"""
8 通道 U-Net 模型

根据 roadmap 描述：
- 输入：脏图 z_t (8 通道) + 时间步 t + 文本特征向量
- 输出：预测的噪声 epsilon_pred (8 通道)
- 使用交叉注意力层（Cross-Attention）融入文本条件
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.unets.unet_2d_blocks import get_down_block, get_mid_block, get_up_block
from diffusers.models.activations import get_activation

class UNet8Channel(nn.Module):
    """
    8 通道 U-Net 模型

    输入：
        - noisy_latent: 带噪联合隐向量 [B, 8, 64, 64]
        - timestep: 时间步 [B]
        - global_features: 全局文本特征 [B, D_global]
        - local_features: 细节文本特征 [B, N, D_local]

    输出：
        - noise_pred: 预测的噪声 [B, 8, 64, 64]

    结构特点：
        - 输入/输出均为 8 通道（4 通道高程 + 4 通道纹理）
        - 使用交叉注意力层融入文本条件
        - 时间步通过嵌入层加入网络
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
        cross_attention_dim: int = 512,
        use_linear_projection: bool = True,
    ):
        """
        初始化 8 通道 U-Net

        Args:
            in_channels: 输入通道数（固定为 8）
            out_channels: 输出通道数（固定为 8）
            down_block_types: 下采样块类型元组
            up_block_types: 上采样块类型元组
            block_out_channels: 各块输出通道数
            layers_per_block: 每个块的 ResNet 层数
            attention_head_dim: 注意力头维度
            cross_attention_dim: 交叉注意力维度（文本特征维度）
            use_linear_projection: 是否使用线性投影
        """
        super().__init__()

        # 保存配置参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block

        # TODO: 时间步嵌入层
        # 将时间步 t 编码为向量，加入到网络中
        # self.time_proj = Timesteps(...)
        # self.time_embedding = TimestepEmbedding(...)

        time_embed_dim = self.block_out_channels[0] << 2

        self.time_proj = Timesteps(num_channels=self.block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)

        self.time_embedding = TimestepEmbedding(
            in_channels=self.block_out_channels[0], 
            time_embed_dim=time_embed_dim, 
            act_fn="silu"
        )

        self.global_proj = nn.Linear(768, time_embed_dim)

        # TODO: 下采样模块 (Down Blocks)
        # 逐步降低空间分辨率，提取多层次特征
        # self.down_blocks = nn.ModuleList([...])

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

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
                temb_channels=time_embed_dim,       # 【关键】把 TODO 1 里的时间广播线接进来
                add_downsample=not is_final_block,  # 是否缩小长宽
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim, # 【关键】文本翻译官的接口尺寸 (如 1024)
                num_attention_heads=attention_head_dim,
            )

            self.down_blocks.append(down_block)

        # TODO: 中间层 (Mid Block)
        # 在最低分辨率下进行更深层的特征提取
        # self.mid_block = UNetMidBlock2DCrossAttn(...)

        self.mid_block = get_mid_block(
            mid_block_type="UNetMidBlock2DCrossAttn",
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,           # 接上时间线
            resnet_eps=1e-5,
            cross_attention_dim=cross_attention_dim, # 接上文本线
            num_attention_heads=attention_head_dim,
        )

        # TODO: 上采样模块 (Up Blocks)
        # 逐步恢复空间分辨率，融合跳跃连接
        # self.up_blocks = nn.ModuleList([...])

        self.up_blocks = nn.ModuleList([])
        
        # 上采样是倒着来的，所以要把通道数列表反转
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            # 这里的 input_channel 包含了跳跃连接拼接过来的额外通道
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type=up_block_type,
                num_layers=layers_per_block + 1,    # 上采样层通常比下采样多一层 ResNet 用于消化拼接的特征
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,    # 最后一层不需要再放大了
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=attention_head_dim,
            )
            self.up_blocks.append(up_block)

        # TODO: 输出层
        # 将特征映射回 8 通道噪声空间
        # self.conv_norm_out = nn.GroupNorm(...)
        # self.conv_act = nn.SiLU()
        # self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        # 2. 扭曲（激活函数）
        self.conv_act = get_activation("silu")
        # 3. 降维输出：从 320 压回 8 通道 (out_channels)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

        # 占位模块（框架代码，实际需要使用 diffusers 的 UNet2DConditionModel）
        # self.placeholder_conv = nn.Conv2d(in_channels, out_channels, 1)

    def loss(self, noise_pred: torch.Tensor, noise: torch.Tensor):

        img_noise_pred = noise_pred[:, :4, :, :]
        dem_noise_pred = noise_pred[:, 4:, :, :]

        img_noise = noise[:, :4, :, :]
        dem_noise = noise[:, 4:, :, :]
        
        loss_img = F.mse_loss(img_noise_pred, img_noise)
        loss_dem = F.mse_loss(dem_noise_pred, dem_noise)
        
        loss = loss_img + 1.5 * loss_dem

        return loss

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
        local_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播（做题）

        Args:
            noisy_latent: 带噪联合隐向量 [B, 8, 64, 64]
            timestep: 时间步 [B]
            global_features: 全局文本特征 [B, D_global]
            local_features: 细节特征向量 [B, N, D_local]

        Returns:
            noise_pred: 预测的噪声 [B, 8, 64, 64]
        """
        # TODO: 完整的前向传播逻辑

        # ==========================================
        # 1. 翻译“时间”与“文本”指令
        # ==========================================
        # 将干瘪的标量 timestep 展开为正余弦波
        t_emb = self.time_proj(timestep)
        # 统一数据类型 (防止 float16 和 float32 冲突)
        t_emb = t_emb.to(dtype=noisy_latent.dtype)
        # 翻译成 U-Net 能听懂的全局控制向量
        t_emb = self.time_embedding(t_emb)

        global_emb = self.global_proj(global_features)

        t_emb += global_emb
        
        # 文本特征就是我们要传给 Cross-Attention 的 hidden_states
        encoder_hidden_states = local_features

        # ==========================================
        # 2. 进门安检 (Initial Convolution)
        # ==========================================
        # [B, 8, 64, 64] -> [B, 320, 64, 64]
        sample = self.conv_in(noisy_latent)

        # ==========================================
        # 3. 压缩流水线 (Down Blocks)
        # ==========================================
        # 准备一个“堆栈”，用来存放要发送给右半边(Up Blocks)的高清特征
        # 注意：进门后的第一个 sample 也要存进去！
        down_block_res_samples = (sample,) 

        for downsample_block in self.down_blocks:
            # 必须严谨地检查当前 block 是不是 Cross-Attention 类型
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=t_emb)

            # 把这一层产生的高清特征，追加到堆栈里
            down_block_res_samples += res_samples

        # ==========================================
        # 4. 决策谷底 (Mid Block)
        # ==========================================
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample, 
                t_emb, 
                encoder_hidden_states=encoder_hidden_states
            )
        else:
            sample = self.mid_block(sample, t_emb)

        # ==========================================
        # 5. 解压拼图流水线 (Up Blocks)
        # ==========================================
        for i, upsample_block in enumerate(self.up_blocks):
            # 核心魔法：从堆栈的末尾（也就是最深层），精确地“切”下当前 Up Block 需要的跳跃特征
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            # 把已经拿走的特征从堆栈里剔除
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples, # 把跳跃连接的特征塞进去融合！
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples,
                )

        # ==========================================
        # 6. 出口装配器 (Output Layer)
        # ==========================================
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        # [B, 320, 64, 64] -> [B, 8, 64, 64]
        noise_pred = self.conv_out(sample)

        return noise_pred
        
        """
        # 框架代码：仅返回占位输出
        batch_size = noisy_latent.shape[0]
        noise_pred = self.placeholder_conv(noisy_latent)

        return noise_pred
        """
        

def build_unet(
    in_channels: int = 8,
    out_channels: int = 8,
) -> UNet8Channel:
    """
    构建 8 通道 U-Net 工厂函数

    Args:
        in_channels: 输入通道数（固定为 8）
        out_channels: 输出通道数（固定为 8）
    Returns:
        unet: 8 通道 U-Net 模型
    """
    return UNet8Channel(
        in_channels=in_channels,
        out_channels=out_channels,
    )
