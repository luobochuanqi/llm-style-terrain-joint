"""
PixArt-alpha 风格的 DiT (Diffusion Transformer)，用于 8 通道联合纹理-高程去噪。

基于 PixArt-alpha XL 架构：
  - Patchify: 8×64×64 → 1024 tokens × 1152 dim (patch_size=2)
  - Transformer blocks × 28: adaLN + Self-Attn + Cross-Attn + FFN
  - 文本注入：全局特征 → adaLN 调制，局部特征 → cross-attn K/V
  - 与 UNet8Channel 保持相同接口（forward + loss 签名完全一致）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class DiTBlock(nn.Module):
    """Single DiT Transformer Block.

    Structure:
        ┌─ adaLN: shift/scale/gate ← MLP(timestep_emb + global_text_emb)
        ├─ Self-Attention: Q/K/V from latent tokens
        ├─ Cross-Attention: Q from tokens, K/V from local_text_features
        └─ FFN: hidden_size → 4*hidden_size → hidden_size

    All sub-layers use pre-norm + residual connection.
    """

    def __init__(self, hidden_size: int, num_heads: int, time_embed_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # adaLN modulation: 9 * hidden_size params (shift/scale/gate × 3 sub-layers)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 9 * hidden_size),
        )

        # Self-Attention (pre-norm)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Cross-Attention (pre-norm)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
            kdim=hidden_size,
            vdim=hidden_size,
        )

        # FFN (pre-norm)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        hidden_states : Tensor [B, N, D]
            Patch tokens.
        conditioning : Tensor [B, time_embed_dim]
            adaLN conditioning vector (timestep emb + global text emb).
        encoder_hidden_states : Tensor [B, 77, D]
            CLIP local features, used as K/V for cross-attn.

        Returns
        -------
        Tensor [B, N, D]
            Processed patch tokens.
        """
        # 1. Generate adaLN modulation parameters
        mod = self.adaLN_modulation(conditioning)  # [B, 9*D]
        shift1, scale1, gate1, \
            shift2, scale2, gate2, \
            shift3, scale3, gate3 = mod.chunk(9, dim=-1)

        def _b(b_tensor):
            return b_tensor.unsqueeze(1)

        # 2. Self-Attention
        normed = self.norm1(hidden_states)
        normed = _b(gate1) * (normed * (1 + _b(scale1)) + _b(shift1))
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        hidden_states = hidden_states + attn_out

        # 3. Cross-Attention
        normed = self.norm2(hidden_states)
        normed = _b(gate2) * (normed * (1 + _b(scale2)) + _b(shift2))
        cross_out, _ = self.cross_attn(
            normed, encoder_hidden_states, encoder_hidden_states, need_weights=False
        )
        hidden_states = hidden_states + cross_out

        # 4. FFN
        normed = self.norm3(hidden_states)
        normed = _b(gate3) * (normed * (1 + _b(scale3)) + _b(shift3))
        ffn_out = self.ffn(normed)
        hidden_states = hidden_states + ffn_out

        return hidden_states


class DiT8Channel(nn.Module):
    """PixArt-alpha 风格的 8 通道 DiT，用于纹理与高程联合去噪。

    参数
    ----------
    in_channels : int, default 8
        输入通道数（4 通道纹理 + 4 通道高程）。
    out_channels : int, default 8
        输出通道数。
    patch_size : int, default 2
        Patch 尺寸，64×64 隐空间 → (64/patch_size)² tokens。
    hidden_size : int, default 1152
        Transformer hidden dimension。
    depth : int, default 28
        Transformer blocks 数量。
    num_heads : int, default 16
        自注意力头数。
    cross_attention_dim : int, default 768
        CLIP 文本编码器隐藏维度。
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        cross_attention_dim: int = 768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.cross_attention_dim = cross_attention_dim

        latent_size = 64
        self.num_patches = (latent_size // patch_size) ** 2  # 1024

        time_embed_dim = hidden_size * 4

        # 1. Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

        # 2. Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        # 3. Timestep encoding
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # 4. Text projections
        self.global_text_proj = nn.Linear(cross_attention_dim, time_embed_dim)
        self.local_text_proj = nn.Linear(cross_attention_dim, hidden_size)

        # 5. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [self._build_transformer_block(hidden_size, num_heads, time_embed_dim)
             for _ in range(depth)]
        )

        # 6. Output layer
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.final_linear = nn.Linear(
            hidden_size, out_channels * patch_size * patch_size
        )

        self.initialize_weights()

    def _build_transformer_block(
        self, hidden_size: int, num_heads: int, time_embed_dim: int
    ) -> nn.Module:
        """Construct a single DiTBlock."""
        return DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
        )

    def initialize_weights(self):
        """Basic weight initialization (pretrained weights loaded separately)."""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.global_text_proj.weight)
        nn.init.zeros_(self.global_text_proj.bias)
        nn.init.xavier_uniform_(self.local_text_proj.weight)
        nn.init.zeros_(self.local_text_proj.bias)
        nn.init.xavier_uniform_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("forward will be implemented in a later task")

    def loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor
    ) -> dict:
        raise NotImplementedError("loss will be implemented in a later task")


def build_dit(
    pretrained_path: str | None = None,
    **kwargs,
) -> DiT8Channel:
    """Factory function: build DiT8Channel and optionally load PixArt-alpha pretrained weights."""
    model = DiT8Channel(**kwargs)
    if pretrained_path is not None:
        model.load_pretrained(pretrained_path)
        print(f"Loaded PixArt-alpha pretrained weights: {pretrained_path}")
    return model
