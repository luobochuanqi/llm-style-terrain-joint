"""
PixArt-alpha 风格的 DiT (Diffusion Transformer)，用于 8 通道联合纹理-高程去噪。
与 UNet8Channel 保持完全相同的 forward() / loss() 接口，可作为 drop-in 替换。

=== 架构概览 ===

输入: [B, 8, 64, 64] 联合隐向量 + timestep [B] + global_features [B,768] + local_features [B,77,768]

  Patch Embed
    8×64×64 ──Conv2d(8,1152,k=2,s=2)──> [B, 1152, 32, 32]
    ──Flatten──> [B, 1024 tokens, 1152 dim]
    ──+ learnable pos_embed──> [B, 1024, 1152]

  Timestep 编码
    timestep ──sinusoidal(256)──> ──MLP(SiLU)──> [B, time_embed_dim=1152]

  全局文本注入 (adaLN 调制)
    global_features [B,768] ──Linear──> [B, 1152]
    ──add with timestep embedding──> conditioning [B, 1152]

  局部文本投影 (交叉注意力)
    local_features [B,77,768] ──Linear──> [B, 77, 1152]
    → 作为 cross-attn 的 K/V

  Transformer Block × 28
    ┌─ adaLN: shift/scale/gate ← MLP(conditioning)
    ├─ Self-Attention: Q/K/V from tokens, 16 heads × 72 dim/head
    ├─ Cross-Attention: Q from tokens, K/V from projected local features
    └─ FFN: 1152 → 4608 → 1152 (4× expansion, GELU)

  输出
    ──Final LayerNorm──> [B, 1024, 1152]
    ──Linear(1152, 32)──> [B, 1024, 32]  (8ch × 2²)
    ──Unpatchify──> [B, 8, 64, 64]  噪声预测

=== 关键参数 ===

in_channels: 8      输入通道 (4 通道纹理 + 4 通道高程)
out_channels: 8     输出通道
patch_size: 2       Patch 尺寸，64→32 grid, 1024 tokens
hidden_size: 1152   Transformer 隐藏维度
depth: 28           Transformer block 数量
num_heads: 16       自注意力头数
cross_attention_dim: 768  CLIP ViT-L/14 隐藏维度
参数量: ~934M (fp16 权重 ~1.87 GB)

=== 预训练权重加载 ===

load_pretrained(path) 从 PixArt-alpha XL (PixArtTransformer2DModel) 加载权重：
  - Transformer blocks: attn1/attn2 的 qkv 合并 (PixArt 分离的 to_q/to_k/to_v → nn.MultiheadAttention 的 in_proj_weight)
  - Patch embed: 4ch→8ch 通道扩展 (前 4ch 加载预训练，后 4ch 零初始化)
  - 跳过层: global_text_proj, local_text_proj (CLIP 768≠T5 4096), pos_embed, adaLN_modulation (随机初始化)

=== 与 UNet8Channel 接口对比 ===

  forward(noisy_latent, timestep, global_features, local_features) → noise_pred
  loss(noise_pred, noise) → {"loss": ..., "loss_img": ..., "loss_dem": ...}

完全一致，train_pipeline.py 通过 model_type 参数切换，无需改动训练循环。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

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

       # 2. Self-Attention (修正：gate 移到最后)
        normed = self.norm1(hidden_states)
        attn_in = normed * (1 + _b(scale1)) + _b(shift1)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        hidden_states = hidden_states + _b(gate1) * attn_out

        # 3. Cross-Attention (修正：gate 移到最后)
        normed = self.norm2(hidden_states)
        cross_in = normed * (1 + _b(scale2)) + _b(shift2)
        cross_out, _ = self.cross_attn(
            cross_in, encoder_hidden_states, encoder_hidden_states, need_weights=False
        )
        hidden_states = hidden_states + _b(gate2) * cross_out

        # 4. FFN (修正：gate 移到最后)
        normed = self.norm3(hidden_states)
        ffn_in = normed * (1 + _b(scale3)) + _b(shift3)
        ffn_out = self.ffn(ffn_in)
        hidden_states = hidden_states + _b(gate3) * ffn_out

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

        time_embed_dim = hidden_size  # must match hidden_size for PixArt-adaln compat

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
        """Forward pass: predict noise added to latent at given timestep.

        Parameters
        ----------
        noisy_latent : Tensor [B, 8, H, W], H=W=64
            Noisy joint latent.
        timestep : Tensor [B]
            Diffusion timestep indices.
        global_features : Tensor [B, 768]
            CLIP pooled output.
        local_features : Tensor [B, 77, 768]
            CLIP last_hidden_state.

        Returns
        -------
        Tensor [B, 8, 64, 64]
            Predicted noise.
        """
        # 1. Patchify: [B, 8, 64, 64] -> [B, 1024, hidden_size]
        tokens = self.patch_embed(noisy_latent)  # [B, hidden_size, 32, 32]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, 1024, hidden_size]
        tokens = tokens + self.pos_embed

        # 2. Timestep encoding
        t_emb = self.time_proj(timestep).to(dtype=noisy_latent.dtype)
        t_emb = self.time_embedding(t_emb)  # [B, time_embed_dim]

        # 3. Global text injection
        global_emb = self.global_text_proj(global_features)
        conditioning = t_emb + global_emb  # [B, time_embed_dim]

        # 4. Local text projection for cross-attention
        encoder_hidden_states = self.local_text_proj(local_features)  # [B, 77, hidden_size]

        # 5. Transformer blocks
        for block in self.transformer_blocks:
            tokens = block(
                hidden_states=tokens,
                conditioning=conditioning,
                encoder_hidden_states=encoder_hidden_states,
            )

        # 6. Output projection and unpatchify
        tokens = self.final_norm(tokens)
        tokens = self.final_linear(tokens)  # [B, 1024, out_channels * patch_size^2]
        out = self._unpatchify(tokens)
        return out

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Rearrange patch tokens back to spatial image.

        Parameters
        ----------
        tokens : Tensor [B, num_patches, out_channels * patch_size^2]

        Returns
        -------
        Tensor [B, out_channels, 64, 64]
        """
        B = tokens.shape[0]
        latent_size = 64
        grid_size = latent_size // self.patch_size  # 32
        out_channels = self.out_channels
        ps = self.patch_size

        tokens = tokens.view(B, grid_size, grid_size, out_channels, ps, ps)
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        tokens = tokens.view(B, out_channels, latent_size, latent_size)
        return tokens

    def load_pretrained(self, pretrained_path: str) -> None:
        """Load pretrained weights from PixArt-alpha XL.

        Handles structural differences between our nn.MultiheadAttention
        (merged qkv weights) and diffusers' Attention class (separate q/k/v
        weights). Also handles channel expansion for patch_embed
        (4-channel pretrained -> 8-channel ours).

        Intentionally skipped (random init):
          - pos_embed (PixArt has no separate pos_embed parameter)
          - global_text_proj (PixArt has no separate global text path)
          - local_text_proj (PixArt uses T5 4096-dim, ours uses CLIP 768-dim)
          - adaLN_modulation.1 (PixArt uses shared adaln-single, ours is per-block)
        """
        from diffusers import PixArtTransformer2DModel

        pretrained = PixArtTransformer2DModel.from_pretrained(
            pretrained_path,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )
        px_state = pretrained.state_dict()
        own_state = self.state_dict()

        loaded, skipped, expanded = 0, 0, 0

        # ---- Top-level direct copies (same shape, different path) ----

        # Time embedding
        for suffix in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"):
            px_key = f"adaln_single.emb.timestep_embedder.{suffix}"
            our_key = f"time_embedding.{suffix}"
            if px_key in px_state and own_state[our_key].shape == px_state[px_key].shape:
                own_state[our_key].copy_(px_state[px_key])
                loaded += 1

        # Final output (same shape: [32, 1152] and [32] for 8ch * 4patch)
        for suffix in ("proj_out.weight", "proj_out.bias"):
            our_key = suffix.replace("proj_out.", "final_linear.")
            px_key = suffix
            if px_key in px_state and own_state[our_key].shape == px_state[px_key].shape:
                own_state[our_key].copy_(px_state[px_key])
                loaded += 1

        # Patch embedding bias (same shape: [1152])
        if "pos_embed.proj.bias" in px_state:
            own_state["patch_embed.bias"].copy_(px_state["pos_embed.proj.bias"])
            loaded += 1

        # Patch embedding weight: 4ch -> 8ch expansion
        if "pos_embed.proj.weight" in px_state:
            px_w = px_state["pos_embed.proj.weight"]  # [1152, 4, 2, 2]
            own_state["patch_embed.weight"].copy_(
                torch.cat([px_w, torch.zeros_like(px_w)], dim=1)  # [1152, 8, 2, 2]
            )
            expanded += 1

        # ---- Block-level transfers ----

        for i in range(self.depth):
            px_prefix = f"transformer_blocks.{i}"
            our_prefix = f"transformer_blocks.{i}"

            # Self-attention qkv: merge PixArt separate q/k/v into in_proj
            self._merge_qkv(
                px_state, own_state,
                f"{px_prefix}.attn1",
                f"{our_prefix}.self_attn",
            )
            loaded += 2  # weight + bias

            # Self-attention output projection
            for param_name in ("out_proj.weight", "out_proj.bias"):
                our_key = f"{our_prefix}.self_attn.{param_name}"
                suffix = param_name.split(".", 1)[1]
                px_key = f"{px_prefix}.attn1.to_out.0.{suffix}"
                if px_key in px_state and own_state[our_key].shape == px_state[px_key].shape:
                    own_state[our_key].copy_(px_state[px_key])
                    loaded += 1

            # Cross-attention qkv
            self._merge_qkv(
                px_state, own_state,
                f"{px_prefix}.attn2",
                f"{our_prefix}.cross_attn",
            )
            loaded += 2

            # Cross-attention output projection
            for param_name in ("out_proj.weight", "out_proj.bias"):
                our_key = f"{our_prefix}.cross_attn.{param_name}"
                suffix = param_name.split(".", 1)[1]
                px_key = f"{px_prefix}.attn2.to_out.0.{suffix}"
                if px_key in px_state and own_state[our_key].shape == px_state[px_key].shape:
                    own_state[our_key].copy_(px_state[px_key])
                    loaded += 1

            # FFN: PixArt ff.net.0.proj -> our ffn.0, PixArt ff.net.2 -> our ffn.2
            ffn_map = {
                (f"{our_prefix}.ffn.0.weight", f"{px_prefix}.ff.net.0.proj.weight"),
                (f"{our_prefix}.ffn.0.bias", f"{px_prefix}.ff.net.0.proj.bias"),
                (f"{our_prefix}.ffn.2.weight", f"{px_prefix}.ff.net.2.weight"),
                (f"{our_prefix}.ffn.2.bias", f"{px_prefix}.ff.net.2.bias"),
            }
            for our_key, px_key in ffn_map:
                if px_key in px_state and own_state[our_key].shape == px_state[px_key].shape:
                    own_state[our_key].copy_(px_state[px_key])
                    loaded += 1

        del pretrained, px_state
        print(
            f"Pretrained weights loaded: {loaded} exact matches, "
            f"{expanded} expanded (channel dim), "
            f"{skipped} skipped (random init)"
        )

    def _merge_qkv(
        self,
        px_state: dict,
        own_state: dict,
        px_attn: str,
        our_attn: str,
    ) -> None:
        """Merge PixArt separate q/k/v weights into nn.MultiheadAttention's
        merged in_proj_weight (dim=0 concatenation: [q, k, v]).
        """
        for param_type in ("weight", "bias"):
            q = px_state.get(f"{px_attn}.to_q.{param_type}")
            k = px_state.get(f"{px_attn}.to_k.{param_type}")
            v = px_state.get(f"{px_attn}.to_v.{param_type}")
            our_key = f"{our_attn}.in_proj_{param_type}"
            if q is not None and k is not None and v is not None:
                own_state[our_key].copy_(torch.cat([q, k, v], dim=0))

    def loss(
        self, noise_pred: torch.Tensor, noise: torch.Tensor
    ) -> dict:
        """Channel-wise MSE loss with DEM channels weighted 1.5x.

        RGB (channels 0-3) and DEM (channels 4-7) computed separately.

        Returns
        -------
        dict
            ``loss``: weighted total loss;
            ``loss_img``: RGB channel MSE;
            ``loss_dem``: DEM channel MSE.
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
