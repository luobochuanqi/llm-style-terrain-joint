# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A diffusion-based text-to-terrain joint generation system: given a natural language prompt (e.g., "广东丹霞地貌，红色平顶方山"), it simultaneously generates a paired 512×512 height map and texture map. Uses the Latent Diffusion Model (LDM) paradigm — diffusion in a compressed latent space rather than pixel space.

## Commands

```bash
# Package management (uv, not pip; Python 3.11)
uv sync                    # install dependencies
uv run python <script>     # execute anything

# Skeleton main entry (all NotImplementedError placeholders)
uv run python main.py --mode train
uv run python main.py --mode inference

# Height VAE training — memory-optimized (fp32, grad_checkpointing, B=1+acc=8, ~3GB VRAM)
uv run python scripts/height_vae/train_height_vae.py --epochs 100
# Height VAE training — quality (AMP fp16, B=2, ~8GB VRAM)
uv run python scripts/height_vae/train_height_vae_full.py --epochs 100
# Height VAE test/reconstruction
uv run python scripts/height_vae/train_height_vae.py --mode test --checkpoint <path>

# U-Net training (requires data_root with rgb/, dem/, txt/ subdirectories)
uv run python scripts/unet/train_unet_full.py --epochs 50
# U-Net test noise prediction
uv run python scripts/unet/train_unet_full.py --mode test --checkpoint <path>
# U-Net resume training
uv run python scripts/unet/train_unet_full.py --epochs 100 --checkpoint ./outputs/unet_8ch/checkpoint.pt

# DiT training (PixArt-alpha XL, 934M params, ~20GB VRAM with B=2 + grad_ckpt)
uv run python scripts/dit/train_dit_full.py --epochs 50
# DiT resume
uv run python scripts/dit/train_dit_full.py --epochs 100 --checkpoint ./outputs/dit_8ch/checkpoint.pt
# DiT test noise prediction
uv run python scripts/dit/train_dit_full.py --mode test --checkpoint ./outputs/dit_8ch/checkpoint.pt

# Data preprocessing pipeline
uv run python scripts/data_process/preprocess/preprocess_heightmaps.py --stage stats     # compute global percentiles
uv run python scripts/data_process/preprocess/preprocess_heightmaps.py --stage transform # apply normalization
uv run python scripts/data_process/verify/scan_heightmaps.py                            # verify output
```

No linter, formatter, type checker, pre-commit hooks, or CI configured.

## Architecture

### Joint latent space

The core idea: height and texture latents are concatenated channel-wise into an 8-channel joint latent, so the U-Net can model cross-modal correlations during denoising:

```
torch.cat([height_latent, texture_latent], dim=1) → [B, 8, 64, 64]
  channels 0-3: height latent (custom HeightMapVAE, 4×64×64)
  channels 4-7: texture latent (SD VAE, 4×64×64, scaled by 0.18215)
```

### Component status

| Component | File | Status |
|-----------|------|--------|
| **HeightMapVAE** | `models/vae/heightmap_vae.py` | Runnable |
| **DualBranchCLIPEncoder** | `models/clip/text_encoder.py` | Runnable (HF-based, frozen) |
| **UNet8Channel** | `models/unet/unet_8ch.py` | Runnable |
| **DiT8Channel** | `models/dit/dit_8ch.py` | Runnable (PixArt-alpha XL, drop-in UNet replacement) |
| **UNetTrainingPipeline** | `train/train_pipeline.py` | Runnable (AMP, grad clipping, checkpoint, CSV logging, viz) |
| **InferencePipeline** | `inference/inference_pipeline.py` | Skeleton (all `NotImplementedError`) |
| **DDIMScheduler** | `utils/latent_utils.py` | Skeleton (all `NotImplementedError`) |
| **main.py** | `main.py` | Skeleton (all TODO) |

The VAE and U-Net training scripts under `scripts/` manipulate `sys.path` to import project modules — they do not use package entry points.

### Dual-branch CLIP text encoding

- **Global branch**: CLIP `pooler_output` [B, 768] → linear projection → added to timestep embedding for global modulation of all ResNet blocks
- **Local branch**: CLIP `last_hidden_state` [B, 77, 768] → linear projection → injected as `encoder_hidden_states` into cross-attention layers (spatially-aware conditioning)

### HeightMapVAE details

Extends `diffusers.AutoencoderKL` with `in_channels=1, out_channels=1, latent_channels=4, sample_size=512, scaling_factor=1.0`. Adds geo-constraint losses:

- **Slope loss**: MSE between Sobel gradient magnitudes of pred vs target
- **Curvature loss**: smooth L1 between Laplacian responses
- Composite: `loss_geo = slope_loss + 0.5 * curvature_loss`, weighted at 0.8 in total VAE loss
- KL divergence is normalized to per-dimension mean

Sobel/Laplacian kernels are registered as non-persistent buffers.

### Data pipeline

```
1081×1081 uint16 PNG → center crop 1080×1080 → Area scale 512×512
  → percentile clip → log transform → linear map [0, 1] → .npy float32
```

- Raw PNG: `data/origin/heightmaps_hf/`
- Processed `.npy`: `data/process/heightmaps_hf/`
- `data/` has `.gitignore` of `*` — all data is local-only
- The `_hf` suffix is hardcoded in all data scripts
- `HeightMapDataset` expects `hmap_*.npy` glob, returns `[1, 512, 512]` float32, augmentation: random hflip + rot90
- `UNetDataset` expects `data_root/{rgb/, dem/, txt/}` with matching filenames; returns `{"rgb": [3,512,512], "dem": [1,512,512], "prompt": str}`

## Critical gotchas

- **AMP + grad_checkpointing are mutually exclusive** — enabling both silently breaks gradient flow. Choose one or the other.
- **fp16 underflow in slope computation**: `torch.sqrt(grad^2 + 1e-8)` with epsilon `1e-8` underflows to 0 in fp16 → `sqrt(0)` has infinite gradient → NaN. Always wrap slope/curvature computation in `torch.autocast(enabled=False)`.
- **KL divergence overflow**: `posterior.kl()` calls `exp(logvar)` which overflows in fp16. Compute KL in fp32 with explicit `torch.autocast(enabled=False)`.
- `torch.autocast` intercepts `F.conv2d` and re-casts inputs to fp16 even if you explicitly create fp32 tensors.
- Config is hardcoded in scripts — no config files exist. README references a `configs/` directory that does not exist.
- Docstrings are in Chinese; keep comments Chinese when modifying existing Chinese files.
- The VAE's static normalization (`x / 3000.0`) differs from the data pipeline's log-transform normalization. The static `/3000` is used only when `norm_params.json` is absent.

## DiT-specific gotchas

- **`time_embed_dim` must equal `hidden_size`** (1152), NOT `hidden_size * 4`. The 4x expansion made `adaLN_modulation` a 4608→10368 Linear per block (~48M x 28 = 1.3B params), bloating the model to ~1.96B.
- **Pretrained loading uses `PixArtTransformer2DModel`** from diffusers. `nn.MultiheadAttention.in_proj_weight` requires concatenating PixArt's separate `to_q`/`to_k`/`to_v` weights.
- **adaLN modulation and pos_embed are randomly initialized** — PixArt uses shared `adaln_single` while our DiTBlock uses per-block modulation, so these can't be loaded from pretrained.
- **`train_pipeline.py` attribute `self.unet`** holds either UNet or DiT — intentionally reused for code compatibility.
- **DiT staged training**: Stage 1 (burn-in, epochs 0-9) freezes backbone, trains only `global_text_proj` + `local_text_proj` at 10x lr. Stage 2 unfreezes all.
