# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A diffusion-based text-to-terrain joint generation system: given a natural language prompt (e.g., "广东丹霞地貌，红色平顶方山"), it simultaneously generates a paired 512×512 height map and texture map. Uses the Latent Diffusion Model (LDM) paradigm — diffusion in a compressed latent space rather than pixel space.

### End goal

**"一句话生成 3D 地形"** — a text-to-dual-modal generation pipeline:

```
Prompt  →  CLIP text encoding  →  DDIM denoising (50 steps)
  →  split 8-ch joint latent  →  VAE decode
  →  512×512 height map (uint16, 0-65535m)  +  512×512 texture map (RGB)
```

The height map provides geometric elevation data and the texture map provides surface color/material — together they form a complete 3D terrain asset, with both modalities semantically and structurally aligned.

| Dimension | Detail |
|---|---|
| **Core task** | Cross-modal text-to-image generation, **dual-modal synchronized output** (height + texture) |
| **Tech paradigm** | Latent Diffusion Model — diffuse/denoise in compressed latent space, not pixel space |
| **Conditioning** | Dual-branch CLIP: global semantics (pooler_output) + local detail (hidden_states) |
| **Target applications** | Procedural 3D terrain generation, game scene asset creation, digital twin environments, VR/AR content authoring |

### Inference pipeline (end-to-end)

1. **Text encode**: Prompt → Dual-branch CLIP → global features [B,768] + local features [B,77,768]
2. **Initialize**: Random Gaussian noise [B, 8, 64, 64]
3. **Denoise loop**: 50-step DDIM, UNet/DiT predicts noise conditioned on text features
4. **Split**: 8-ch latent → texture latent [B,4,64,64] + height latent [B,4,64,64]
5. **Decode**: SD VAE → RGB texture | HeightMapVAE → elevation (denormalized to physical meters)

### Training pipeline (how it learns)

1. **Encode**: Real height map + texture map → VAE encoders → 8-ch joint latent $z_0$
2. **Add noise**: Random timestep $t$, mix Gaussian noise into $z_0$ → noisy $z_t$
3. **Predict noise**: UNet/DiT receives $z_t$, timestep $t$, text features → predicts noise $\hat{\epsilon}$
4. **Loss**: MSE($\hat{\epsilon}$, $\epsilon$), DEM channels weighted 1.5×
5. **Update**: Backprop through UNet/DiT; VAEs and CLIP frozen

## Runnable status

| Component | File | Status |
|---|---|---|
| HeightMapVAE | `models/vae/heightmap_vae.py` | Runnable (extend `AutoencoderKL`, 1-ch, latent [4,64,64]) |
| DualBranchCLIPEncoder | `models/clip/text_encoder.py` | Runnable (HF CLIPTextModel, frozen) |
| UNet8Channel | `models/unet/unet_8ch.py` | Runnable (diffusers 2D U-Net, 8-in/8-out) |
| DiT8Channel | `models/dit/dit_8ch.py` | Runnable (PixArt-α XL port, ~934M, drop-in UNet replacement) |
| UNetTrainingPipeline | `train/train_pipeline.py` | **搁置**（实际训练用 `scripts/unet/unet_full.py`） |
| HeightMapDataset, UNetDataset | `dataset/` | Runnable |
| **InferencePipeline** | `inference/inference_pipeline.py` | Skeleton (`NotImplementedError`) |
| **DDIMScheduler** | `utils/latent_utils.py` | Skeleton (`NotImplementedError`；推理用 diffusers 原生版） |
| **main.py** | `main.py` | Skeleton（暂时弃用） |

## Commands

All scripts are standalone — they `sys.path.insert(0, PROJECT_ROOT)`. No package entry points.

```bash
# Package management (uv, not pip; Python 3.11)
uv sync                    # install dependencies
uv run python <script>     # execute anything

# Height VAE training — memory-optimized (fp32, grad_checkpointing, B=1+acc=8, ~3GB VRAM)
uv run python scripts/height_vae/train_height_vae.py --epochs 100
# Height VAE training — quality (AMP fp16, B=2, ~8GB VRAM)
uv run python scripts/height_vae/train_height_vae_full.py --epochs 100
# Height VAE test/reconstruction
uv run python scripts/height_vae/train_height_vae.py --mode test --checkpoint <path>

# U-Net training (requires data_root with rgb/, dem/, txt/ subdirectories; B=4, ~8GB VRAM)
uv run python scripts/unet/unet_full.py --mode train --epochs 50
uv run python scripts/unet/unet_full.py --mode train --epochs 100 --checkpoint <path>   # resume
uv run python scripts/unet/unet_full.py --mode test --checkpoint <path>

# DiT training — 待重构（scripts/dit/train_dit_full.py 暂不可用）
# uv run python scripts/dit/train_dit_full.py --epochs 50

# Data preprocessing
uv run python scripts/data_process/preprocess/preprocess_heightmaps.py --stage stats     # compute global percentiles
uv run python scripts/data_process/preprocess/preprocess_heightmaps.py --stage transform # apply normalization
uv run python scripts/data_process/preprocess/preprocess_unet.py                         # U-Net DEM prep
uv run python scripts/data_process/verify/scan_heightmaps.py                             # verify output

# NotImplementedError skeletons
uv run python main.py --mode train|inference
```

No linter, formatter, type checker, pre-commit hooks, or CI configured.

## Architecture

### Joint latent space

The core idea: height and texture latents are concatenated channel-wise into an 8-channel joint latent, so the denoiser can model cross-modal correlations:

```
torch.cat([texture_latent, height_latent], dim=1) → [B, 8, 64, 64]
  channels 0-3: texture latent (SD VAE, 4×64×64, scaling_factor=0.18215)
  channels 4-7: height latent (custom HeightMapVAE, 4×64×64, scaling_factor=1.0)
```

### Dual-branch CLIP text encoding

- **Global branch**: CLIP `pooler_output` [B, 768] → linear projection → added to timestep embedding for global modulation
  - 注意：当前 SD UNet2DConditionModel 训练中 global_features 路径暂未启用，仅 DiT 和自定义 UNet8Channel 使用
- **Local branch**: CLIP `last_hidden_state` [B, 77, 768] → injected as `encoder_hidden_states` into cross-attention layers (spatially-aware conditioning)

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
- `data/` has `.gitignore` of `*` — all data is local-only, never committed
- The `_hf` suffix is hardcoded in all data scripts
- `HeightMapDataset` globs `hmap_*.npy` → `[1, 512, 512]` float32, augmentation: random hflip + rot90
- `UNetDataset` expects `data_root/{rgb/, dem/, txt/}` with matching filenames; returns `{"rgb": [3,512,512], "dem": [1,512,512], "prompt": str}`

## Critical gotchas

- **AMP + grad_checkpointing are mutually exclusive** — enabling both silently breaks gradient flow. Choose one or the other.
- **fp16 underflow in slope computation**: `torch.sqrt(grad^2 + 1e-8)` with epsilon `1e-8` underflows to 0 in fp16 → `sqrt(0)` has infinite gradient → NaN. Always wrap slope/curvature computation in `torch.autocast(enabled=False)`.
- **KL divergence overflow**: `posterior.kl()` calls `exp(logvar)` which overflows in fp16. Compute KL in fp32 with explicit `torch.autocast(enabled=False)`.
- `torch.autocast` intercepts `F.conv2d` and re-casts inputs to fp16 even if you explicitly create fp32 tensors.
- **VAE normalization mismatch**: static `x / 3000.0` in HeightMapVAE vs log-transform in data pipeline. Static `/3000` used only when `norm_params.json` is absent.
- Config is hardcoded in scripts — no config files exist. README references a `configs/` directory that does not exist.

## DiT-specific gotchas

- **`time_embed_dim` must equal `hidden_size`** (1152), NOT `hidden_size * 4`. The 4x expansion made `adaLN_modulation` a 4608→10368 Linear per block (~48M x 28 = 1.3B params), bloating the model to ~1.96B.
- **Pretrained loading uses `PixArtTransformer2DModel`** from diffusers. `nn.MultiheadAttention.in_proj_weight` requires concatenating PixArt's separate `to_q`/`to_k`/`to_v` weights.
- **adaLN modulation and pos_embed are randomly initialized** — PixArt uses shared `adaln_single` while our DiTBlock uses per-block modulation, so these can't be loaded from pretrained.
- **DiT staged training** (via `scripts/dit/train_dit_full.py` — 待重构): Stage 1 (burn-in, epochs 0-9) freezes backbone, trains only `global_text_proj` + `local_text_proj` at 10x lr. Stage 2 unfreezes all.

## Conventions

- No linter, formatter, type checker, pre-commit hooks, or CI.
- All docstrings are Chinese; keep comments Chinese when touching existing Chinese files.
- Project planning: `docs/` (`roadmap.md`, `now.md`, `todolist.md`, `goals.md`, `plans/`, `superpowers/`). Check `docs/now.md` for current focus.
- `AGENTS.md` points to this file (`@CLAUDE.md`); `CLAUDE.md` is the single source of truth.
