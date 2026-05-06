# AGENTS.md — llm-style-terrain-joint

## Project status

Early scaffold. Only `HeightMapVAE` (inherits `diffusers.AutoencoderKL`) is runnable. UNet, CLIP text encoder, DDIM scheduler, training pipeline, inference pipeline are all `NotImplementedError` / placeholder skeletons.

## Entry points

```bash
python main.py --mode train       # skeleton (all TODO)
python main.py --mode inference   # skeleton (all TODO)
python scripts/height_vae/train_height_vae.py --mode train   # runnable VAE train
python scripts/height_vae/train_height_vae.py --mode test --checkpoint <path>
python scripts/height_vae/train_height_vae_full.py            # quality-oriented variant (73M params, needs 16GB+ VRAM)
```

Two VAE training scripts exist: `train_height_vae.py` (memory-optimized: fp32, grad_checkpointing, B=1+acc=8, 55M params, ~3GB VRAM) and `train_height_vae_full.py` (quality: AMP fp16, B=2, 73M params, ~8GB VRAM).

## Package management

Uses **uv** (not pip). Python 3.11 required (`uv sync` to install, `uv run python ...` to execute).

## Data pipeline

```bash
python scripts/data_process/preprocess/preprocess_heightmaps.py --stage stats   # compute global percentiles
python scripts/data_process/preprocess/preprocess_heightmaps.py --stage transform  # apply normalization
python scripts/data_process/verify/scan_heightmaps.py   # verify
```

Pipeline: 1081×1081 uint16 PNG → center crop 1080×1080 → Area scale 512×512 → percentile clip → log transform → linear map [0, 1] → `.npy` float32.

All scripts use `_hf`-suffixed directories (`heightmaps_hf`, `origin/heightmaps_hf`). Raw PNG in `data/origin/heightmaps_hf/`, processed `.npy` in `data/process/heightmaps_hf/`. The `_hf` suffix is hardcoded in every script. `data/` has `.gitignore` of just `*` — all data is local-only.

`HeightMapDataset` (`dataset/height_map_dataset.py`) expects `hmap_*.npy` glob, returns `[1, 512, 512]` float32 tensors normalized to [0, 1]. Augmentation: random hflip + rot90.

## Architecture

- **Joint latent**: `torch.cat([height_latent, texture_latent], dim=1)` → `[B, 8, 64, 64]`
  - channels 0-3: height latent (custom VAE)
  - channels 4-7: texture latent (SD VAE, `texture_vae = None` — placeholder)
- **HeightMapVAE** (`models/vae/heightmap_vae.py:30`): `AutoencoderKL(in_channels=1, out_channels=1, latent_channels=4, sample_size=512, scaling_factor=1.0)`
  - Normalization: `x / 3000.0` ↔ `x * 3000.0` (note: this differs from data pipeline's log-transform — the VAE's static `/3000` is used in `denormalize_to_elevation` only when `norm_params.json` is absent)
  - Geo loss: `slope_loss + 0.5 * curvature_loss` at weight 0.8 in total VAE loss
- **UNet** (`models/unet/unet_8ch.py`): 8-in 8-out, placeholder only (pass-through `nn.Conv2d(8,8,1)`)
- **Text encoder** (`models/clip/text_encoder.py`): dual-branch CLIP (`openai/clip-vit-base-patch32`) — raises `NotImplementedError`
- **DDIM** (`utils/latent_utils.py`): skeleton only — `set_timesteps` and `step` both raise `NotImplementedError`

## Important gotchas

- No linter, formatter, type checker, pre-commit hooks, or CI configured.
- `ENABLE_GRAD_CHECKPOINTING=True` and `USE_AMP=True` are **mutually exclusive** — enabling both silently breaks gradient flow.
- Config is hardcoded in `main.py` and both `train_height_vae*.py` files — no config files.
- README references a `configs/` directory that does not exist.
- `docs/` contains planning reference docs (`roadmap.md`, `now.md`, `todolist.md`, `plans/`).
- All docstrings are in Chinese; keep comments Chinese when modifying existing Chinese files.