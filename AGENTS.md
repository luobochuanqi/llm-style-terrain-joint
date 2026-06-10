# AGENTS.md

> Architecture deep-dive → `CLAUDE.md` (single source of truth).

## Runtime

- **Python 3.11, `uv`** for deps (`uv sync` / `uv run python ...`). No pip, conda, or poetry.
- No linter, formatter, type checker, pre-commit, or CI configured.

## Entrypoints (scripts are standalone, not importable packages)

| What | Command | Status |
|---|---|---|
| U-Net train | `uv run python scripts/unet/unet_full.py --mode train --epochs 50` | Runnable |
| U-Net resume | `... --checkpoint <path>` | Runnable |
| U-Net test/infer | `... --mode test --checkpoint <path>` | Runnable |
| HeightMapVAE train (mem) | `uv run python scripts/height_vae/train_height_vae.py --epochs 100` | Runnable |
| HeightMapVAE train (quality) | `uv run python scripts/height_vae/train_height_vae_full.py --epochs 100` | Runnable |
| HeightMapVAE test | `... --mode test --checkpoint <path>` | Runnable |
| Data preprocess | `uv run python scripts/data_process/preprocess/preprocess_heightmaps.py --stage stats\|transform` | Runnable |
| Data verify | `uv run python scripts/data_process/verify/scan_heightmaps.py` | Runnable |
| DiT train | `scripts/dit/train_dit_full.py` | **Stalled** (needs refactor to UNetTrainer pattern) |
| `main.py` / `train/train_pipeline.py` / `inference/inference_pipeline.py` / `utils/latent_utils.py` | — | **Skeletons** (not importable) |

## Architecture essentials

- **Joint latent** `[B, 8, 64, 64]`: ch 0-3 = texture (SD VAE, ×0.18215), ch 4-7 = height (custom HeightMapVAE, ×1.0). `torch.cat([rgb_latent, dem_latent], dim=1)`.
- **Dual-branch CLIP**: global (pooler_output → timestep embed) + local (hidden_states → cross-attn). Frozen during training.
- **U-Net**: built from SD1.5 `UNet2DConditionModel`, conv_in/conv_out expanded 4→8 channels. Staged unfreeze strategy.
- **HeightMapVAE**: extends diffusers `AutoencoderKL` (1→ch→4 latent →1 ch), adds slope loss (Sobel) + curvature loss (Laplacian) at 0.8× weight.
- **Data pipeline**: `data/origin/heightmaps_hf/` → percentile clip → log transform → `data/process/heightmaps_hf/`. Training data expects `data_root/{rgb/, dem/, txt/}`.

## Critical gotchas

- **AMP + grad_checkpointing are mutually exclusive** — enabling both silently breaks gradient flow.
- **fp16 safety**: wrap slope/curvature/KL computation in `torch.autocast(enabled=False)` — `F.conv2d` re-casts to fp16 even with explicit fp32 tensors.
- **Config is hardcoded** in each script — no `configs/` directory exists.
- **No CI / tests** — run scripts directly to verify.
- All docstrings and comments are Chinese; keep existing Chinese untouched.
- `data/` is gitignored (`*`) — all data is local-only.

## References

- `CLAUDE.md` — project overview, full architecture, history, DiT gotchas
- `docs/now.md` — current focus and remaining gaps