"""
DiT 8 通道训练脚本 — 已搁置

基于 PixArt-alpha XL 架构的 DiT。train/train_pipeline.py 已清除为占位，
本脚本依赖的 UNetTrainingPipeline 不再可用。

重构方向：参照 scripts/unet/unet_full.py 的 UNetTrainer 模式，
独立构建模型、训练循环和 checkpoint 管理。

历史默认超参数和工具函数保留供参考。
"""

import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

import torch

# =============================================================================
# 历史默认超参数 (供独立重构参考)
# =============================================================================

_DEFAULT_DATA_ROOT = "./data/unet_training"
_DEFAULT_DEM_VAE_CKPT = "./data/vae_model_data/best_checkpoint.pt"
_DEFAULT_DIT_PRETRAINED = "PixArt-alpha/PixArt-XL-2-1024-MS"
_DEFAULT_OUTPUT_DIR = "./outputs/dit_8ch"
_DEFAULT_EPOCHS = 50
_DEFAULT_BATCH_SIZE = 2
_DEFAULT_LEARNING_RATE = 5e-5
_DEFAULT_WEIGHT_DECAY = 1e-4
_DEFAULT_NUM_WORKERS = 2
_DEFAULT_USE_AMP = True
_DEFAULT_SAVE_STEPS = 1000
_DEFAULT_VIZ_INTERVAL = 1
_DEFAULT_WARMUP_EPOCHS = 5
_DEFAULT_STAGE1_EPOCHS = 10
_DEFAULT_SEED = 45
_DEFAULT_SPLIT = 0.05
_DEFAULT_NUM_SAMPLES = 5
_DEFAULT_CFG_DROP_RATE = 0.1
_DEFAULT_MIN_SNR_GAMMA = 5.0
_DEFAULT_GUIDANCE_SCALE = 4.0


# =============================================================================
# 历史工具函数 (供独立重构参考)
# =============================================================================


def _get_adapter_param_names(model: torch.nn.Module) -> set:
    """返回适配层参数名的集合（用于 10x lr）。

    PixArt 预训练层：transformer_blocks.*, pos_embed, time_proj, time_embedding,
                     patch_embed (部分), final_norm, final_linear (部分)
    CLIP 适配层：global_text_proj, local_text_proj
    """
    adapter_patterns = ["global_text_proj", "local_text_proj"]
    adapter_params = set()
    for name, _ in model.named_parameters():
        if any(pat in name for pat in adapter_patterns):
            adapter_params.add(name)
    return adapter_params


def setup_staged_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    stage: int = 1,
) -> torch.optim.AdamW:
    """构建分阶段优化器。

    Stage 1: 冻结预训练层，仅训练 adapter 层 (global_text_proj, local_text_proj)，
             使用 10x lr。
    Stage 2: 全部解冻，adapter 层 10x lr，预训练层 base lr。
    """
    adapter_names = _get_adapter_param_names(model)

    if stage == 1:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name in adapter_names:
                param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        params_to_opt = [{"params": trainable, "lr": lr * 10}]
        print(
            f"Stage 1: 训练 {len(trainable)}/"
            f"{sum(1 for _ in model.parameters())} 个参数组（仅适配层）"
        )
    else:
        for param in model.parameters():
            param.requires_grad = True

        adapter_param_objs = []
        pretrained_param_objs = []
        for name, param in model.named_parameters():
            if name in adapter_names:
                adapter_param_objs.append(param)
            else:
                pretrained_param_objs.append(param)

        params_to_opt = [
            {"params": pretrained_param_objs, "lr": lr},
            {"params": adapter_param_objs, "lr": lr * 10},
        ]
        print(
            f"Stage 2: adapter {len(adapter_param_objs)} 组 @ {lr * 10:.2e}, "
            f"pretrained {len(pretrained_param_objs)} 组 @ {lr:.2e}"
        )

    return torch.optim.AdamW(params_to_opt, weight_decay=weight_decay)


def main():
    raise NotImplementedError(
        "DiT 训练已搁置。train_pipeline.py 已被清除为占位模块，\n"
        "DiT 训练需参照 scripts/unet/unet_full.py 的 UNetTrainer 独立重构。\n"
        "如需 U-Net 训练请使用: python scripts/unet/unet_full.py --mode train"
    )


if __name__ == "__main__":
    main()
