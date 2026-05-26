"""
8 通道 U-Net 训练脚本（效果优先版）

基于 DDPM 的纹理 + 高程联合去噪训练。CLIP 文本编码、VAE 隐空间映射、
训练流水线均已拆分到对应模块（models/clip/、train/），本脚本仅为入口。

用法：
    # 全新训练
    python scripts/unet/train_unet_full.py --epochs 50

    # 断点续训
    python scripts/unet/train_unet_full.py --epochs 100 --checkpoint ./outputs/unet_8ch/checkpoint.pt

    # 测试噪声预测精度
    python scripts/unet/train_unet_full.py --mode test --checkpoint ./outputs/unet_8ch/checkpoint.pt

数据目录结构：
    data_root/
      ├── rgb/      # RGB 纹理图 (.png / .jpg)
      ├── dem/      # DEM 高度图 (.npy / .png / .tif)
      └── txt/      # 文本提示词 (.txt)，与图片同名
"""

import argparse
import os
import sys
# import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# warnings.filterwarnings("ignore")

# 将项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from train.train_pipeline import UNetTrainingPipeline, test_noise_prediction


# =============================================================================
# 默认超参数（可通过命令行覆盖）
# =============================================================================

_DEFAULT_DATA_ROOT = "./data/unet_training"
_DEFAULT_DEM_VAE_CKPT = ""
_DEFAULT_OUTPUT_DIR = "./outputs/unet_8ch"
_DEFAULT_EPOCHS = 50
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_LEARNING_RATE = 1e-4
_DEFAULT_WEIGHT_DECAY = 1e-4
_DEFAULT_NUM_WORKERS = 4
_DEFAULT_USE_AMP = True
_DEFAULT_SAVE_STEPS = 1000
_DEFAULT_VIZ_INTERVAL = 1
_DEFAULT_WARMUP_EPOCHS = 5


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="8 通道 U-Net 训练脚本（效果优先版）")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="运行模式：train（训练）或 test（噪声预测精度测试）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=_DEFAULT_DATA_ROOT,
        help="数据集根目录",
    )
    parser.add_argument(
        "--dem_vae_ckpt",
        type=str,
        default=_DEFAULT_DEM_VAE_CKPT,
        help="高度图 VAE 权重路径（为空则用 RGB VAE 替代）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help="模型输出根目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_DEFAULT_EPOCHS,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="批次大小",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=_DEFAULT_LEARNING_RATE,
        help="学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=_DEFAULT_WEIGHT_DECAY,
        help="权重衰减系数",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help="数据加载进程数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=_DEFAULT_SAVE_STEPS,
        help="每隔多少步保存一次快照",
    )
    parser.add_argument(
        "--viz_interval",
        type=int,
        default=_DEFAULT_VIZ_INTERVAL,
        help="每隔几个 epoch 绘制 loss 曲线（0 表示不绘制）",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=_DEFAULT_WARMUP_EPOCHS,
        help="学习率 warmup epoch 数",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=_DEFAULT_USE_AMP,
        help="启用 fp16 混合精度训练",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="禁用混合精度训练",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径（用于断点续训或测试）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="测试模式的样本数",
    )

    return parser


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        print("=== 8 通道 U-Net 训练（效果优先版） ===")

        pipeline = UNetTrainingPipeline(args)

        start_epoch = 0
        if args.checkpoint:
            start_epoch = pipeline.load_checkpoint(args.checkpoint)

        pipeline.train(start_epoch=start_epoch)

    elif args.mode == "test":
        print("=== 8 通道 U-Net 噪声预测精度测试 ===")

        if not args.checkpoint:
            print("错误：测试模式需要指定 --checkpoint 参数")
            sys.exit(1)

        pipeline = UNetTrainingPipeline(args)
        pipeline.load_checkpoint(args.checkpoint)
        test_noise_prediction(pipeline, num_samples=args.num_samples)
