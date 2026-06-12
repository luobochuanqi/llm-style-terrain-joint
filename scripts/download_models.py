"""
下载 CLIP 和 SD VAE 模型到本地指定文件夹。

用法:
    # 下载所有模型到默认路径 ./data/models/
    python scripts/download_models.py

    # 指定目标目录
    python scripts/download_models.py --models_dir /path/to/models

    # 仅下载 CLIP
    python scripts/download_models.py --clip_only

    # 仅下载 VAE
    python scripts/download_models.py --vae_only

下载内容:
    CLIP:   openai/clip-vit-large-patch14  → {models_dir}/clip-vit-large-patch14/
    VAE:    stabilityai/sd-vae-ft-mse      → {models_dir}/sd-vae-ft-mse/
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

DEFAULT_MODELS_DIR = "./data/models"


def download_clip(models_dir: str):
    from transformers import CLIPTextModel, CLIPTokenizer

    save_path = os.path.join(models_dir, "clip-vit-large-patch14")
    model_id = "openai/clip-vit-large-patch14"

    print(f"[CLIP] 正在下载 {model_id} ...")
    model = CLIPTextModel.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"[CLIP] 已保存至: {save_path}")


def download_vae(models_dir: str):
    from diffusers import AutoencoderKL

    save_path = os.path.join(models_dir, "sd-vae-ft-mse")
    model_id = "stabilityai/sd-vae-ft-mse"

    print(f"[VAE] 正在下载 {model_id} ...")
    vae = AutoencoderKL.from_pretrained(model_id)

    os.makedirs(save_path, exist_ok=True)
    vae.save_pretrained(save_path)

    print(f"[VAE] 已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="下载 CLIP 和 VAE 模型到本地")
    parser.add_argument(
        "--models_dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"模型保存目录 (默认: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--clip_only", action="store_true", help="仅下载 CLIP")
    parser.add_argument("--vae_only", action="store_true", help="仅下载 VAE")
    args = parser.parse_args()

    download_all = not args.clip_only and not args.vae_only

    print(f"模型将保存至: {os.path.abspath(args.models_dir)}")
    print("-" * 50)

    if download_all or args.clip_only:
        download_clip(args.models_dir)

    if download_all or args.vae_only:
        download_vae(args.models_dir)

    print("-" * 50)
    print("下载完成!")


if __name__ == "__main__":
    main()
