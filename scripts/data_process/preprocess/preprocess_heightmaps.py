"""
高度图数据集预处理脚本

按照 docs/plans/heightmap_dataset_process_plan.md 方案，
将 data/origin/heightmaps 的 1081×1081 uint16 PNG 瓦片处理为
data/process/heightmaps 的 512×512 float32 归一化数组。

管线：
  1. 空间重采样：中心裁剪 1080×1080 → Area 缩放 → 512×512
  2. 全局百分位截断（p_low, p_high）
  3. 对数变换：h_log = log(h_clipped - p_low + 1)
  4. 线性映射到 [0, 1]
  5. 保存为 .npy 文件 + 归一化参数 JSON

用法：
    # 第一步：统计全局参数（必须）
    python scripts/data_process/preprocess/preprocess_heightmaps.py --stage stats

    # 第二步：应用归一化并保存
    python scripts/data_process/preprocess/preprocess_heightmaps.py --stage transform

    # 一步完成
    python scripts/data_process/preprocess/preprocess_heightmaps.py --stage all
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
INPUT_DIR = "data/origin/heightmaps_hf"
OUTPUT_DIR = "data/process/heightmaps_hf"
PARAMS_FILE = "data/process/heightmaps_hf/norm_params.json"

ORIGINAL_SIZE = 1081
CROP_SIZE = 1080
TARGET_SIZE = 512

PERCENTILE_LOW = 0.1  # 下截断百分位
PERCENTILE_HIGH = 99.9  # 上截断百分位


def get_input_files(input_dir: str):
    return sorted(glob.glob(os.path.join(input_dir, "hmap_*.png")))


# ============================================================
# Stage 1: 全局统计
# ============================================================
def compute_global_stats(input_dir: str, sample_ratio: float = 1.0):
    """
    扫描全部文件，采样像素，计算：
      - 全局像素累积分布 → p_low, p_high 百分位值
      - 截断并 log 变换后的 min_log, max_log
    """
    files = get_input_files(input_dir)
    print(f"找到 {len(files)} 个文件")

    if sample_ratio < 1.0:
        rng = np.random.RandomState(42)
        n_sample = max(1, int(len(files) * sample_ratio))
        sampled = rng.choice(files, n_sample, replace=False)
        print(f"采样 {n_sample}/{len(files)} 个文件用于统计")
        files = list(sampled)

    # 第一遍：收集全局像素分布（采样像素以节省内存）
    all_values = []
    for fpath in tqdm(files, desc="扫描像素分布"):
        arr = np.array(Image.open(fpath))
        # 空间重采样
        arr = center_crop_resize(arr)
        arr_flat = arr.ravel()
        # 每张图采样最多 256×256 = 65536 像素
        if len(arr_flat) > 65536:
            indices = np.linspace(0, len(arr_flat) - 1, 65536, dtype=np.int64)
            arr_flat = arr_flat[indices]
        all_values.append(arr_flat)

    all_values = np.concatenate(all_values)
    print(f"总采样像素: {len(all_values):,}")

    # 计算百分位
    p_low_val = float(np.percentile(all_values, PERCENTILE_LOW))
    p_high_val = float(np.percentile(all_values, PERCENTILE_HIGH))
    print(f"像素范围: [{all_values.min()}, {all_values.max()}]")
    print(f"百分位 {PERCENTILE_LOW}%: {p_low_val:.1f}")
    print(f"百分位 {PERCENTILE_HIGH}%: {p_high_val:.1f}")

    # 第二遍：截断 + log变换后计算min/max
    log_vals = np.log(np.clip(all_values, p_low_val, p_high_val) - p_low_val + 1)
    min_log = float(log_vals.min())
    max_log = float(log_vals.max())
    print(f"log空间范围: [{min_log:.4f}, {max_log:.4f}]")

    params = {
        "original_size": ORIGINAL_SIZE,
        "target_size": TARGET_SIZE,
        "p_low": p_low_val,
        "p_high": p_high_val,
        "min_log": min_log,
        "max_log": max_log,
        "description": "对数变换归一化参数，用于高程图的归一化/反归一化",
    }

    # 保存参数
    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n归一化参数已保存到: {PARAMS_FILE}")

    return params


def center_crop_resize(arr: np.ndarray) -> np.ndarray:
    """中心裁剪到 1080×1080 → Area 缩放 → 512×512 float32"""
    h, w = arr.shape
    assert (
        h == ORIGINAL_SIZE and w == ORIGINAL_SIZE
    ), f"期望 {ORIGINAL_SIZE}×{ORIGINAL_SIZE}，实际 {h}×{w}"

    # 中心裁剪
    margin = (ORIGINAL_SIZE - CROP_SIZE) // 2
    cropped = arr[margin : margin + CROP_SIZE, margin : margin + CROP_SIZE]

    # PIL Area 插值缩放（抗混叠）
    img = Image.fromarray(cropped)
    resized = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.float32)


def normalize(arr: np.ndarray, params: dict) -> np.ndarray:
    """
    对数变换归一化
      h_log = log(h_clipped - p_low + 1)
      h_norm = (h_log - min_log) / (max_log - min_log)
      clamp to [0, 1]
    """
    clipped = np.clip(arr, params["p_low"], params["p_high"])
    log_h = np.log(clipped - params["p_low"] + 1)
    norm_h = (log_h - params["min_log"]) / (params["max_log"] - params["min_log"])
    return np.clip(norm_h, 0.0, 1.0)


# ============================================================
# Stage 2: 逐文件处理
# ============================================================
def transform_all(input_dir: str, output_dir: str, params_file: str):
    """逐文件做 resample + normalize，保存为 .npy"""
    with open(params_file, "r") as f:
        params = json.load(f)

    files = get_input_files(input_dir)
    print(f"处理 {len(files)} 个文件")

    os.makedirs(output_dir, exist_ok=True)
    stats = {"total": len(files), "saved": 0, "errors": []}

    for fpath in tqdm(files, desc="resample + normalize"):
        basename = os.path.basename(fpath)
        try:
            arr = np.array(Image.open(fpath))
            arr = center_crop_resize(arr)
            arr_norm = normalize(arr, params)

            out_name = os.path.splitext(basename)[0] + ".npy"
            np.save(os.path.join(output_dir, out_name), arr_norm.astype(np.float32))
            stats["saved"] += 1
        except Exception as e:
            stats["errors"].append(f"{basename}: {e}")

    # 也复制一份 norm_params.json 到输出目录（如果不在同一位置）
    import shutil

    dst_params = os.path.join(output_dir, "norm_params.json")
    if os.path.abspath(params_file) != os.path.abspath(dst_params):
        shutil.copy(params_file, dst_params)

    print(f"\n完成: 保存 {stats['saved']}/{stats['total']} 个文件到 {output_dir}")
    if stats["errors"]:
        print(f"错误: {len(stats['errors'])} 个")
        for e in stats["errors"][:5]:
            print(f"  {e}")


# ============================================================
# 反归一化（供 decoder 输出后使用）
# ============================================================
def denormalize(norm_arr: np.ndarray, params: dict) -> np.ndarray:
    """
    反归一化至物理高程（uint16）
      h_log = norm_arr * (max_log - min_log) + min_log
      h = exp(h_log) + p_low - 1
      round & clip to [0, 65535]
    """
    log_h = norm_arr * (params["max_log"] - params["min_log"]) + params["min_log"]
    h = np.exp(log_h) + params["p_low"] - 1
    return np.round(np.clip(h, 0, 65535)).astype(np.uint16)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="高度图数据集预处理（对数变换归一化）")
    parser.add_argument(
        "--stage",
        choices=["stats", "transform", "all"],
        default="all",
        help="stats=仅统计参数 / transform=仅变换 / all=全部",
    )
    parser.add_argument(
        "--input_dir",
        default=INPUT_DIR,
        help=f"原始 PNG 目录 (默认: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help=f"处理结果输出目录 (默认: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--params_file",
        default=PARAMS_FILE,
        help=f"归一化参数 JSON 路径 (默认: {PARAMS_FILE})",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.3,
        help="stats 阶段像素采样比例 (0-1, 默认 0.3)",
    )

    args = parser.parse_args()

    if args.stage in ("stats", "all"):
        compute_global_stats(args.input_dir, sample_ratio=args.sample_ratio)

    if args.stage in ("transform", "all"):
        transform_all(args.input_dir, args.output_dir, args.params_file)

    if args.stage == "all":
        print("\n全部完成！")


if __name__ == "__main__":
    main()
