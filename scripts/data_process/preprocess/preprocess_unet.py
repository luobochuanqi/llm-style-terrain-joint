"""
U-Net 训练数据预处理：将原始 DEM/卫星图转为标准化的 .npy 文件。

DEM: 原始高程 PNG → 对数归一化 [0,1] (与 HeightMapVAE 预处理一致)
卫星图: 原始 TIFF/PNG → RGB uint8 .npy

用法：
    python scripts/data_process/preprocess/preprocess_unet.py
"""

import glob
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# 配置

DEM_INPUT_DIR = "./data/origin/unet_training/dem"
DEM_OUTPUT_DIR = "./data/unet_training/dem"
DEM_EXT = ".png"

SAT_INPUT_DIR = "./data/origin/unet_training/rgb"
SAT_OUTPUT_DIR = "./data/unet_training/rgb"
SAT_EXT = ".tif"

PARAMS_FILE = "data/process/heightmaps_hf/norm_params.json"
TARGET_SIZE = 512


# 对数归一化 (与 HeightMapVAE 预处理一致)
def normalize_dem(arr: np.ndarray, params: dict) -> np.ndarray:
    """逐元素: clip → log(1 + h - p_low) → min-max → [0, 1]"""
    clipped = np.clip(arr, params["p_low"], params["p_high"])
    log_h = np.log(clipped - params["p_low"] + 1)
    norm_h = (log_h - params["min_log"]) / (params["max_log"] - params["min_log"])
    return np.clip(norm_h, 0.0, 1.0)


# 核心转换函数
def process_folder(input_dir, output_dir, mode="dem", file_ext=".png", params=None):
    """批量转换图片为标准化 .npy。mode="dem" → float32 [0,1]; mode="sat" → uint8 RGB。"""
    if not os.path.exists(input_dir):
        print(f"跳过: {input_dir} 不存在。")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, f"*{file_ext}")))
    if not image_files:
        print(f"跳过: {input_dir} 中未找到 {file_ext} 文件。")
        return

    print(
        f"转换 {mode.upper()}: {len(image_files)} 个文件 ({input_dir} → {output_dir})"
    )

    success_count = 0
    for fpath in tqdm(image_files, desc=mode.upper()):
        basename = os.path.basename(fpath)
        out_path = os.path.join(output_dir, os.path.splitext(basename)[0] + ".npy")

        try:
            img = Image.open(fpath)

            if img.size != (TARGET_SIZE, TARGET_SIZE):
                tqdm.write(
                    f"跳过 {basename}: 尺寸 {img.size} ≠ {TARGET_SIZE}×{TARGET_SIZE}"
                )
                continue

            if mode == "dem":
                if params is None:
                    raise ValueError("DEM 处理需要 norm_params.json")
                arr = np.array(img, dtype=np.float32)
                arr = normalize_dem(arr, params)

            elif mode == "sat":
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
            else:
                raise ValueError(f"未知 mode: {mode}")

            np.save(out_path, arr)
            success_count += 1

        except Exception as e:
            tqdm.write(f"处理 {basename} 失败: {e}")

    print(f"{mode.upper()} 完成: {success_count}/{len(image_files)} → {output_dir}\n")


# 主程序
if __name__ == "__main__":
    if not os.path.exists(PARAMS_FILE):
        raise FileNotFoundError(f"找不到 {PARAMS_FILE}，请先运行 VAE 预处理脚本")
    with open(PARAMS_FILE, "r") as f:
        global_params = json.load(f)
    print(
        f"DEM 归一化参数: p_low={global_params['p_low']:.2f}, "
        f"p_high={global_params['p_high']:.2f}, "
        f"min_log={global_params['min_log']:.4f}, "
        f"max_log={global_params['max_log']:.4f}\n"
    )

    process_folder(
        DEM_INPUT_DIR,
        DEM_OUTPUT_DIR,
        mode="dem",
        file_ext=DEM_EXT,
        params=global_params,
    )

    process_folder(
        SAT_INPUT_DIR,
        SAT_OUTPUT_DIR,
        mode="sat",
        file_ext=SAT_EXT,
    )

    print("全部完成。")
