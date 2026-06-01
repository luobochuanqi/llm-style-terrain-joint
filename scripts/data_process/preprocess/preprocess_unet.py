import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. 配置文件夹路径 (请根据你的实际情况修改)
# ==========================================
# DEM 文件夹配置
DEM_INPUT_DIR = r"D:\WorkSpace\Data\unet\dem"
DEM_OUTPUT_DIR = r"D:\WorkSpace\Data\unet\dem_npy"

# 卫星图文件夹配置
SAT_INPUT_DIR = r""
SAT_OUTPUT_DIR = r"data/unet_training/rgb"

# VAE 预处理时生成的全局参数文件路径
# 【极其重要】：必须指向 VAE 训练用的那个 JSON 文件！
PARAMS_FILE = "data/process/heightmaps_hf/norm_params.json" 

# 目标分辨率 (用于安全检查)
TARGET_SIZE = 512

# ==========================================
# 2. 高程图归一化函数 (与 VAE 保持绝对一致)
# ==========================================
def normalize_dem(arr: np.ndarray, params: dict) -> np.ndarray:
    """
    使用 VAE 统计出的全局参数，进行对数变换和 min-max 归一化
    """
    clipped = np.clip(arr, params["p_low"], params["p_high"])
    log_h = np.log(clipped - params["p_low"] + 1)
    norm_h = (log_h - params["min_log"]) / (params["max_log"] - params["min_log"])
    return np.clip(norm_h, 0.0, 1.0)


# ==========================================
# 3. 核心转换函数
# ==========================================
def process_folder(input_dir, output_dir, mode="dem", file_ext=".png", params=None):
    if not os.path.exists(input_dir):
        print(f"警告: 输入文件夹 {input_dir} 不存在，跳过该任务。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 动态匹配扩展名
    image_files = sorted(glob.glob(os.path.join(input_dir, f"*{file_ext}")))
    if not image_files:
        print(f"警告: 在 {input_dir} 中没有找到 {file_ext} 文件。")
        return

    print(f"\n开始转换 {mode.upper()} 数据: {len(image_files)} 张图片")
    
    success_count = 0
    for fpath in tqdm(image_files, desc=f"处理 {mode.upper()}"):
        basename = os.path.basename(fpath)
        out_name = os.path.splitext(basename)[0] + ".npy"
        out_path = os.path.join(output_dir, out_name)
        
        try:
            img = Image.open(fpath)
            
            # 尺寸安全检查
            if img.size != (TARGET_SIZE, TARGET_SIZE):
                tqdm.write(f"跳过 {basename}: 尺寸为 {img.size}，不是目标尺寸 {TARGET_SIZE}x{TARGET_SIZE}")
                continue
            
            if mode == "dem":
                # DEM: 转为 float32 并应用与 VAE 一致的对数归一化
                arr = np.array(img, dtype=np.float32)
                if params is None:
                    raise ValueError("处理 DEM 时必须提供 norm_params.json 的参数！")
                arr = normalize_dem(arr, params)
                # 转回 float32 确保精度兼容
                arr = arr.astype(np.float32) 

            elif mode == "sat":
                # 卫星图: 强制转为 RGB，使用 uint8
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
            else:
                raise ValueError("未知的 mode 类型")

            # 保存为独立的无压缩 .npy 文件
            np.save(out_path, arr)
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"处理文件 {basename} 失败: {e}")
            
    print(f"{mode.upper()} 转换完成！成功保存 {success_count} 个 NPY 文件到 {output_dir}\n")


# ==========================================
# 4. 运行主程序
# ==========================================
if __name__ == "__main__":
    
    # 1. 读取 VAE 训练时生成的全局分布参数
    if not os.path.exists(PARAMS_FILE):
        raise FileNotFoundError(f"找不到参数文件 {PARAMS_FILE}！请确认 VAE 的预处理脚本已经成功运行。")
        
    with open(PARAMS_FILE, "r") as f:
        global_params = json.load(f)
        print("成功加载 DEM 归一化参数: ")
        print(f"  p_low: {global_params['p_low']:.2f}, p_high: {global_params['p_high']:.2f}")
        print(f"  min_log: {global_params['min_log']:.4f}, max_log: {global_params['max_log']:.4f}\n")

    # 2. 执行 DEM 转换 (传入 parameters)
    process_folder(DEM_INPUT_DIR, DEM_OUTPUT_DIR, mode="dem", file_ext=".png", params=global_params)
    
    # 3. 执行 卫星图 转换 (卫星图不需要 params，传 None 即可)
    process_folder(SAT_INPUT_DIR, SAT_OUTPUT_DIR, mode="sat", file_ext=".tif", params=None)
    
    print("所有转换任务已全部结束！")