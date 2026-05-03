"""
扫描 data/origin/heightmaps 并输出统计数据（中英文）
"""

import argparse
import glob
import os
import re

import numpy as np
from PIL import Image


def scan_heightmaps(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "hmap_*.png")))
    total = len(files)
    if total == 0:
        print("未找到 hmap_*.png 文件 / No hmap_*.png files found.")
        return

    print(f"文件总数 / Total files: {total}")
    print("─" * 60)

    # ── 文件名解析 / naming pattern ──
    lons, lats, idxs = [], [], []
    for fpath in files:
        m = re.match(r"hmap_(-?\d+)_(-?\d+)__(\d+)\.png", os.path.basename(fpath))
        if m:
            lons.append(int(m.group(1)))
            lats.append(int(m.group(2)))
            idxs.append(int(m.group(3)))
        else:
            print(f"  [警告/WARN] 无法解析文件名: {os.path.basename(fpath)}")

    print("文件名格式 / Filename pattern: hmap_{lon}_{lat}__{idx}.png")
    print(
        f"  经度范围 / Longitude range: {min(lons)} .. {max(lons)}  "
        f"(不重复数/unique: {len(set(lons))})"
    )
    print(
        f"  纬度范围 / Latitude range:  {min(lats)} .. {max(lats)}   "
        f"(不重复数/unique: {len(set(lats))})"
    )
    idx_set = set(idxs)
    print(
        f"  idx 值 / idx values:      {sorted(idx_set)}  "
        f"(不重复数/unique: {len(idx_set)})"
    )
    print("─" * 60)

    # ── 逐文件统计 / per-file statistics ──
    sizes = {}
    modes = {}
    dtypes = set()
    min_vals, max_vals, mean_vals, std_vals = [], [], [], []
    uniq_counts = []
    pixel_counts = []

    for fpath in files:
        img = Image.open(fpath)
        sz = img.size
        md = img.mode
        sizes[sz] = sizes.get(sz, 0) + 1
        modes[md] = modes.get(md, 0) + 1

        arr = np.array(img)
        dtypes.add(str(arr.dtype))
        min_vals.append(int(arr.min()))
        max_vals.append(int(arr.max()))
        mean_vals.append(float(arr.mean()))
        std_vals.append(float(arr.std()))
        uniq_counts.append(len(np.unique(arr)))
        pixel_counts.append(arr.size)

    print("图像尺寸 / Image size (W×H):")
    for sz, cnt in sorted(sizes.items()):
        print(
            f"  {sz[0]} × {sz[1]}  —  {cnt} 个文件/files  "
            f"({cnt / total * 100:.1f}%)"
        )
    print()

    print("像素模式/位深 / Pixel mode / bit depth:")
    for md, cnt in sorted(modes.items()):
        print(f"  {md}  —  {cnt} 个文件/files")
    print()

    print("NumPy 数据类型 / NumPy dtype:")
    for dt in sorted(dtypes):
        print(f"  {dt}")
    print()

    print("像素值范围 / Pixel value range（每文件统计 / per-file）:")
    print(f"  全局最小值 / Global min:              {min(min_vals)}")
    print(f"  全局最大值 / Global max:              {max(max_vals)}")
    print(f"  文件级最小值均值 / Mean of file-wise mins:  {np.mean(min_vals):.1f}")
    print(f"  文件级最大值均值 / Mean of file-wise maxs:  {np.mean(max_vals):.1f}")
    print(f"  文件级最大值中位数 / Median of file-wise maxs: {np.median(max_vals):.1f}")
    print(f"  文件级最大值标准差 / Std of file-wise maxs:   {np.std(max_vals):.1f}")
    print()

    print("像素值集中趋势 / Pixel value central tendency（每文件平均 / averaged）:")
    print(f"  文件均值之均值 / Mean of file-wise means: {np.mean(mean_vals):.1f}")
    print(f"  文件标准差之均值 / Mean of file-wise stds:  {np.mean(std_vals):.1f}")
    print()

    print("空间分辨率 / Spatial resolution（从文件名推断 / inferred from filename）:")
    print(f"  每块瓦片覆盖 1° × 1°（经/纬度均为整数网格索引）")
    print(f"  图像尺寸 {1081} × {1081} px → ~{1081 / 3600:.1f} px/角秒 (arc-second)")
    print(f"  (1° ≈ 111 km → ~{1081 / 111:.0f} m/像素 at equator)")
    print()

    print("不重复像素值数 / Unique pixel value count（每文件 / per-file）:")
    print(f"  均值 / Mean:  {np.mean(uniq_counts):.0f}")
    print(f"  最小值 / Min:   {min(uniq_counts)}")
    print(f"  最大值 / Max:   {max(uniq_counts)}")
    print(f"  中位数 / Median: {np.median(uniq_counts):.0f}")
    print()

    total_pixels = sum(pixel_counts)
    print(f"全部文件像素总数 / Total pixels across all files: {total_pixels:,}")
    print(f"总数据量 / Total data (uint16): {total_pixels * 2 / (1024**3):.2f} GiB")
    print("─" * 60)

    # ── 坐标网格分布 / coordinate grid summary ──
    unique_lons = sorted(set(lons))
    unique_lats = sorted(set(lats))
    gap_lons = [
        unique_lons[i + 1] - unique_lons[i] for i in range(len(unique_lons) - 1)
    ]
    gap_lats = [
        unique_lats[i + 1] - unique_lats[i] for i in range(len(unique_lats) - 1)
    ]
    print()
    print("坐标网格间距 / Coordinate grid spacing:")
    if gap_lons:
        print(
            f"  经度步长 / Longitude step:  {min(gap_lons)} .. {max(gap_lons)}  "
            f"(中位数/median {np.median(gap_lons):.0f})"
        )
    if gap_lats:
        print(
            f"  纬度步长 / Latitude step:   {min(gap_lats)} .. {max(gap_lats)}  "
            f"(中位数/median {np.median(gap_lats):.0f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="扫描高程图 PNG 文件并输出统计 / Scan heightmap PNG files and print statistics"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/origin/heightmaps_hf",
        help="高程图目录路径 / Path to heightmaps directory",
    )
    args = parser.parse_args()
    scan_heightmaps(args.data_dir)


if __name__ == "__main__":
    main()
