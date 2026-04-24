"""
生成测试用高度图数据

用于快速验证训练流程是否正常
生成包含陡崖、山峰、山谷等地形特征的高度图
"""

import numpy as np
import os
from pathlib import Path


def generate_terrain_with_cliff(size: int = 512) -> np.ndarray:
    """
    生成包含陡崖的地形

    Args:
        size: 图像尺寸
    Returns:
        height_map: 高度图 [size, size]，单位米
    """
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)

    # 基础地形：多个正弦波叠加
    Z = (
        200 * np.sin(X * 2) * np.cos(Y * 2)
        + 100 * np.sin(X * 5 + Y * 3)
        + 50 * np.cos(X * 8 - Y * 5)
    )

    # 添加陡崖特征（高程突变）
    cliff_mask = (X > 4) & (X < 6)
    Z[cliff_mask] += 800 * (X[cliff_mask] - 4) / 2

    # 添加山峰
    center_x, center_y = size // 2, size // 2
    distance = np.sqrt((X - 5) ** 2 + (Y - 5) ** 2)
    peak = 500 * np.exp(-(distance**2) / 2)
    Z += peak

    # 添加山谷
    valley = -300 * np.exp(-((X - 3) ** 2) / 1.5)
    Z += valley

    # 确保高程在合理范围内 [0, 2000]
    Z = Z - Z.min()
    Z = Z / Z.max() * 2000

    return Z.astype(np.float32)


def generate_simple_cliff(size: int = 512) -> np.ndarray:
    """
    生成简单的陡崖测试图

    Args:
        size: 图像尺寸
    Returns:
        height_map: 高度图 [size, size]，单位米
    """
    Z = np.zeros((size, size), dtype=np.float32)

    # 左半部分低海拔
    Z[:, : size // 3] = 100

    # 中间陡崖
    cliff_width = size // 6
    for i in range(cliff_width):
        Z[:, size // 3 + i] = 100 + (i / cliff_width) * 1500

    # 右半部分高海拔
    Z[:, size // 3 + cliff_width :] = 1600

    # 添加一些噪声
    noise = np.random.normal(0, 20, Z.shape)
    Z += noise

    # 裁剪到 [0, 2000]
    Z = np.clip(Z, 0, 2000)

    return Z


def generate_dataset(output_dir: str, num_samples: int = 20):
    """
    生成测试数据集

    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"生成 {num_samples} 个测试高度图...")

    for i in range(num_samples):
        if i % 2 == 0:
            # 复杂地形
            height_map = generate_terrain_with_cliff()
        else:
            # 简单陡崖
            height_map = generate_simple_cliff()

        # 保存为 .npy 文件
        filename = output_path / f"terrain_{i:03d}.npy"
        np.save(filename, height_map)

        # 打印统计信息
        print(
            f"  {filename.name}: "
            f"min={height_map.min():.1f}m, "
            f"max={height_map.max():.1f}m, "
            f"mean={height_map.mean():.1f}m"
        )

    print(f"\n数据集保存到：{output_path}")
    print(f"共 {num_samples} 个文件")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成测试高度图数据")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/height_maps",
        help="输出目录",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="样本数量",
    )

    args = parser.parse_args()

    generate_dataset(args.output_dir, args.num_samples)

    print("\n使用方法:")
    print(
        f"  python scripts/train_height_vae.py --data_root {args.output_dir} --epochs 100"
    )
