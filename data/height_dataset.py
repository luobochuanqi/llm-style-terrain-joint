"""
高度图数据集

加载高度图数据用于训练 VAE
支持的数据格式：
- .npy (NumPy 数组)
- .exr (OpenEXR 高程格式)
- .png / .tiff (灰度图)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path


class HeightMapDataset(Dataset):
    """
    高度图数据集

    从指定目录加载高度图文件
    所有高度图会被归一化到 [-1, 1] 范围
    """

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {".npy", ".exr", ".png", ".tiff", ".tif"}

    def __init__(
        self,
        data_root: str,
        image_size: int = 512,
        h_max: float = 3000.0,
        augment: bool = False,
    ):
        """
        初始化数据集

        Args:
            data_root: 数据根目录
            image_size: 图像尺寸（正方形）
            h_max: 全局最大高程用于归一化（米）
            augment: 是否使用数据增强
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.h_max = h_max
        self.augment = augment

        # 收集所有高度图文件路径
        self.file_paths = self._collect_file_paths()

        if len(self.file_paths) == 0:
            raise ValueError(f"在 {data_root} 目录下未找到支持的高度图文件")

        print(f"加载到 {len(self.file_paths)} 个高度图文件")

    def _collect_file_paths(self) -> List[Path]:
        """收集所有支持的高度图文件"""
        file_paths = []

        # 遍历目录（包括子目录）
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_paths.append(Path(root) / file)

        return sorted(file_paths)

    def _load_height_map(self, file_path: Path) -> np.ndarray:
        """
        加载单个高度图文件

        Args:
            file_path: 文件路径
        Returns:
            height_map: 高度图数组 [H, W] 或 [1, H, W]
        """
        ext = file_path.suffix.lower()

        if ext == ".npy":
            height_map = np.load(file_path)

        elif ext == ".exr":
            # TODO: 使用 OpenEXR 库加载
            # import OpenEXR
            # import Imath
            # exr_file = OpenEXR.InputFile(str(file_path))
            # ...
            raise NotImplementedError("EXR 格式加载待实现")

        elif ext in {".png", ".tiff", ".tif"}:
            # 使用 PIL 或 opencv 加载
            try:
                from PIL import Image

                img = Image.open(file_path)
                height_map = np.array(img)
            except ImportError:
                import cv2

                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                height_map = img

        else:
            raise ValueError(f"不支持的文件格式：{ext}")

        # 确保是 2D 数组
        if height_map.ndim == 3:
            if height_map.shape[0] == 1:
                height_map = height_map[0]
            elif height_map.shape[2] == 1:
                height_map = height_map[:, :, 0]
            else:
                # 如果是 RGB，转灰度
                height_map = np.mean(height_map, axis=2)

        return height_map.astype(np.float32)

    def _normalize(self, height_map: np.ndarray) -> torch.Tensor:
        """
        归一化高度图到 [-1, 1]

        Args:
            height_map: 原始高度图 [H, W]
        Returns:
            normalized: 归一化后的高度图 [1, H, W]
        """
        # 归一化到 [0, 1]
        normalized = height_map / self.h_max

        # 映射到 [-1, 1]
        normalized = normalized * 2 - 1

        # 添加通道维度 [1, H, W]
        normalized = np.expand_dims(normalized, axis=0)

        return torch.from_numpy(normalized)

    def _augment(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        数据增强

        Args:
            height_map: 高度图 [1, H, W]
        Returns:
            augmented: 增强后的高度图
        """
        # 随机水平翻转
        if torch.rand(1).item() > 0.5:
            height_map = torch.flip(height_map, dims=[2])

        # 随机垂直翻转
        if torch.rand(1).item() > 0.5:
            height_map = torch.flip(height_map, dims=[1])

        # 随机旋转 90 度
        k = torch.randint(0, 4, (1,)).item()
        height_map = torch.rot90(height_map, k, dims=[1, 2])

        return height_map

    def _resize(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        调整图像尺寸

        Args:
            height_map: 高度图 [1, H, W]
        Returns:
            resized: 调整后的图像 [1, image_size, image_size]
        """
        from torch.nn import functional as F

        # 添加 batch 维度 [1, 1, H, W]
        height_map = height_map.unsqueeze(0)

        # 双线性插值缩放
        resized = F.interpolate(
            height_map,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # 移除 batch 维度 [1, H, W]
        return resized.squeeze(0)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        获取单个样本

        Args:
            idx: 样本索引
        Returns:
            height_map: 归一化后的高度图 [1, image_size, image_size]
            info: 包含元信息的字典
        """
        # 加载文件
        file_path = self.file_paths[idx]
        height_map = self._load_height_map(file_path)

        # 归一化
        height_map = self._normalize(height_map)

        # 调整尺寸
        height_map = self._resize(height_map)

        # 数据增强（仅训练时）
        if self.augment:
            height_map = self._augment(height_map)

        # 元信息
        info = {
            "file_path": str(file_path),
            "h_max": self.h_max,
        }

        return height_map, info


def create_dataloader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = False,
) -> DataLoader:
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        augment: 是否使用数据增强
    Returns:
        dataloader: PyTorch 数据加载器
    """
    dataset = HeightMapDataset(
        data_root=data_root,
        image_size=image_size,
        augment=augment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
