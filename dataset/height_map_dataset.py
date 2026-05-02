import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class HeightMapDataset(Dataset):
    """
    高度图数据集

    从预处理后的 .npy 文件加载 512×512 归一化高度图。

    预处理管线参见 scripts/data_process/preprocess/preprocess_heightmaps.py：
      1081×1081 uint16 → 中心裁剪 1080×1080 → Area 缩放 512×512 →
      百分位截断 → 对数变换 → 线性映射 [0, 1]

    Returns:
        tensor: float32 [1, H, W]，归一化范围 [0, 1]
        info: dict，包含 file_path、lon、lat、idx 等元信息
    """

    def __init__(
        self,
        data_root: str = "data/process/heightmaps_hf",
        image_size: int = 512,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rot90_prob: float = 0.5,
    ):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob

        self.file_list = sorted(glob.glob(os.path.join(data_root, "hmap_*.npy")))
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"未找到 hmap_*.npy 文件: {data_root}")

        self._parse_metadata()

    def _parse_metadata(self):
        """预解析文件名中的坐标和索引信息"""
        import re

        self.metadata = []
        for fpath in self.file_list:
            basename = os.path.basename(fpath)
            m = re.match(r"hmap_(-?\d+)_(-?\d+)__(\d+)\.npy", basename)
            if m:
                self.metadata.append(
                    {
                        "file_path": fpath,
                        "lon": int(m.group(1)),
                        "lat": int(m.group(2)),
                        "idx": int(m.group(3)),
                    }
                )
            else:
                self.metadata.append({"file_path": fpath})

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        info = self.metadata[idx].copy()

        arr = np.load(self.file_list[idx])  # [H, W] float32

        # 可选 resize（数据已为 target_size 时跳过）
        if arr.shape != (self.image_size, self.image_size):
            from PIL import Image

            img = Image.fromarray(arr)
            img = img.resize(
                (self.image_size, self.image_size),
                Image.Resampling.LANCZOS,
            )
            arr = np.array(img, dtype=np.float32)

        # 添加通道维度 [H, W] → [1, H, W]
        tensor = torch.from_numpy(arr).unsqueeze(0)

        # 数据增强（仅在训练模式下）
        if self.augment:
            tensor = self._apply_augment(tensor)

        return tensor, info

    def _apply_augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """随机水平/垂直翻转 + 90° 倍数旋转"""
        if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
            tensor = torch.flip(tensor, dims=[-1])  # 水平翻转

        if self.vflip_prob > 0 and torch.rand(1).item() < self.vflip_prob:
            tensor = torch.flip(tensor, dims=[-2])  # 垂直翻转

        if self.rot90_prob > 0 and torch.rand(1).item() < self.rot90_prob:
            k = torch.randint(0, 4, (1,)).item()
            tensor = torch.rot90(tensor, k, dims=[-2, -1])

        return tensor
