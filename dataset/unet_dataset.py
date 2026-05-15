import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class UNetDataset(Dataset):
    """
    8通道联合生成数据集 (RGB + DEM + Prompt)
    自动通过文件名匹配 RGB、DEM 和对应的 txt 提示词文件。
    """
    def __init__(
        self, 
        data_root: str, 
        image_size: int = 512, 
        augment: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5
    ):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5]) # 将 [0, 1] 映射到 [-1, 1]

        # 自动解析元数据
        self._parse_metadata()

    def _parse_metadata(self):
        """
        核心查找逻辑：假设文件结构如下
        data_root/
          ├── rgb/     (存放 .png / .jpg)
          ├── dem/     (存放 .npy / .png / .tif)
          └── txt/     (存放 .txt 提示词)
        """
        rgb_dir = os.path.join(self.data_root, "rgb")
        dem_dir = os.path.join(self.data_root, "dem")
        txt_dir = os.path.join(self.data_root, "txt")

        # 扫描所有的彩色图
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.*")))
        if len(rgb_files) == 0:
            raise FileNotFoundError(f"在 {rgb_dir} 下未找到任何图片文件！")

        self.metadata = []
        for rgb_path in rgb_files:
            basename = os.path.basename(rgb_path)
            name_without_ext = os.path.splitext(basename)[0]

            # 寻找同名的 DEM 文件 (用通配符兼容 .npy 和 .png)
            dem_pattern = os.path.join(dem_dir, f"{name_without_ext}.*")
            dem_matches = glob.glob(dem_pattern)
            
            if not dem_matches:
                print(f"警告：找不到 {name_without_ext} 对应的 DEM，跳过该样本。")
                continue
            dem_path = dem_matches[0]

            # 寻找同名的 txt 提示词文件
            txt_path = os.path.join(txt_dir, f"{name_without_ext}.txt")
            
            self.metadata.append({
                "rgb_path": rgb_path,
                "dem_path": dem_path,
                "txt_path": txt_path,
                "basename": name_without_ext
            })
            
        print(f"成功扫描并匹配了 {len(self.metadata)} 组联合数据！")

    def __len__(self):
        return len(self.metadata)

    def _load_dem_to_tensor(self, dem_path: str) -> torch.Tensor:
        """兼容读取 .npy 和常见图像格式的 DEM，并统一转为 [1, 512, 512] 的 Tensor"""
        if dem_path.endswith(".npy"):
            # 复用你 VAE dataset 里的 numpy 读取逻辑
            arr = np.load(dem_path) # [H, W] float32
            if arr.shape != (self.image_size, self.image_size):
                img = Image.fromarray(arr).resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0) # [1, H, W]
        else:
            # 读取普通单通道图像
            img = Image.open(dem_path).convert("L")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensor = self.to_tensor(img) # to_tensor 自动除以 255 归一化到 [0,1]
            
        return tensor

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        
        # 1. 读取 Prompt 文本
        prompt = ""
        if os.path.exists(entry["txt_path"]):
            with open(entry["txt_path"], "r", encoding="utf-8") as f:
                prompt = f.read().strip()

        # 2. 读取 RGB 彩图并缩放
        rgb_image = Image.open(entry["rgb_path"]).convert("RGB")
        rgb_image = rgb_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb_tensor = self.to_tensor(rgb_image) # [3, 512, 512], 值域 [0, 1]

        # 3. 读取 DEM 高程图并缩放
        dem_tensor = self._load_dem_to_tensor(entry["dem_path"]) # [1, 512, 512], 值域 [0, 1]

        # ==========================================
        # 4. 同步数据增强 (在 Tensor 级别操作，绝不错位)
        # ==========================================
        if self.augment:
            if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
                rgb_tensor = torch.flip(rgb_tensor, dims=[-1])
                dem_tensor = torch.flip(dem_tensor, dims=[-1])

            if self.vflip_prob > 0 and torch.rand(1).item() < self.vflip_prob:
                rgb_tensor = torch.flip(rgb_tensor, dims=[-2])
                dem_tensor = torch.flip(dem_tensor, dims=[-2])

            if self.rot90_prob > 0 and torch.rand(1).item() < self.rot90_prob:
                k = torch.randint(0, 4, (1,)).item()
                rgb_tensor = torch.rot90(rgb_tensor, k, dims=[-2, -1])
                dem_tensor = torch.rot90(dem_tensor, k, dims = [-2, -1])

        # 5. 最终映射到扩散模型所需的 [-1, 1] 区间
        # to_tensor() 或我们读入的 numpy 数据系 [0, 1] 范围
        rgb_tensor = self.normalize(rgb_tensor) 
        dem_tensor = self.normalize(dem_tensor) 

        return {
            "rgb": rgb_tensor,
            "dem": dem_tensor,
            "prompt": prompt,
            "basename": entry["basename"]
        }