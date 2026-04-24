# LLM 风格地形生成联合模型

根据文本 Prompt 生成匹配的高程图和纹理图。

## 项目结构

```
.
├── main.py                     # 主入口文件
├── configs/                    # 配置文件
│   ├── training_config.py      # 训练超参数配置
│   └── inference_config.py     # 推理超参数配置
├── models/                     # 模型定义
│   ├── clip/                   # 双分支 CLIP 文本编码器
│   │   └── text_encoder.py
│   ├── unet/                   # 8 通道 U-Net 模型
│   │   └── unet_8ch.py
│   └── vae/                    # VAE 编码器/解码器
│       └── heightmap_vae.py    # 高程图专用 VAE
├── train/                      # 训练流水线
│   └── train_pipeline.py       # 训练过程实现
├── inference/                  # 推理流水线
│   └── inference_pipeline.py   # 推理过程实现
└── utils/                      # 工具函数
    └── latent_utils.py         # 隐空间操作工具
```

## 快速开始

### 训练模式

```bash
python main.py --mode train
```

### 推理模式

```bash
python main.py --mode inference
```

### 查看帮助

```bash
python main.py --help
```

## 核心流程

### 训练过程

1. **文本编码**：Prompt 进入双分支 CLIP，输出全局特征和细节特征
2. **图像压缩**：真实高度图和纹理图分别通过 VAE 编码，拼接为 8 通道联合隐向量
3. **前向加噪**：随机时间步 t，将高斯噪声混入联合隐向量
4. **模型预测**：U-Net 根据文本条件和脏图预测噪声
5. **反向传播**：计算预测噪声与真实噪声的 MSE 损失，更新权重

### 推理过程

1. **文本编码**：Prompt 翻译为向量指令
2. **准备初始噪声**：随机生成 8x64x64 高斯噪声
3. **循环降噪**：DDIM 采样器 50 步迭代去噪
4. **拆分输出**：8 通道隐向量拆分为高度和纹理，分别解码

## 配置说明

### 训练配置 (`configs/training_config.py`)

- `DataConfig`: 数据集路径、图像尺寸、批次大小
- `ModelConfig`: CLIP、VAE、U-Net 模型参数
- `NoiseSchedulerConfig`: DDPM 噪声调度参数
- `OptimizerConfig`: AdamW 优化器配置
- `TrainingConfig`: 训练轮数、梯度裁剪、混合精度
- `CheckpointConfig`: 检查点保存策略
- `LoggingConfig`: 日志记录配置

### 推理配置 (`configs/inference_config.py`)

- `ModelPathConfig`: 预训练模型权重路径
- `ModelConfig`: 模型结构参数
- `SamplerConfig`: DDIM 采样器参数
- `OutputConfig`: 输出格式和路径
- `DeviceConfig`: 运行设备配置
- `MemoryConfig`: 显存优化配置

## 技术细节

### 8 通道联合隐空间

- 前 4 通道：高程隐向量（魔改 VAE 编码/解码）
- 后 4 通道：纹理隐向量（SD VAE 编码/解码）
- 拼接方式：`torch.cat([height_latent, texture_latent], dim=1)`

### 双分支 CLIP

- 全局特征：捕捉整体语义（如"广东丹霞地貌"）
- 细节特征：捕捉局部描述（如"红色平顶方山"）
- 通过交叉注意力层注入 U-Net

### 高程图 VAE

- 输入：1 通道 512x512 高程图（归一化到 [-1, 1]）
- 输出：4 通道 64x64 隐向量
- 几何约束损失：坡度损失 + 曲率损失
