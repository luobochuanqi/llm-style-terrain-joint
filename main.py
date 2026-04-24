"""
LLM 风格地形生成联合模型

主入口文件
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.train_pipeline import TrainingPipeline
from inference.inference_pipeline import InferencePipeline
from models.clip.text_encoder import build_text_encoder
from models.unet.unet_8ch import build_unet
from models.vae.heightmap_vae import HeightMapVAE


# =============================================================================
# 训练配置（硬编码在代码中）
# =============================================================================

# 数据配置
DATA_ROOT = "./data"
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 4

# 模型配置
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GLOBAL_FEATURE_DIM = 512
LOCAL_FEATURE_DIM = 512

# VAE 配置
HEIGHT_VAE_LATENT_CHANNELS = 4
HEIGHT_H_MAX = 3000.0  # 全局最大高程用于归一化
TEXTURE_VAE_PRETRAINED = "stabilityai/stable-diffusion-2-1"

# U-Net 配置
UNET_IN_CHANNELS = 8  # 4 高度 + 4 纹理
UNET_OUT_CHANNELS = 8
UNET_BLOCK_OUT_CHANNELS = (320, 640, 1280, 1280)
UNET_LAYERS_PER_BLOCK = 2

# 噪声调度器配置
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.012

# 优化器配置
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

# 训练流程配置
NUM_EPOCHS = 100
MAX_GRAD_NORM = 1.0
DEVICE = "cuda"

# 检查点配置
OUTPUT_DIR = "./outputs"
CHECKPOINTING_STEPS = 5000

# 日志配置
LOGGING_STEPS = 10
VALIDATION_STEPS = 500


# =============================================================================
# 推理配置（硬编码在代码中）
# =============================================================================

# 采样器配置
SAMPLER_TYPE = "DDIM"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
DDIM_ETA = 0.0

# 输出配置
OUTPUT_FORMAT_HEIGHT = "exr"
OUTPUT_FORMAT_TEXTURE = "png"

# 设备配置
USE_HALF_PRECISION = True


def build_models():
    """构建所有模型"""
    # 构建文本编码器
    text_encoder = build_text_encoder(CLIP_MODEL_NAME)

    # 构建高程 VAE
    height_vae = HeightMapVAE()

    # 构建纹理 VAE（使用 SD 预训练 VAE）
    # TODO: 加载预训练 SD VAE
    # texture_vae = AutoencoderKL.from_pretrained(TEXTURE_VAE_PRETRAINED, subfolder="vae")
    texture_vae = None  # 占位

    # 构建 U-Net
    unet = build_unet(UNET_IN_CHANNELS, UNET_OUT_CHANNELS)

    return text_encoder, height_vae, texture_vae, unet


def run_training():
    """运行训练流程"""
    print("=== 训练模式 ===")

    # 构建模型
    text_encoder, height_vae, texture_vae, unet = build_models()

    # 构建优化器
    # TODO: 只优化 U-Net 参数
    # params_to_optimize = unet.parameters()
    # optimizer = torch.optim.AdamW(
    #     params_to_optimize,
    #     lr=LEARNING_RATE,
    #     betas=(ADAM_BETA1, ADAM_BETA2),
    #     weight_decay=WEIGHT_DECAY,
    # )
    optimizer = None  # 占位

    # 构建数据加载器
    # TODO: 实现数据集和数据加载器
    # train_dataset = TerrainDataset(DATA_ROOT, IMAGE_SIZE)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS,
    # )
    train_dataloader = None  # 占位

    # 创建训练流水线
    pipeline = TrainingPipeline(
        unet=unet,
        text_encoder=text_encoder,
        height_vae=height_vae,
        texture_vae=texture_vae,
        optimizer=optimizer,
        device=DEVICE,
    )

    # 开始训练循环
    print(f"训练配置：{NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("训练框架已搭建，具体实现待完成")

    # TODO: 完整训练循环
    # for epoch in range(NUM_EPOCHS):
    #     loss_dict = pipeline.train_epoch(train_dataloader, epoch)
    #     if epoch % LOGGING_STEPS == 0:
    #         print(f"Epoch {epoch}: loss={loss_dict['avg_loss']}")


def run_inference():
    """运行推理流程"""
    print("=== 推理模式 ===")

    # 构建模型
    text_encoder, height_vae, texture_vae, unet = build_models()

    # TODO: 加载预训练权重
    # unet.load_state_dict(torch.load("path/to/unet.pth"))
    # text_encoder.load_state_dict(torch.load("path/to/text_encoder.pth"))
    # height_vae.load_state_dict(torch.load("path/to/height_vae.pth"))

    # 创建推理流水线
    pipeline = InferencePipeline(
        unet=unet,
        text_encoder=text_encoder,
        height_vae=height_vae,
        texture_vae=texture_vae,
        device=DEVICE,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    # 推理配置
    print(f"推理配置：steps={NUM_INFERENCE_STEPS}, guidance={GUIDANCE_SCALE}")
    print("推理框架已搭建，具体实现待完成")

    # TODO: 执行生成
    # prompt = "广东丹霞地貌，红色平顶方山"
    # outputs = pipeline.generate(prompt)
    # height_map = outputs["height_map"]
    # texture_map = outputs["texture_map"]


def main():
    """主函数"""

    if torch.cuda.is_available():
        print("当前设备是 GPU")
        print("GPU 型号：", torch.cuda.get_device_name(0))  # 输出显卡名称
    else:
        print("当前设备是 CPU，无可用 GPU")

    import argparse

    parser = argparse.ArgumentParser(description="LLM 风格地形生成联合模型")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="运行模式",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "inference":
        run_inference()
    else:
        raise ValueError(f"未知模式：{args.mode}")


if __name__ == "__main__":
    main()
