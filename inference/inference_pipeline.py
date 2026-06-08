"""
推理流水线 — 已搁置

本模块为骨架占位。当前推理通过训练脚本的 --mode test 完成：
    python scripts/unet/unet_full.py --mode test --checkpoint <path>

待 U-Net/DiT 训练方案确定后，统一在此处总写完整的推理流水线。
"""


class InferencePipeline:
    """
    推理流水线骨架 (已搁置)。

    原始设计目标:
        输入: Prompt 字符串
        输出: (512x512 高度图, 512x512 纹理图)

    流程: 文本编码 → 初始化噪声 → DDIM 循环降噪 → VAE 解码
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "InferencePipeline 已搁置。\n"
            "请使用: python scripts/unet/unet_full.py --mode test --checkpoint <path>"
        )
