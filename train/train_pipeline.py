"""
8 通道去噪模型训练流水线 — 已搁置

当前 U-Net 训练使用 scripts/unet/unet_full.py（独立 UNetTrainer）。
待 U-Net/DiT 训练方案确定后，统一在此处总写训练流水线。
"""


class UNetTrainingPipeline:
    """
    训练流水线占位。

    该流水线已搁置。当前 U-Net 训练请使用 scripts/unet/unet_full.py。
    待训练方案确定后，统一在此处总写训练流水线。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "UNetTrainingPipeline 已搁置。\n"
            "U-Net 训练请使用: python scripts/unet/unet_full.py --mode train"
        )


def test_noise_prediction(*args, **kwargs):
    """
    噪声预测测试占位。

    该函数已搁置。测试请使用:
        python scripts/unet/unet_full.py --mode test --checkpoint <path>
    """
    raise NotImplementedError(
        "test_noise_prediction 已搁置。\n"
        "请使用: python scripts/unet/unet_full.py --mode test --checkpoint <path>"
    )
