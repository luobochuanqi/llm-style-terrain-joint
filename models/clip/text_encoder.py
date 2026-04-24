"""
双分支 CLIP 文本编码器

根据 roadmap 描述：
- Prompt 进入双分支 CLIP
- 输出：全局特征向量 和 细节特征向量
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class DualBranchCLIPEncoder(nn.Module):
    """
    双分支 CLIP 文本编码器

    将文本 Prompt 编码为两种特征：
    1. 全局特征向量：捕捉整体语义（如"广东丹霞地貌"）
    2. 细节特征向量：捕捉局部细节（如"红色平顶方山"）
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        global_dim: int = 512,
        local_dim: int = 512,
    ):
        """
        初始化双分支 CLIP 编码器

        Args:
            model_name: CLIP 模型名称
            global_dim: 全局特征维度
            local_dim: 细节特征维度
        """
        super().__init__()

        # TODO: 加载预训练 CLIP 模型
        # self.clip_model = CLIPEncoder.from_pretrained(model_name)

        # 全局特征投影层
        # 将 CLIP 的 [CLS] token 投影到全局特征空间
        self.global_proj = nn.Linear(512, global_dim)

        # 细节特征投影层
        # 将所有 token 序列投影到细节特征空间
        self.local_proj = nn.Linear(512, local_dim)

    def forward(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            prompts: Prompt 字符串列表
        Returns:
            global_features: 全局特征向量 [B, global_dim]
            local_features: 细节特征向量 [B, N, local_dim]
        """
        # TODO: 实现 CLIP 文本编码

        # 伪代码示意：
        # 1. 使用 CLIP tokenizer 对 prompts 进行分词
        # text_inputs = self.tokenizer(
        #     prompts,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=77,
        #     return_tensors="pt",
        # )
        #
        # 2. CLIP 文本编码器前向传播
        # outputs = self.clip_model(**text_inputs)
        #
        # 3. 提取全局特征（[CLS] token 或 pooled output）
        # global_features = outputs.pooler_output  # [B, 512]
        # global_features = self.global_proj(global_features)  # [B, global_dim]
        #
        # 4. 提取细节特征（所有 token 的 hidden states）
        # local_features = outputs.last_hidden_state  # [B, N, 512]
        # local_features = self.local_proj(local_features)  # [B, N, local_dim]

        raise NotImplementedError("CLIP 文本编码功能待实现")


def build_text_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
) -> DualBranchCLIPEncoder:
    """
    构建文本编码器工厂函数

    Args:
        model_name: CLIP 模型名称
    Returns:
        text_encoder: 双分支 CLIP 编码器
    """
    return DualBranchCLIPEncoder(model_name=model_name)
