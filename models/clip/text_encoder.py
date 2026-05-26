"""
CLIP 双分支文本编码器

将文本 Prompt 编码为两组特征：
    1. 全局特征向量（pooled 输出）—— 捕捉整体语义
    2. 序列级特征（hidden states）—— 捕捉局部细节，供交叉注意力使用

基于 HuggingFace transformers 的 CLIPTextModel 实现。
"""

import torch
import torch.nn as nn
from typing import Tuple

from transformers import CLIPTextModel, CLIPTokenizer


class DualBranchCLIPEncoder(nn.Module):
    """
    CLIP 双分支文本编码器。

    全局分支：返回 CLIP 的 pooler_output [B, hidden_size]，
    序列分支：返回 last_hidden_state [B, seq_len, hidden_size]。
    两者均可经可选投影层映射到自定义维度，默认使用 CLIP 原生维度。

    参数
    ----------
    model_name : str
        HuggingFace CLIP 模型名称（默认 ``openai/clip-vit-large-patch14``，hidden_size=768）。
    global_dim : int or None
        全局特征输出维度，None 则使用 CLIP 原生维度。
    local_dim : int or None
        序列特征输出维度，None 则使用 CLIP 原生维度。
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        global_dim: int | None = None,
        local_dim: int | None = None,
    ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        hidden_size = self.text_encoder.config.hidden_size
        self.hidden_size = hidden_size
        self._global_dim = global_dim or hidden_size
        self._local_dim = local_dim or hidden_size

        self.global_proj = (
            nn.Linear(hidden_size, self._global_dim)
            if self._global_dim != hidden_size
            else nn.Identity()
        )
        self.local_proj = (
            nn.Linear(hidden_size, self._local_dim)
            if self._local_dim != hidden_size
            else nn.Identity()
        )

    @property
    def cross_attention_dim(self) -> int:
        """交叉注意力所需的文本特征维度。"""
        return self._local_dim

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def freeze(self) -> None:
        """冻结 CLIP 参数，仅保留投影层可训练（如有）。"""
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

    def _tokenize(self, prompts: list[str]) -> dict:
        """分词并移到模型所在设备。"""
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(
        self, prompts: list[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本 Prompt。

        参数
        ----------
        prompts : list[str]
            文本字符串列表。

        返回
        -------
        global_features : Tensor
            全局特征 [B, global_dim]。
        local_features : Tensor
            序列特征 [B, seq_len, local_dim]。
        """
        tokenized = self._tokenize(prompts)
        outputs = self.text_encoder(**tokenized)

        global_features = self.global_proj(outputs.pooler_output)
        local_features = self.local_proj(outputs.last_hidden_state)

        return global_features, local_features


def build_text_encoder(
    model_name: str = "openai/clip-vit-large-patch14",
    global_dim: int | None = None,
    local_dim: int | None = None,
) -> DualBranchCLIPEncoder:
    """
    工厂函数：构建并冻结 CLIP 双分支编码器。

    参数
    ----------
    model_name : str
        HuggingFace CLIP 模型名称。
    global_dim : int or None
        全局特征维度。
    local_dim : int or None
        序列特征维度。

    返回
    -------
    DualBranchCLIPEncoder
        已冻结的编码器实例。
    """
    encoder = DualBranchCLIPEncoder(
        model_name=model_name,
        global_dim=global_dim,
        local_dim=local_dim,
    )
    encoder.freeze()
    return encoder
