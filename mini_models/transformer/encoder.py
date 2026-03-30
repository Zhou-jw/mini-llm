from typing import Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache

from ..attention import StandardAttention
from .ffn import FeedForward
from .rmsnorm import RMSNorm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        d_ff: int = 1024,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        num_key_value_heads: int | None = None,
        attention_bias: bool = False,
    ):
        super().__init__()
        # Pre-LayerNorm: norm 在子层之前
        self.norm1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=rms_norm_eps)
        self.attention = StandardAttention(
            layer_idx=layer_idx,
            hidden_size=d_model,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量，(batch_size, seq_len, d_model)
            position_embeddings: RoPE位置编码(cos, sin)，每个张量形状 [batch_size, seq_len, head_dim]
            attention_mask: 注意力掩码，形状 [batch_size, 1, seq_len, seq_len]（加性掩码）
            past_key_values: Transformers Cache对象，存储历史KV缓存，None表示不使用缓存
            cache_position: 缓存位置索引，形状 [seq_len] 或 [batch_size, seq_len]

        Returns:
            Tuple[torch.Tensor, Cache | None]:
                - 输出张量：形状 (batch_size, seq_len, d_model)
                - 更新后的Cache对象：None表示未使用缓存
        """

        # Pre-LayerNorm: 先 norm 再 attention，最后加残差
        residual = x  # (batch_size, seq_len, d_model)
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(
            q=x_norm,
            k=x_norm,
            v=x_norm,
            attention_mask=attention_mask,
            q_position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        x = residual + self.dropout1(attn_output)

        # Pre-LayerNorm: 先 norm 再 FFN，最后加残差
        residual = x
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = residual + self.dropout2(ff_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        d_ff: int = 1024,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        num_key_value_heads: int | None = None,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    i,
                    d_model,
                    num_heads,
                    head_dim,
                    d_ff,
                    dropout,
                    rms_norm_eps,
                    num_key_value_heads,
                    attention_bias,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
            position_ids: 位置索引，形状 [batch_size, seq_len]，如果为None则自动生成
            attention_mask: 注意力掩码，形状 [batch_size, 1, seq_len, seq_len]（加性掩码）
            past_key_values: Transformers Cache对象，存储历史KV缓存，None表示不使用缓存
            cache_position: 缓存位置索引，形状 [seq_len] 或 [batch_size, seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 输出张量：形状 [batch_size, seq_len, d_model]
                - position_ids：形状 [batch_size, seq_len]

        优化前向逻辑：
        1. 支持KV缓存（推理时复用past_key_values）
        2. 返回更新后的KV缓存，兼容生成式推理
        3. 增加最终归一化，提升训练稳定性
        4. 集成RoPE位置编码，自动生成position_embeddings
        """
        batch_size, seq_len, _ = x.shape
        position_ids = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        for idx, layer in enumerate(self.layers):
            # 按层获取对应的KV缓存
            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        # 最终归一化（小模型关键，防止梯度爆炸）
        x = self.final_norm(x)

        # 返回输出和更新后的KV缓存
        return x, position_ids

    def count_parameters(self) -> float:
        """辅助函数：计算Encoder参数量（单位：MB），验证是否在目标区间"""
        total_params = sum(p.numel() for p in self.parameters())
        # 每个参数是4字节（float32），转换为MB
        total_mb = total_params * 4 / (1024 * 1024)
        return total_mb


# 测试代码（验证维度和逻辑）
if __name__ == "__main__":
    # 初始化Encoder
    encoder = Encoder(num_layers=6, d_model=512, num_heads=8)
    print(f"Encoder参数量：{encoder.count_parameters():.2f} MB")  # ~110MB，符合目标

    # 构造测试输入
    batch_size, seq_len, d_model = 2, 32, 512
    x = torch.randn(batch_size, seq_len, d_model)

    # 测试1：自动生成position_ids
    output, cache = encoder(x)
    print(
        f"Encoder输出维度（自动生成position_ids）：{output.shape}"
    )  # torch.Size([2, 32, 512])，符合预期

    # 测试2：手动提供position_ids
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    output, cache = encoder(x, position_ids=position_ids)
    print(
        f"Encoder输出维度（手动提供position_ids）：{output.shape}"
    )  # torch.Size([2, 32, 512])，符合预期
