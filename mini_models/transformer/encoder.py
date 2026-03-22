import torch
from torch import nn
from typing import Any, Tuple, List
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
        max_position_embeddings: int = 2048,
        num_key_value_heads: int | None = None,
        attention_bias: bool = False,
        rope_theta: float = 10000.0,
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
            max_position_embeddings=max_position_embeddings,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rope_theta=rope_theta,
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position=None,
    ) -> torch.Tensor:
        # Pre-LayerNorm: 先 norm 再 attention，最后加残差
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(
            x_norm,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
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
        max_position_embeddings: int = 2048,
        num_key_value_heads: int | None = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
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
                    max_position_embeddings,
                    num_key_value_heads,
                    False,
                    rope_theta,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None= None,
        cache_position: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[Any] | None]:
        """
        优化前向逻辑：
        1. 支持KV缓存（推理时复用past_key_values）
        2. 返回更新后的KV缓存，兼容生成式推理
        3. 增加最终归一化，提升训练稳定性
        """
        current_past_key_values = []
        for idx, layer in enumerate(self.layers):
            # 按层获取对应的KV缓存
            layer_past = past_key_values[idx] if past_key_values is not None else None
            x, layer_past = layer(
                x,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=layer_past,
                cache_position=cache_position,
            )
            current_past_key_values.append(layer_past)
        
        # 最终归一化（小模型关键，防止梯度爆炸）
        x = self.final_norm(x)
        
        # 返回输出和更新后的KV缓存
        return (x, current_past_key_values if past_key_values is not None else None)

    def count_parameters(self) -> float:
        """辅助函数：计算Encoder参数量（单位：MB），验证是否在目标区间"""
        total_params = sum(p.numel() for p in self.parameters())
        # 每个参数是4字节（float32），转换为MB
        total_mb = total_params * 4 / (1024 * 1024)
        return total_mb
