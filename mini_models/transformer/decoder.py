from typing import Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache

from ..attention import StandardAttention
from .ffn import FeedForward
from .rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
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
        """
        Decoder block:
          x ->  norm -> attention -> residule -> norm -> attention -> residule -> norm -> ffn -> residule

        """
        super().__init__()
        self.norm1 = RMSNorm(d_model, rms_norm_eps)
        self.attn1 = StandardAttention(
            layer_idx=layer_idx,
            hidden_size=d_model,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
        )
        self.norm2 = RMSNorm(d_model, rms_norm_eps)
        self.attn2 = StandardAttention(
            layer_idx=layer_idx,
            hidden_size=d_model,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
        )
        self.norm3 = RMSNorm(d_model, rms_norm_eps)
        self.ffn = FeedForward(
            hidden_size=d_model,
            intermediate_size=d_ff,
            dropout_rate=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cross_past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # 1. Self-attention block
        residual = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn1(
            q=x_norm,
            k=x_norm,
            v=x_norm,
            attention_mask=self_attn_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        x = residual + self.dropout1(x_attn)

        # 2. Cross-attention block
        residual = x
        x_norm = self.norm2(x)
        x_attn, _ = self.attn2(
            q=x_norm,
            k=enc_output,
            v=enc_output,
            attention_mask=cross_attn_mask,
            past_key_values=cross_past_key_values,
            cache_position=None,
        )
        x = residual + self.dropout2(x_attn)

        # 3. Feed-forward block
        residual = x
        x_norm = self.norm3(x)
        x_ffn = self.ffn(x_norm)
        x = residual + self.dropout3(x_ffn)
        return x


class Decoder(nn.Module):
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
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    layer_idx=i,
                    d_model=d_model,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    d_ff=d_ff,
                    dropout=dropout,
                    rms_norm_eps=rms_norm_eps,
                    num_key_value_heads=num_key_value_heads,
                    attention_bias=attention_bias,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cross_past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        for idx, layer in enumerate(self.layers):
            x = layer(
                x=x,
                enc_output=enc_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                past_key_values=past_key_values,
                cross_past_key_values=cross_past_key_values,
                cache_position=cache_position,
            )

        # 最终归一化（小模型关键，防止梯度爆炸）
        x = self.norm(x)

        # 返回输出和更新后的KV缓存
        return x
