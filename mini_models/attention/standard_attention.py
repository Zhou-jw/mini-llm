from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache

from ..rope import apply_rope
from .utils import repeat_kv


class StandardAttention(nn.Module):
    """
    Args:
        layer_idx (int): 层索引
        hidden_size (int): 隐藏层维度
        num_attention_heads (int): 注意力头数
        head_dim (int): 每个头的维度
        rms_norm_eps (float): RMSNorm正则化系数
        attention_bias (bool): 是否使用注意力偏置
        # use_cache (bool): 是否使用KV Cache
        max_position_embeddings (int): 最大位置编码长度
        rope_theta (float): ROPE的底数
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        max_position_embeddings: int,
        attention_bias: bool = False,
        num_key_value_heads: Optional[int] = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # 线性变换层
        # x = (b, s, h)
        self.q_proj = nn.Linear(
            hidden_size, self.num_attention_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, hidden_size, bias=attention_bias
        )

        # 注意力缩放因子
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        兼容 transformers 的 attention 前向传播

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, dim)
            position_embeddings (Optional[tuple[Tensor, Tensor]]): 预计算 (cos, sin) 表, 形状 (batch_size, seq_len, head_dim)
            attention_mask (Optional[torch.Tensor]): 通常为 (batch, 1, q_len, kv_len) 的加性掩码
            past_key_values (Optional[Cache]): transformers 缓存对象, 此处仅做占位兼容
            cache_position (Optional[LongTensor]): 当前位置索引 (q_len,) 或 (batch, q_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出张量 (attn_output, attn_weights)
        """
        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_len)
        hidden_shape = (
            *input_shape,
            self.num_attention_heads,
            self.head_dim,
        )  # (batch_size, seq_len, num_attention_heads, head_dim)

        # 1. build query, key, value
        query_states = (
            self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )  # (batch_size, num_attention_heads, seq_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 2. apply rope embeddings
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")
            
        cos, sin = position_embeddings
        query_states = apply_rope(query_states, (cos, sin))
        key_states = apply_rope(key_states, (cos, sin))
            
        # 3. update cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # 4. compute attention
        key_states = repeat_kv(key_states, self.num_key_value_groups) # (batch_size, num_attention_heads, k_len, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        atten_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        atten_weights = torch.softmax(atten_weights, dim=-1, dtype=torch.float32)
        
        output = torch.matmul(atten_weights, value_states) # (batch_size, num_attention_heads, seq_len, head_dim)
        
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(*input_shape, -1).contiguous() # (batch_size, seq_len, num_attention_heads * head_dim)
        output = self.o_proj(output) # (batch_size, seq_len, hidden_size)
        
        return output, atten_weights
        
