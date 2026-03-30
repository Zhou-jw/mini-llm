import torch
from torch import nn
from transformers.cache_utils import Cache

from ..rope import RotaryEmbedding, apply_rope
from .utils import repeat_kv


class StandardAttention(nn.Module):
    """
    Args:
        layer_idx (int): 层索引
        hidden_size (int): 隐藏层维度
        num_attention_heads (int): 注意力头数
        head_dim (int): 每个头的维度
        attention_bias (bool): 是否使用注意力偏置
        # use_cache (bool): 是否使用KV Cache
        max_position_embeddings (int): 最大位置编码长度
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        num_key_value_heads: int | None,
        attention_bias: bool = False,
        max_position_embeddings: int = 2048,
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
        self.attention_bias = attention_bias

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

        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        # 注意力缩放因子
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
        q_position_ids: torch.Tensor | None,
        kv_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        兼容 transformers 的 attention 前向传播

        Args:
            q (torch.Tensor): 查询张量, 形状 (batch_size, q_len, dim)
            k (torch.Tensor): 键张量, 形状 (batch_size, k_len, dim)
            v (torch.Tensor): 值张量, 形状 (batch_size, k_len, dim)
            position_embeddings (Optional[tuple[Tensor, Tensor]]): 预计算 (cos, sin) 表, 形状 (batch_size, seq_len, head_dim)
            attention_mask (Optional[torch.Tensor]): 通常为 (batch, 1, q_len, kv_len) 的加性掩码
            past_key_values (Optional[Cache]): transformers 缓存对象, 此处仅做占位兼容
            cache_position (Optional[LongTensor]): 当前位置索引 (q_len,) 或 (batch, q_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出张量 (attn_output, attn_weights)
        """
        batch_size, q_len, _ = q.shape  # (batch_size, seq_len)
        _, k_len, _ = k.shape
        # (batch_size, q_len, num_attention_heads, head_dim)
        q_shape = (
            batch_size,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        )
        k_shape = (
            batch_size,
            k_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # 1. build query, key, value
        # (batch_size, num_attention_heads, seq_len, head_dim)
        query_states = self.q_proj(q).view(q_shape).transpose(1, 2)
        # (batch_size, num_kv_heads, seq_len, head_dim)
        key_states = self.k_proj(k).view(k_shape).transpose(1, 2)
        value_states = self.v_proj(v).view(k_shape).transpose(1, 2)

        if q_position_ids is None:
            q_position_ids = (
            torch.arange(q_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
            )
        cos, sin = self.rope(query_states, position_ids=q_position_ids)
        query_states = apply_rope(query_states, (cos, sin))

        if kv_position_ids is None:
            kv_position_ids = q_position_ids
        cos, sin = self.rope(key_states, position_ids=kv_position_ids)
        key_states = apply_rope(key_states, (cos, sin))
        # 3. update cache
        if past_key_values is not None:
            # 缓存 kwargs（传递给 CacheLayerMixin 的 update 方法）
            cache_kwargs = {"cache_position": cache_position}

            # 唯一正确的缓存更新方式：调用 Cache.update
            # 该方法内部会自动拼接历史 K/V + 当前 K/V，并返回拼接后的完整 K/V
            key_states, value_states = past_key_values.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=self.layer_idx,
                cache_kwargs=cache_kwargs,
            )

        # 4. compute attention
        key_states = repeat_kv(
            key_states, self.num_key_value_groups
        )  # (batch_size, num_attention_heads, k_len, head_dim)
        value_states = repeat_kv(
            value_states, self.num_key_value_groups
        )  # (batch_size, num_attention_heads, k_len, head_dim)

        atten_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        )  # (batch_size, num_attention_heads, seq_len, k_len)

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(
                    1
                )  # (batch_size, 1, seq_len, k_len)
            atten_weights = atten_weights + attention_mask

        atten_weights = torch.softmax(
            atten_weights, dim=-1, dtype=torch.float32
        )  # (batch_size, num_attention_heads, seq_len, k_len)

        output = torch.matmul(
            atten_weights, value_states
        )  # (batch_size, num_attention_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(
            batch_size, q_len, -1
        ).contiguous()  # (batch_size, seq_len, num_heads * head_dim)
        output = self.o_proj(output)  # (batch_size, seq_len, hidden_size)

        return (output, atten_weights)
