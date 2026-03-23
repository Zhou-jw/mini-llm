"""
位置编码模块

提供 RoPE (Rotary Position Embedding) 的便捷接口
"""

from ..rope import RotaryEmbedding, apply_rope, precompute_freqs_cos_sin

__all__ = ["RotaryEmbedding", "apply_rope", "precompute_freqs_cos_sin"]
