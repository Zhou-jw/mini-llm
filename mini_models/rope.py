from typing import Tuple

import torch
# from torch import nn


# ---------------------------- Position Embeddings ---------------------------- #

def apply_rope(
    x: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    impl: str = "real",
) -> torch.Tensor:
    """
    应用旋转位置编码, 可选复数实现方式或实数实现方式
    复数实现方式更加直观, 但复数数据类型可能兼容有限, 复数实现中使用的是相邻维度为一组
    实数实现方式更通用, 但不那么直观, 实数实现中使用的是拆分后, 对应的维度为一组

    Args:
        x (torch.Tensor): 输入张量 (batch, heads, seq_len, head_dim)
        position_embeddings (Tuple[torch.Tensor, torch.Tensor] | torch.Tensor): 预计算的余弦表和正弦表元组 (cos, sin), 每个表的形状为 (seq_len, dim), 需根据输入 x 的形状和位置切片好
        impl (str): 实现方式, 默认为 "real", 可选 "real" 和 "complex"

    Returns:
        torch.Tensor: 应用了 RoPE 的输出张量
    """
    if impl == "real":
        cos, sin = position_embeddings # (batch, heads_num, seq_len, head_dim // 2)
        return apply_rope_real(x, cos, sin)
    else:
        raise ValueError(f"Unknown impl: {impl}")

def apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    在实数域应用旋转位置编码, 基本原理如下:

    对于每一对维度 (a, b)，旋转角度 θ 后的新向量 (a', b') 是:

        ⎡ a'⎤ = ⎡ cos(θ)  -sin(θ)⎤ ⎡ a ⎤
        ⎣ b'⎦ = ⎣ sin(θ)  cos(θ) ⎦ ⎣ b ⎦

    展开得:

        a' = a * cos(θ) - b * sin(θ)
        b' = b * cos(θ) + a * sin(θ)

    Args:
        x (torch.Tensor): 输入张量 (batch, heads, seq_len, head_dim)
        cos (torch.Tensor): 预计算的余弦表, 形状为 (seq_len, head_dim // 2)
        sin (torch.Tensor): 预计算的正弦表, 形状为 (seq_len, head_dim // 2)

    Returns:
        torch.Tensor: 应用了RoPE的输出张量
    """
    batch, heads, seq_len, head_dim = x.shape
    dtype = x.dtype
    
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype) # (1, 1, seq_len, head_dim // 2)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype)
    
    # 1. group into pairs (even, odd)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    
    x_even_rot = x_even * cos - x_odd * sin # (batch, heads, seq_len, head_dim // 2)
    x_odd_rot = x_even * sin + x_odd * cos
    
    x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1).reshape(batch, heads, seq_len, head_dim)
    
    return x_rot
    

def precompute_freqs_cos_sin(head_dim: int, seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的余弦表和正弦表.
    PE(pos, 2i) = sin(pos / theta^(2i / head_dim))
    PE(pos, 2i+1) = cos(pos / theta^(2i / head_dim))
    
    Args:
        head_dim (int): 头维度
        seq_len (int): 序列长度
        theta (float): RoPE 的底, 默认为 10000.0

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 余弦表和正弦表, 形状均为 (seq_len, head_dim)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim)) # (head_dim // 2)
    pos = torch.arange(seq_len, dtype=torch.float32) # (seq_len)
    freqs = torch.outer(pos, freqs) # (seq_len, head_dim // 2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin
    
# class RotaryEmbedding(nn.Module):
#     """
#     旋转位置编码 (Rotary Position Embedding, RoPE)

#     Args:
#         max_position_embeddings (int): 最大位置编码长度
#         head_dim (int): 每个头的维度
#         rope_theta (float): RoPE 的底数, 默认为 10000.0
#     """

#     inv_freq: torch.Tensor  # 用于类型标注(type hint)

#     def __init__(self, max_position_embeddings: int, head_dim: int, rope_theta: float = 10000.0):
#         super().__init__()
#         self.max_seq_len_cached = max_position_embeddings
#         self.head_dim = head_dim
#         self.rope_theta = rope_theta

#         inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))  # (head_dim//2,)
#         # 仅缓存 inv_freq，而不是 cos、sin，能够节省缓存，且支持动态适应
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         执行前向传播会计算产生 cos、sin 表, 可以自适应序列长度, 可以处理训练时未见过的序列长度

#         Args:
#             x (torch.Tensor): 输入的 embeddings, 形状为 (batch, seq_len, hidden_size)
#             position_ids (torch.Tensor): 位置索引, 形状为 (batch, seq_len)

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: 输出 cos、sin 表, 形状为 (batch, seq_len, head_dim)
#         """
#         # 调整形状为后续外积计算做准备
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)  # (batch, head_dim//2, 1)
#         position_ids_expanded = position_ids[:, None, :].float()  # (batch, 1, seq_len)

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         # 关闭自动混合精度，强制使用 float32 计算，确保 cos、sin 表的精度
#         with torch.autocast(device_type=device_type, enabled=False):
#             # 批量矩阵乘法，freqs 是每个 token 在每个频率维度上的旋转角度
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # (batch, seq_len, head_dim//2)
#             emb = torch.cat((freqs, freqs), dim=-1)  # (batch, seq_len, head_dim) 详见 precompute_cos_sin_tables 函数中的注释
#             cos = emb.cos()  # (batch, seq_len, head_dim)
#             sin = emb.sin()  # (batch, seq_len, head_dim)

#         return cos.to(x.dtype), sin.to(x.dtype)
    
# ---------------------------- 测试示例 ---------------------------- #
if __name__ == "__main__":
    # 设定测试参数
    batch_size = 1
    num_heads = 1
    seq_len = 2
    head_dim = 4  # 必须是偶数
    
    # 1. 创建测试输入张量 (batch, heads, seq_len, head_dim)
    x = torch.tensor([[[
        [1.0, 2.0, 3.0, 4.0],  # 位置0的特征
        [5.0, 6.0, 7.0, 8.0]   # 位置1的特征
    ]]])
    print("原始输入 x:")
    print(x)
    print(f"x形状: {x.shape}\n")
    
    # 2. 预计算cos/sin表
    cos, sin = precompute_freqs_cos_sin(head_dim, seq_len)
    print("cos表 (seq_len, head_dim//2):")
    print(cos)
    print("sin表 (seq_len, head_dim//2):")
    print(sin)
    print(f"cos形状: {cos.shape}, sin形状: {sin.shape}\n")
    
    # 3. 应用RoPE
    x_rot = apply_rope(x, (cos, sin))
    print("应用RoPE后的输出 x_rot:")
    print(x_rot)
    print(f"x_rot形状: {x_rot.shape}\n")
    
    # 4. 手动验证位置0的计算（以θ=0为例，简化计算）
    print("=== 手动验证位置0的计算 ===")
    # 位置0的cos/sin值
    cos0 = cos[1]  # [1.0, 1.0] (因为freqs[0]=0, cos(0)=1)
    sin0 = sin[1]  # [0.0, 0.0] (sin(0)=0)
    print(f"位置0的cos值: {cos0}")
    print(f"位置0的sin值: {sin0}")
    
    # 位置0的原始特征: [1,2,3,4]
    x0 = x[0,0,1]  # [1,2,3,4]
    x1_0 = x0[::2]  # [1,3] (偶数维度)
    x2_0 = x0[1::2]  # [2,4] (奇数维度)
    print(f"x0 = {x0}")
    
    # 手动计算旋转后的值
    x1_rot_0 = x1_0 * cos0 - x2_0 * sin0  # [1*1 - 2*0, 3*1 - 4*0] = [1,3]
    x2_rot_0 = x2_0 * cos0 + x1_0 * sin0  # [2*1 + 1*0, 4*1 + 3*0] = [2,4]
    x_rot_0 = torch.stack([x1_rot_0, x2_rot_0], dim=-1).reshape(-1)
    
    print(f"位置0手动计算结果: {x_rot_0}")
    print(f"代码计算位置0结果: {x_rot[0,0,1]}")