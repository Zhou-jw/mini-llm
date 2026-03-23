"""
验证 RotaryEmbedding.forward 和 precompute_freqs_cos_sin 的一致性
"""

import torch
from mini_models.rope import RotaryEmbedding, precompute_freqs_cos_sin


def test_consistency():
    """测试两种计算方法是否一致"""

    # 参数
    head_dim = 64
    seq_len = 16
    theta = 10000.0
    batch_size = 2

    print("=" * 60)
    print("验证 RotaryEmbedding.forward 和 precompute_freqs_cos_sin 的一致性")
    print("=" * 60)

    # 方法1: 使用 precompute_freqs_cos_sin（静态预计算）
    cos_precompute, sin_precompute = precompute_freqs_cos_sin(head_dim, seq_len, theta)
    print("\n方法1 - precompute_freqs_cos_sin:")
    print(f"  cos 形状: {cos_precompute.shape}")
    print(f"  sin 形状: {sin_precompute.shape}")
    print(f"  cos[0:3, 0:4]:\n{cos_precompute[0:3, 0:4]}")

    # 方法2: 使用 RotaryEmbedding.forward（动态计算）
    rope = RotaryEmbedding(
        head_dim=head_dim, max_position_embeddings=512, rope_theta=theta
    )

    # 创建假输入（只是为了获取 device 和 dtype）
    x = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    cos_rope, sin_rope = rope(x, position_ids=position_ids)
    print("\n方法2 - RotaryEmbedding.forward:")
    print(f"  cos 形状: {cos_rope.shape}")
    print(f"  sin 形状: {sin_rope.shape}")
    print(f"  cos[0, 0:3, 0:4]:\n{cos_rope[0, 0:3, 0:4]}")

    # 验证一致性
    print("\n一致性验证:")

    # 对于 batch 中的每个样本，cos/sin 应该相同（因为 position_ids 是 [0,1,2,...,seq_len-1]）
    # 所以 cos_rope[0] 应该等于 cos_precompute
    cos_match = torch.allclose(cos_rope[0], cos_precompute, atol=1e-6)
    sin_match = torch.allclose(sin_rope[0], sin_precompute, atol=1e-6)

    print(f"  cos 匹配: {cos_match}")
    print(f"  sin 匹配: {sin_match}")

    if cos_match and sin_match:
        print("\n✓ 两种方法计算结果一致！")
    else:
        print("\n✗ 两种方法计算结果不一致！")
        print("\n差异分析:")
        print(f"  cos 最大差异: {(cos_rope[0] - cos_precompute).abs().max()}")
        print(f"  sin 最大差异: {(sin_rope[0] - sin_precompute).abs().max()}")

    # 展示计算过程对比
    print("\n" + "=" * 60)
    print("计算过程对比")
    print("=" * 60)

    print("\n方法1 - precompute_freqs_cos_sin:")
    print("  1. inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))")
    print("  2. pos = arange(seq_len)")
    print("  3. freqs = outer(pos, inv_freq)  # (seq_len, head_dim//2)")
    print("  4. cos = cos(freqs)")
    print("  5. sin = sin(freqs)")

    print("\n方法2 - RotaryEmbedding.forward:")
    print("  1. inv_freq (缓存在 self.inv_freq)")
    print("  2. inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1)")
    print("  3. position_ids_expanded = position_ids[:, None, :]")
    print("  4. freqs = inv_freq_expanded @ position_ids_expanded")
    print("  5. freqs = freqs.transpose(1, 2)  # (batch, seq_len, head_dim//2)")
    print("  6. cos = cos(freqs)")
    print("  7. sin = sin(freqs)")

    print("\n数学本质:")
    print("  freqs[i, j] = position_ids[i, k] * inv_freq[j]")
    print("  对于 position_ids = [0, 1, 2, ..., seq_len-1]:")
    print("  freqs[k, j] = k * inv_freq[j]  # 与方法1的 outer(pos, inv_freq) 相同")

    print("\n" + "=" * 60)
    print("为什么需要 RotaryEmbedding.forward？")
    print("=" * 60)
    print("""
1. 支持非连续位置:
   - precompute_freqs_cos_sin 只能计算 [0, 1, 2, ..., seq_len-1]
   - RotaryEmbedding.forward 可以处理任意 position_ids
   
2. 支持 KV Cache:
   - 推理时，新 token 的位置可能不是从 0 开始
   - 例如: cache_position = [5, 6, 7, 8] 表示这是第 6-9 个 token
   
3. 动态序列长度:
   - 无需预先知道最大序列长度
   - 可以处理训练时未见过的序列长度
   
4. 设备和类型管理:
   - 自动处理 device 和 dtype
   - 支持混合精度训练
""")


if __name__ == "__main__":
    test_consistency()
