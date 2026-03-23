"""
Encoder 测试脚本

验证 Encoder 的 RoPE 集成功能
"""

import torch
from mini_models.transformer.encoder import Encoder

def test_encoder():
    """测试 Encoder 的基本功能"""
    print("=" * 50)
    print("测试 Encoder 的 RoPE 集成功能")
    print("=" * 50)

    # 1. 初始化 Encoder
    encoder = Encoder(
        num_layers=2,
        d_model=256,
        num_heads=4,
        head_dim=64,
        d_ff=512,
        max_position_embeddings=512,
    )
    print(f"Encoder 参数量：{encoder.count_parameters():.2f} MB")

    # 2. 测试基本前向传播
    batch_size, seq_len, d_model = 2, 16, 256
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n输入形状：{x.shape}")

    # 自动生成 position_ids
    output, _ = encoder(x)
    print(f"输出形状（自动生成 position_ids）：{output.shape}")

    # 手动提供 position_ids
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    output, _ = encoder(x, position_ids=position_ids)
    print(f"输出形状（手动提供 position_ids）：{output.shape}")

    # 3. 测试不同的序列长度
    print("\n测试不同的序列长度：")
    for seq_len in [8, 16, 32, 64]:
        x = torch.randn(1, seq_len, d_model)
        output, _ = encoder(x)
        print(f"  seq_len={seq_len}: 输出形状 {output.shape}")

    # 4. 测试 cache_position
    print("\n测试 cache_position：")
    cache_position = torch.arange(8)
    x = torch.randn(1, 8, d_model)
    output, _ = encoder(x, cache_position=cache_position)
    print(f"  cache_position={cache_position.tolist()}: 输出形状 {output.shape}")

    print("\n" + "=" * 50)
    print("所有测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    test_encoder()
