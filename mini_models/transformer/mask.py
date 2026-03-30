import torch


def new_padding_mask(
    seq: torch.Tensor, pad_token_id: int, is_bool_mask: bool = False
) -> torch.Tensor:
    """
    创建填充掩码，用于屏蔽序列中的填充部分。
        Args:
        seq: 输入序列，形状为 [batch_size, seq_len]
        pad_token_id: 用于填充的特殊字符<PAD>所对应的token_id
        return_bool: 是否返回布尔类型掩码，默认为False，返回int类型掩码

    Returns:
        padding_mask: [batch_size, seq_len]
        0 = keep, -inf = mask
    """
    mask = seq != pad_token_id
    if is_bool_mask:
        return mask
    padding_mask = torch.where(mask, 0.0, float("-inf"))
    return padding_mask


def new_sequence_mask(seq: torch.Tensor, is_bool_mask: bool = False) -> torch.Tensor:
    """
    创建 Sequence 掩码（下三角掩码，屏蔽未来位置，用于 Decoder 自注意力）
    作用：防止模型看到未来的词
    Args:
        seq: 输入序列，形状为 [batch_size, seq_len]
        is_bool_mask: 是否返回布尔类型掩码，默认为False，返回int类型掩码

    Returns:
        序列掩码，形状为 [batch_size, seq_len]
        0 = keep, -inf = mask
    """
    batch_size, seq_len = seq.shape
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=seq.device)
    )  # (seq_len, seq_len), 下三角为1, 上三角为0
    mask = mask.unsqueeze(0).expand(
        batch_size, -1, -1
    )  # (batch_size, seq_len, seq_len)

    if is_bool_mask:
        return mask.bool()
    return torch.where(mask == 1, 0.0, float("-inf"))


def new_self_attn_mask(
    seq: torch.Tensor, pad_token_id: int, is_bool_mask: bool = False
) -> torch.Tensor:
    """
    Args:
        seq: 输入序列，形状为 [batch_size, seq_len]
        pad_token_id: 填充标记的ID
        is_bool_mask: 是否返回布尔类型掩码，默认为False，返回int类型掩码

    Returns:
        编码器掩码，形状为 [batch_size, 1, 1, seq_len]
        后续通过广播成 [batch_size, num_heads, seq_len, seq_len]
        0 = keep, -inf = mask
    """
    batch_size, seq_len = seq.shape
    padding_mask = new_padding_mask(
        seq, pad_token_id=pad_token_id
    )  # (batch_size, seq_len)
    enc_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
    return enc_mask


def new_decoder_self_attn_mask(
    seq: torch.Tensor, pad_token_id: int, is_bool_mask: bool = False
) -> torch.Tensor:
    """
    padding mask + sequence mask
    Args:
        seq: 输入序列，形状为 [batch_size, seq_len]
        pad_token_id: 填充标记的ID
        is_bool_mask: 是否返回布尔类型掩码，默认为False，返回int类型掩码

    Returns:
        解码器自注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
        0 = keep, -inf = mask
    """
    # padding mask
    batch_size, seq_len = seq.shape
    padding_mask = new_padding_mask(
        seq, pad_token_id=pad_token_id
    )  # (batch_size, seq_len)
    dec_mask = padding_mask.unsqueeze(1).expand(
        batch_size, seq_len, seq_len
    )  # (batch_size, seq_len, seq_len)
    seq_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=seq.device)
    )  # (seq_len, seq_len), 下三角为1, 上三角为0
    seq_mask = torch.where(seq_mask == 1, 0.0, float("-inf"))
    dec_mask = dec_mask + seq_mask
    if is_bool_mask:
        return dec_mask.bool()
    return dec_mask


if __name__ == "__main__":
    print(" ==== padding mask, sequence mask test ====")
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    pad_mask = new_padding_mask(seq, pad_token_id=0)
    seq_mask = new_sequence_mask(seq)
    print("pad_mask:\n", pad_mask)
    print("seq_mask:\n", seq_mask)

    print("\n==== encoder mask test ====")
    encoder_token = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
    )
    enc_mask = new_self_attn_mask(encoder_token, pad_token_id=0)
    print("enc_mask:\n", enc_mask)
    print(enc_mask.shape)

    print("\n==== decoder mask test ====")
    decoder_token = torch.tensor(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
            ],  # mask matrix last 4 values of each row should be 0
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ]  # mask matrix last 5 values of each row should be 0
    )
    dec_mask = new_decoder_self_attn_mask(decoder_token, pad_token_id=0)
    print("dec_mask:\n", dec_mask)
    print(dec_mask.shape)
