import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        dst_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Transformer model with rotary positional embeddings.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            dst_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimension of the model.
            num_layer (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.dst_embedding = nn.Embedding(dst_vocab_size, d_model)


        self.encoder = Encoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, dst_vocab_size)

    def forward(
        self,
        src_tokens: torch.Tensor,
        dst_tokens: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
        dec_cross_attn_mask: torch.Tensor | None = None,
        dec_self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """    
        Input:
            src_tokens: [batch_size, src_seq_len] 源序列的token
            dst_tokens: [batch_size, dst_seq_len] 目标序列的token
            enc_mask: [batch_size, src_seq_len] 编码器多头自注意力的mask
            dec_self_attn_mask: [batch_size, dst_seq_len, dst_seq_len] 解码器多头自注意力mask
            dec_cross_attn_mask: [batch_size, dst_seq_len, src_seq_len] 交叉注意力的mask
            
        Output:
            dst_states: [batch_size, dst_seq_len, hidden_size] 目标序列的状态
        """

        src_embeddings = self.src_embedding(src_tokens)
        src_states = self.dropout(src_embeddings)
        enc_output, enc_position_ids = self.encoder(src_states, enc_mask)

        dst_embeddings = self.dst_embedding(dst_tokens)
        dst_states = self.dropout(dst_embeddings)
        dec_output = self.decoder(
            dst_states,
            enc_output,
            self_attn_mask=dec_self_attn_mask,
            cross_attn_mask=dec_cross_attn_mask,
            enc_position_ids=enc_position_ids,
        )

        output = self.fc(dec_output)
        return output

