import torch
from torch import nn

from ..rope import RotaryEmbedding, apply_rope
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
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

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
        dec_self_attn_mask: torch.Tensor | None = None,
        dec_cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        src_tokens: [batch_size, src_seq_len] 
        dst_tokens: [batch_size, dst_seq_len] 
        src_mask: [batch_size, src_seq_len] 编码器多头自注意力的mask
        self_attn_mask: [batch_size, dst_seq_len, dst_seq_len] 解码器多头自注意力mask
        cross_attn_mask: [batch_size, dst_seq_len, src_seq_len] 交叉注意力的mask
        """
        src_pos_ids = (
            torch.arange(src_tokens.size(1), device=src_tokens.device)
            .unsqueeze(0)
            .expand(src_tokens.size(0), -1)
        )
        src_embeddings = self.src_embedding(src_tokens)
        src_cos, src_sin = self.rope(src_embeddings, position_ids=src_pos_ids)
        src_states = apply_rope(src_embeddings, (src_cos, src_sin))
        src_states = self.dropout(src_states)
        enc_output = self.encoder(src_states, enc_mask)

        dst_pos_ids = (
            torch.arange(src_tokens.size(1), device=dst_tokens.device)
            .unsqueeze(0)
            .expand(src_tokens.size(0), -1)
        )
        dst_embeddings = self.dst_embedding(dst_tokens)
        dst_cos, dst_sin = self.rope(dst_embeddings, position_ids=dst_pos_ids)
        dst_states = apply_rope(dst_embeddings, (dst_cos, dst_sin))
        dst_states = self.dropout(dst_states)
        dec_output = self.decoder(
            dst_states,
            enc_output,
            self_attn_mask=dec_self_attn_mask,
            cross_attn_mask=dec_cross_attn_mask,
        )

        output = self.fc(dec_output)
        return output

