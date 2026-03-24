#         self.rope = RotaryEmbedding(
#     head_dim=head_dim,
#     max_position_embeddings=max_position_embeddings,
#     rope_theta=rope_theta,
# )
#         # 自动生成position_ids（如果未提供）
# if position_ids is None:
#     if cache_position is not None:
#         position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
#     else:
#         position_ids = (
#             torch.arange(seq_len, device=x.device)
#             .unsqueeze(0)
#             .expand(batch_size, -1)
#         )

# 使用RotaryEmbedding生成position_embeddings
# position_embeddings = self.rope(x, position_ids=position_ids)


# # 2. apply rope embeddings
#         if position_embeddings is None:
#             raise ValueError("position_embeddings must be provided")
            
#         cos, sin = position_embeddings
#         query_states = apply_rope(query_states, (cos, sin))
#         key_states = apply_rope(key_states, (cos, sin))
         