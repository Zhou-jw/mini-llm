import torch
from demo.dataloader import get_dataloader

from mini_models.transformer.transformer import Transformer

if __name__ == "__main__":
    train_loader, val_loader, en_vocab, cn_vocab = get_dataloader()
    en_vocab_size = len(en_vocab)
    cn_vocab_size = len(cn_vocab)

    transformer = Transformer(
        src_vocab_size=en_vocab_size,  # 7192
        dst_vocab_size=cn_vocab_size,  # 2839
        d_model=256,
        num_layers=3,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
    )

    test_src = torch.randint(0, en_vocab_size, (32, 45))  # 在词典范围内生成随机数
    test_tgt = torch.randint(0, cn_vocab_size, (32, 50))
    print(test_src.shape, test_tgt.shape)
    
    test_out = transformer(test_src, test_tgt)
    print(test_out.shape)
    
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")