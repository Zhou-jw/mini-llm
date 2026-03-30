import torch
from demo.dataloader import get_dataloader
from mini_models.transformer.transformer import Transformer
from mini_models.transformer.mask import new_self_attn_mask, new_decoder_self_attn_mask

# ====================== 超参数 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "transformer_from_scratch.pt"
MAX_EN_LEN = 45
MAX_CN_LEN = 50
# ====================================================

# 加载词典
train_loader, val_loader, en_vocab, cn_vocab, en_id2token, cn_id2token = get_dataloader()
EN_VOCAB_SIZE = len(en_vocab)
CN_VOCAB_SIZE = len(cn_vocab)

# 模型
model = Transformer(
    src_vocab_size=EN_VOCAB_SIZE,
    dst_vocab_size=CN_VOCAB_SIZE,
    d_model=256,
    num_layers=3,
    num_heads=8,
    d_ff=1024,
    dropout=0.1,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


# ====================== 工具函数 ======================
def tokenize_en(text: str):
    from demo.dataloader import en_tokenizer
    tokens = en_tokenizer(text)
    ids = [en_vocab[token] for token in tokens]
    ids = [0] + ids + [1]
    ids = ids[:MAX_EN_LEN]
    ids += [2] * (MAX_EN_LEN - len(ids))
    return torch.tensor([ids]).long().to(DEVICE)


def ids2cn(ids: list[int]):
    tokens = []
    for i in ids:
        if i == 1:
            break
        if i not in [0, 1, 2]:
            tokens.append(cn_id2token[i])
    return "".join(tokens)


# ====================== 正确推理（含MASK） ======================
@torch.no_grad()
def translate_en2cn(text_en: str):
    src = tokenize_en(text_en)

    # 初始化 decoder 输入
    tgt = torch.zeros((1, MAX_CN_LEN), dtype=torch.long).to(DEVICE)
    tgt[0, 0] = 0  # <sos>

    for step in range(1, MAX_CN_LEN):
        # 输入只取前面的部分
        tgt_input = tgt[:, :step]

        # ======================
        # ✅ 正确构造 mask（全部放 cuda！）
        # ======================
        src_mask = new_self_attn_mask(seq=src, pad_token_id=2).to(DEVICE)
        tgt_mask = new_decoder_self_attn_mask(seq=tgt_input, pad_token_id=2).to(DEVICE)

        # 前向
        output = model(
            src_tokens=src,
            dst_tokens=tgt_input,
            enc_mask=src_mask,
            dec_self_attn_mask=tgt_mask,
            dec_cross_attn_mask=src_mask,
        )

        # 取最后一个词
        last_token = output[:, -1, :].argmax(-1)
        tgt[0, step] = last_token

        if last_token.item() == 1:
            break

    return ids2cn(tgt[0].tolist())


# ====================== 测试 ======================
if __name__ == "__main__":
    print("✅ 模型加载完成！输入英文，输出中文翻译")
    print("输入 'quit' 退出\n")

    while True:
        user_input = input("输入英文：")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        cn_trans = translate_en2cn(user_input)
        print(f"翻译结果：{cn_trans}\n")