import json
from collections import Counter

from tqdm import tqdm

from demo.utils import *


def en_tokenizer(text: str) -> list[str]:
    text = text.lower().strip()
    punctuations = r""".,!?;:'"()[]{}"""
    for p in punctuations:
        text = text.replace(p, f" {p} ")
    return [t for t in text.split() if t]


def cn_tokenizer(text: str) -> list[str]:
    return list(text)


if __name__ == "__main__":
    """
    build cn && en vocab
    """
    en_max_len = 0
    cn_max_len = 0
    en_vocab = []
    cn_vocab = []
    dataset = get_dataset()

    import os

    SAVE_DIR = os.path.join(os.path.dirname(__file__), "dataset")

    for data in tqdm(dataset, desc="Building Vocabulary"):
        en_text = data["english"]
        cn_text = data["chinese"]

        en_tokens = en_tokenizer(en_text)
        cn_tokens = cn_tokenizer(cn_text)
        en_max_len = max(en_max_len, len(en_tokens))
        cn_max_len = max(cn_max_len, len(cn_tokens))

        en_vocab.extend(en_tokens)
        cn_vocab.extend(cn_tokens)

    en_counter = dict(Counter(en_vocab))
    cn_counter = dict(Counter(cn_vocab))

    # 保存词频统计
    with open(f"{SAVE_DIR}/en_counter.json", "w", encoding="utf-8") as f:
        json.dump(en_counter, f, ensure_ascii=False, indent=4)
    with open(f"{SAVE_DIR}/cn_counter.json", "w", encoding="utf-8") as f:
        json.dump(cn_counter, f, ensure_ascii=False, indent=4)

    # 为简单起见，将数据集所有的token都添加到词典中，不考虑词频和未知token
    # 定义特殊字符
    start_token = "<sos>"
    end_token = "<eos>"
    pad_token = "<pad>"

    special_tokens = [start_token, end_token, pad_token]
    en_vocab = special_tokens + list(en_counter.keys())
    cn_vocab = special_tokens + list(cn_counter.keys())

    # 构建词典
    en_dict_token2id = {token: i for i, token in enumerate(en_vocab)}
    en_dict_id2token = {i: token for i, token in enumerate(en_vocab)}
    cn_dict_token2id = {token: i for i, token in enumerate(cn_vocab)}
    cn_dict_id2token = {i: token for i, token in enumerate(cn_vocab)}

    # 分别保存token到token_id和token_id到token的词典
    with open(f"{SAVE_DIR}/en_dict_token2id.json", "w") as f:
        json.dump(en_dict_token2id, f, ensure_ascii=False, indent=4)
    with open(f"{SAVE_DIR}/cn_dict_token2id.json", "w") as f:
        json.dump(cn_dict_token2id, f, ensure_ascii=False, indent=4)
    with open(f"{SAVE_DIR}/en_dict_id2token.json", "w") as f:
        json.dump(en_dict_id2token, f, ensure_ascii=False, indent=4)
    with open(f"{SAVE_DIR}/cn_dict_id2token.json", "w") as f:
        json.dump(cn_dict_id2token, f, ensure_ascii=False, indent=4)

    # 计算词典大小
    en_vocab_size = len(en_vocab)
    cn_vocab_size = len(cn_vocab)
    print(f"英文字典大小为：{en_vocab_size}")
    print(f"英文最长序列长度为：{en_max_len}")
    print(f"中文字典大小为：{cn_vocab_size}")
    print(f"中文最长序列长度为：{cn_max_len}")
    """
    Building Vocabulary: 100%|██████████████████████████████████████████████████████████████| 29909/29909 [00:00<00:00, 342966.21it/s]
    英文字典大小为：7195
    英文最长序列长度为：38
    中文字典大小为：2839
    中文最长序列长度为：44
    """
