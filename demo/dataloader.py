import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from .utils import *
from .tokenizer import cn_tokenizer, en_tokenizer
from torch.utils.data import DataLoader, Dataset, random_split

# 由token转换为token_id
def text2id(
    text,
    language,
    dict=None,
    dict_path="./demo/dataset/cmn-eng",
    en_max_len=45,
    cn_max_len=50,
) -> List[int]:
    """将一段文本转换为该词典下对应的token_id, 并根据max_len补全pad, 这里将中文填充为50, 英文填充为45

    Args:
        text : 输入文本
        language : 语言, 中文或英文
        dict : 词典, 如果为None, 则从dict_path中加载词典
        dict_path : 词典路径

    Returns:
        list : 列表, 里面是每个token的token_id
    """
    if language == "cn":
        token = cn_tokenizer(text)
        max_len = cn_max_len
    if language == "en":
        token = en_tokenizer(text)
        max_len = en_max_len
    if dict is None:
        with open(f"{dict_path}/{language}_dict_token2id.json", "r") as f:
            dict = json.load(f)

    try:
        token_id = [dict[t] for t in token]
    except KeyError as e:
        # 明确告诉你哪个词不在词典里
        print(f"\n❌ 报错：词【{e}】不在词典中！ token={token}")
        raise  # 继续抛出错误，方便定位
    # token_id = [dict[t] for t in token]
    token_id = [dict["<sos>"]] + token_id + [dict["<eos>"]]
    if len(token_id) < max_len:
        token_id += [dict["<pad>"]] * (max_len - len(token_id))
    return token_id


# 由token_id转换为token
def id2text(
    token_id, language, dict=None, dict_path="./demo/dataset/cmn-eng"
) -> str | None:
    """将一个列表中的token_id转换为对应的文本, 并去掉<sos>、<eos>、<pad>

    Args:
        token_id : 装有token_id的列表
        language : 语言, 中文或英文
        dict : 词典, 如果为None, 则从dict_path中加载词典
        dict_path : 词典路径

    Returns:
        str : 文本
    """
    if dict is None:
        with open(f"{dict_path}/{language}_dict_id2token.json", "r") as f:
            dict = json.load(f)
            dict = {
                int(k): v for k, v in dict.items()
            }  # 词典保存为json后，键会变成字符串, 转换为int

    token = [dict[i] for i in token_id if i not in [0, 1, 2]]
    if language == "cn":
        return "".join(token)
    if language == "en":
        text = ""
        CAP = False  # 调整是否大写
        # 调整英文单词、符号之间空格的有无
        for i, t in enumerate(token):
            if i == 0:
                text += t.capitalize()  # 首字母大写
            else:
                if t in ",.!?;:)}]'\"":
                    text += t
                    if t in ".?!":
                        CAP = True
                else:
                    if CAP:
                        t = t.capitalize()
                        CAP = False
                    text += " " + t
        return text


class TranslateDataset(Dataset):
    def __init__(
        self,
        dataset,
        en_vocab: Dict[str, int],
        cn_vocab: Dict[str, int],
    ):
        """构建中英翻译数据集

        Args:
            dataset : [{'english': '...', 'chinese': '...'}, ...]
            en_dict_token2id : 英语字典, token -> id
            cn_dict_token2id : 中文字典, token -> id
        """
        self.dataset = dataset
        self.en_vocab = en_vocab
        self.cn_vocab = cn_vocab

        en_tokens: List[List[int]] = []
        cn_tokens: List[List[int]] = []

        for data in self.dataset:
            en_tokens.append(text2id(data["english"], language="en", dict=en_vocab))
            cn_tokens.append(text2id(data["chinese"], language="cn", dict=cn_vocab))

        self.en_tokens = np.array(en_tokens)  # (total_data_size, en_seq_len)
        self.cn_tokens = np.array(cn_tokens)  # (total_data_size, cn_seq_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        en_seq = self.en_tokens[index]
        cn_seq = self.cn_tokens[index]
        return torch.from_numpy(en_seq), torch.from_numpy(cn_seq)


def get_dataloader(batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Dict, Dict]:
    dataset = get_dataset()
    en_vocab = get_en_vocab()
    cn_vocab = get_cn_vocab()

    translate_dataset = TranslateDataset(dataset, en_vocab=en_vocab, cn_vocab=cn_vocab)
    val_size = 1000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(translate_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, en_vocab, cn_vocab


if __name__ == "__main__":
    train_loader, val_loader, en_vocab, cn_vocab = get_dataloader()
