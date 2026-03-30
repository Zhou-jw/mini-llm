import json
import os
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

def get_dataset():
    with open(f"{DATASET_DIR}/cmn-eng/dataset.json", "r") as f:
        dataset = []
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                dataset.append(data)
    return dataset

def get_en_vocab():
    with open(f"{DATASET_DIR}/en_dict_token2id.json", "r") as f:
        en_vocab = json.load(f)
    return en_vocab

def get_cn_vocab():
    with open(f"{DATASET_DIR}/cn_dict_token2id.json", "r") as f:
        en_vocab = json.load(f)
    return en_vocab
    
def get_en_id2token():
    with open(f"{DATASET_DIR}/en_dict_id2token.json", "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def get_cn_id2token():
    with open(f"{DATASET_DIR}/cn_dict_id2token.json", "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}