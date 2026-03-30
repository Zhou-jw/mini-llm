# Transformer From Scratch

从零实现 Transformer 模型，用于英中翻译任务。

## 参考

- [手写一个 Transformer](https://wkq9411.github.io/2026-01-01/Code-Transformer.html)

## 项目结构

```
mini-llm/
├── demo/                      # 演示代码
│   ├── train.py              # 训练脚本
│   ├── infer.py              # 推理脚本
│   ├── dataloader.py         # 数据加载
│   ├── tokenizer.py          # 分词器
│   ├── dataset/              # 数据集目录
│   └── README.md             # 数据集下载说明
├── mini_models/              # 模型实现
│   ├── transformer/          # Transformer 架构
│   │   ├── encoder.py        # 编码器
│   │   ├── decoder.py        # 解码器
│   │   ├── transformer.py    # 主模型
│   │   ├── mask.py           # 注意力掩码
│   │   ├── ffn.py            # 前馈网络
│   │   ├── rmsnorm.py        # RMSNorm
│   │   └── pos_emb.py        # 位置编码
│   ├── attention/           # 注意力机制
│   │   ├── standard_attention.py  # 标准注意力 (含 RoPE)
│   │   └── utils.py
│   └── rope.py              # RoPE 位置编码实现
├── transformer_from_scratch.pt  # 训练好的模型权重
└── README.md
```

## 模型架构

- **编码器-解码器结构**：经典 Transformer 架构
- **位置编码**：RoPE (Rotary Position Embedding)
- **归一化**：RMSNorm
- **注意力机制**：Multi-Head Attention with RoPE

## 环境依赖

```
torch
transformers
tqdm
matplotlib
```

## 数据集

使用 [CMU's Chinese-English Corpus](https://www.manythings.org/anki/cmn-eng.zip)：

```bash
wget -P ./demo/dataset https://www.manythings.org/anki/cmn-eng.zip
unzip ./demo/dataset/cmn-eng.zip -d ./demo/dataset/
```

## 使用方法

### 训练

```bash
python -m demo.train
```

### 推理

```bash
python -m demo.infer
```

## 模型超参数

| 参数 | 值 |
|------|-----|
| d_model | 256 |
| num_layers | 3 |
| num_heads | 8 |
| d_ff | 1024 |
| dropout | 0.1 |
