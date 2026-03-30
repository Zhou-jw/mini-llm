import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from mini_models.transformer.mask import new_decoder_self_attn_mask, new_self_attn_mask
from mini_models.transformer.transformer import Transformer

from .dataloader import get_dataloader, id2text

if __name__ == "__main__":
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, en_vocab, cn_vocab, en_id2token, cn_id2token = get_dataloader()
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
    transformer = transformer.to(device)

    loss_func = torch.nn.CrossEntropyLoss(
        ignore_index=2
    )  # 计算损失时，忽略掉pad_id部分的计算
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.8
    )  # 每隔固定数量的epoch将学习率减少一个固定的比例

    train_loss_curve = []
    val_loss_curve = []
    lr_curve = []
    # 训练和验证
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        transformer.train()
        loss_sum = 0.0

        # 训练----------------------------------------------------
        for step, (src, tgt) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # src: (batch_size, 45)
            # tgt: (batch_size, 50)

            ####################################################
            if step % (len(train_loader) - 1) == 0 and step != 0:
                print(id2text(src[0].tolist(), "en", en_id2token))
                print(id2text(tgt[0].tolist(), "cn", cn_id2token))
            ####################################################

            # 构造mask
            src_mask = new_self_attn_mask(seq=src, pad_token_id=2)
            memory_mask = new_self_attn_mask(seq=src, pad_token_id=2)
            tgt_mask = new_decoder_self_attn_mask(seq=tgt[:, :-1], pad_token_id=2)

            ####################################################
            # mask可视化
            if epoch == 0 and step == 0:
                # print(src_mask.shape)
                # print(src_mask)
                # print(memory_mask.shape)
                # print(memory_mask)
                # print(tgt_mask.shape)
                # print(tgt_mask)
                plt.imshow(
                    src_mask.squeeze().numpy(), cmap="viridis", interpolation="nearest"
                )  # (batch_size, seq_len)
                plt.colorbar()
                plt.title("src_mask")
                plt.show(block=False)
                plt.imshow(
                    memory_mask.squeeze().numpy(),
                    cmap="viridis",
                    interpolation="nearest",
                )  # (batch_size, seq_len)
                plt.colorbar()
                plt.title("memory_mask")
                plt.show(block=False)
                plt.imshow(
                    tgt_mask[0].squeeze().numpy(),
                    cmap="viridis",
                    interpolation="nearest",
                )  # 取了batch中的第一个，(seq_len, seq_len)
                plt.colorbar()
                plt.title("tgt_mask")
                plt.show(block=False)
            ####################################################

            src = src.to(device).long()
            tgt = tgt.to(device).long()
            src_mask = src_mask.to(device)
            memory_mask = memory_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            # 训练时，是由输入的tgt预测下一个字符，因此输入为tgt[:, :-1]，每一个位置的字符在看见前面已有字符的情况下预测下一个字符
            # 例如，tgt为: <sos> a b c d e <eos> <pad> <pad>，那么输入为：<sos> a b c d e <eos> <pad>，真值为：a b c d e <eos> <pad> <pad>
            # 假设预测输出为：a' b' c' d' e' <eos> <pad> <pad>，该预测输出需要与真实值进行交叉熵计算损失，为避免<pad>对有效token的影响，计算损失时<pad>位置不参与
            # 因此实际需要计算的是：a b c d e <eos> 与 a' b' c' d' e' <eos>的对应字符位置损失
            pred = transformer(
                src_tokens = src,
                dst_tokens = tgt[:, :-1],
                enc_mask=src_mask,
                dec_cross_attn_mask=memory_mask,
                dec_self_attn_mask=tgt_mask,
            )

            ####################################################
            # 查看训练时翻译效果
            if step % (len(train_loader) - 1) == 0 and step != 0:
                test_pred = pred[0]  # (seq_len, vocab_size)
                test_pred = test_pred.argmax(dim=1)  # (seq_len,)
                test_pred = test_pred.tolist()  # 转换成装了token_id的列表
                if 1 in test_pred:
                    eos_index = test_pred.index(1)  # 找到<eos>索引
                    test_pred = test_pred[: eos_index + 1]
                print(id2text(test_pred, "cn", cn_id2token))
                print("pred_len:", len(test_pred))
            ####################################################

            # 调整形状以计算损失
            pred = pred.contiguous().view(
                -1, pred.shape[-1]
            )  # (batch_size, seq_len, cn_vocab_size) -> (batch_size * seq_len, cn_vocab_size)
            target = (
                tgt[:, 1:].contiguous().view(-1)
            )  # (batch_size, seq_len) -> (batch_size * seq_len)
            loss = loss_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()  # 当前epoch的累计损失

        train_avg_loss = loss_sum / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        train_loss_curve.append(train_avg_loss)
        lr_curve.append(lr)

        scheduler.step()

        # 验证----------------------------------------------------
        transformer.eval()
        loss_sum = 0.0
        for step, (src, tgt) in enumerate(val_loader):
            # 构造mask
            src_mask = new_self_attn_mask(seq=src, pad_token_id=2)
            memory_mask = new_self_attn_mask(seq=src, pad_token_id=2)
            tgt_mask = new_decoder_self_attn_mask(seq=tgt[:, :-1], pad_token_id=2)

            src = src.to(device).long()
            tgt = tgt.to(device).long()
            src_mask = src_mask.to(device)
            memory_mask = memory_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            pred = transformer(
                src_tokens = src,
                dst_tokens = tgt[:, :-1],
                enc_mask=src_mask,
                dec_cross_attn_mask=memory_mask,
                dec_self_attn_mask=tgt_mask,
            )
            pred = pred.contiguous().view(-1, pred.shape[-1])
            target = tgt[:, 1:].contiguous().view(-1)
            loss = loss_func(pred, target)

            loss_sum += loss.item()

        val_avg_loss = loss_sum / len(val_loader)
        val_loss_curve.append(val_avg_loss)
        print(
            f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | LR: {lr:.6f}"
        )

    # 保存模型
    torch.save(transformer.state_dict(), "transformer_from_scratch.pt")

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_loss_curve, label="Train Loss", color="blue")
    plt.plot(val_loss_curve, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show(block=False)

    # 绘制学习率曲线
    plt.figure()
    plt.plot(lr_curve, label="Learning Rate", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.show(block = False)
