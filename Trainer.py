import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from Embedding import get_tokenizer
from Seq2SeqModel import Seq2SeqModel  # the training module we just created
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd

# ------------------ Demo DataFrame ------------------
df = pd.read_csv("synthetic_text_completion.csv")

print("\nDataFrame shape:", df.shape)


class Seq2SeqDataset(Dataset):
    """
    Dataset for encoder-decoder training.
    - Encoder input: tokenized src text
    - Decoder input: [BOS] + target
    - Labels: target + [EOS] with -100 for padding
    """
    def __init__(self, tokenizer, df, max_length=128):
        self.tokenizer = tokenizer
        self.src_texts = df["text"].tolist()
        self.tgt_texts = df["completion"].tolist()
        self.max_length = max_length

        # Ensure special tokens exist
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src, tgt = self.src_texts[idx], self.tgt_texts[idx]

        # ---------------- Encoder ----------------
        src_enc = self.tokenizer(
            src,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        src_ids = src_enc["input_ids"].squeeze(0)
        src_mask = src_enc["attention_mask"].squeeze(0)

        # ---------------- Decoder ----------------
        tgt_enc = self.tokenizer(
            tgt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,  # reserve BOS+EOS
            return_tensors="pt"
        )
        tgt_ids_raw = tgt_enc["input_ids"].squeeze(0)

        # Decoder input: [BOS] + target
        tgt_ids = torch.cat(
            [torch.tensor([self.bos_id]), tgt_ids_raw], dim=0
        )

        # Labels: target + [EOS]
        labels = torch.cat(
            [tgt_ids_raw, torch.tensor([self.eos_id])], dim=0
        )

        # ---------------- Padding ----------------
        # pad decoder input with pad_id
        if len(tgt_ids) < self.max_length:
            pad_len = self.max_length - len(tgt_ids)
            tgt_ids = torch.cat([tgt_ids, torch.full((pad_len,), self.pad_id)])
        else:
            tgt_ids = tgt_ids[:self.max_length]

        # pad labels with -100 (ignored by loss)
        if len(labels) < self.max_length:
            pad_len = self.max_length - len(labels)
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            labels = labels[:self.max_length]

        # tgt_mask (for attention)
        tgt_mask = (tgt_ids != self.pad_id).long()

        return {
            "src_ids": src_ids.long(),
            "src_mask": src_mask.long(),
            "tgt_ids": tgt_ids.long(),
            "tgt_mask": tgt_mask.long(),
            "labels": labels.long()
        }



# ---------------- Setup ----------------
tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
vocab_size = len(tokenizer)
pad_id = tokenizer.pad_token_id



dataset = Seq2SeqDataset(tokenizer, df, max_length=32)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# ---------------- Lightning Model ----------------
model = Seq2SeqModel(
    vocab_size=vocab_size,
    d_model=256,          # smaller d_model for demo
    max_positions=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_ff=128,
    dropout=0.1,
    pad_token_id=pad_id,
    lr=1e-3
)



checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'BestModel',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss_epoch',
    mode = 'min'
)



# ---------------- Trainer ----------------
trainer = pl.Trainer(
    max_epochs=100,
    check_val_every_n_epoch=1,
    devices=-1,
    accelerator="gpu",  # change to 'gpu' if available
    callbacks=[checkpoint_callback]
)

# ---------------- Run Training ----------------
trainer.fit(model, train_loader, val_loader)
