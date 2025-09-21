import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from Embedding import get_tokenizer
from DecoderOnlySeq2SeqModel import DecoderOnlyModel  # the training module we just created
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd

pl.seed_everything(42)

# ------------------ Demo DataFrame ------------------
df = pd.read_csv("versatile_dataset_2000.csv")

print(f"\n---------DataFrame shape: {df.shape}---------\n")

class DecoderOnlyDataset(Dataset):
    """
    Dataset for decoder-only (GPT-style) training.
    - Input: [BOS] + text
    - Labels: text + [EOS], with -100 for padding
    """
    def __init__(self, tokenizer, df, max_length=128):
        self.tokenizer = tokenizer
        self.texts = (df["text"] + " " + df["completion"]).tolist()  # merge prompt + target
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
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize target text (without BOS/EOS for now)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length - 2,  # reserve BOS + EOS
            return_tensors="pt",
            add_special_tokens=False
        )
        ids = enc["input_ids"].squeeze(0)

        # Input IDs: [BOS] + text
        input_ids = torch.cat([torch.tensor([self.bos_id]), ids], dim=0)

        # Labels: text + [EOS]
        labels = torch.cat([ids, torch.tensor([self.eos_id])], dim=0)

        # Pad input_ids
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_id)])
        else:
            input_ids = input_ids[:self.max_length]

        # Pad labels with -100 (ignore index for loss)
        if len(labels) < self.max_length:
            pad_len = self.max_length - len(labels)
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids.long(),
            "labels": labels.long()
        }


# ---------------- Setup ----------------
tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
vocab_size = len(tokenizer)
pad_id = tokenizer.pad_token_id



dataset = DecoderOnlyDataset(tokenizer, df, max_length=64)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# ---------------- Lightning Model ----------------


model = DecoderOnlyModel(
    vocab_size=vocab_size,
    d_model=256,          # smaller d_model for demo
    max_positions=64,
    num_layers=4,
    num_heads=4,
    d_ff=128,
    tokenizer=tokenizer,
    dropout=0.1,
    pad_token_id=pad_id,
    lr=1e-3
)



checkpoint_callback = ModelCheckpoint(
    dirpath = 'DecoderOnlyCheckpoints',
    filename = 'DecoderOnlyBestModel',
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
