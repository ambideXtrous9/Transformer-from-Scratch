import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from Embedding import get_tokenizer, tokenize_batch
from Seq2SeqModel import Seq2SeqModel  # the training module we just created
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd

# ------------------ Demo DataFrame ------------------
df = pd.read_csv("synthetic_text_completion_dataset.csv")

print("\nDataFrame shape:", df.shape)


class Seq2SeqDataset(Dataset):
    """
    Dataset for sequence-to-sequence task using a DataFrame with 'text' and 'completion' columns.
    """
    def __init__(self, tokenizer, df, max_length=32):
        """
        df: pandas DataFrame with columns 'text' (source) and 'completion' (target)
        tokenizer: a HuggingFace tokenizer
        max_length: max length for both source and target sequences
        """
        self.tokenizer = tokenizer
        self.src_texts = df["text"].tolist()
        self.tgt_texts = df["completion"].tolist()
        self.max_length = max_length

        # tokenize source and target separately
        self.src_batch = tokenize_batch(tokenizer, self.src_texts, max_length=max_length)
        self.tgt_batch = tokenize_batch(tokenizer, self.tgt_texts, max_length=max_length)

        self.src_ids = self.src_batch["input_ids"]
        self.src_mask = self.src_batch["attention_mask"]
        self.tgt_ids = self.tgt_batch["input_ids"]
        self.tgt_mask = self.tgt_batch["attention_mask"]

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return {
            "src_ids": self.src_ids[idx],
            "tgt_ids": self.tgt_ids[idx],
            "src_mask": self.src_mask[idx],
            "tgt_mask": self.tgt_mask[idx]
        }

# ---------------- Setup ----------------
tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
vocab_size = len(tokenizer)
pad_id = tokenizer.pad_token_id



dataset = Seq2SeqDataset(tokenizer, df, max_length=32)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
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
    max_epochs=5,
    check_val_every_n_epoch=1,
    devices=-1,
    accelerator="gpu",  # change to 'gpu' if available
    callbacks=[checkpoint_callback]
)

# ---------------- Run Training ----------------
trainer.fit(model, train_loader, val_loader)
