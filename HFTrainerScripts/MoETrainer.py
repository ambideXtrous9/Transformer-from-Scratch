"""
HF Trainer — Decoder-Only with Mixture-of-Experts (MoE)

Original: TrainerScripts/DecoderMoETrainer.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from core.Embedding import get_tokenizer
from models.DecoderMoE import DecoderOnlyMoEModel
from HFTrainerScripts.hf_wrapper import HFModelWrapper, SaveBestModelCallback

torch.manual_seed(42)

MAX_LENGTH = 64
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "HF_MoECheckpoints")


class DecoderOnlyDataset(Dataset):
    def __init__(self, tokenizer, df, max_length=64):
        self.tokenizer = tokenizer
        self.texts = (df["text"] + " " + df["completion"]).tolist()
        self.max_length = max_length

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
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length - 2,
            return_tensors="pt", add_special_tokens=False,
        )
        ids = enc["input_ids"].squeeze(0)

        input_ids = torch.cat([torch.tensor([self.bos_id]), ids])
        labels = torch.cat([ids, torch.tensor([self.eos_id])])

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_id)])
        else:
            input_ids = input_ids[:self.max_length]

        if len(labels) < self.max_length:
            pad_len = self.max_length - len(labels)
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            labels = labels[:self.max_length]

        return {"input_ids": input_ids.long(), "labels": labels.long()}


if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "versatile_dataset_2000.csv"))
    print(f"\n---------DataFrame shape: {df.shape}---------\n")

    dataset = DecoderOnlyDataset(tokenizer, df, max_length=MAX_LENGTH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    base_model = DecoderOnlyMoEModel(
        vocab_size=vocab_size, d_model=256, max_positions=MAX_LENGTH,
        num_layers=4, num_heads=4, d_ff=512, tokenizer=tokenizer,
        dropout=0.1, pad_token_id=pad_id, lr=1e-3,
        num_experts=4, top_k=2,
    )
    model = HFModelWrapper(base_model)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_callback = SaveBestModelCallback(save_dir=os.path.join(OUTPUT_DIR, "best"))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        callbacks=[best_callback],
    )

    print("\n" + "=" * 50)
    print("Starting Training — DecoderOnly (MoE)")
    print("=" * 50 + "\n")

    trainer.train()
