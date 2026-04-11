import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from core.Embedding import get_tokenizer
from models.DecoderOnlyMLAModel import DecoderOnlyMLAModel
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
import config
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
import wandb

MAX_LENGTH = config.MAX_LENGTH

class GSM8KDataset(Dataset):
    """
    Dataset for decoder-only (GPT-style) training on GSM8K.
    - Input: [BOS] + text
    - Labels: text + [EOS], with -100 for padding
    """
    def __init__(self, tokenizer, hf_dataset, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        for sample in hf_dataset:
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
            self.texts.append(text)

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
            text,
            truncation=True,
            max_length=self.max_length - 2,
            return_tensors="pt",
            add_special_tokens=False
        )
        ids = enc["input_ids"].squeeze(0)

        input_ids = torch.cat([torch.tensor([self.bos_id]), ids], dim=0)
        labels = torch.cat([ids, torch.tensor([self.eos_id])], dim=0)

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

        return {
            "input_ids": input_ids.long(),
            "labels": labels.long()
        }


if __name__ == '__main__':
    pl.seed_everything(config.SEED)
    
    # ---------------- Setup ----------------
    print("Loading GSM8K dataset from HuggingFace...")
    gsm8k = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    train_data = gsm8k["train"]
    test_data = gsm8k["test"]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")

    tokenizer = get_tokenizer(config.TOKENIZER_NAME, add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    train_dataset = GSM8KDataset(tokenizer, train_data, max_length=MAX_LENGTH)
    val_dataset = GSM8KDataset(tokenizer, test_data, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # ---------------- Lightning Model ----------------
    model = DecoderOnlyMLAModel(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        max_positions=MAX_LENGTH,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_compress=config.D_COMPRESS,
        d_ff=config.D_FF,
        tokenizer=tokenizer,
        dropout=config.DROPOUT,
        pad_token_id=pad_id,
        lr=config.LEARNING_RATE
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath = config.CHECKPOINTS["mla"],
        filename = 'DecoderOnlyMLABestModel',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss_epoch',
        mode = 'min'
    )

    # ---------------- Trainer ----------------
    wandb_logger = WandbLogger(project=config.WANDB_PROJECT, name="DecoderOnly-MLA", log_model=False)
    wandb_logger.experiment.config.update({
        "architecture": "DecoderOnly-MLA",
        "d_model": config.D_MODEL,
        "num_layers": config.NUM_LAYERS,
        "num_heads": config.NUM_HEADS,
        "d_compress": config.D_COMPRESS,
        "d_ff": config.D_FF,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "max_epochs": config.MAX_EPOCHS,
    })

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        check_val_every_n_epoch=1,
        devices=-1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    # ---------------- Run Training ----------------
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
