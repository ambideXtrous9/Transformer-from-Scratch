import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from core.Embedding import get_tokenizer
from models.CrossAttentionSeq2SeqModel import CrossAttentionSeq2SeqModel
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
import config
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
import wandb

pl.seed_everything(config.SEED)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

MAX_LENGTH = config.MAX_LENGTH


class GSM8KSeq2SeqDataset(Dataset):
    """
    Dataset for encoder-decoder training on GSM8K.
    - Encoder input: tokenized question
    - Decoder input: [BOS] + answer
    - Labels: answer + [EOS] with -100 for padding
    """
    def __init__(self, tokenizer, hf_dataset, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.src_texts = []
        self.tgt_texts = []
        for sample in hf_dataset:
            self.src_texts.append(f"Question: {sample['question']}")
            self.tgt_texts.append(sample['answer'])

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
            max_length=self.max_length - 2,
            return_tensors="pt"
        )
        tgt_ids_raw = tgt_enc["input_ids"].squeeze(0)

        tgt_ids = torch.cat(
            [torch.tensor([self.bos_id]), tgt_ids_raw], dim=0
        )

        labels = torch.cat(
            [tgt_ids_raw, torch.tensor([self.eos_id])], dim=0
        )

        # ---------------- Padding ----------------
        if len(tgt_ids) < self.max_length:
            pad_len = self.max_length - len(tgt_ids)
            tgt_ids = torch.cat([tgt_ids, torch.full((pad_len,), self.pad_id)])
        else:
            tgt_ids = tgt_ids[:self.max_length]

        if len(labels) < self.max_length:
            pad_len = self.max_length - len(labels)
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            labels = labels[:self.max_length]

        tgt_mask = (tgt_ids != self.pad_id).long()

        return {
            "src_ids": src_ids.long(),
            "src_mask": src_mask.long(),
            "tgt_ids": tgt_ids.long(),
            "tgt_mask": tgt_mask.long(),
            "labels": labels.long()
        }


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

train_dataset = GSM8KSeq2SeqDataset(tokenizer, train_data, max_length=MAX_LENGTH)
val_dataset = GSM8KSeq2SeqDataset(tokenizer, test_data, max_length=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)

# ---------------- Lightning Model ----------------
model = CrossAttentionSeq2SeqModel(
    vocab_size=vocab_size,
    d_model=config.D_MODEL,
    max_positions=MAX_LENGTH,
    num_encoder_layers=config.NUM_ENCODER_LAYERS,
    num_decoder_layers=config.NUM_DECODER_LAYERS,
    num_heads=config.NUM_HEADS,
    d_ff=config.SEQ2SEQ_D_FF,
    tokenizer=tokenizer,
    dropout=config.DROPOUT,
    pad_token_id=pad_id,
    lr=config.LEARNING_RATE
)

checkpoint_callback = ModelCheckpoint(
    dirpath = config.CHECKPOINTS["seq2seq"],
    filename = 'CrossAttentionSeq2SeqBestModel',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss_epoch',
    mode = 'min'
)

# ---------------- Trainer ----------------
wandb_logger = WandbLogger(project=config.WANDB_PROJECT, name="Seq2Seq-CrossAttention", log_model=False)
wandb_logger.experiment.config.update({
    "architecture": "CrossAttention-Seq2Seq",
    "d_model": config.D_MODEL,
    "num_encoder_layers": config.NUM_ENCODER_LAYERS,
    "num_decoder_layers": config.NUM_DECODER_LAYERS,
    "num_heads": config.NUM_HEADS,
    "d_ff": config.SEQ2SEQ_D_FF,
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
