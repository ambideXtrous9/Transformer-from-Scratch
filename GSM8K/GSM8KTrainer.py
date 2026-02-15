import sys
import os

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from Embedding import get_tokenizer
from DecoderOnlySeq2SeqModel import DecoderOnlyModel
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset

pl.seed_everything(42)


# ==================== Model Subclass (no BERTScore) ====================
class GSM8KModel(DecoderOnlyModel):
    """
    Subclass of DecoderOnlyModel that removes heavy metrics (BERTScore, ROUGE, etc.)
    during validation to avoid loading RoBERTa and associated warnings.
    Only tracks validation loss.
    """

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        logits, _ = self(input_ids)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.val_epoch_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_epoch_losses).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n----------------------------------------------\n"
              f"  Validation loss epoch: {avg_loss.item():.4f}\n"
              f"----------------------------------------------\n")
        self.val_epoch_losses = []


# ==================== Dataset ====================
class GSM8KDataset(Dataset):
    """
    Dataset for training a Decoder-Only model on GSM8K math problems.

    Format per sample:
        Input:  [BOS] Question: {question}\nAnswer: {answer}
        Labels: Question: {question}\nAnswer: {answer} [EOS]

    The model learns to generate step-by-step math reasoning.
    """
    def __init__(self, tokenizer, hf_dataset, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Format each sample as "Question: ...\nAnswer: ..."
        self.texts = []
        for sample in hf_dataset:
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
            self.texts.append(text)

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

        # Tokenize (without BOS/EOS for now)
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


if __name__ == "__main__":
    # ==================== Load GSM8K Dataset ====================
    print("Loading GSM8K dataset from HuggingFace...")
    gsm8k = load_dataset("openai/gsm8k", "main")

    train_data = gsm8k["train"]
    test_data = gsm8k["test"]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")
    print(f"\nSample question:\n{train_data[0]['question']}")
    print(f"\nSample answer:\n{train_data[0]['answer']}")

    # ==================== Setup ====================
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    MAX_LENGTH = 256

    train_dataset = GSM8KDataset(tokenizer, train_data, max_length=MAX_LENGTH)
    val_dataset = GSM8KDataset(tokenizer, test_data, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=2)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ==================== Model ====================
    model = GSM8KModel(
        vocab_size=vocab_size,
        d_model=256,
        max_positions=MAX_LENGTH,
        num_layers=4,
        num_heads=4,
        d_ff=512,
        tokenizer=tokenizer,
        dropout=0.1,
        pad_token_id=pad_id,
        lr=1e-3
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== Checkpointing ====================
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'GSM8KCheckpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='GSM8K-DecoderOnly-{epoch:02d}-{val_loss_epoch:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss_epoch',
        mode='min'
    )

    # ==================== Trainer ====================
    trainer = pl.Trainer(
        max_epochs=100,
        check_val_every_n_epoch=1,
        devices=-1,
        accelerator="gpu",
        callbacks=[checkpoint_callback]
    )

    # ==================== Run Training ====================
    print("\n" + "=" * 50)
    print("Starting GSM8K Training")
    print("=" * 50 + "\n")

    trainer.fit(model, train_loader, val_loader)
