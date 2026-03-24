"""
HF Trainer — Decoder-Only with Multi-Head Latent Attention (MLA) on GSM8K
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from core.Embedding import get_tokenizer
from models.DecoderOnlyMLAModel import DecoderOnlyMLAModel
from HFTrainerScripts.hf_wrapper import HFModelWrapper, SaveBestModelCallback, make_compute_metrics, make_compute_perplexity, preprocess_logits_for_metrics

import config

torch.manual_seed(config.SEED)

MAX_LENGTH = config.MAX_LENGTH
OUTPUT_DIR = config.CHECKPOINTS["hf_mla"]


class GSM8KDataset(Dataset):
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

    base_model = DecoderOnlyMLAModel(
        vocab_size=vocab_size, d_model=config.D_MODEL, max_positions=MAX_LENGTH,
        num_layers=config.NUM_LAYERS, num_heads=config.NUM_HEADS, d_compress=config.D_COMPRESS, d_ff=config.D_FF, tokenizer=tokenizer,
        dropout=config.DROPOUT, pad_token_id=pad_id, lr=config.LEARNING_RATE,
    )
    model = HFModelWrapper(base_model)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_callback = SaveBestModelCallback(save_dir=os.path.join(OUTPUT_DIR, "best"))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=config.MAX_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.VAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=config.NUM_WORKERS,
        seed=config.SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[best_callback, make_compute_perplexity()],
    )

    print("\n" + "=" * 50)
    print("Starting Training — DecoderOnly (MLA)")
    print("=" * 50 + "\n")

    trainer.train()
