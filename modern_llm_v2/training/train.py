"""
PyTorch Lightning Trainer Script for Modern LLM (200M Parameters)

Production-level training script with:
- PyTorch Lightning Trainer
- Weights & Biases (W&B) logging
- Learning rate scheduling with warmup
- Gradient clipping
- Mixed precision training (bf16)
- Model checkpointing
- Multi-GPU support
- Dataset loading (OpenWebText or custom)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set MPS memory environment variable before importing torch
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import datasets
import math
from typing import Optional, Dict, Any

from models.ModernLLM import ModernDecoderOnlyModel
import config_200m as config


class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling.
    
    Creates input_ids, labels, and attention_mask from batched sequences.
    Labels are shifted by one position for next-token prediction.
    Padding tokens are masked with -100.
    """
    
    def __init__(self, tokenizer, mlm: bool = False):
        """
        Args:
            tokenizer: Tokenizer instance
            mlm: Use masked language modeling (False for causal LM)
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    
    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Collate batch into model inputs.
        
        Args:
            batch: List of token sequences
        
        Returns:
            Dictionary with input_ids, labels, and attention_mask
        """
        # Stack sequences
        input_ids = torch.stack(batch)
        
        # Create attention mask (True for valid tokens, False for padding)
        # Must be boolean for masked_fill operations
        attention_mask = (input_ids != self.pad_token_id)  # Boolean tensor
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        
        # Mask padding tokens in labels with -100
        labels[labels == self.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


class TextDataset(Dataset):
    """
    Simple text dataset that tokenizes on-the-fly.
    
    More memory-efficient than loading all tokens at once.
    """
    
    def __init__(self, texts: list, tokenizer, max_length: int = 1024):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Tokenize and return a fixed-length sequence."""
        text = self.texts[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return encoded['input_ids'].squeeze(0)


class ConstantLengthDataset(Dataset):
    """
    Dataset that returns fixed-length chunks from a text corpus.
    
    This is more efficient than variable-length sequences and avoids padding.
    Used in production LLM training (e.g., GPT-Neo, Pythia).
    """
    
    def __init__(self, dataset, tokenizer, block_size: int, split: str = "train"):
        """
        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer instance
            block_size: Fixed sequence length
            split: Dataset split name
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize all text in the split
        print(f"[DATASET] Tokenizing {split} split...")
        texts = dataset[split]["text"][:10000]  # Limit for demo
        
        # Batch tokenize
        tokenized = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_tensors="pt"
        )
        
        # Concatenate all token IDs
        self.token_ids = torch.cat(tokenized["input_ids"], dim=0)
        print(f"[DATASET] Total tokens: {len(self.token_ids):,}")
        
        # Calculate number of complete blocks
        self.num_blocks = len(self.token_ids) // block_size
        print(f"[DATASET] Number of {block_size}-token blocks: {self.num_blocks:,}")
    
    def __len__(self) -> int:
        return self.num_blocks
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a fixed-length chunk."""
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        return self.token_ids[start_idx:end_idx]


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    Standard schedule for training LLMs:
    1. Linear warmup for warmup_steps
    2. Cosine decay to minimum learning rate
    """
    
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, 
                 min_lr_ratio: float = 0.1):
        """
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            min_lr_ratio: Minimum LR as fraction of initial LR
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        step = self._step_count
        
        # Linear warmup
        if step < self.warmup_steps:
            lr_scale = step / max(1, self.warmup_steps)
        # Cosine decay
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


class TextGenerationCallback(Callback):
    """
    Callback to generate and log text samples during training.
    """
    
    def __init__(self, tokenizer, prompt: str = "Hello, I'm a language model",
                 max_new_tokens: int = 50, log_every_n_epochs: int = 1):
        """
        Args:
            tokenizer: Tokenizer instance
            prompt: Text prompt for generation
            max_new_tokens: Maximum tokens to generate
            log_every_n_epochs: Generate text every N epochs
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate text at end of validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        
        pl_module.eval()
        
        with torch.no_grad():
            # Encode prompt
            input_ids = self.tokenizer.encode(
                self.prompt, return_tensors="pt", truncation=True
            ).to(pl_module.device)
            
            # Generate
            generated = pl_module.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            
            # Decode
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Log to W&B
            if isinstance(trainer.logger, WandbLogger):
                import wandb
                trainer.logger.experiment.log({
                    "generated_text": wandb.Html(f"<pre>{generated_text}</pre>"),
                    "epoch": trainer.current_epoch
                })
        
        pl_module.train()


def create_datamodule(tokenizer, batch_size: int, max_length: int,
                      num_workers: int = 4, dataset_name: str = "roneneldan/TinyStories"):
    """
    Create training and validation dataloaders.
    
    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of DataLoader workers
        dataset_name: HuggingFace dataset name
    
    Returns:
        train_dataloader, val_dataloader
    """
    print(f"[DATA] Loading dataset: {dataset_name}")
    
    # Load dataset with appropriate configuration
    if dataset_name == "roneneldan/TinyStories":
        # TinyStories: Small dataset (~100MB), great for testing
        print("[DATA] Loading TinyStories dataset...")
        dataset = datasets.load_dataset("roneneldan/TinyStories")
        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", dataset["train"].select(range(min(1000, len(dataset["train"])))))
        text_column = "text"
        
    elif dataset_name == "karpathy/tiny_shakespeare":
        # Tiny Shakespeare: Very small (~1MB), quick testing
        print("[DATA] Loading Tiny Shakespeare dataset...")
        dataset = datasets.load_dataset("karpathy/tiny_shakespeare")
        train_dataset = dataset["train"]
        val_dataset = dataset["train"].select(range(min(500, len(dataset["train"]))))
        text_column = "text"
        
    elif dataset_name == "Salesforce/wikitext":
        # WikiText-103: Medium size (~200MB)
        print("[DATA] Loading WikiText dataset...")
        dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", dataset["test"])
        text_column = "text"
        
    else:
        # Generic dataset loading
        print(f"[DATA] Loading custom dataset: {dataset_name}")
        dataset = datasets.load_dataset(dataset_name)
        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", dataset.get("test", train_dataset))
        text_column = "text"  # Adjust based on your dataset
    
    # Limit dataset size if configured
    if hasattr(config, 'DATASET_MAX_SAMPLES') and config.DATASET_MAX_SAMPLES is not None:
        max_samples = config.DATASET_MAX_SAMPLES
        print(f"[DATA] Limiting dataset to {max_samples} samples")
        if len(train_dataset) > max_samples:
            train_dataset = train_dataset.select(range(max_samples))
        if len(val_dataset) > max_samples // 10:
            val_dataset = val_dataset.select(range(min(max_samples // 10, len(val_dataset))))
    
    # Extract text column
    print("[DATA] Extracting text...")
    train_texts = train_dataset[text_column]
    val_texts = val_dataset[text_column]
    
    # Create datasets
    print("[DATA] Creating tokenized datasets")
    train_data = TextDataset(train_texts, tokenizer, max_length)
    val_data = TextDataset(val_texts, tokenizer, max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer)
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer)
    )
    
    print(f"[DATA] Train samples: {len(train_data):,}")
    print(f"[DATA] Validation samples: {len(val_data):,}")
    print(f"[DATA] Train batches: {len(train_dataloader)}")
    print(f"[DATA] Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader


def create_model(vocab_size: int, **model_kwargs) -> ModernDecoderOnlyModel:
    """
    Create the modern LLM model.
    
    Args:
        vocab_size: Vocabulary size
        **model_kwargs: Additional model hyperparameters
    
    Returns:
        ModernDecoderOnlyModel instance
    """
    model = ModernDecoderOnlyModel(
        vocab_size=vocab_size,
        **model_kwargs
    )
    
    # Print model info
    model.print_model_info()
    
    return model


def create_trainer(output_dir: str, max_steps: int, gradient_clip_val: float = 1.0,
                   precision: str = "bf16-mixed", accumulate_grad_batches: int = 4,
                   val_check_interval: float = 0.5, log_every_n_steps: int = 50,
                   checkpoint_every_n_epochs: int = 1, save_top_k: int = 3,
                   wandb_project: str = "modern-llm-200m", wandb_entity: str = None,
                   **trainer_kwargs) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer with W&B logging.
    
    Args:
        output_dir: Directory for checkpoints
        max_steps: Maximum training steps
        gradient_clip_val: Gradient clipping value
        precision: Training precision ("32", "16-mixed", "bf16-mixed")
        accumulate_grad_batches: Gradient accumulation steps
        val_check_interval: Validation frequency (fraction of epoch)
        log_every_n_steps: Logging frequency
        checkpoint_every_n_epochs: Checkpoint frequency
        save_top_k: Number of best checkpoints to save
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team)
        **trainer_kwargs: Additional trainer arguments
    
    Returns:
        PyTorch Lightning Trainer
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # W&B Logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        save_dir=output_dir,
        log_model=True if wandb_entity else False,
    )
    
    # Watch gradients if requested
    watch_gradients = trainer_kwargs.pop("watch_gradients", False)
    if watch_gradients:
        # We'll watch gradients after model is available
        pass
    
    print(f"[TRAINER] W&B Logger initialized: {wandb_logger.name}")
    print(f"[TRAINER] W&B Project: {wandb_project}")
    if wandb_entity:
        print(f"[TRAINER] W&B Entity: {wandb_entity}")
    
    # Model Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="modern-llm-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=save_top_k,
        every_n_epochs=checkpoint_every_n_epochs,
        save_last=True
    )
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        **trainer_kwargs  # Pass remaining kwargs
    )
    
    return trainer


def train(model: ModernDecoderOnlyModel, train_dataloader: DataLoader,
          val_dataloader: DataLoader, trainer: pl.Trainer,
          warmup_steps: int = 2000, max_steps: int = 100000,
          watch_gradients: bool = False):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        trainer: PyTorch Lightning Trainer
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        watch_gradients: Watch gradient norms in W&B
    """
    print("\n[TRAIN] Starting training...")
    print(f"[TRAIN] Warmup steps: {warmup_steps:,}")
    print(f"[TRAIN] Max steps: {max_steps:,}")
    
    # Watch gradients if requested
    if watch_gradients and isinstance(trainer.logger, WandbLogger):
        import wandb
        try:
            trainer.logger.watch(model, log="all", log_freq=100)
            print("[TRAIN] W&B gradient watching enabled")
        except Exception as e:
            print(f"[TRAIN] Warning: Could not enable gradient watching: {e}")
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    print("[TRAIN] Training complete!")
    print(f"[TRAIN] Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    return trainer


def main():
    """Main training function."""
    print("\n" + "=" * 80)
    print("Modern LLM Training (200M Parameters)")
    print("=" * 80)
    
    # Initialize tokenizer
    print("\n[TOKENIZER] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"[TOKENIZER] Vocabulary size: {vocab_size:,}")
    
    # Create model
    print("\n[MODEL] Creating model...")
    model = create_model(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_kv_heads=config.NUM_KV_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        max_seq_length=config.MAX_LENGTH,
        dropout=config.DROPOUT,
        use_flash_attention=config.USE_FLASH_ATTENTION,
        attention_bias=config.ATTENTION_BIAS,
        ffn_bias=config.FFN_BIAS,
        rope_theta=config.ROPE_THETA,
        norm_eps=config.NORM_EPS,
        scale_embeddings=True,
        tie_word_embeddings=config.TIE_WORD_EMBEDDINGS,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Create dataloaders
    print("\n[DATA] Creating dataloaders...")
    
    # Use auto-detected batch size (recommended) or manual override
    batch_size = config.TRAIN_BATCH_SIZE or config.AUTO_BATCH_SIZE
    val_batch_size = config.VAL_BATCH_SIZE or config.AUTO_BATCH_SIZE
    
    print(f"[DATA] Auto-detected batch size: {config.AUTO_BATCH_SIZE}")
    print(f"[DATA] Using train batch size: {batch_size}")
    print(f"[DATA] Using validation batch size: {val_batch_size}")
    
    train_dataloader, val_dataloader = create_datamodule(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=config.MAX_LENGTH,
        num_workers=config.NUM_WORKERS,
        dataset_name=config.DATASET_NAME
    )
    
    # Create trainer
    print("\n[TRAINER] Creating trainer...")
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "modern-llm-200m")
    
    trainer = create_trainer(
        output_dir=checkpoint_dir,
        max_steps=config.MAX_STEPS,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        precision=config.PRECISION,
        accumulate_grad_batches=config.GRADIENT_ACCUMULATION_STEPS,
        val_check_interval=config.VAL_CHECK_INTERVAL,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        checkpoint_every_n_epochs=config.CHECKPOINT_EVERY_N_EPOCHS,
        save_top_k=config.SAVE_TOP_K,
        wandb_project=config.WANDB_PROJECT,
        wandb_entity=config.WANDB_ENTITY
    )
    
    # Add learning rate scheduler hook
    def configure_optimizers_with_scheduler(model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY
        )
        
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=config.WARMUP_STEPS,
            max_steps=config.MAX_STEPS
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    # Override model's configure_optimizers
    model.configure_optimizers = lambda: configure_optimizers_with_scheduler(model)
    
    # Train
    trainer = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        trainer=trainer,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=config.MAX_STEPS,
        watch_gradients=config.WANDB_WATCH_GRADIENTS
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    trainer.save_checkpoint(os.path.join(final_model_path, "checkpoint.ckpt"))
    print(f"\n[SAVE] Final model saved to: {final_model_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(final_model_path)
    print(f"[SAVE] Tokenizer saved to: {final_model_path}")
    
    print("\n" + "=" * 80)
    print("Training complete! Model and checkpoints saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
