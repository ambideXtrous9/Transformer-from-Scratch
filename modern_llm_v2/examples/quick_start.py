"""
Quick Start Example - Train with Small Dataset

This script demonstrates how to quickly train the model with a small dataset
for testing and experimentation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

# Import training utilities
from training.train import create_model, create_datamodule, create_trainer, train
import config_200m as config


def quick_test_run():
    """
    Quick test run with tiny dataset and minimal epochs.
    Perfect for verifying the setup works.
    """
    print("\n" + "=" * 80)
    print("Quick Test Run - Tiny Shakespeare Dataset")
    print("=" * 80)
    
    # Override config for quick testing
    original_max_samples = config.DATASET_MAX_SAMPLES
    original_dataset = config.DATASET_NAME
    
    # Use tiny Shakespeare dataset
    config.DATASET_NAME = "karpathy/tiny_shakespeare"
    config.DATASET_MAX_SAMPLES = 500  # Very small
    config.MAX_LENGTH = 128  # Shorter sequences
    config.TRAIN_BATCH_SIZE = 4  # Small batch size
    config.MAX_STEPS = 100  # Just 100 steps
    config.WARMUP_STEPS = 10  # Minimal warmup
    config.VAL_CHECK_INTERVAL = 0.5  # Validate twice per epoch
    
    try:
        # Initialize tokenizer
        print("\n[TOKENIZER] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = len(tokenizer)
        
        # Create smaller model for testing
        print("\n[MODEL] Creating model...")
        model = create_model(
            vocab_size=vocab_size,
            d_model=256,  # Smaller model for testing
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,  # Just 2 layers
            d_ff=512,
            max_seq_length=config.MAX_LENGTH,
            dropout=0.1
        )
        
        # Create dataloaders
        print("\n[DATA] Creating dataloaders...")
        train_dl, val_dl = create_datamodule(
            tokenizer=tokenizer,
            batch_size=config.TRAIN_BATCH_SIZE,
            max_length=config.MAX_LENGTH,
            num_workers=0,  # No workers for testing
            dataset_name=config.DATASET_NAME
        )
        
        # Create trainer
        print("\n[TRAINER] Creating trainer...")
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "quick-test")
        
        trainer = create_trainer(
            output_dir=checkpoint_dir,
            max_steps=config.MAX_STEPS,
            gradient_clip_val=config.GRADIENT_CLIP_VAL,
            precision="32",  # No mixed precision for testing
            accumulate_grad_batches=1,
            val_check_interval=config.VAL_CHECK_INTERVAL,
            log_every_n_steps=10,
            wandb_project="modern-llm-test",
            wandb_entity=None  # Disable W&B for test
        )
        
        # Add learning rate scheduler
        from training.train import CosineWarmupScheduler
        
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
        
        model.configure_optimizers = lambda: configure_optimizers_with_scheduler(model)
        
        # Train
        trainer = train(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            trainer=trainer,
            warmup_steps=config.WARMUP_STEPS,
            max_steps=config.MAX_STEPS
        )
        
        print("\n✅ Quick test run completed successfully!")
        print(f"📁 Checkpoint saved to: {checkpoint_dir}")
        
    finally:
        # Restore original config
        config.DATASET_NAME = original_dataset
        config.DATASET_MAX_SAMPLES = original_max_samples


def small_training_run():
    """
    Small training run with TinyStories dataset.
    Good for experimentation without long wait times.
    """
    print("\n" + "=" * 80)
    print("Small Training Run - TinyStories Dataset")
    print("=" * 80)
    
    # Override config for small training
    original_max_samples = config.DATASET_MAX_SAMPLES
    original_dataset = config.DATASET_NAME
    original_max_steps = config.MAX_STEPS
    original_warmup = config.WARMUP_STEPS
    
    # Use TinyStories with limited samples
    config.DATASET_NAME = "roneneldan/TinyStories"
    config.DATASET_MAX_SAMPLES = 5000  # 5k samples
    config.MAX_LENGTH = 256  # Medium sequences
    config.TRAIN_BATCH_SIZE = 8
    config.MAX_STEPS = 1000  # 1k steps
    config.WARMUP_STEPS = 100
    config.VAL_CHECK_INTERVAL = 0.25  # Validate 4 times per epoch
    
    try:
        # Initialize tokenizer
        print("\n[TOKENIZER] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = len(tokenizer)
        
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
            dropout=config.DROPOUT
        )
        
        # Create dataloaders
        print("\n[DATA] Creating dataloaders...")
        train_dl, val_dl = create_datamodule(
            tokenizer=tokenizer,
            batch_size=config.TRAIN_BATCH_SIZE,
            max_length=config.MAX_LENGTH,
            num_workers=2,
            dataset_name=config.DATASET_NAME
        )
        
        # Create trainer
        print("\n[TRAINER] Creating trainer...")
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "small-training")
        
        trainer = create_trainer(
            output_dir=checkpoint_dir,
            max_steps=config.MAX_STEPS,
            gradient_clip_val=config.GRADIENT_CLIP_VAL,
            precision=config.PRECISION,
            accumulate_grad_batches=2,
            val_check_interval=config.VAL_CHECK_INTERVAL,
            log_every_n_steps=50,
            wandb_project="modern-llm-small",
            wandb_entity=None  # Set your entity to enable W&B
        )
        
        # Add learning rate scheduler
        from training.train import CosineWarmupScheduler
        
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
        
        model.configure_optimizers = lambda: configure_optimizers_with_scheduler(model)
        
        # Train
        trainer = train(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            trainer=trainer,
            warmup_steps=config.WARMUP_STEPS,
            max_steps=config.MAX_STEPS
        )
        
        print("\n✅ Small training run completed successfully!")
        print(f"📁 Checkpoint saved to: {checkpoint_dir}")
        
    finally:
        # Restore original config
        config.DATASET_NAME = original_dataset
        config.DATASET_MAX_SAMPLES = original_max_samples
        config.MAX_STEPS = original_max_steps
        config.WARMUP_STEPS = original_warmup


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick start examples")
    parser.add_argument("--mode", choices=["test", "small"], default="test",
                       help="Run mode: 'test' for quick test, 'small' for small training")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        quick_test_run()
    else:
        small_training_run()
