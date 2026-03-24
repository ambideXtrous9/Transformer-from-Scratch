# DecoderMoEGQATrainer.py Module Documentation

## 1. Overview

This script trains the `DecoderOnlyMoEGQAModel` (MoE + GQA) on GSM8K using **PyTorch Lightning**. It is the PL equivalent of `HFTrainerScripts/MoEGQATrainer.py`.

## 2. Dependencies
-   `DecoderMoEGQA.py` -> `DecoderOnlyMoEGQAModel`
-   `Embedding.py` -> `get_tokenizer`
-   `config.py` -> Centralized hyperparameters

## 3. Architecture

```mermaid
graph TD
    GSM8K[GSM8K Dataset] --> Dataset[GSM8KDataset]
    Dataset --> TrainLoader[Train DataLoader] & ValLoader[Val DataLoader]
    TrainLoader & ValLoader --> Trainer[PL Trainer]
    Trainer --> Model[DecoderOnlyMoEGQAModel]
    Trainer --> Checkpoint[ModelCheckpoint]
    Checkpoint --> Disk[DecoderMoEGQACheckpoints/]
```

## 4. Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 512 | `config.D_MODEL` |
| num_heads | 8 | `config.NUM_HEADS` |
| num_kv_heads | 2 | `config.NUM_KV_HEADS` |
| num_layers | 6 | `config.NUM_LAYERS` |
| d_ff | 2048 | `config.D_FF` |
| num_experts | 4 | `config.NUM_EXPERTS` |
| top_k | 2 | `config.TOP_K` |
| max_epochs | 100 | `config.MAX_EPOCHS` |

## 5. Usage

```bash
python PLTrainerScripts/DecoderMoEGQATrainer.py
```
