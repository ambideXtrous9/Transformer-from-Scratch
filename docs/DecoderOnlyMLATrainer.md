# DecoderOnlyMLATrainer.py Module Documentation

## 1. Overview

This script handles the **training pipeline** for the `DecoderOnlyMLAModel` (Multi-Head Latent Attention). It mirrors `DecoderOnlyTrainer.py` but trains a model that uses `MultiHeadLatentAttention` instead of standard `MultiHeadSelfAttention`, enabling low-rank KV compression for a smaller KV cache.

## 2. Modules Involved

-   **torch**: Tensor operations and data utilities.
-   **torch.utils.data**: `Dataset`, `DataLoader`, `random_split`.
-   **pytorch_lightning**: `Trainer`, `ModelCheckpoint`.
-   **datasets**: Loading GSM8K dataset.

### Dependencies
-   `Embedding.py` -> `get_tokenizer`: Provides the tokenizer.
-   `DecoderOnlyMLAModel.py` -> `DecoderOnlyMLAModel`: The MLA model being trained.

## 3. Architecture

```mermaid
graph TD
    GSM8K[GSM8K openai/gsm8k] --> Dataset[GSM8KDataset]
    Dataset --> Split[80/20 Split]
    Split --> TL[Train Loader]
    Split --> VL[Val Loader]

    TL & VL --> Trainer[PL Trainer]
    Trainer --> Model[DecoderOnlyMLAModel]
    Trainer --> CK[Checkpoint]
    CK --> Disk[MLACheckpoints/]
```

## 4. Class: `GSM8KDataset`

### `__getitem__` Logic Step-by-Step

1.  **Read**: Get `question` and `answer` from GSM8K row `idx`.
2.  **Combine**: `full_text = question + " " + answer`.
3.  **Tokenize**: `tokenizer(full_text, max_length=254, truncation=True)` (reserves 2 for BOS + EOS).
4.  **Input IDs**: `[BOS] + tokenized_ids`, padded to `max_length` with `pad_token_id`.
5.  **Labels**: `tokenized_ids + [EOS]`, padded with `-100` (ignored by loss).
6.  **Return**: Dict with `input_ids` and `labels`.

## 5. Dry Run Trace (Single Training Step)

**GSM8K Row**: `question="What is 2+2?"`, `answer="4"`.

| Step | Operation | Shape / Value |
|------|-----------|---------------|
| 1 | Combine | `"What is 2+2? 4"` |
| 2 | Tokenize | `[2061, 318, 362, 10, 17, 30, 604]` (len=7) |
| 3 | Input IDs | `[BOS, 2061, 318, 362, 10, 17, 30, 604, PAD, ...]` (len=256) |
| 4 | Labels | `[2061, 318, 362, 10, 17, 30, 604, EOS, -100, ...]` (len=256) |
| 5 | Forward | `logits (1, 256, vocab_size)` |
| 6 | Loss | CE between logits and labels (ignoring -100) |
| 7 | Backward | Compute gradients |
| 8 | Update | AdamW step |

## 6. Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| num_layers | 4 |
| num_heads | 4 |
| d_compress | 64 |
| d_ff | 512 |
| max_positions | 256 |
| max_epochs | 100 |
| batch_size (train) | 4 |
| batch_size (val) | 2 |
| Learning Rate | 1e-3 |

## 7. Usage

```bash
cd MLA && python DecoderOnlyMLATrainer.py
```

Requires the `datasets` library (GSM8K is loaded via `load_dataset("openai/gsm8k")`). Metrics: eval_loss, Perplexity.
