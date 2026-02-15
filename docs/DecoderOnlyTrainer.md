# DecoderOnlyTrainer.py Module Documentation

## 1. Overview

This script handles the **training pipeline** for the standard `DecoderOnlyModel` (non-MoE). It mirrors `DecoderMoETrainer.py` but trains a model with a dense FFN instead of Mixture of Experts.

## 2. Modules Involved

-   **torch**: Tensor operations and data utilities.
-   **torch.utils.data**: `Dataset`, `DataLoader`, `random_split`.
-   **pytorch_lightning**: `Trainer`, `ModelCheckpoint`.
-   **pandas**: Reading CSV data.

### Dependencies
-   `Embedding.py` → `get_tokenizer`: Provides the tokenizer.
-   `DecoderOnlySeq2SeqModel.py` → `DecoderOnlyModel`: The model being trained.

## 3. Architecture

```mermaid
graph TD
    CSV[synthetic_text_completion.csv] --> DF[DataFrame]
    DF --> Dataset[DecoderOnlyDataset]
    Dataset --> Split[80/20 Split]
    Split --> TL[Train Loader]
    Split --> VL[Val Loader]
    
    TL & VL --> Trainer[PL Trainer]
    Trainer --> Model[DecoderOnlyModel]
    Trainer --> CK[Checkpoint]
    CK --> Disk[DecoderOnlyCheckpoints/]
```

## 4. Class: `DecoderOnlyDataset`

### `__getitem__` Logic Step-by-Step

1.  **Read**: Get `text` and `completion` from row `idx`.
2.  **Combine**: `full_text = text + " " + completion`.
3.  **Tokenize**: `tokenizer(full_text, max_length=32, padding="max_length")`.
4.  **Labels**: Copy `input_ids`, set padded positions to `-100`.
5.  **Return**: Dict with `input_ids` and `labels`.

## 5. Dry Run Trace (Single Training Step)

**CSV Row**: `text="Hello"`, `completion="world"`.

| Step | Operation | Shape / Value |
|------|-----------|---------------|
| 1 | Combine | `"Hello world"` |
| 2 | Tokenize | `[15496, 995, PAD, PAD, ...]` (len=32) |
| 3 | Labels | `[15496, 995, -100, -100, ...]` |
| 4 | Forward | `logits (1, 32, vocab_size)` |
| 5 | Loss | CE between logits and labels (ignoring -100) |
| 6 | Backward | Compute gradients |
| 7 | Update | AdamW step |

## 6. Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| num_layers | 2 |
| num_heads | 4 |
| d_ff | 128 |
| max_epochs | 100 |
| batch_size (train) | 4 |
| Learning Rate | 1e-3 |

## 7. Usage

```bash
python DecoderOnlyTrainer.py
```

Requires `synthetic_text_completion.csv`.
