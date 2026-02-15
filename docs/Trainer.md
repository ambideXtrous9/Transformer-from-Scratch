# Trainer.py

## Overview

The `Trainer.py` script orchestrates the training process for the **Encoder-Decoder (Seq2Seq)** model. It prepares the source-target pairs, initializes the `CrossAttentionSeq2SeqModel`, and uses PyTorch Lightning for the training loop.

## Data Pipeline

### `Seq2SeqDataset`

Prepares paired data for translation or sequence-to-sequence tasks.

-   **Inputs**: CSV file with `text` (Source) and `completion` (Target).
-   **Encoder Processing**:
    -   Tokenizes source text.
    -   Pads to `max_length`.
    -   Returns `src_ids` and `src_mask`.
-   **Decoder Processing**:
    -   Tokenizes target text.
    -   **Input**: `[BOS]` + Target.
    -   **Labels**: Target + `[EOS]`.
    -   Pads to `max_length`.

### Mermaid Diagram: Seq2Seq Data Flow

```mermaid
graph TD
    Raw[CSV Data] --> Dataset[Seq2SeqDataset]
    
    subgraph "GetItem"
        Dataset --> Src[Source Text]
        Dataset --> Tgt[Target Text]
        
        Src --> TokSrc[Tokenize source]
        TokSrc --> EncIn[Encoder Inputs]
        
        Tgt --> TokTgt[Tokenize target]
        TokTgt --> DecIn[Decoder Input <br> (BOS + Target)]
        TokTgt --> Labels[Labels <br> (Target + EOS)]
    end
    
    EncIn & DecIn & Labels --> Batch
    Batch --> Model[CrossAttentionSeq2SeqModel]
```

## Training Configuration

-   **Model**: `CrossAttentionSeq2SeqModel`.
-   **Hyperparameters**:
    -   `d_model`: 256
    -   `num_encoder_layers`: 2
    -   `num_decoder_layers`: 2
    -   `num_heads`: 4
    -   `d_ff`: 128
-   **Trainer**:
    -   `max_epochs`: 100
    -   `accelerator`: GPU.
    -   `callbacks`: ModelCheckpoint (saves to `CrossAttentionSeq2SeqCheckpoints`).

## Usage

Run the script to start training:

```bash
python Trainer.py
```

Prerequisites:
-   `versatile_dataset_2000.csv`
-   `CrossAttentionSeq2SeqModel.py`
