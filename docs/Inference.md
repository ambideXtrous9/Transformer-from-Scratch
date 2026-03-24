# Inference.py Module Documentation

## 1. Overview

This script performs **inference** for the **Encoder-Decoder (Seq2Seq) Transformer** model (`CrossAttentionSeq2SeqModel`). It uses greedy decoding with separate encoder and decoder stages—unlike the decoder-only inference scripts which only have one model call per step.

## 2. Modules Involved

-   **os, glob**: Filesystem operations.
-   **torch**: Core PyTorch.

### Dependencies
-   `CrossAttentionSeq2SeqModel.py` → `Seq2SeqModel` (aliased): The Encoder-Decoder model.
-   `Embedding.py` → `get_tokenizer`, `tokenize_batch`: Tokenization utilities.

### Key Difference from Decoder-Only Inference
| Aspect | Encoder-Decoder (this file) | Decoder-Only |
|--------|---------------------------|--------------|
| Encoding | Separate Encoder pass | N/A |
| Decoding | Decoder with cross-attention to Encoder output | Self-attention only |
| Input to Decoder | BOS token (grows each step) | Full prompt (grows each step) |

## 3. Architecture

```mermaid
graph TD
    SrcText[Source Text] --> Tokenize[Tokenize src]
    Tokenize --> Encoder[Encoder Forward Pass \n \\(done once\\)]
    Encoder --> EncOut[Encoder Output \n (Key/Value)]
    
    BOS[BOS Token] --> DecInput[Decoder Input]
    
    subgraph "Decoding Loop"
        DecInput --> Decoder[Decoder Forward Pass]
        EncOut -.->|Cross-Attention| Decoder
        Decoder --> Classifier[Linear Classifier]
        Classifier --> Logits[Last Position Logits]
        Logits --> ArgMax[ArgMax]
        ArgMax --> NextToken[Next Token]
        NextToken --> Append[Append to DecInput]
        Append --> Check{EOS?}
        Check -- No --> DecInput
    end
    
    Check -- Yes --> Output[Decoded Text]
```

## 4. Functions

### `greedy_decode(model, tokenizer, src_text, max_len, device, bos_token_id, eos_token_id)`

-   **Encode once**: Tokenize source → pass through `model.encoder` → `enc_out`.
-   **Decode iteratively**: Start with `[BOS]`, pass through `model.decoder` + `model.classifier` each step.

### `load_latest_checkpoint(checkpoint_dir, vocab_size, tokenizer)`

-   Loads the most recent `.ckpt` from `Seq2SeqCheckpoints/`.

## 5. Step-by-Step Logic

1.  **Encode** (done once):
    -   Tokenize `src_text` → `src_ids`, `src_mask`.
    -   `enc_out, _ = model.encoder(src_ids, src_mask)` → `(1, src_len, d_model)`.

2.  **Initialize Decoder**: `tgt_ids = [[BOS]]`.

3.  **Decode Loop** (up to `max_len` iterations):
    -   `dec_out, _, _ = model.decoder(tgt_ids, enc_out, tgt_mask=None, memory_mask=src_mask)`.
    -   `logits = model.classifier(dec_out)` → `(1, current_tgt_len, vocab_size)`.
    -   `next_token = logits[:, -1, :].argmax(dim=-1)` → greedy pick.
    -   Append `next_token` to `tgt_ids`.
    -   If `next_token == eos_token_id` → **STOP**.

4.  **Decode**: `tokenizer.decode(tgt_ids)`.

## 6. Dry Run Trace

**Source**: `"Question: If a train travels 60 miles in 2 hours, how fast is it going?\nAnswer:"` → Token IDs `[24361, 25, 1002, ...]`

| Step | tgt_ids | Decoder sees | Next Token | Action |
|------|---------|-------------|------------|--------|
| Init | `[BOS]` | 1 token + enc_out | — | — |
| Iter 1 | `[BOS]` | cross-attn to `[24361, 25, ...]` | 464 ("The") | Append |
| Iter 2 | `[BOS, 464]` | cross-attn to `[24361, 25, ...]` | 4644 ("train") | Append |
| Iter 3 | `[BOS, 464, 4644]` | cross-attn to `[24361, 25, ...]` | EOS | **STOP** |
| Output | `"The train"` | — | — | Decoded |

**Note**: At each step, the Decoder attends to the Encoder output (the source sentence) via cross-attention, which guides the generation.

## 7. Usage

```bash
python Inference.py
```

Requires a trained checkpoint in `Seq2SeqCheckpoints/`.
