# DecoderOnlyInference.py Module Documentation

## 1. Overview

This script performs **text generation** using a trained standard `DecoderOnlyModel` (non-MoE). It mirrors `DecoderMoEInference.py` in structure but loads a different model class.

## 2. Modules Involved

-   **os, glob**: Filesystem utilities.
-   **torch**: Core PyTorch library.

### Dependencies
-   `DecoderOnlySeq2SeqModel.py` → `DecoderOnlyModel`: The standard decoder-only model.
-   `Embedding.py` → `get_tokenizer`: Tokenizer provider.

## 3. Architecture

```mermaid
graph TD
    Prompt[User Prompt] --> Tokenize[Tokenize + Prepend BOS]
    Tokenize --> Loop

    subgraph "Greedy Decoding Loop"
        Loop[Current Sequence] --> Model[DecoderOnlyModel]
        Model --> Logits[Last Position Logits]
        Logits --> ArgMax[ArgMax]
        ArgMax --> Append[Append Token]
        Append --> Check{EOS?}
        Check -- No --> Loop
    end
    
    Check -- Yes --> Decode[Decode to String]
    Decode --> Output[Generated Text]
```

## 4. Functions

### `greedy_decode(model, tokenizer, prompt, max_len, device, bos_token_id, eos_token_id)`

Identical logic to `DecoderMoEInference.greedy_decode`, but operates on `DecoderOnlyModel`.

### `load_latest_checkpoint(checkpoint_dir, vocab_size, tokenizer)`

Loads the latest `.ckpt` file from `DecoderOnlyCheckpoints/`.

## 5. Step-by-Step Logic

1.  **Set eval mode**: Disables dropout and batch norm training behavior.
2.  **Tokenize prompt**: Convert text to token IDs.
3.  **Prepend BOS**: Add beginning-of-sequence token.
4.  **Loop**:
    -   Forward pass through entire sequence.
    -   Extract logits at the last position.
    -   Select token with highest probability (`argmax`).
    -   Append to the sequence.
    -   Check for EOS → stop condition.
5.  **Decode**: Convert final token ID sequence to human-readable text.

## 6. Dry Run Trace

**Prompt**: `"Question: If a train travels 60 miles in 2 hours, how fast is it going?\nAnswer:"`

| Step | Sequence | Next Token | Notes |
|------|----------|------------|-------|
| Init | `[BOS, 24361, 25, 1002, ...]` | — | Tokenized GSM8K-style prompt |
| Iter 1 | `[BOS, 24361, 25, ...]` | 464 | "The" |
| Iter 2 | `[BOS, ..., 464]` | 4644 | "train" |
| Iter 3 | `[BOS, ..., 4644]` | EOS | **STOP** |
| Output | — | — | `"Question: If a train travels 60 miles...\nAnswer: The train"` |

## 7. Usage

```bash
python DecoderOnlyInference.py
```

Requires a trained checkpoint in `DecoderOnlyCheckpoints/`.
