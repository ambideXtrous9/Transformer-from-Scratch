# DecoderOnlyMLAInference.py Module Documentation

## 1. Overview

This script performs **text generation** using a trained `DecoderOnlyMLAModel` (Multi-Head Latent Attention). It mirrors `DecoderOnlyInference.py` in structure but loads a model that uses `MultiHeadLatentAttention` instead of standard `MultiHeadSelfAttention`.

## 2. Modules Involved

-   **os, glob**: Filesystem utilities.
-   **torch**: Core PyTorch library.

### Dependencies
-   `DecoderOnlyMLAModel.py` -> `DecoderOnlyMLAModel`: The MLA decoder-only model.
-   `Embedding.py` -> `get_tokenizer`: Tokenizer provider.

## 3. Architecture

```mermaid
graph TD
    Prompt[User Prompt] --> Tokenize[Tokenize + Prepend BOS]
    Tokenize --> Loop

    subgraph "Greedy Decoding Loop"
        Loop[Current Sequence] --> Model[DecoderOnlyMLAModel]
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

Identical logic to `DecoderOnlyInference.greedy_decode`, but operates on `DecoderOnlyMLAModel`.

### `load_latest_checkpoint(checkpoint_dir, vocab_size, tokenizer)`

Loads the latest `.ckpt` file from `MLACheckpoints/`.

## 5. Step-by-Step Logic

1.  **Set eval mode**: Disables dropout and batch norm training behavior.
2.  **Tokenize prompt**: Convert text to token IDs.
3.  **Prepend BOS**: Add beginning-of-sequence token.
4.  **Loop**:
    -   Forward pass through entire sequence.
    -   Extract logits at the last position.
    -   Select token with highest probability (`argmax`).
    -   Append to the sequence.
    -   Check for EOS -> stop condition.
5.  **Decode**: Convert final token ID sequence to human-readable text.

## 6. Dry Run Trace

**Prompt**: `"Question: A store sells apples for $2 each. If you buy 5 apples, how much do you pay?\nAnswer:"`

| Step | Sequence | Next Token | Notes |
|------|----------|------------|-------|
| Init | `[BOS, 24361, 25, 317, ...]` | --- | Tokenized GSM8K-style prompt |
| Iter 1 | `[BOS, 24361, 25, 317, ...]` | 1639 | "You" |
| Iter 2 | `[BOS, ..., 1639]` | 1414 | "pay" |
| Iter 3 | `[BOS, ..., 1414]` | EOS | **STOP** |
| Output | --- | --- | `"Question: A store sells apples...\nAnswer: You pay"` |

## 7. Usage

```bash
cd MLA && python DecoderOnlyMLAInference.py
```

Requires a trained checkpoint in `MLACheckpoints/`.
