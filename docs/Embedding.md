# Embedding.py Module Documentation

## 1. Overview

The `Embedding` module converts raw token IDs into dense vector representations. It contains tokenization utilities and the core embedding infrastructure that combines **Token Embeddings** with **Positional Encodings**—the first step in every Transformer pipeline.

## 2. Modules Involved

-   **torch**, **torch.nn**: Embedding layers and tensor ops.
-   **pytorch_lightning**: LightningModule base class.
-   **transformers** (HuggingFace): `AutoTokenizer` for loading pre-trained tokenizers.
-   **math**: For sinusoidal encoding computation.

### Dependencies
This module has **no dependencies** on other custom modules. It is the foundational module that all others depend on:
-   Used by: `Encoder.py`, `Decoder.py`, `DecoderMoE.py`, `DecoderOnlySeq2SeqModel.py`, `FFN.py`, `MultiHeadSelfAttention.py`, `AddNorm.py`.

## 3. Architecture

```mermaid
graph TD
    InputIDs[Input Token IDs \n (B, L)] --> TokenEmb[Token Embedding \n nn.Embedding]
    TokenEmb --> Scale["Scale by √d_model"]
    
    Positions[Position Indices \n 0, 1, 2, ...] --> PosEmb{Positional Embedding}
    PosEmb -->|Sinusoidal| Fixed[Fixed sin/cos Encoding]
    PosEmb -->|Learnable| Learned[nn.Embedding]
    
    Scale --> Add((+))
    Fixed --> Add
    Learned --> Add
    
    Add --> Dropout
    Dropout --> Output[Embedded Tensor \n (B, L, d_model)]
```

## 4. Component Definitions

### `get_tokenizer(name, add_pad_token_if_missing)`
-   Loads a HuggingFace tokenizer (default: `"gpt2"`).
-   Optionally adds a pad token if missing (GPT-2 doesn't have one by default).

### `tokenize_batch(tokenizer, texts, max_length)`
-   Tokenizes a list of strings with padding and truncation.
-   Returns a dict with `input_ids` and `attention_mask` tensors.

### `sinusoidal_positional_encoding(max_len, d_model)`
-   Generates fixed positional encodings using sine/cosine functions.
-   **Formula**:
    -   Even dims: $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
    -   Odd dims: $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
-   Returns tensor of shape `(max_len, d_model)`.

### `TokenEmbedding(nn.Module)`
-   Wraps `nn.Embedding(vocab_size, d_model)`.
-   Forward: `embedding(x) * sqrt(d_model)` (scaling prevents small values).

### `PositionalEmbedding(nn.Module)`
-   **Sinusoidal mode**: Registers a buffer (no gradients). Shape `(1, max_positions, d_model)`.
-   **Learnable mode**: Uses `nn.Embedding(max_positions, d_model)`.

### `TokenEmbeddingModule(LightningModule)`
-   Combines `TokenEmbedding` + `PositionalEmbedding` + `Dropout`.
-   Optionally masks padding positions.

## 5. Step-by-Step Logic (TokenEmbeddingModule.forward)

1.  **Token Embedding**: Look up each token ID in the embedding table → `(B, L, d_model)`.
2.  **Scale**: Multiply by $\sqrt{d_{model}}$ to balance magnitude with positional encoding.
3.  **Positional Encoding**: Add position-dependent vectors → each position gets a unique signature.
4.  **Dropout**: Regularization.
5.  **(Optional) Mask**: Zero out positions where `attention_mask == 0`.

## 6. Dry Run Trace

**Scenario**: `d_model=4`, `vocab_size=100`, sequence `[5, 12]`.

| Step | Operation | Result (Shape) |
|------|-----------|----------------|
| 1 | Token Embed | `[[e5_0, e5_1, e5_2, e5_3], [e12_0, e12_1, e12_2, e12_3]]` → `(1, 2, 4)` |
| 2 | Scale (×√4=×2) | Each value doubled |
| 3 | Pos Embed (sinusoidal) | Pos 0: `[sin(0), cos(0), sin(0), cos(0)]` = `[0, 1, 0, 1]` |
| | | Pos 1: `[sin(1/c), cos(1/c), sin(1/c²), cos(1/c²)]` where c=10000^(2/4) |
| 4 | Add | Token embeddings + positional embeddings |
| 5 | Dropout | Some values randomly zeroed (during training) |
| **Output** | | `(1, 2, 4)` — Each token now has a position-aware representation |

## 7. Code Example

```python
import torch
from Embedding import TokenEmbeddingModule, get_tokenizer, tokenize_batch

tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
texts = ["Hello world"]
batch = tokenize_batch(tokenizer, texts, max_length=8)

embed = TokenEmbeddingModule(
    vocab_size=len(tokenizer),
    d_model=256,
    max_positions=8,
    use_sinusoidal_pos=True
)

output = embed(batch["input_ids"], batch["attention_mask"])
print("Shape:", output.shape)  # (1, 8, 256)
```
