# Embedding.py

## Overview

The `Embedding.py` module handles the transformation of raw text into dense vector representations. It includes utilities for tokenization and implements the **Embedding Layer** of the Transformer, which consists of both **Token Embeddings** and **Positional Encodings**.

## Architecture

The `TokenEmbeddingModule` combines two types of embeddings:
1.  **Token Embedding**: Maps each token ID to a dense vector of size $d_{model}$. Scaled by $\sqrt{d_{model}}$.
2.  **Positional Encoding**: Adds information about the position of tokens in the sequence (either learnable or fixed sinusoidal).

### Mermaid Diagram

```mermaid
graph LR
    Input[Input IDs] --> TokenEmb[Token Embedding]
    TokenEmb --> Scale[Scale by sqrt(d_model)]
    
    Position[Position Indices] --> PosEmb[Positional Embedding]
    
    Scale --> Add(+)
    PosEmb --> Add
    
    Add --> Dropout
    Dropout --> Output[Final Embeddings]
```

## detailed Components

### 1. Tokenization Utilities

-   `get_tokenizer`: Loads a pre-trained tokenizer (default: GPT-2).
-   `tokenize_batch`: Tokenizes a list of strings with padding and truncation.

### 2. `sinusoidal_positional_encoding`

Generates fixed positional encodings using sine and cosine functions:

$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

### 3. `TokenEmbedding`

Wrapper around `nn.Embedding`.
-   **Forward**: `embedding(x) * sqrt(d_model)`

### 4. `PositionalEmbedding`

Supports two modes:
-   **Sinusoidal**: Uses fixed sine/cosine waves (not learnable).
-   **Learnable**: Uses `nn.Embedding(max_positions, d_model)`.

### 5. `TokenEmbeddingModule`

Combined module used by Encoder and Decoder.
-   **Inputs**: `input_ids`
-   **Operations**:
    1.  Get token embeddings.
    2.  Add positional embeddings.
    3.  Apply Dropout.
    4.  (Optional) Apply attention mask masking (rarely used here, usually done in attention).

## Usage Example

```python
import torch
from Embedding import TokenEmbeddingModule, get_tokenizer, tokenize_batch

# 1. Prepare Data
tokenizer = get_tokenizer("gpt2")
texts = ["Transformer embeddings are cool."]
batch = tokenize_batch(tokenizer, texts, max_length=10)
input_ids = batch["input_ids"]

# 2. Initialize Module
embed_module = TokenEmbeddingModule(
    vocab_size=len(tokenizer),
    d_model=256,
    max_positions=10,
    dropout=0.1,
    use_sinusoidal_pos=True
)

# 3. Forward
embeddings = embed_module(input_ids)
print("Embedding Shape:", embeddings.shape)
# Expected: torch.Size([1, 10, 256])
```
