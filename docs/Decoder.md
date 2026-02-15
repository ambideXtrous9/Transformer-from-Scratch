# Decoder.py Module Documentation

## 1. Overview

The `Decoder` module implements the **Transformer Decoder**, a stack of layers that consumes a target sequence (and optionally an encoder's output) to generate the next token in the sequence. It is designed to be autoregressive, meaning it cannot look ahead at future tokens during training (masked self-attention).

## 2. Modules Involved

-   **torch**: Core PyTorch library.
-   **torch.nn**: Neural network building blocks.
-   **pytorch_lightning**: LightningModule base class.

### Dependencies
This module relies on the following custom components:
-   `Embedding` (for `TokenEmbeddingModule`): To convert token IDs into dense vectors.
-   `MultiHeadSelfAttention` (from `MultiHeadSelfAttention.py`): For both Masked Self-Attention and Cross-Attention.
-   **AddNorm** (from `AddNorm.py`): For residual connections and layer normalization.
-   **FFN** (from `FFN.py`): For the Position-wise Feed-Forward Network.

## 3. Architecture

The Decoder consists of an Embedding layer followed by $N$ identical `DecoderBlock`s. Each block has three sub-layers:
1.  **Masked Multi-Head Self-Attention**: Attends to previous positions in the target sequence.
2.  **Cross-Attention (Encoder-Decoder Attention)**: Attends to the Encoder's output.
3.  **Feed-Forward Network (FFN)**: Processes positions independently.

### Architecture Diagram

```mermaid
graph TD
    Input[Target Input IDs] --> Embed[Token + Positional Embedding]
    Embed --> Block1[Decoder Block 1]
    
    subgraph "Decoder Block"
        BlockInput[Input from Prev Layer] --> MHSA[Masked Self-Attention]
        MHSA --> AddNorm1[Add & Norm]
        
        AddNorm1 --> Cross[Cross-Attention]
        EncOut[Encoder Output \n (Key/Value)] -.-> Cross
        Cross --> AddNorm2[Add & Norm]
        
        AddNorm2 --> FFN[Feed-Forward Network]
        FFN --> AddNorm3[Add & Norm]
    end
    
    Block1 --> Block2[Decoder Block 2]
    Block2 --> Norm[Final Layer Norm]
    Norm --> Output[Contextualized Output]
```

## 4. Class Definitions

### `class DecoderBlock(LightningModule)`

#### `__init__(self, d_model, num_heads, d_ff, dropout)`
-   Initializes the three sub-layers and their corresponding `AddNorm` modules.
    -   `self.mhsa`: `causal=True`
    -   `self.cross_attn`: `causal=False` (but uses `enc_out` as KV)
    -   `self.ffn`

#### `forward(self, x, enc_out, tgt_mask, memory_mask)`
-   **x**: Input from previous layer `(Batch, Seq_Len, d_model)`.
-   **enc_out**: Encoder output `(Batch, Src_Len, d_model)`.
-   **tgt_mask**: Causal mask for self-attention.
-   **memory_mask**: Padding mask for cross-attention (masks source padding).

### `class Decoder(LightningModule)`

#### `__init__(self, vocab_size, d_model, ...)`
-   Initializes the `TokenEmbeddingModule`, a `ModuleList` of `DecoderBlock`s, and the final `LayerNorm`.

#### `forward(self, input_ids, enc_out, tgt_mask, memory_mask)`
-   **input_ids**: Target sequence IDs `(Batch, Tgt_Len)`.
-   **enc_out**: Encoder representations.
-   **Returns**:
    -   `x`: Output tensor `(Batch, Tgt_Len, d_model)`.
    -   `self_attn_maps`: List of self-attention weights from each layer.
    -   `cross_attn_maps`: List of cross-attention weights from each layer.

## 5. Step-by-Step Logic

1.  **Embedding**: 
    -   `input_ids` are converted to vectors and summed with positional encodings.
    -   Shape: `(Batch, Tgt_Len)` -> `(Batch, Tgt_Len, d_model)`.

2.  **Layer Stacking**:
    -   The tensor passes through each `DecoderBlock` sequentially.
    -   **Inside Block**:
        1.  **Masked Self-Attention**: The model attends to itself. The `tgt_mask` ensures position $i$ can only attend to positions $0...i$.
            -   `x = AddNorm(x, MHSA(x))`
        2.  **Cross-Attention**: The model attends to the Encoder Output.
            -   Query = `x` (from previous step).
            -   Key/Value = `enc_out`.
            -   `x = AddNorm(x, CrossAttn(x, kv=enc_out))`
        3.  **FFN**:
            -   `x = AddNorm(x, FFN(x))`

3.  **Final Normalization**:
    -   The output of the last block is normalized using `LayerNorm`.

## 6. Dry Run Trace

**Scenario**:
-   `Batch` = 1
-   `Tgt_Len` = 2 (`[BOS, Token1]`)
-   `d_model` = 4
-   `enc_out` shape = `(1, 3, 4)` (Source Length 3)

**Trace**:

1.  **Input**:
    -   `input_ids`: `[[1, 10]]`
    -   `enc_out`: Random tensor `(1, 3, 4)`
    
2.  **Embedding**:
    -   Output `x`: `(1, 2, 4)`

3.  **Decoder Block 1**:
    -   **Masked Self-Attn**:
        -   Pos 0 attends to Pos 0.
        -   Pos 1 attends to Pos 0, 1.
        -   Output `(1, 2, 4)`.
        -   AddNorm -> `x` `(1, 2, 4)`.
    -   **Cross-Attn**:
        -   Query `(1, 2, 4)` vs Key `(1, 3, 4)`.
        -   Attention Matrix `(1, Heads, 2, 3)` (Each target token attends to all source tokens).
        -   Output `(1, 2, 4)`.
        -   AddNorm -> `x`.
    -   **FFN**:
        -   Linear(4->16) -> Relu -> Linear(16->4).
        -   Output `(1, 2, 4)`.
        -   AddNorm -> `x`.

4.  **Decoder Block 2** (if num_layers=2):
    -   Repeat steps above with input from Block 1.
    -   Output `x`: `(1, 2, 4)`.

5.  **Final Norm**:
    -   Returns normalized `(1, 2, 4)`.

## 7. Code Example

```python
import torch
from Decoder import Decoder

# Init
decoder = Decoder(vocab_size=100, d_model=4, num_layers=1, num_heads=2)

# Data
tgt_ids = torch.randint(0, 100, (1, 5)) # Batch 1, Seq 5
enc_out = torch.randn(1, 8, 4)          # Batch 1, Src 8, Dim 4
tgt_mask = torch.tril(torch.ones(5, 5)).view(1, 1, 5, 5) # Causal Mask

# Forward
out, self_attns, cross_attns = decoder(
    input_ids=tgt_ids, 
    enc_out=enc_out, 
    tgt_mask=tgt_mask
)

print("Output Shape:", out.shape)
# Expected: torch.Size([1, 5, 4])
```
