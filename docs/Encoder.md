# Encoder.py Module Documentation

## 1. Overview

The `Encoder` module implements the **Transformer Encoder**, which processes an input sequence into contextualized representations. Unlike the Decoder, the Encoder uses **bidirectional attention** — every token can attend to every other token, enabling full contextual understanding.

## 2. Modules Involved

-   **torch**, **torch.nn**: Core PyTorch.
-   **pytorch_lightning**: LightningModule base class.

### Dependencies
-   `Embedding.py` → `TokenEmbeddingModule`: For token + positional embeddings.
-   `MultiHeadSelfAttention.py` → `MultiHeadSelfAttention`: With `causal=False`.
-   `AddNorm.py` → `AddNorm`: Residual + Layer Norm.
-   `FFN.py` → `PositionwiseFeedForward`: Position-wise feed-forward.

### Used By
-   `CrossAttentionSeq2SeqModel.py`: As the encoder half of the Seq2Seq model.
-   `Decoder.py`: In the usage example (to provide encoder output).

## 3. Architecture

```mermaid
graph TD
    Input[Source Token IDs \n (B, L)] --> Embed[Token + Positional Embedding]
    Embed --> Block1[Encoder Block 1]
    
    subgraph "Encoder Block (x N)"
        EI[Input] --> MHSA[Multi-Head Self-Attention \n causal=False, bidirectional]
        MHSA --> AN1[Add & Norm]
        AN1 --> FFN[Feed-Forward Network]
        FFN --> AN2[Add & Norm]
    end
    
    Block1 --> BlockN[Encoder Block N]
    BlockN --> Norm[Final Layer Norm]
    Norm --> Output[Encoded Representations \n (B, L, d_model)]
```

## 4. Class Definitions

### `class EncoderBlock(LightningModule)`

-   **Sub-layers**:
    1.  `self.mhsa`: `MultiHeadSelfAttention(causal=False)` — bidirectional.
    2.  `self.addnorm1`: After self-attention.
    3.  `self.ffn`: `PositionwiseFeedForward`.
    4.  `self.addnorm2`: After FFN.
-   **Forward**: Returns `(x, attn_weights)`.

### `class Encoder(LightningModule)`

-   **Components**: `TokenEmbeddingModule` → N × `EncoderBlock` → `LayerNorm`.
-   **Forward**: Returns `(x, list_of_attn_maps)`.

## 5. Step-by-Step Logic

1.  **Embed**: `input_ids` → Token + Positional embeddings → `(B, L, d_model)`.
2.  **For each EncoderBlock**:
    -   **Self-Attention**:
        -   Q, K, V all come from the same input `x`.
        -   No causal mask → every token sees every other token.
        -   Padding mask applied if provided.
    -   **Add & Norm 1**: `x = LN(x + Dropout(attn_out))`.
    -   **FFN**: SwiGLU gated FFN (Swish gate).
    -   **Add & Norm 2**: `x = LN(x + Dropout(ffn_out))`.
3.  **Final Norm**: `x = LayerNorm(x)`.

## 6. Dry Run Trace

**Scenario**: `Batch=1`, `Seq=3` (`[Hello, World, PAD]`), `d_model=4`, `num_heads=2`.

| Step | Shape | Description |
|------|-------|-------------|
| Input | `(1, 3)` | Token IDs: `[15496, 2159, 50257]` |
| Embed | `(1, 3, 4)` | Token embed + pos embed |
| **Block 1** | | |
| Self-Attn Q,K,V | `(1, 2, 3, 2)` | 2 heads, d_k=2 |
| Attn Scores | `(1, 2, 3, 3)` | Full matrix (bidirectional) |
| Padding Mask | `[[1,1,0]]` | PAD position masked to -inf |
| Attn Output | `(1, 3, 4)` | Weighted values |
| AddNorm1 | `(1, 3, 4)` | Residual + LN |
| FFN | `(1, 3, 4)` | Through d_ff and back |
| AddNorm2 | `(1, 3, 4)` | Residual + LN |
| **Final Norm** | `(1, 3, 4)` | Output |

**Key Observation**: Token at PAD position still gets a representation, but downstream modules will ignore it via masking.

## 7. Code Example

```python
import torch
from Encoder import Encoder

encoder = Encoder(
    vocab_size=50258,
    d_model=256,
    num_layers=2,
    num_heads=8,
    d_ff=1024
)

input_ids = torch.randint(0, 50258, (2, 10))
mask = torch.ones(2, 10)

output, attn_maps = encoder(input_ids, mask)
print("Output:", output.shape)       # (2, 10, 256)
print("Layers:", len(attn_maps))     # 2
print("Attn:", attn_maps[0].shape)   # (2, 8, 10, 10)
```
