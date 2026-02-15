# MultiHeadSelfAttention.py

## Overview

The `MultiHeadSelfAttention.py` module implements the **Multi-Head Self-Attention (MHSA)** mechanism, the core component of the Transformer. It allows the model to jointly attend to information from different representation subspaces at different positions.

## Mechanism

### Scaled Dot-Product Attention

The basic attention mechanism is defined as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where:
-   $Q$ (Query), $K$ (Key), $V$ (Value) are matrices.
-   $d_k$ is the dimension of the keys (used for scaling to prevent vanishing gradients).

### Multi-Head Attention

Multi-head attention runs $h$ attention mechanisms (heads) in parallel:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$

Where each head is:
$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

### Mermaid Diagram

```mermaid
graph TD
    Input[Input Embeddings] --> Proj[Linear Projections Q, K, V]
    Proj --> Split[Split into h Heads]
    
    subgraph "Per Head"
        Split --> Q[Q]
        Split --> K[K]
        Split --> V[V]
        Q & K --> Matmul1[Q * K^T]
        Matmul1 --> Scale[Scale by 1/sqrt(d_k)]
        Scale --> Mask[Apply Mask]
        Mask --> Softmax
        Softmax --> Drop[Dropout]
        Drop & V --> Matmul2[Attn * V]
    end
    
    Matmul2 --> Concat[Concatenate Heads]
    Concat --> Linear[Final Linear Projection]
    Linear --> Output
```

## Class Definition: `MultiHeadSelfAttention`

Inherits from `pl.LightningModule`.

### `__init__`

-   **Parameters**:
    -   `d_model`: Total dimension of the model.
    -   `num_heads`: Number of parallel attention heads.
    -   `dropout`: Dropout probability.
    -   `causal`: If `True`, applies a causal mask (prevents attending to future tokens), used in Decoders.

### `forward`

-   **Args**:
    -   `x`: Input tensor for Query (and Key/Value if `kv` is None).
    -   `mask`: Padding mask (prevents attending to pad tokens).
    -   `kv`: Optional input for Key/Value (used for **Cross-Attention** where Q comes from Decoder and K/V from Encoder).
    
-   **Logic**:
    1.  Project inputs to Q, K, V.
    2.  Split heads/shapes: `(Batch, Seq_Len, d_model)` -> `(Batch, Num_Heads, Seq_Len, d_k)`.
    3.  Compute Scaled Dot-Product Attention.
    4.  Apply masks (if provided).
    5.  Concatenate heads.
    6.  Final linear projection.

## Usage Example

```python
import torch
from MultiHeadSelfAttention import MultiHeadSelfAttention

# 1. Config
d_model = 256
num_heads = 8

# 2. Init
mhsa = MultiHeadSelfAttention(
    d_model=d_model,
    num_heads=num_heads,
    causal=False
)

# 3. Dummy Input
x = torch.randn(2, 10, d_model) # (Batch, Seq, Dim)
mask = torch.ones(2, 10) # Attention Mask

# 4. Forward
output, attn_weights = mhsa(x, mask=mask)

print("Output Shape:", output.shape)
# Expected: torch.Size([2, 10, 256])
```
