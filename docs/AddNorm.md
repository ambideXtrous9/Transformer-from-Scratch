# AddNorm.py

## Overview

The `AddNorm.py` module implements the **Add & Norm** component, which is a critical building block in the Transformer architecture. It combines a residual connection with Layer Normalization to stabilize training and allow gradients to flow through deep networks.

## Architecture

The `AddNorm` module takes two inputs:
1.  **x**: The input tensor from the previous layer (residual connection base).
2.  **sublayer_out**: The output tensor from the current sublayer (e.g., Multi-Head Self-Attention or Feed-Forward Network).

It performs the following operations:
1.  Applies **Dropout** to the `sublayer_out`.
2.  Adds the result to the residual base `x` (Element-wise Addition).
3.  Applies **Layer Normalization** to the sum.

### Mermaid Diagram

```mermaid
graph LR
    X[Input x <br> (Residual Base)] --> Add(+)
    SubOut[Sublayer Output] --> Dropout
    Dropout --> Add
    Add --> LayerNorm
    LayerNorm --> Output
```

## Class Definition: `AddNorm`

Inherits from `pytorch_lightning.LightningModule`.

### `__init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5)`

Initializes the module.

-   **Parameters:**
    -   `d_model` (int): The hidden dimension of the model (e.g., 512, 768).
    -   `dropout` (float): The dropout probability applied to the sublayer output. Default is 0.1.
    -   `eps` (float): Epsilon value for numerical stability in Layer Normalization. Default is 1e-5.

-   **Attributes:**
    -   `self.layer_norm`: `nn.LayerNorm` module.
    -   `self.dropout`: `nn.Dropout` module.

### `forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor`

Performs the forward pass.

-   **Args:**
    -   `x`: Input tensor acting as the residual connection base. Shape: `(Batch_Size, Sequence_Length, d_model)`.
    -   `sublayer_out`: Output from the sublayer (MHSA or FFN). Shape: `(Batch_Size, Sequence_Length, d_model)`.

-   **Returns:**
    -   The normalized and residual-added tensor. Shape: `(Batch_Size, Sequence_Length, d_model)`.

-   **Logic:**
    ```python
    return self.layer_norm(x + self.dropout(sublayer_out))
    ```

## Example Usage

The following example demonstrates how to use the `AddNorm` module in conjunction with an Embedding layer and a Multi-Head Self-Attention layer.

```python
import torch
from AddNorm import AddNorm
from MultiHeadSelfAttention import MultiHeadSelfAttention

# 1. Simulate Inputs
batch_size = 2
seq_len = 10
d_model = 256

# Random embeddings (representing 'x' or the residual base)
embeddings = torch.randn(batch_size, seq_len, d_model)

# 2. Simulate Sublayer Output (e.g., from MHSA)
mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=8, causal=False)
attention_mask = torch.ones(batch_size, seq_len) # Simple mask
# mhsa_out corresponds to 'sublayer_out'
mhsa_out, _ = mhsa(embeddings, attention_mask)

# 3. Apply Add & Norm
addnorm = AddNorm(d_model=d_model, dropout=0.1)
output = addnorm(x=embeddings, sublayer_out=mhsa_out)

print("Input 'x' shape:", embeddings.shape)
print("Sublayer output shape:", mhsa_out.shape)
print("AddNorm output shape:", output.shape)

# Expected Output:
# Input 'x' shape: torch.Size([2, 10, 256])
# Sublayer output shape: torch.Size([2, 10, 256])
# AddNorm output shape: torch.Size([2, 10, 256])
```
