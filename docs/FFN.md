# FFN.py

## Overview

The `FFN.py` module implements the **Position-wise Feed-Forward Network (FFN)**. In the Transformer architecture, this network is applied to each position separately and identically. It consists of two linear transformations with a non-linear activation function in between.

## Architecture

The FFN performs the following operations:
1.  Projects input from $d_{model}$ to a larger dimension $d_{ff}$ (usually $4 \times d_{model}$).
2.  Applies a non-linear activation function (ReLU or GELU).
3.  Applies Dropout.
4.  Projects back from $d_{ff}$ to $d_{model}$.

### Mathematical Formulation

$$ FFN(x) = \text{Linear}_2(\text{Dropout}(\text{Activation}(\text{Linear}_1(x)))) $$

### Mermaid Diagram

```mermaid
graph LR
    Input[Input x <br> (d_model)] --> Linear1[Linear 1 <br> (d_model -> d_ff)]
    Linear1 --> Activation[Activation <br> (ReLU/GELU)]
    Activation --> Dropout
    Dropout --> Linear2[Linear 2 <br> (d_ff -> d_model)]
    Linear2 --> Output
```

## Class Definition: `PositionwiseFeedForward`

Inherits from `pl.LightningModule`.

### `__init__`

-   **Parameters**:
    -   `d_model`: Input/output dimension.
    -   `d_ff`: Hidden dimension.
    -   `dropout`: Dropout probability.
    -   `activation`: "relu" or "gelu".

### `forward`

-   **Args**:
    -   `x`: Input tensor of shape `(Batch, Seq_Len, d_model)`.
-   **Returns**:
    -   Output tensor of the same shape.

## Usage Example

```python
import torch
from FFN import PositionwiseFeedForward

# 1. Config
d_model = 256
d_ff = 1024

# 2. Init
ffn = PositionwiseFeedForward(
    d_model=d_model,
    d_ff=d_ff,
    dropout=0.1,
    activation="gelu"
)

# 3. Dummy Input
x = torch.randn(2, 10, d_model)

# 4. Forward
output = ffn(x)
print("Output Shape:", output.shape)
# Expected: torch.Size([2, 10, 256])
```
