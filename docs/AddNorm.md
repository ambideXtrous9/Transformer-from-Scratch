# AddNorm.py Module Documentation

## 1. Overview

The `AddNorm` module implements the **Residual Connection** followed by **Layer Normalization**, a standard design pattern in Transformer architectures (specifically the "Post-LN" variant). It ensures stable gradient flow and normalizes the input to a standard distribution, enabling deeper networks to train effectively.

## 2. Modules Involved

-   **torch**: Core PyTorch library for tensor operations.
-   **torch.nn**: Neural network layers (`nn.LayerNorm`, `nn.Dropout`).
-   **pytorch_lightning**: LightningModule base class (though this module is often used as a standard `nn.Module` within a larger Lightning system).

### Dependencies
This module is a fundamental building block used by:
-   `Encoder.py` (in `EncoderBlock`)
-   `Decoder.py` (in `DecoderBlock`)
-   `DecoderMoE.py` (in `DecoderBlockMoE`)
-   `DecoderOnlySeq2SeqModel.py` (in `DecoderBlock`)

It does not depend on other custom modules in the repository.

## 3. Architecture

The `AddNorm` block wraps a sub-layer (like Self-Attention or FFN).

1.  **Input**:
    -   `x`: The residual connection (input to the sub-layer).
    -   `sublayer_out`: The output of the sub-layer (e.g., Attention(x)).
2.  **Dropout**: Applied to `sublayer_out`.
3.  **Add**: `x + Dropout(sublayer_out)`.
4.  **Normalize**: `LayerNorm(Sum)`.

### Architecture Diagram

```mermaid
graph TD
    subgraph "Add & Norm Block"
        InputX[Input x <br>(Residual Base)]
        InputSub[Sublayer Output <br>(e.g. Attention/FFN)]
        
        InputSub --> Dropout[Dropout]
        InputX --> Add((+))
        Dropout --> Add
        
        Add --> LayerNorm[Layer Normalization]
        LayerNorm --> Output[Output Tensor]
    end
    
    style InputX fill:#e1f5fe
    style InputSub fill:#e1f5fe
    style Output fill:#e1f5fe
    style LayerNorm fill:#fff9c4
```

## 4. Class Definition

### `class AddNorm(LightningModule)`

#### `__init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5)`

-   **d_model**: The feature dimension size (e.g., 512).
-   **dropout**: Probability of zeroing elements in the sublayer output.
-   **eps**: Epsilon value for numerical stability in LayerNorm.

#### `forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor`

-   **x**: Input tensor, shape `(Batch, Seq_Len, d_model)`.
-   **sublayer_out**: Tensor to be added, shape `(Batch, Seq_Len, d_model)`.
-   **Returns**: Normalized tensor of shape `(Batch, Seq_Len, d_model)`.

## 5. Step-by-Step Logic

1.  **Dropout Application**: The `sublayer_out` is passed through a Dropout layer. This acts as a regularizer, preventing reliance on specific neurons.
    $$ \text{drop\_out} = \text{Dropout}(\text{sublayer\_out}) $$
2.  **Residual Addition**: The original input `x` is added to the dropped output.
    $$ \text{res\_sum} = x + \text{drop\_out} $$
3.  **Layer Normalization**: The sum is normalized over the last dimension ($d_{model}$).
    $$ \text{Output} = \frac{\text{res\_sum} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$
    Where $\mu$ and $\sigma^2$ are the mean and variance of `res_sum`, and $\gamma, \beta$ are learnable affine parameters.

## 6. Dry Run Example

Let's trace a minimal example.

**Settings:**
-   `d_model` = 4
-   `dropout` = 0.0 (for deterministic trace)
-   `eps` = 0.0

**Inputs:**
-   `x` (Batch=1, Seq=1): `[10.0, 10.0, 10.0, 10.0]`
-   `sublayer_out`: `[2.0, 4.0, 6.0, 8.0]`

**Execution:**

1.  **Dropout**: (Assuming eval mode or p=0)
    -   `dropped` = `[2.0, 4.0, 6.0, 8.0]`
2.  **Add**:
    -   `sum` = `[10+2, 10+4, 10+6, 10+8]` = `[12.0, 14.0, 16.0, 18.0]`
3.  **Layer Norm Statistics**:
    -   Mean ($\mu$) of `[12, 14, 16, 18]`: $(12+14+16+18)/4 = 60/4 = 15.0$
    -   Variance ($\sigma^2$):
        -   $(12-15)^2 = 9$
        -   $(14-15)^2 = 1$
        -   $(16-15)^2 = 1$
        -   $(18-15)^2 = 9$
        -   Avg Var = $(9+1+1+9)/4 = 20/4 = 5.0$
    -   Std Dev ($\sqrt{\sigma^2}$): $\sqrt{5} \approx 2.236$
4.  **Normalize**:
    -   Element 1: $(12 - 15) / 2.236 = -3 / 2.236 \approx -1.341$
    -   Element 2: $(14 - 15) / 2.236 = -1 / 2.236 \approx -0.447$
    -   Element 3: $(16 - 15) / 2.236 = 1 / 2.236 \approx 0.447$
    -   Element 4: $(18 - 15) / 2.236 = 3 / 2.236 \approx 1.341$
    -   Output: `[-1.341, -0.447, 0.447, 1.341]` (assuming $\gamma=1, \beta=0$)

## 7. Code Example

```python
import torch
from AddNorm import AddNorm

# Initialize
add_norm = AddNorm(d_model=4, dropout=0.0)

# Inputs
x = torch.tensor([[[10.0, 10.0, 10.0, 10.0]]])
sub_out = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])

# Forward
output = add_norm(x, sub_out)

print("Output:\n", output)
# Output should be close to [-1.34, -0.45, 0.45, 1.34]
```
