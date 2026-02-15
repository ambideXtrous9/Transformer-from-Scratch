# DecoderOnlySeq2SeqModel.py

## Overview

The `DecoderOnlySeq2SeqModel.py` module implements a **Decoder-Only Transformer** (similar to GPT). Unlike the Encoder-Decoder architecture, this model processes sequences auto-regressively using only masked self-attention. It is primarily used for tasks like language modeling and text generation.

## Architecture

The model consists of a stack of identical **Decoder Blocks**.

### Decoder Block Logic
1.  **Masked Multi-Head Self-Attention**: Allows the model to attend to previous tokens but prevents attending to future tokens (causal masking).
2.  **Add & Norm**: Residual connection followed by Layer Normalization.
3.  **Position-wise Feed-Forward Network**: Processes each position independently.
4.  **Add & Norm**: Residual connection followed by Layer Normalization.

### Mermaid Diagram

```mermaid
graph TD
    Input[Input ID Sequence] --> Embed[Token + Positional Embedding]
    Embed --> Block1[Decoder Block 1]
    Block1 --> Block2[Decoder Block 2]
    Block2 --> Norm[Final Layer Norm]
    Norm --> Classifier[Linear Head]
    Classifier --> Logits[Logits (Vocab Size)]
    
    subgraph "Decoder Block"
        B_In[Input] --> MHSA[Masked Self-Attention]
        MHSA --> AddNorm1[Add & Norm]
        AddNorm1 --> FFN[Feed-Forward Network]
        FFN --> AddNorm2[Add & Norm]
        AddNorm2 --> B_Out[Output]
    end
```

## Class Definitions

### 1. `DecoderBlock`

A single layer of the Transformer.

-   **Components**:
    -   `self.mhsa`: `MultiHeadSelfAttention` (causal=True).
    -   `self.ffn`: `PositionwiseFeedForward`.
    -   `self.addnorm1`, `self.addnorm2`.

### 2. `DecoderOnlyModel`

The complete PyTorch Lightning module.

-   **Parameters**:
    -   `vocab_size`, `d_model`, `num_layers`, `num_heads`, `d_ff`, etc.
    
-   **Forward Pass**:
    1.  Embed inputs.
    2.  Pass through `self.layers` (ModuleList of `DecoderBlock`).
    3.  Normalize and project to logits.
    
-   **Metrics**:
    -   Logs `train_loss` and `val_loss`.
    -   Computes **BLEU**, **ROUGE**, **METEOR**, and **BERTScore** on the validation set by decoding predictions.

## Usage Example

```python
import torch
from DecoderOnlySeq2SeqModel import DecoderOnlyModel

# 1. Config
vocab_size = 5000
d_model = 256

# 2. Init
model = DecoderOnlyModel(
    vocab_size=vocab_size,
    tokenizer=None,
    d_model=d_model,
    num_layers=4,
    num_heads=8
)

# 3. Dummy Input
input_ids = torch.randint(0, vocab_size, (2, 20)) # (Batch, Seq)

# 4. Forward
logits, attn_maps = model(input_ids)
print("Logits:", logits.shape) 
# Expected: torch.Size([2, 20, 5000])
```
