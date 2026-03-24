# DecoderOnlyMQAModel.py Module Documentation

## 1. Overview

The `DecoderOnlyMQAModel` module implements a **GPT-style Decoder-Only Transformer with Multi-Query Attention (MQA)**. Unlike standard Multi-Head Attention where each head has its own Q, K, V projections, MQA uses multiple query heads but a **single shared Key head and a single shared Value head**. This drastically reduces the KV cache size during autoregressive inference while retaining most of the representational power.

## 2. Modules Involved

-   **torch**, **torch.nn**: Core PyTorch and neural network layers.
-   **pytorch_lightning**: LightningModule training framework.
-   **Metrics**: `sacrebleu`, `rouge_score`, `nltk`, `bert_score`, Perplexity.

### Dependencies
-   `Embedding.py` → `TokenEmbeddingModule`: Token and positional embeddings.
-   `MultiQueryAttention.py` → `MultiQueryAttention`: Multi-Query Attention (`causal=True`).
-   `AddNorm.py` → `AddNorm`: Residual connection + normalization.
-   `FFN.py` → `PositionwiseFeedForward`: Standard feed-forward network.

## 3. Architecture

```mermaid
graph TD
    Input[Input Token IDs] --> Embed[Token + Positional Embedding]
    Embed --> Block1[Decoder Block 1]

    subgraph "Decoder Block (x N)"
        B_In[Input] --> MQA[Multi-Query Attention \n causal=True]
        MQA --> AN1[Add & Norm]
        AN1 --> FFN[Feed-Forward Network]
        FFN --> AN2[Add & Norm]
    end

    Block1 --> BlockN[Decoder Block N]
    BlockN --> Norm[Final Layer Norm]
    Norm --> Classifier[Linear → Vocab Size]
    Classifier --> Logits[Output Logits]
```

### Key Difference from Standard Multi-Head Attention
-   **Standard MHA**: Each of `H` heads has its own Q, K, V projections → `W_q`, `W_k`, `W_v` are all `(d_model, d_model)`.
-   **Multi-Query Attention (MQA)**: All `H` query heads share a **single** K head and a **single** V head → `W_q` is `(d_model, d_model)`, but `W_k` and `W_v` are only `(d_model, d_k)`.
-   **Broadcast**: The single K and V are expanded to match the number of query heads via `unsqueeze(1) + expand()` — this is memory-efficient (no data copy).
-   **Benefit**: Reduces KV cache size by a factor of `H` during inference, enabling faster autoregressive decoding.

## 4. Class Definitions

### `class DecoderBlock(nn.Module)`

A single decoder layer with two sub-layers.

-   `self.mhsa`: `MultiQueryAttention(causal=True)`.
-   `self.addnorm1`: After multi-query attention.
-   `self.ffn`: `PositionwiseFeedForward`.
-   `self.addnorm2`: After FFN.

### `class DecoderOnlyMQAModel(pl.LightningModule)`

The complete model.

-   **Components**: Embedding → N × DecoderBlock → LayerNorm → Linear Classifier.
-   **Loss**: `CrossEntropyLoss(ignore_index=-100)`.
-   **Optimizer**: AdamW.

## 5. Step-by-Step Logic (Forward Pass)

1.  **Embedding**:
    -   `input_ids` `(B, L)` → `TokenEmbeddingModule` → `x` `(B, L, d_model)`.
    -   Adds positional encoding.

2.  **Decoder Blocks** (repeated N times):
    -   **Multi-Query Attention**:
        -   Compute Q from `x` → `(B, L, num_heads * d_k)`.
        -   Compute K, V from `x` → `(B, L, d_k)` each (single head).
        -   Reshape Q → `(B, num_heads, L, d_k)`.
        -   Expand K, V → `unsqueeze(1)` to `(B, 1, L, d_k)` then `expand` to `(B, num_heads, L, d_k)`.
        -   Apply causal mask (lower triangular).
        -   Output: attention-weighted values.
    -   **Add & Norm**: `x = LayerNorm(x + Dropout(attn_out))`.
    -   **FFN**: SwiGLU gated FFN (Swish gate).
    -   **Add & Norm**: `x = LayerNorm(x + Dropout(ffn_out))`.

3.  **Final Norm**: `x = LayerNorm(x)`.

4.  **Classifier**: `logits = Linear(x)` → `(B, L, vocab_size)`.

## 6. Dry Run Trace

**Scenario**: `Batch=1`, `Seq=3`, `d_model=4`, `num_heads=2`, `d_k=2`, `vocab_size=10`.

**Input**: `input_ids = [[5, 12, 8]]`

| Step | Shape | Description |
|------|-------|-------------|
| Embed | `(1, 3, 4)` | Token embed + positional embed for IDs [5, 12, 8] |
| **Block 1 - Multi-Query Attn** | | |
| Q projection | `(1, 3, 4)` | W_q: (4, 4) → full d_model output |
| K projection | `(1, 3, 2)` | W_k: (4, 2) → single head of size d_k |
| V projection | `(1, 3, 2)` | W_v: (4, 2) → single head of size d_k |
| Q reshape | `(1, 2, 3, 2)` | Split into num_heads=2 heads |
| K unsqueeze | `(1, 1, 3, 2)` | Add head dimension |
| K expand | `(1, 2, 3, 2)` | Broadcast to match num_heads (no copy) |
| V unsqueeze | `(1, 1, 3, 2)` | Add head dimension |
| V expand | `(1, 2, 3, 2)` | Broadcast to match num_heads (no copy) |
| Causal Mask | `[[1,0,0],[1,1,0],[1,1,1]]` | Lower triangular |
| Attn Scores | `(1, 2, 3, 3)` | QK^T / sqrt(d_k), masked |
| Attn Output | `(1, 3, 4)` | Weighted V, concatenated heads |
| AddNorm1 | `(1, 3, 4)` | Residual + LayerNorm |
| **Block 1 - FFN** | | |
| Linear1 | `(1, 3, 16)` | d_model → d_ff |
| SwiGLU/Swish | `(1, 3, 16)` | Activation |
| Linear2 | `(1, 3, 4)` | d_ff → d_model |
| AddNorm2 | `(1, 3, 4)` | Residual + LayerNorm |
| **Final** | | |
| Norm | `(1, 3, 4)` | Final LayerNorm |
| Classifier | `(1, 3, 10)` | Linear(4 → 10) |

**Output**: `logits` of shape `(1, 3, 10)`. Each position predicts the next token.

## 7. Code Example

```python
import torch
from DecoderOnlyMQAModel import DecoderOnlyMQAModel
from Embedding import get_tokenizer

tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)

model = DecoderOnlyMQAModel(
    vocab_size=len(tokenizer),
    tokenizer=tokenizer,
    d_model=256,
    num_layers=4,
    num_heads=4
)

input_ids = torch.randint(0, len(tokenizer), (1, 10))
logits, attn_maps = model(input_ids)
print("Logits:", logits.shape)  # (1, 10, vocab_size)
```
