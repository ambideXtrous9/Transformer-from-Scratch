# Transformer from Scratch

This repository contains a PyTorch implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The implementation is modular, well-documented, and follows PyTorch Lightning best practices.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Components](#components)
   - [Token Embedding](#token-embedding)
   - [Positional Encoding](#positional-encoding)
   - [Multi-Head Self-Attention](#multi-head-self-attention)
   - [Feed-Forward Network](#feed-forward-network)
   - [Add & Norm](#add--norm)
   - [Encoder](#encoder)
   - [Decoder](#decoder)
   - [Seq2Seq Model](#seq2seq-model)
3. [Mathematical Formulations](#mathematical-formulations)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Training](#training)
7. [References](#references)

## Architecture Overview

The Transformer follows the encoder-decoder architecture with self-attention mechanisms. Key components include:

- **Encoder**: Processes the input sequence
- **Decoder**: Generates the output sequence
- **Multi-Head Attention**: Captures relationships between different positions
- **Position-wise Feed-Forward Networks**: Applies non-linearity
- **Residual Connections & Layer Normalization**: Aids training

## Components

### Token Embedding

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.scale = math.sqrt(d_model)
```

- Maps token indices to dense vectors
- Scales embeddings by √d_model
- Handles padding tokens

### Positional Encoding

```python
def sinusoidal_positional_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

- Adds positional information to token embeddings
- Uses sine and cosine functions of different frequencies
- Enables the model to use sequence order information

### Multi-Head Self-Attention

```python
class MultiHeadSelfAttention(LightningModule):
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
```

- Implements scaled dot-product attention
- Supports both self-attention and cross-attention
- Handles padding and causal masking
- Projects inputs to multiple representation subspaces

### Feed-Forward Network

```python
class PositionwiseFeedForward(LightningModule):
    def __init__(self, d_model: int = 256, d_ff: int = 1024, 
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
```

- Two linear transformations with ReLU/GELU activation
- Applied to each position separately and identically
- Expands to d_ff dimensions then projects back to d_model

### Add & Norm

```python
class AddNorm(LightningModule):
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(sublayer_out))
```

- Residual connection around each sub-layer
- Layer normalization for stable training
- Dropout for regularization

### Encoder

```python
class Encoder(LightningModule):
    def __init__(self, vocab_size: int, d_model: int = 256, max_positions: int = 512,
                 num_layers: int = 6, num_heads: int = 8, d_ff: int = 1024,
                 dropout: float = 0.1, pad_token_id: Optional[int] = None,
                 use_sinusoidal_pos: bool = True):
        # Initialization code...
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Forward pass code...
```

- Stack of N identical layers
- Each layer has:
  - Multi-head self-attention
  - Position-wise feed-forward network
  - Residual connections & layer norm
- Outputs contextualized representations

### Decoder

```python
class Decoder(LightningModule):
    def __init__(self, vocab_size: int, d_model: int = 256, max_positions: int = 512,
                 num_layers: int = 6, num_heads: int = 8, d_ff: int = 1024,
                 dropout: float = 0.1, pad_token_id: Optional[int] = None,
                 use_sinusoidal_pos: bool = True):
        # Initialization code...
        
    def forward(self, input_ids: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None):
        # Forward pass code...
```

- Similar to encoder but with masked self-attention
- Additional cross-attention to encoder outputs
- Generates output sequence auto-regressively

## Mathematical Formulations

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

Where:
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### Position-wise Feed-Forward Network

$$
\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
$$

### Layer Normalization

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

## Usage

```python
# Initialize model
model = Seq2SeqModel(
    vocab_size=vocab_size,
    d_model=256,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    d_ff=1024,
    dropout=0.1,
    pad_token_id=tokenizer.pad_token_id
)

# Forward pass
logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
```

## Training

Training is handled through PyTorch Lightning's `Trainer`:

```python
from pytorch_lightning import Trainer

# Initialize trainer
trainer = Trainer(
    max_epochs=10,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=1.0,
    enable_checkpointing=True
)

# Train model
trainer.fit(model, train_dataloader, val_dataloader)
```

## Examples

### Translation Task

```python
# Example for machine translation
src_texts = ["Hello world", "How are you?"]
tgt_texts = ["Hola mundo", "¿Cómo estás?"]

# Tokenize inputs
src_encodings = tokenizer(src_texts, padding=True, return_tensors="pt")
tgt_encodings = tokenizer(tgt_texts, padding=True, return_tensors="pt")

# Forward pass
logits = model(
    src_encodings["input_ids"],
    tgt_encodings["input_ids"][:, :-1],  # Shift right for teacher forcing
    src_mask=src_encodings["attention_mask"],
    tgt_mask=tgt_encodings["attention_mask"][:, :-1]
)
```

### Text Generation

```python
# Greedy decoding
def generate(model, src_ids, max_length=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Encode source
        memory, _ = model.encoder(src_ids)
        
        # Initialize target with <sos> token
        tgt_ids = torch.tensor([[tokenizer.bos_token_id]], device=src_ids.device)
        
        for _ in range(max_length):
            # Get next token probabilities
            logits = model.decoder(tgt_ids, memory)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Stop if <eos> is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Append to sequence
            tgt_ids = torch.cat([tgt_ids, next_token], dim=-1)
            
    return tgt_ids[0].tolist()
```

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is all you need". *Advances in neural information processing systems*, 30.
2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
3. PyTorch Lightning: https://pytorch-lightning.readthedocs.io/
