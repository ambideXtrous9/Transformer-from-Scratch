# file: tokenize_and_embedding.py
import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer

# ---------- Tokenizer helper ----------
def get_tokenizer(name: str = "gpt2", add_pad_token_if_missing: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if add_pad_token_if_missing and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

def tokenize_batch(tokenizer, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

# ---------- Sinusoidal positional encoding ----------
def sinusoidal_positional_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)  # (n_pos, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (n_pos, d_model)

# ---------- Token Embedding ----------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.scale = math.sqrt(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * self.scale  # (B, L, d_model)

# ---------- Positional Embedding ----------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_positions: int, use_sinusoidal: bool = False):
        super().__init__()
        self.use_sinusoidal = use_sinusoidal

        if use_sinusoidal:
            pe = sinusoidal_positional_encoding(max_positions, d_model)  # (max_positions, d_model)
            self.register_buffer("positional_encoding", pe, persistent=False)
            self.positional_embedding = None
        else:
            self.positional_embedding = nn.Embedding(max_positions, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        if self.use_sinusoidal:
            pos_enc = self.positional_encoding[:L, :]  # (L, d_model)
            return pos_enc.unsqueeze(0).to(x.device)   # (1, L, d_model) → broadcast
        else:
            pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            return self.positional_embedding(pos_ids)  # (B, L, d_model)

# ---------- Combined Embedding Module ----------
class TokenEmbeddingModule(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_positions: int = 2048,
        dropout: float = 0.1,
        pad_token_id: Optional[int] = None,
        use_sinusoidal_pos: bool = False,
    ):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model, pad_token_id)
        self.pos_emb = PositionalEmbedding(d_model, max_positions, use_sinusoidal=use_sinusoidal_pos)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        token_out = self.token_emb(input_ids)         # (B, L, d_model)
        pos_out = self.pos_emb(token_out)             # (B, L, d_model) or (1, L, d_model)
        x = token_out + pos_out
        x = self.dropout(x)

        # optional mask application
        if attention_mask is not None and self.pad_token_id is None:
            attention_mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            x = x * attention_mask

        return x  # (B, L, d_model)

# ---------- Usage Example ----------
if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    texts = ["Hello world", "This is a longer example for embedding demo."]
    batch = tokenize_batch(tokenizer, texts, max_length=32)
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

    model = TokenEmbeddingModule(
        vocab_size=vocab_size, d_model=256, max_positions=32,
        pad_token_id=pad_id, use_sinusoidal_pos=True
    )
    emb = model(input_ids, attention_mask)
    print("Embeddings:", emb.shape)  # torch.Size([2, seq_len, 256])
