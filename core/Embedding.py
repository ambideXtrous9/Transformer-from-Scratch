# file: tokenize_and_embedding.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer

from core.PositionalEmbedding import PositionalEmbedding

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

# ---------- Token Embedding ----------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_token_id: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.scale = math.sqrt(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * self.scale  # (B, L, d_model)

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
