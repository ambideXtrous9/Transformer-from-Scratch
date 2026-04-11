import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(n_pos, d_model)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)  # (n_pos, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (n_pos, d_model)


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


# ---------- Usage Example ----------
if __name__ == "__main__":
    # Test sinusoidal positional encoding
    print("Testing sinusoidal positional encoding...")
    pe = sinusoidal_positional_encoding(10, 512)
    print(f"Sinusoidal PE shape: {pe.shape}")  # (10, 512)

    # Test PositionalEmbedding with sinusoidal
    print("\nTesting PositionalEmbedding with sinusoidal encoding...")
    pos_emb_sin = PositionalEmbedding(d_model=256, max_positions=100, use_sinusoidal=True)
    x = torch.randn(2, 10, 256)  # (batch=2, seq_len=10, d_model=256)
    out_sin = pos_emb_sin(x)
    print(f"Sinusoidal output shape: {out_sin.shape}")  # (1, 10, 256)

    # Test PositionalEmbedding with learned embedding
    print("\nTesting PositionalEmbedding with learned embedding...")
    pos_emb_learned = PositionalEmbedding(d_model=256, max_positions=100, use_sinusoidal=False)
    out_learned = pos_emb_learned(x)
    print(f"Learned output shape: {out_learned.shape}")  # (2, 10, 256)

    print("\nAll tests passed!")
