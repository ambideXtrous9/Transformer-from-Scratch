"""
Token Embedding Module

Modern implementation with:
- Learnable token embeddings
- Optional sqrt(d_model) scaling (GPT-2 style)
- Weight tying support for input/output embeddings
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytorch_lightning as pl


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional scaling.
    
    Converts token IDs to dense vector representations.
    """
    
    def __init__(self, vocab_size: int, d_model: int, scale_embeddings: bool = False):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Embedding dimension
            scale_embeddings: If True, scale embeddings by sqrt(d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.scale_embeddings = scale_embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (B, L)
        
        Returns:
            Token embeddings of shape (B, L, d_model)
        """
        emb = self.embedding(x)  # (B, L, d_model)
        
        if self.scale_embeddings:
            emb = emb * (self.d_model ** 0.5)
        
        return emb


class TokenEmbeddingModule(pl.LightningModule):
    """
    Complete token embedding module combining token embeddings with optional scaling.
    LightningModule wrapper for standalone testing.
    """
    
    def __init__(self, vocab_size: int, d_model: int, scale_embeddings: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, scale_embeddings)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (B, L)
        
        Returns:
            Token embeddings of shape (B, L, d_model)
        """
        return self.token_embedding(x)


if __name__ == "__main__":
    # Test token embedding
    vocab_size = 50257
    d_model = 768
    batch_size = 2
    seq_length = 1024
    
    print("Testing Token Embedding Module")
    print("=" * 60)
    
    # Create embedding
    embedding = TokenEmbeddingModule(vocab_size, d_model, scale_embeddings=True)
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    emb = embedding(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {emb.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_length}, {d_model})")
    print(f"Number of parameters: {sum(p.numel() for p in embedding.parameters()):,}")
    print(f"Parameter count: {vocab_size * d_model:,} (vocab_size * d_model)")
    
    # Test embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {emb.mean().item():.6f}")
    print(f"  Std: {emb.std().item():.6f}")
    print(f"  Min: {emb.min().item():.6f}")
    print(f"  Max: {emb.max().item():.6f}")
