"""
Positional Encoding Module

Modern implementation with multiple positional encoding strategies:
- RoPE (Rotary Position Embedding) - Default for modern LLMs (LLaMA, PaLM)
- Learned Positional Embeddings
- Sinusoidal (Fixed) Positional Encoding

RoPE is the recommended choice for production LLMs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Implements rotary positional encoding as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    This is the default positional encoding for modern LLMs (LLaMA, PaLM, etc.)
    """
    
    def __init__(self, d_model: int, max_seq_length: int, theta: float = 10000.0):
        """
        Args:
            d_model: Model dimension (must be even)
            max_seq_length: Maximum sequence length
            theta: Base frequency for RoPE (default: 10000.0)
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.theta = theta
        
        # Precompute frequency bands
        # Each pair of dimensions gets a frequency
        freqs = self._compute_freqs(d_model, max_seq_length, theta)
        self.register_buffer("freqs_cis", freqs)
    
    def _compute_freqs(self, d_model: int, max_seq_length: int, theta: float) -> torch.Tensor:
        """
        Compute frequency bands for RoPE.
        
        Returns:
            freqs_cis: Complex frequency tensor of shape (max_seq_length, d_model/2)
        """
        # Compute frequencies: 1/theta^(2i/d_model) for i in [0, d_model/2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Compute positions
        t = torch.arange(max_seq_length, dtype=torch.float)
        
        # Compute outer product: (seq_len, d_model/2)
        freqs = torch.outer(t, freqs)
        
        # Convert to complex: cos + i*sin
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis  # (max_seq_length, d_model/2)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Get RoPE frequency tensor for application to Q/K vectors.
        
        Args:
            x: Input tensor (not used, for API compatibility)
            seq_len: Current sequence length (defaults to max_seq_length)
        
        Returns:
            freqs_cis: Complex frequency tensor of shape (seq_len, d_model/2)
        """
        if seq_len is None:
            seq_len = self.max_seq_length
        
        return self.freqs_cis[:seq_len]  # (seq_len, d_model/2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple:
    """
    Apply RoPE to query and key tensors.
    
    Args:
        q: Query tensor of shape (B, num_heads, L, d_k)
        k: Key tensor of shape (B, num_kv_heads, L, d_k)
        freqs_cis: Complex frequency tensor of shape (L, d_k/2)
    
    Returns:
        q_rotated: RoPE-applied query tensor
        k_rotated: RoPE-applied key tensor
    """
    # Get shapes
    q_shape = q.shape
    k_shape = k.shape
    d_k = q.shape[-1]
    
    # Reshape for complex multiplication: (B, heads, L, d_k/2, 2)
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    # Reshape freqs_cis for broadcasting: (1, 1, L, d_k/2)
    # freqs_cis shape: (L, d_k/2) -> (1, 1, L, d_k/2)
    freqs_cis_broadcast = freqs_cis.view(1, 1, *freqs_cis.shape)
    
    # Apply rotation via complex multiplication
    q_rotated = torch.view_as_real(q_complex * freqs_cis_broadcast).flatten(3)
    k_rotated = torch.view_as_real(k_complex * freqs_cis_broadcast).flatten(3)
    
    # Restore original shapes and types
    q_out = q_rotated.reshape(q_shape).type_as(q)
    k_out = k_rotated.reshape(k_shape).type_as(k)
    
    return q_out, k_out


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embeddings.
    
    Simple learnable positional encoding (GPT-2 style).
    """
    
    def __init__(self, d_model: int, max_seq_length: int):
        """
        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (not used, for API compatibility)
            seq_len: Current sequence length
        
        Returns:
            Positional embeddings of shape (seq_len, d_model)
        """
        if seq_len is None:
            seq_len = x.shape[1] if len(x.shape) > 1 else self.max_seq_length
        
        positions = torch.arange(0, seq_len, device=x.device).long()
        return self.pos_embedding(positions)  # (seq_len, d_model)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal (Fixed) Positional Encoding.
    
    Original Transformer positional encoding using sin/cos functions.
    """
    
    def __init__(self, d_model: int, max_seq_length: int):
        """
        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create positional encoding
        pe = self._create_pe(d_model, max_seq_length)
        self.register_buffer("pe", pe)
    
    def _create_pe(self, d_model: int, max_seq_length: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe  # (max_seq_length, d_model)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (not used, for API compatibility)
            seq_len: Current sequence length
        
        Returns:
            Positional embeddings of shape (seq_len, d_model)
        """
        if seq_len is None:
            seq_len = x.shape[1] if len(x.shape) > 1 else self.max_seq_length
        
        return self.pe[:seq_len]  # (seq_len, d_model)


class PositionalEncoding(nn.Module):
    """
    Unified Positional Encoding interface.
    
    Factory class that creates the appropriate positional encoding module
    based on the specified type.
    """
    
    def __init__(self, encoding_type: str = "rope", d_model: int = 768, 
                 max_seq_length: int = 1024, theta: float = 10000.0):
        """
        Args:
            encoding_type: Type of positional encoding ("rope", "learned", "sinusoidal")
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            theta: Base frequency for RoPE (only used if encoding_type="rope")
        """
        super().__init__()
        self.encoding_type = encoding_type
        
        if encoding_type == "rope":
            self.encoding = RotaryPositionEmbedding(d_model, max_seq_length, theta)
        elif encoding_type == "learned":
            self.encoding = LearnedPositionalEmbedding(d_model, max_seq_length)
        elif encoding_type == "sinusoidal":
            self.encoding = SinusoidalPositionalEmbedding(d_model, max_seq_length)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Get positional encoding."""
        return self.encoding(x, seq_len)


class PositionalEncodingModule(pl.LightningModule):
    """
    LightningModule wrapper for positional encoding.
    Used for standalone testing.
    """
    
    def __init__(self, encoding_type: str = "rope", d_model: int = 768,
                 max_seq_length: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.save_hyperparameters()
        self.positional_encoding = PositionalEncoding(
            encoding_type, d_model, max_seq_length, theta
        )
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Get positional encoding."""
        return self.positional_encoding(x, seq_len)


if __name__ == "__main__":
    # Test positional encoding modules
    d_model = 768
    max_seq_length = 1024
    batch_size = 2
    
    print("Testing Positional Encoding Modules")
    print("=" * 60)
    
    # Test RoPE
    print("\n1. Rotary Position Embedding (RoPE)")
    print("-" * 60)
    rope = PositionalEncodingModule("rope", d_model, max_seq_length)
    x = torch.randn(batch_size, max_seq_length, d_model)
    freqs_cis = rope(x)
    print(f"Frequency tensor shape: {freqs_cis.shape}")
    print(f"Expected shape: ({max_seq_length}, {d_model // 2})")
    
    # Test RoPE application
    num_heads = 12
    d_k = d_model // num_heads  # 64
    q = torch.randn(batch_size, num_heads, 32, d_k)
    k = torch.randn(batch_size, num_heads, 32, d_k)
    
    # Create RoPE with d_k dimension (not d_model)
    rope_test = RotaryPositionEmbedding(d_k, max_seq_length=1024)
    freqs_cis_short = rope_test(x, seq_len=32)
    
    q_rot, k_rot = apply_rope(q, k, freqs_cis_short)
    print(f"\nRoPE application test:")
    print(f"  Query shape: {q.shape} -> {q_rot.shape}")
    print(f"  Key shape: {k.shape} -> {k_rot.shape}")
    
    # Test Learned
    print("\n2. Learned Positional Embedding")
    print("-" * 60)
    learned = PositionalEncodingModule("learned", d_model, max_seq_length)
    pos_emb = learned(x)
    print(f"Positional embedding shape: {pos_emb.shape}")
    print(f"Expected shape: ({max_seq_length}, {d_model})")
    print(f"Number of parameters: {sum(p.numel() for p in learned.parameters()):,}")
    
    # Test Sinusoidal
    print("\n3. Sinusoidal Positional Embedding")
    print("-" * 60)
    sinusoidal = PositionalEncodingModule("sinusoidal", d_model, max_seq_length)
    pos_emb = sinusoidal(x)
    print(f"Positional embedding shape: {pos_emb.shape}")
    print(f"Expected shape: ({max_seq_length}, {d_model})")
    print(f"Number of parameters: {sum(p.numel() for p in sinusoidal.parameters()):,} (should be 0)")
