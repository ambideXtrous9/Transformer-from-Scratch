"""
Decoder Block Module

Modern pre-norm decoder block architecture combining:
- Grouped Query Attention (GQA) with RoPE
- SwiGLU Feed-Forward Network
- RMSNorm with pre-normalization
- Residual connections

Architecture (Pre-Norm, standard in modern LLMs):
    x = x + Attention(RMSNorm(x))  # Pre-norm attention
    x = x + FFN(RMSNorm(x))        # Pre-norm FFN

This matches LLaMA, Mistral, and other production LLMs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple

from core.attention.GroupQueryAttention import GroupQueryAttention
from core.FFN import SwiGLU
from core.Normalization import RMSNorm


class DecoderBlock(nn.Module):
    """
    Modern Decoder Block with Pre-Normalization
    
    Implements a single transformer decoder layer with:
    - Pre-norm architecture (better training stability)
    - Grouped Query Attention (GQA) with RoPE
    - SwiGLU Feed-Forward Network
    - RMSNorm
    
    This is the building block for the complete LLM.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, dropout: float = 0.1, max_seq_length: int = 1024,
                 use_flash_attention: bool = True, attention_bias: bool = False,
                 ffn_bias: bool = False, rope_theta: float = 10000.0,
                 norm_eps: float = 1e-5):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA)
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            use_flash_attention: Use Flash Attention if available
            attention_bias: Use bias in attention projections
            ffn_bias: Use bias in FFN projections
            rope_theta: RoPE base frequency
            norm_eps: RMSNorm epsilon
        """
        super().__init__()
        
        # Pre-norm layers
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        
        # Attention: GQA with RoPE
        self.attention = GroupQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_flash_attention=use_flash_attention,
            attention_bias=attention_bias,
            rope_theta=rope_theta
        )
        
        # Feed-Forward Network: SwiGLU
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_bias=ffn_bias
        )
        
        # Dropout for residual connections
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_kv: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            mask: Optional padding mask (B, L)
            is_causal: Use causal masking
            past_kv: Optional cached (key, value) from previous steps
            return_kv: Return key-value for caching
        
        Returns:
            x: Output tensor of shape (B, L, d_model)
            past_kv: Optional cached (key, value)
            attn_weights: Optional attention weights
        """
        # Pre-norm attention: x = x + Dropout(Attention(Norm(x)))
        attn_input = self.attn_norm(x)
        attn_output, past_kv, attn_weights = self.attention(
            attn_input, mask=mask, is_causal=is_causal,
            past_kv=past_kv, return_kv=return_kv
        )
        x = x + self.resid_dropout(attn_output)
        
        # Pre-norm FFN: x = x + Dropout(FFN(Norm(x)))
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.resid_dropout(ffn_output)
        
        return x, past_kv, attn_weights


class DecoderBlockModule(pl.LightningModule):
    """
    LightningModule wrapper for DecoderBlock.
    Used for standalone testing.
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, num_kv_heads: int = 4,
                 d_ff: int = 2048, dropout: float = 0.1, max_seq_length: int = 1024,
                 use_flash_attention: bool = True, attention_bias: bool = False,
                 ffn_bias: bool = False, rope_theta: float = 10000.0,
                 norm_eps: float = 1e-5):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            use_flash_attention: Use Flash Attention if available
            attention_bias: Use bias in attention projections
            ffn_bias: Use bias in FFN projections
            rope_theta: RoPE base frequency
            norm_eps: RMSNorm epsilon
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.decoder_block = DecoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_flash_attention=use_flash_attention,
            attention_bias=attention_bias,
            ffn_bias=ffn_bias,
            rope_theta=rope_theta,
            norm_eps=norm_eps
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_kv: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass."""
        return self.decoder_block(x, mask, is_causal, past_kv, return_kv)


if __name__ == "__main__":
    # Test decoder block
    d_model = 768
    num_heads = 12
    num_kv_heads = 4  # GQA: 3 query heads per KV head
    d_ff = 2048
    batch_size = 2
    seq_length = 64
    dropout = 0.1
    
    print("Testing Decoder Block")
    print("=" * 60)
    
    # Create decoder block
    decoder_block = DecoderBlockModule(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_length, d_model)
    output, past_kv, attn_weights = decoder_block(x, is_causal=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in decoder_block.parameters()):,}")
    
    # Parameter breakdown
    print(f"\nParameter breakdown:")
    print(f"  Attention: {sum(p.numel() for n, p in decoder_block.named_parameters() if 'attention' in n):,}")
    print(f"  FFN: {sum(p.numel() for n, p in decoder_block.named_parameters() if 'ffn' in n):,}")
    print(f"  Norm: {sum(p.numel() for n, p in decoder_block.named_parameters() if 'norm' in n):,}")
    
    if attn_weights is not None:
        print(f"\nAttention weights shape: {attn_weights.shape}")
    
    # Test with KV caching
    print("\nKV Caching Test:")
    print("-" * 60)
    
    # First pass
    x1 = torch.randn(batch_size, 10, d_model)
    output1, kv1, _ = decoder_block(x1, is_causal=True, return_kv=True)
    print(f"First pass - Input: {x1.shape}, Output: {output1.shape}")
    
    # Second pass with caching
    x2 = torch.randn(batch_size, 5, d_model)
    output2, kv2, _ = decoder_block(x2, is_causal=True, past_kv=kv1, return_kv=True)
    print(f"Second pass - Input: {x2.shape}, Output: {output2.shape}")
    print(f"KV cache 1 - Key shape: {kv1[0].shape}, Value shape: {kv1[1].shape}")
    print(f"KV cache 2 - Key shape: {kv2[0].shape}, Value shape: {kv2[1].shape}")
    
    # Test with Flash Attention
    from core.attention.GroupQueryAttention import FLASH_ATTN_AVAILABLE
    print(f"\nFlash Attention Available: {FLASH_ATTN_AVAILABLE}")
