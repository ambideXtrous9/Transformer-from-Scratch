"""
Normalization Modules

Modern implementation with:
- RMSNorm (Root Mean Square Layer Normalization) - Default for modern LLMs
- AddNorm (Residual connection + Dropout + Normalization)

RMSNorm is used in LLaMA, PaLM, and other modern architectures as it's
more computationally efficient than LayerNorm while maintaining performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytorch_lightning as pl


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    Simplified alternative to LayerNorm that only uses RMS statistic
    without mean and variance normalization.
    
    Formula: y = (x / RMS(x)) * gamma, where RMS(x) = sqrt(mean(x^2) + eps)
    
    Used in: LLaMA, PaLM, T5, and other modern architectures.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Args:
            d_model: Model dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        
        # Learnable scaling parameter (no bias in RMSNorm)
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor of shape (B, L, d_model) or (B, d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Keep dimension for broadcasting
        variance = x.pow(2).mean(dim=-1, keepdim=True)  # (B, L, 1) or (B, 1)
        x_normalized = x * torch.rsqrt(variance + self.eps)  # (B, L, d_model)
        
        # Apply learnable scaling
        return self.weight * x_normalized  # (B, L, d_model)


class AddNorm(pl.LightningModule):
    """
    Add & Norm layer with dropout.
    
    Implements residual connection followed by dropout and normalization.
    Supports both LayerNorm and RMSNorm.
    
    Usage:
        output = add_norm(x, sublayer_output)
        where sublayer_output is the result of attention/FFN
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, 
                 norm_type: str = "rms", eps: float = 1e-5):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            norm_type: Type of normalization ("rms" or "layer")
            eps: Epsilon for normalization
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.dropout = nn.Dropout(dropout)
        
        if norm_type == "rms":
            self.norm = RMSNorm(d_model, eps)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(d_model, eps=eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Apply residual connection, dropout, and normalization.
        
        Args:
            x: Input tensor (residual) of shape (B, L, d_model)
            sublayer_output: Output from sublayer of shape (B, L, d_model)
        
        Returns:
            Normalized output of shape (B, L, d_model)
        """
        # Residual connection with dropout
        residual = x + self.dropout(sublayer_output)
        
        # Normalization
        return self.norm(residual)


class PreNormDecoderBlock(pl.LightningModule):
    """
    Pre-Normalization architecture for decoder blocks.
    
    Modern LLMs use pre-norm architecture where normalization is applied
    BEFORE the sublayer (attention/FFN), which improves training stability.
    
    Architecture:
        x = x + Attention(Norm(x))  # Pre-norm attention
        x = x + FFN(Norm(x))        # Pre-norm FFN
    
    This is the standard in LLaMA, GPT-2, and other modern architectures.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1,
                 norm_type: str = "rms", eps: float = 1e-5):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            norm_type: Type of normalization
            eps: Epsilon for normalization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Pre-norm layers
        if norm_type == "rms":
            self.attn_norm = RMSNorm(d_model, eps)
            self.ffn_norm = RMSNorm(d_model, eps)
        elif norm_type == "layer":
            self.attn_norm = nn.LayerNorm(d_model, eps=eps)
            self.ffn_norm = nn.LayerNorm(d_model, eps=eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_output: torch.Tensor, 
                ffn_output: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-norm decoder block.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            attn_output: Raw attention output (before residual)
            ffn_output: Raw FFN output (before residual)
        
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        # Pre-norm attention: x = x + Dropout(Attention(Norm(x)))
        x = x + self.dropout(attn_output)
        
        # Pre-norm FFN: x = x + Dropout(FFN(Norm(x)))
        x = x + self.dropout(ffn_output)
        
        return x


if __name__ == "__main__":
    # Test normalization modules
    d_model = 768
    batch_size = 2
    seq_length = 32
    dropout = 0.1
    
    print("Testing Normalization Modules")
    print("=" * 60)
    
    # Test RMSNorm
    print("\n1. RMSNorm")
    print("-" * 60)
    rms_norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_length, d_model)
    x_normalized = rms_norm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_normalized.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in rms_norm.parameters()):,}")
    print(f"Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
    print(f"Output mean: {x_normalized.mean().item():.6f}, std: {x_normalized.std().item():.6f}")
    
    # Test AddNorm
    print("\n2. AddNorm (RMSNorm)")
    print("-" * 60)
    add_norm_rms = AddNorm(d_model, dropout, norm_type="rms")
    sublayer_output = torch.randn(batch_size, seq_length, d_model)
    output_rms = add_norm_rms(x, sublayer_output)
    
    print(f"Input shape: {x.shape}")
    print(f"Sublayer output shape: {sublayer_output.shape}")
    print(f"Output shape: {output_rms.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in add_norm_rms.parameters()):,}")
    
    # Test AddNorm with LayerNorm
    print("\n3. AddNorm (LayerNorm)")
    print("-" * 60)
    add_norm_layer = AddNorm(d_model, dropout, norm_type="layer")
    output_layer = add_norm_layer(x, sublayer_output)
    
    print(f"Output shape: {output_layer.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in add_norm_layer.parameters()):,}")
    
    # Test PreNormDecoderBlock
    print("\n4. PreNormDecoderBlock")
    print("-" * 60)
    pre_norm = PreNormDecoderBlock(d_model, dropout, norm_type="rms")
    attn_output = torch.randn(batch_size, seq_length, d_model)
    ffn_output = torch.randn(batch_size, seq_length, d_model)
    output = pre_norm(x, attn_output, ffn_output)
    
    print(f"Input shape: {x.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    print(f"FFN output shape: {ffn_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in pre_norm.parameters()):,}")
