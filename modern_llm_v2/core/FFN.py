"""
Feed-Forward Network Modules

Modern implementation with:
- SwiGLU (Swish-Gated Linear Unit) - Default for modern LLMs
- GELU activation (GPT-2 style)
- ReLU activation (Original Transformer)

SwiGLU is used in LLaMA, PaLM, Mistral, and other modern architectures.
It provides better performance than standard FFNs at the cost of additional parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class SwiGLU(pl.LightningModule):
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network
    
    SwiGLU uses a gating mechanism with Swish activation:
    FFN(x) = (Swish(xW1) * (xW3))W2
    
    Where:
    - W1, W3 are "up" projections (create two representations)
    - W2 is the "down" projection (combine and project back)
    - Swish(x) = x * sigmoid(x) (also known as SiLU)
    
    This is 3x the parameters of a standard FFN but provides better performance.
    Used in: LLaMA, PaLM, Mistral, and other modern LLMs.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 use_bias: bool = False):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_bias: Use bias in linear projections
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Three linear projections (SwiGLU uses 3x parameters)
        self.W_gate = nn.Linear(d_model, d_ff, bias=use_bias)  # Gate projection (W3)
        self.W_up = nn.Linear(d_model, d_ff, bias=use_bias)    # Up projection (W1)
        self.W_down = nn.Linear(d_ff, d_model, bias=use_bias)  # Down projection (W2)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.W_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_down.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU FFN.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
        
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        # Gate and up projections
        gate = self.W_gate(x)  # (B, L, d_ff)
        up = self.W_up(x)      # (B, L, d_ff)
        
        # Swish gating: Swish(x) = x * sigmoid(x)
        # F.silu() is equivalent to x * sigmoid(x)
        gated = F.silu(gate) * up  # (B, L, d_ff)
        
        # Apply dropout and down projection
        gated = self.dropout(gated)
        output = self.W_down(gated)  # (B, L, d_model)
        
        return output


class PositionwiseFeedForward(pl.LightningModule):
    """
    Standard Position-wise Feed-Forward Network
    
    FFN(x) = Activation(xW1 + b1)W2 + b2
    
    Supports multiple activation functions:
    - GELU (GPT-2 style)
    - ReLU (Original Transformer)
    - SiLU/Swish
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu", use_bias: bool = False):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function ("gelu", "relu", "silu")
            use_bias: Use bias in linear projections
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.W1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.W2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Select activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W2.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFN.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
        
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        # Up projection with activation
        hidden = self.activation(self.W1(x))  # (B, L, d_ff)
        hidden = self.dropout(hidden)
        
        # Down projection
        output = self.W2(hidden)  # (B, L, d_model)
        
        return output


class FFNFactory(pl.LightningModule):
    """
    Factory class to create the appropriate FFN type.
    
    Usage:
        ffn = FFNFactory(ffn_type="swiglu", d_model=768, d_ff=2048)
    """
    
    def __init__(self, ffn_type: str = "swiglu", d_model: int = 768,
                 d_ff: int = 2048, dropout: float = 0.1,
                 activation: str = "gelu", use_bias: bool = False):
        """
        Args:
            ffn_type: Type of FFN ("swiglu", "standard")
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function for standard FFN
            use_bias: Use bias in linear projections
        """
        super().__init__()
        self.save_hyperparameters()
        
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, dropout, use_bias)
        elif ffn_type == "standard":
            self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation, use_bias)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN."""
        return self.ffn(x)


if __name__ == "__main__":
    # Test FFN modules
    d_model = 768
    d_ff = 2048
    batch_size = 2
    seq_length = 32
    dropout = 0.1
    
    print("Testing Feed-Forward Network Modules")
    print("=" * 60)
    
    # Test SwiGLU
    print("\n1. SwiGLU FFN")
    print("-" * 60)
    swiglu = SwiGLU(d_model, d_ff, dropout)
    x = torch.randn(batch_size, seq_length, d_model)
    output = swiglu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in swiglu.parameters()):,}")
    print(f"Parameter breakdown:")
    print(f"  W_gate: {swiglu.W_gate.weight.numel():,} ({d_model} x {d_ff})")
    print(f"  W_up: {swiglu.W_up.weight.numel():,} ({d_model} x {d_ff})")
    print(f"  W_down: {swiglu.W_down.weight.numel():,} ({d_ff} x {d_model})")
    print(f"  Total: {3 * d_model * d_ff:,} (3x standard FFN)")
    
    # Test Standard FFN with GELU
    print("\n2. Standard FFN (GELU)")
    print("-" * 60)
    ffn_gelu = PositionwiseFeedForward(d_model, d_ff, dropout, activation="gelu")
    output = ffn_gelu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in ffn_gelu.parameters()):,}")
    print(f"Parameter breakdown:")
    print(f"  W1: {ffn_gelu.W1.weight.numel():,} ({d_model} x {d_ff})")
    print(f"  W2: {ffn_gelu.W2.weight.numel():,} ({d_ff} x {d_model})")
    print(f"  Total: {2 * d_model * d_ff:,} (2x d_model * d_ff)")
    
    # Test Standard FFN with ReLU
    print("\n3. Standard FFN (ReLU)")
    print("-" * 60)
    ffn_relu = PositionwiseFeedForward(d_model, d_ff, dropout, activation="relu")
    output = ffn_relu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in ffn_relu.parameters()):,}")
    
    # Compare parameter counts
    print("\n4. Parameter Comparison")
    print("-" * 60)
    print(f"SwiGLU parameters: {sum(p.numel() for p in swiglu.parameters()):,}")
    print(f"Standard FFN parameters: {sum(p.numel() for p in ffn_gelu.parameters()):,}")
    print(f"Ratio: {sum(p.numel() for p in swiglu.parameters()) / sum(p.numel() for p in ffn_gelu.parameters()):.2f}x")
