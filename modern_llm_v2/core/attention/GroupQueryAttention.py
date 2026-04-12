"""
Multi-Head Attention with Grouped Query Attention (GQA)

Modern implementation with:
- Grouped Query Attention (GQA) - Default for efficiency
- Flash Attention 2 support (if available)
- RoPE (Rotary Position Embedding) integration
- Causal masking for decoder-only architectures
- KV caching for efficient inference

GQA provides a good trade-off between MHA quality and MQA efficiency.
Used in: LLaMA-2, Mistral, and other modern LLMs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from typing import Optional, Tuple

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[WARNING] Flash Attention not available. Using standard attention.")


class GroupQueryAttention(pl.LightningModule):
    """
    Grouped Query Attention (GQA)
    
    GQA uses multiple query heads but fewer key-value heads, which are shared
    across query heads. This provides a good trade-off between:
    - Multi-Head Attention (MHA): num_kv_heads = num_heads
    - Multi-Query Attention (MQA): num_kv_heads = 1
    
    Example: num_heads=32, num_kv_heads=8 means each KV head is shared by 4 query heads.
    
    References:
    - "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    - Used in LLaMA-2, Mistral, etc.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, 
                 dropout: float = 0.1, max_seq_length: int = 1024,
                 use_flash_attention: bool = True, attention_bias: bool = False,
                 rope_theta: float = 10000.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads (must divide num_heads)
            dropout: Attention dropout probability
            max_seq_length: Maximum sequence length for RoPE
            use_flash_attention: Use Flash Attention if available
            attention_bias: Use bias in linear projections
            rope_theta: RoPE base frequency
        """
        super().__init__()
        self.save_hyperparameters()
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.rope_theta = rope_theta
        
        # Precompute RoPE frequencies
        self._init_rope()
        
        # Linear projections (no bias in modern architectures)
        self.W_q = nn.Linear(d_model, d_model, bias=attention_bias)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=attention_bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=attention_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=attention_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_rope(self):
        """Initialize RoPE frequency tensor."""
        # Compute frequency bands
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.hparams.max_seq_length if hasattr(self, 'hparams') else 1024, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_v.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02)
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            q: Query tensor of shape (B, num_heads, L, head_dim)
            k: Key tensor of shape (B, num_kv_heads, L, head_dim)
        
        Returns:
            q_rotated, k_rotated
        """
        seq_len = q.shape[2]
        freqs_cis = self.freqs_cis[:seq_len]  # (L, head_dim/2)
        
        # Reshape for complex multiplication
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        
        # Broadcast freqs_cis: (1, 1, L, head_dim/2)
        freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
        
        # Apply rotation
        q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(3)
        k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(3)
        
        return q_rotated.type_as(q), k_rotated.type_as(k)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match query heads.
        
        Args:
            x: Key or value tensor of shape (B, num_kv_heads, L, head_dim)
        
        Returns:
            Repeated tensor of shape (B, num_heads, L, head_dim)
        """
        if self.num_queries_per_kv == 1:
            return x
        
        B, num_kv_heads, L, head_dim = x.shape
        
        # Repeat via interpolation: (B, num_kv_heads, L, head_dim) -> (B, num_heads, L, head_dim)
        x = x[:, :, None, :, :].expand(B, num_kv_heads, self.num_queries_per_kv, L, head_dim)
        x = x.reshape(B, num_kv_heads * self.num_queries_per_kv, L, head_dim)
        
        return x
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal (autoregressive) mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Causal mask tensor
        """
        # Lower triangular matrix: 1 for allowed, -inf for masked
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        return mask
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_kv: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            mask: Optional padding mask (B, L) - True for valid tokens
            is_causal: Use causal masking
            past_kv: Optional (key, value) from previous steps for caching
            return_kv: Return key-value for caching
        
        Returns:
            output: Attention output of shape (B, L, d_model)
            past_kv: Optional cached (key, value) tuples
            attn_weights: Optional attention weights (not returned with flash attention)
        """
        B, L, _ = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x)  # (B, L, d_model)
        k = self.W_k(x)  # (B, L, num_kv_heads * head_dim)
        v = self.W_v(x)  # (B, L, num_kv_heads * head_dim)
        
        # Reshape to (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, L, head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, L, head_dim)
        
        # Apply RoPE
        q, k = self._apply_rope(q, k)  # (B, num_heads, L, head_dim), (B, num_kv_heads, L, head_dim)
        
        # KV caching
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        past_kv_out = (k, v) if return_kv else None
        
        # Repeat KV heads if needed
        k_expanded = self._repeat_kv(k)  # (B, num_heads, L_kv, head_dim)
        v_expanded = self._repeat_kv(v)  # (B, num_heads, L_kv, head_dim)
        
        # Attention computation
        if self.use_flash_attention and is_causal and mask is None:
            # Flash Attention path
            # Reshape for flash attention: (B, L, num_heads, head_dim)
            q = q.transpose(1, 2)
            k_expanded = k_expanded.transpose(1, 2)
            v_expanded = v_expanded.transpose(1, 2)
            
            # Flash Attention (automatically applies causal mask)
            output = flash_attn_func(
                q, k_expanded, v_expanded,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
            output = output.reshape(B, L, self.d_model)
            
            attn_weights = None  # Flash attention doesn't return weights
        else:
            # Standard attention path
            # Scale queries
            q = q / math.sqrt(self.head_dim)
            
            # Compute attention scores: (B, num_heads, L_q, L_kv)
            attn_scores = torch.matmul(q, k_expanded.transpose(-2, -1))
            
            # Apply causal mask
            if is_causal:
                L_kv = k_expanded.shape[2]
                causal_mask = self._create_causal_mask(L_kv, q.device)
                # Adjust for query length != key length (e.g., during generation)
                if L != L_kv:
                    causal_mask = causal_mask[-L:, :]
                attn_scores = attn_scores + causal_mask  # (B, num_heads, L, L_kv)
            
            # Apply padding mask
            if mask is not None:
                # mask: (B, L) -> (B, 1, 1, L_kv)
                mask_expanded = mask[:, None, None, :].expand_as(attn_scores)
                attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))
            
            # Softmax and dropout
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to values
            output = torch.matmul(attn_weights, v_expanded)  # (B, num_heads, L, head_dim)
            output = output.transpose(1, 2).contiguous()  # (B, L, num_heads, head_dim)
            output = output.reshape(B, L, self.d_model)
        
        # Output projection
        output = self.W_o(output)  # (B, L, d_model)
        output = self.resid_dropout(output)
        
        return output, past_kv_out, attn_weights


class MultiHeadAttention(GroupQueryAttention):
    """
    Standard Multi-Head Attention (MHA)
    
    Special case of GQA where num_kv_heads = num_heads.
    Used in original Transformer and BERT.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 max_seq_length: int = 1024, use_flash_attention: bool = True,
                 attention_bias: bool = False, rope_theta: float = 10000.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Attention dropout
            max_seq_length: Maximum sequence length
            use_flash_attention: Use Flash Attention if available
            attention_bias: Use bias in linear projections
            rope_theta: RoPE base frequency
        """
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,  # MHA: num_kv_heads = num_heads
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_flash_attention=use_flash_attention,
            attention_bias=attention_bias,
            rope_theta=rope_theta
        )


class MultiQueryAttention(GroupQueryAttention):
    """
    Multi-Query Attention (MQA)
    
    Special case of GQA where num_kv_heads = 1.
    All query heads share a single key-value head.
    Very efficient for inference but may sacrifice some quality.
    
    Used in: PaLM, Falcon
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 max_seq_length: int = 1024, use_flash_attention: bool = True,
                 attention_bias: bool = False, rope_theta: float = 10000.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            dropout: Attention dropout
            max_seq_length: Maximum sequence length
            use_flash_attention: Use Flash Attention if available
            attention_bias: Use bias in linear projections
            rope_theta: RoPE base frequency
        """
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=1,  # MQA: num_kv_heads = 1
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_flash_attention=use_flash_attention,
            attention_bias=attention_bias,
            rope_theta=rope_theta
        )


if __name__ == "__main__":
    # Test attention modules
    d_model = 768
    num_heads = 12
    num_kv_heads = 4  # GQA: 3 query heads per KV head
    batch_size = 2
    seq_length = 64
    dropout = 0.1
    
    print("Testing Attention Modules")
    print("=" * 60)
    
    # Test GQA
    print("\n1. Grouped Query Attention (GQA)")
    print("-" * 60)
    gqa = GroupQueryAttention(d_model, num_heads, num_kv_heads, dropout)
    x = torch.randn(batch_size, seq_length, d_model)
    output, past_kv, attn_weights = gqa(x, is_causal=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in gqa.parameters()):,}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test MHA
    print("\n2. Multi-Head Attention (MHA)")
    print("-" * 60)
    mha = MultiHeadAttention(d_model, num_heads, dropout)
    output, _, _ = mha(x, is_causal=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mha.parameters()):,}")
    
    # Test MQA
    print("\n3. Multi-Query Attention (MQA)")
    print("-" * 60)
    mqa = MultiQueryAttention(d_model, num_heads, dropout)
    output, _, _ = mqa(x, is_causal=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mqa.parameters()):,}")
    
    # Test with KV caching
    print("\n4. KV Caching Test")
    print("-" * 60)
    gqa = GroupQueryAttention(d_model, num_heads, num_kv_heads, dropout)
    
    # First pass
    x1 = torch.randn(batch_size, 10, d_model)
    output1, kv1, _ = gqa(x1, is_causal=True, return_kv=True)
    print(f"First pass - Input: {x1.shape}, Output: {output1.shape}")
    
    # Second pass with caching
    x2 = torch.randn(batch_size, 5, d_model)
    output2, kv2, _ = gqa(x2, is_causal=True, past_kv=kv1, return_kv=True)
    print(f"Second pass - Input: {x2.shape}, Output: {output2.shape}")
    print(f"KV cache 1 - Key shape: {kv1[0].shape}, Value shape: {kv1[1].shape}")
    print(f"KV cache 2 - Key shape: {kv2[0].shape}, Value shape: {kv2[1].shape}")
    
    # Flash Attention status
    print(f"\n5. Flash Attention Available: {FLASH_ATTN_AVAILABLE}")
