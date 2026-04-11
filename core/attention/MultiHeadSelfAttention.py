# file: mhsa.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


import torch.nn as nn
# Removed LightningModule import - not needed for a standalone module
from core.Embedding import get_tokenizer, tokenize_batch, TokenEmbeddingModule


class MultiHeadSelfAttention(nn.Module):  # ✅ Changed to nn.Module
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
        # ✅ Precompute scale factor for efficiency
        self.register_buffer("scale", torch.tensor(self.d_k ** -0.5))
        
        # ✅ Optional: cache for causal mask to avoid reallocation
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, kv: torch.Tensor = None, return_attn: bool = False):
        """
        x: (B, L, d_model) → query
        kv: (B, L_kv, d_model) → optional key/value (for cross-attention)
        mask: 
            - (B, L_kv) padding mask (most common)
            - (B, 1, 1, L_kv) or (B, 1, L, L_kv) already broadcast
        return_attn: if True, returns (output, attention_weights); else just output
        """
        B, L, _ = x.size()
        kv = kv if kv is not None else x
        L_kv = kv.size(1)

        # 1. Linear projections
        Q = self.W_q(x)       # (B, L, d_model)
        K = self.W_k(kv)      # (B, L_kv, d_model)
        V = self.W_v(kv)      # (B, L_kv, d_model)

        # 2. Split heads
        def split_heads(tensor, seq_len):
            return tensor.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, seq_len, d_k)

        Q = split_heads(Q, L)
        K = split_heads(K, L_kv)
        V = split_heads(V, L_kv)

        # 3. Scaled dot-product attention
        # ✅ Use precomputed scale
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, L, L_kv)

        # ---- Padding mask ----
        if mask is not None:
            if mask.dim() == 2:  # (B, L_kv) - most common case
                mask = mask.unsqueeze(1).unsqueeze(2)  # → (B, 1, 1, L_kv)
            elif mask.dim() == 3:  # (B, 1, L_kv) or (B, L, L_kv)
                if mask.size(1) == 1:
                    mask = mask.unsqueeze(2)  # → (B, 1, 1, L_kv)
                # else assume (B, L, L_kv) already compatible
            # 4D masks assumed already correctly broadcasted
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # ---- Causal mask (decoder) ----
        if self.causal:
            # ✅ Safety check: causal mask only valid for self-attention
            if kv is not x and L != L_kv:
                raise ValueError(
                    f"Causal masking requires self-attention (kv=None or L==L_kv). "
                    f"Got L={L}, L_kv={L_kv}"
                )
            # ✅ Cache or create causal mask
            if self.causal_mask is None or self.causal_mask.size(-2) < L or self.causal_mask.size(-1) < L_kv:
                mask_template = torch.tril(torch.ones(L, L_kv, device=x.device))
                self.causal_mask = mask_template.unsqueeze(0).unsqueeze(0)  # (1,1,L,L_kv)
            # Slice to actual sequence lengths (handles variable-length batches)
            causal_mask = self.causal_mask[:, :, :L, :L_kv]
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # 4. Softmax + dropout
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # 5. Weighted sum
        out = torch.matmul(attn_weights, V)  # (B, num_heads, L, d_k)

        # 6. Concatenate heads + final projection
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, d_model)
        out = self.W_o(out)

        # ✅ Return attention weights only if requested
        if return_attn:
            return out, attn_weights
        return out


if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    texts = ["Hello world", "This is a longer example for embedding demo."]
    batch = tokenize_batch(tokenizer, texts, max_length=32)
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

    print("\nInput IDs:", input_ids.shape)
    print("\nAttention mask:", attention_mask.shape)

    embed_module = TokenEmbeddingModule(
        vocab_size=vocab_size, 
        d_model=256, 
        max_positions=32,
        pad_token_id=pad_id, 
        use_sinusoidal_pos=True,
    )

    embeddings = embed_module(input_ids, attention_mask)
    print("\nEmbeddings Shape:", embeddings.shape)

    mhsa = MultiHeadSelfAttention(d_model=256, num_heads=8, causal=False)
    
    # ✅ Test without attention return
    out = mhsa(embeddings, attention_mask)
    print("\nOutput shape:", out.shape)
    
    # ✅ Test with attention return
    out, attn = mhsa(embeddings, attention_mask, return_attn=True)
    print("\nAttention shape:", attn.shape)
    
    # ✅ Test causal mode
    mhsa_causal = MultiHeadSelfAttention(d_model=256, num_heads=8, causal=True)
    out_causal = mhsa_causal(embeddings, attention_mask)
    print("\nCausal output shape:", out_causal.shape)