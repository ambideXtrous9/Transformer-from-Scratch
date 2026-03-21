# file: MultiHeadLatentAttention.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from core.Embedding import get_tokenizer, tokenize_batch, TokenEmbeddingModule


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadLatentAttention(LightningModule):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2 (2024).

    Key idea: compress KV into a low-rank latent representation to
    drastically reduce KV cache size during inference.

    - Q path:  x → W_dq (down-project to d_compress) → RMSNorm → W_uq (up-project to num_heads * d_k)
    - KV path: x → W_dkv (down-project to d_compress) → RMSNorm → W_uk / W_uv (up-project to num_heads * d_k each)

    During inference, only the compressed latent c_kv (of dimension d_compress)
    needs to be cached instead of the full K and V tensors.

    - d_compress == d_model → no compression (similar to standard MHA)
    - d_compress << d_model → significant KV cache reduction
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_compress: int = 64,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_compress = d_compress
        self.causal = causal

        # Q path: down-project → RMSNorm → up-project
        self.W_dq = nn.Linear(d_model, d_compress)
        self.q_norm = RMSNorm(d_compress)
        self.W_uq = nn.Linear(d_compress, num_heads * self.d_k)

        # KV path: down-project → RMSNorm → up-project K and V
        self.W_dkv = nn.Linear(d_model, d_compress)
        self.kv_norm = RMSNorm(d_compress)
        self.W_uk = nn.Linear(d_compress, num_heads * self.d_k)
        self.W_uv = nn.Linear(d_compress, num_heads * self.d_k)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, kv: torch.Tensor = None):
        """
        x: (B, L, d_model) → query source
        kv: (B, L_kv, d_model) → optional key/value source (for cross-attention)
        mask:
            - (B, L_kv) padding mask
            - (B, 1, 1, L_kv) or (B, 1, L, L_kv) already broadcast
        """
        B, L, _ = x.size()
        kv_input = kv if kv is not None else x
        L_kv = kv_input.size(1)

        # 1. Q path: down-project → RMSNorm → up-project
        c_q = self.W_dq(x)          # (B, L, d_compress)
        c_q = self.q_norm(c_q)      # (B, L, d_compress)
        Q = self.W_uq(c_q)          # (B, L, num_heads * d_k)

        # 2. KV path: down-project → RMSNorm → up-project K and V
        c_kv = self.W_dkv(kv_input)  # (B, L_kv, d_compress) ← this is what gets cached
        c_kv = self.kv_norm(c_kv)    # (B, L_kv, d_compress)
        K = self.W_uk(c_kv)          # (B, L_kv, num_heads * d_k)
        V = self.W_uv(c_kv)         # (B, L_kv, num_heads * d_k)

        # 3. Split into heads
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)         # (B, num_heads, L, d_k)
        K = K.view(B, L_kv, self.num_heads, self.d_k).transpose(1, 2)     # (B, num_heads, L_kv, d_k)
        V = V.view(B, L_kv, self.num_heads, self.d_k).transpose(1, 2)     # (B, num_heads, L_kv, d_k)

        # 4. Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, num_heads, L, L_kv)

        # ---- Padding mask ----
        if mask is not None:
            if mask.dim() == 2:   # (B, L_kv)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_kv)
            elif mask.dim() == 3:  # (B, 1, L_kv)
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # ---- Causal mask (decoder) ----
        if self.causal:
            causal_mask = torch.tril(torch.ones(L, L_kv, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1,1,L,L_kv)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # 5. Softmax + dropout
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # 6. Weighted sum
        out = torch.matmul(attn_weights, V)  # (B, num_heads, L, d_k)

        # 7. Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, d_model)
        out = self.W_o(out)  # final linear projection

        return out, attn_weights


if __name__ == "__main__":

    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    texts = ["Hello world", "This is a longer example for embedding demo."]
    batch = tokenize_batch(tokenizer, texts, max_length=32)
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

    print("Input IDs:", input_ids.shape)
    print("Attention mask:", attention_mask.shape)

    embed_module = TokenEmbeddingModule(
        vocab_size=vocab_size,
        d_model=256,
        max_positions=32,
        pad_token_id=pad_id,
        use_sinusoidal_pos=True,
    )

    embeddings = embed_module(input_ids, attention_mask)
    print("Embeddings:", embeddings.shape)

    # MLA: 8 heads, d_compress=64 (low-rank latent dimension)
    mla = MultiHeadLatentAttention(d_model=256, num_heads=8, d_compress=64, causal=False)
    out, attn = mla(embeddings, attention_mask)

    print("Output shape:", out.shape)      # torch.Size([2, seq_len, 256])
    print("Attention shape:", attn.shape)   # torch.Size([2, 8, seq_len, seq_len])

    # Compare parameter counts
    from core.attention.MultiHeadSelfAttention import MultiHeadSelfAttention
    mha = MultiHeadSelfAttention(d_model=256, num_heads=8)
    mha_params = sum(p.numel() for p in mha.parameters())
    mla_params = sum(p.numel() for p in mla.parameters())
    print(f"\nMHA params: {mha_params:,}")
    print(f"MLA params: {mla_params:,}")
    print(f"MLA saves {mha_params - mla_params:,} params ({(1 - mla_params/mha_params)*100:.1f}% reduction)")

    # KV cache size comparison (per token, per layer)
    seq_len = input_ids.size(1)
    mha_kv_cache = 2 * 256 * seq_len  # K + V, each (seq_len, d_model)
    mla_kv_cache = 64 * seq_len       # only c_kv (seq_len, d_compress)
    print(f"\nKV cache per layer (sequence length={seq_len}):")
    print(f"  MHA: {mha_kv_cache:,} floats (K + V)")
    print(f"  MLA: {mla_kv_cache:,} floats (compressed latent c_kv only)")
    print(f"  MLA KV cache is {mla_kv_cache/mha_kv_cache*100:.1f}% of MHA ({mha_kv_cache//mla_kv_cache}x smaller)")
