# file: GroupQueryAttention.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from core.Embedding import get_tokenizer, tokenize_batch, TokenEmbeddingModule


class GroupQueryAttention(LightningModule):
    """
    Group Query Attention (GQA).

    Instead of each query head having its own KV head (standard MHA),
    multiple query heads share a single KV head.

    - num_heads: number of query heads
    - num_kv_heads: number of key/value heads (must divide num_heads evenly)
      * num_kv_heads == num_heads  → standard Multi-Head Attention
      * num_kv_heads == 1          → Multi-Query Attention (MQA)
      * 1 < num_kv_heads < num_heads → Group Query Attention (GQA)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads  # how many Q heads share one KV head
        self.causal = causal

        # Query projection: full num_heads
        self.W_q = nn.Linear(d_model, num_heads * self.d_k)
        # Key and Value projections: only num_kv_heads
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
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
        kv = kv if kv is not None else x
        L_kv = kv.size(1)

        # 1. Linear projections
        Q = self.W_q(x)   # (B, L, num_heads * d_k)
        K = self.W_k(kv)  # (B, L_kv, num_kv_heads * d_k)
        V = self.W_v(kv)  # (B, L_kv, num_kv_heads * d_k)

        # 2. Reshape into heads
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)         # (B, num_heads, L, d_k)
        K = K.view(B, L_kv, self.num_kv_heads, self.d_k).transpose(1, 2)   # (B, num_kv_heads, L_kv, d_k)
        V = V.view(B, L_kv, self.num_kv_heads, self.d_k).transpose(1, 2)   # (B, num_kv_heads, L_kv, d_k)

        # 3. Expand KV heads to match query heads by repeating
        #    Each KV head is shared by (num_queries_per_kv) query heads
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_heads, L_kv, d_k)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_heads, L_kv, d_k)

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

    # GQA: 8 query heads, 2 KV heads → each KV head shared by 4 query heads
    gqa = GroupQueryAttention(d_model=256, num_heads=8, num_kv_heads=2, causal=False)
    out, attn = gqa(embeddings, attention_mask)

    print("Output shape:", out.shape)      # torch.Size([2, seq_len, 256])
    print("Attention shape:", attn.shape)   # torch.Size([2, 8, seq_len, seq_len])

    # Compare parameter counts
    from core.attention.MultiHeadSelfAttention import MultiHeadSelfAttention
    mha = MultiHeadSelfAttention(d_model=256, num_heads=8)
    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    print(f"\nMHA params: {mha_params:,}")
    print(f"GQA params: {gqa_params:,}")
    print(f"GQA saves {mha_params - gqa_params:,} params ({(1 - gqa_params/mha_params)*100:.1f}% reduction in KV projection)")
