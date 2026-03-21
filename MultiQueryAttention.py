# file: MultiQueryAttention.py
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from Embedding import get_tokenizer, tokenize_batch, TokenEmbeddingModule


class MultiQueryAttention(LightningModule):
    """
    Multi-Query Attention (MQA).

    From "Fast Transformer Decoding" (Shazeer, 2019).

    All query heads share a SINGLE Key head and a SINGLE Value head.
    This drastically reduces the KV cache size during autoregressive
    inference while retaining most of the representational power of
    standard Multi-Head Attention.

    - num_heads query heads, each with dimension d_k
    - 1 key head, 1 value head (shared across all query heads)
    - K and V are expanded via unsqueeze + expand (no memory copy)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        # Query projection: full num_heads
        self.W_q = nn.Linear(d_model, num_heads * self.d_k)
        # Key and Value projections: single head
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
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
        K = self.W_k(kv)  # (B, L_kv, d_k)
        V = self.W_v(kv)  # (B, L_kv, d_k)

        # 2. Reshape into heads
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        K = K.unsqueeze(1)  # (B, 1, L_kv, d_k)
        V = V.unsqueeze(1)  # (B, 1, L_kv, d_k)

        # 3. Expand single KV head to match query heads (memory-efficient, no copy)
        K = K.expand(B, self.num_heads, L_kv, self.d_k)  # (B, num_heads, L_kv, d_k)
        V = V.expand(B, self.num_heads, L_kv, self.d_k)  # (B, num_heads, L_kv, d_k)

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

    # MQA: 8 query heads, 1 shared KV head
    mqa = MultiQueryAttention(d_model=256, num_heads=8, causal=False)
    out, attn = mqa(embeddings, attention_mask)

    print("Output shape:", out.shape)      # torch.Size([2, seq_len, 256])
    print("Attention shape:", attn.shape)   # torch.Size([2, 8, seq_len, seq_len])

    # Compare parameter counts with MHA
    from MultiHeadSelfAttention import MultiHeadSelfAttention
    mha = MultiHeadSelfAttention(d_model=256, num_heads=8)
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())
    print(f"\nMHA params: {mha_params:,}")
    print(f"MQA params: {mqa_params:,}")
    print(f"MQA saves {mha_params - mqa_params:,} params ({(1 - mqa_params/mha_params)*100:.1f}% reduction in KV projection)")
