import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import Optional

from Embedding import TokenEmbeddingModule, get_tokenizer, tokenize_batch
from MultiHeadSelfAttention import MultiHeadSelfAttention
from AddNorm import AddNorm
from FFN import PositionwiseFeedForward



# ---------------- Encoder Block ----------------
class EncoderBlock(LightningModule):
    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, causal=False)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, activation="gelu")
        self.addnorm2 = AddNorm(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # MHSA + Add&Norm
        mhsa_out, attn_weights = self.mhsa(x, mask)
        x = self.addnorm1(x, mhsa_out)

        # FFN + Add&Norm
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)

        return x, attn_weights


# ---------------- Encoder (with embedding) ----------------
class Encoder(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        max_positions: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_token_id: Optional[int] = None,
        use_sinusoidal_pos: bool = True,
    ):
        super().__init__()
        # 1. Embedding module
        self.embedding = TokenEmbeddingModule(
            vocab_size=vocab_size,
            d_model=d_model,
            max_positions=max_positions,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_sinusoidal_pos=use_sinusoidal_pos
        )

        # 2. Encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 3. Final LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 1. Get embeddings
        x = self.embedding(input_ids, attention_mask)  # (B, L, d_model)

        # 2. Pass through stacked EncoderBlocks
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, attention_mask)
            attn_maps.append(attn)

        # 3. Final LayerNorm
        x = self.norm(x)

        return x, attn_maps



# ---------- Usage Example ----------
if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    texts = ["Hello world", "This is a longer example for embedding demo."]
    batch = tokenize_batch(tokenizer, texts, max_length=32)
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

    print("Input IDs:", input_ids.shape)       # torch.Size([2, seq_len])
    print("Attention mask:", attention_mask.shape)  # torch.Size([2, seq_len])

    embed_module = TokenEmbeddingModule(
        vocab_size=vocab_size, 
        d_model=256, 
        max_positions=32,
        pad_token_id=pad_id, 
        use_sinusoidal_pos=True,
    )

    embeddings = embed_module(input_ids, attention_mask)
    print("Embeddings:", embeddings.shape)  # torch.Size([2, seq_len, 256])

    mhsa = MultiHeadSelfAttention(d_model=256, num_heads=8, causal=False)
    out, attn = mhsa(embeddings, attention_mask)

    print("Output shape:", out.shape)       # torch.Size([2, seq_len, 256])
    print("Attention shape:", attn.shape)   # torch.Size([2, 8, seq_len, seq_len])

    addnorm = AddNorm(d_model=256, dropout=0.1)
    addnorm_out = addnorm(embeddings, out)
    print("AddNorm output shape:", addnorm_out.shape)  # torch.Size([2, seq_len, 256])

    ffn = PositionwiseFeedForward(d_model=256, d_ff=1024, dropout=0.1, activation="gelu")
    ffn_out = ffn(addnorm_out)
    print("FFN output shape:", ffn_out.shape)  # torch.Size([2, seq_len, 256])

    addnorm = AddNorm(d_model=256, dropout=0.1)
    addnorm_out = addnorm(addnorm_out, ffn_out)
    print("AddNorm output shape:", addnorm_out.shape)  # torch.Size([2, seq_len, 256])
    
    encoder = Encoder(vocab_size=vocab_size, 
                    d_model=256, 
                    max_positions=32, 
                    num_layers=6, 
                    num_heads=8, 
                    d_ff=1024, 
                    dropout=0.1, 
                    pad_token_id=pad_id, 
                    use_sinusoidal_pos=True)

    out, attn_maps = encoder(input_ids, attention_mask)
    print("Encoder output shape:", out.shape)  # torch.Size([2, seq_len, 256])
    print("Attention maps shape:", [attn.shape for attn in attn_maps])  # torch.Size([2, 8, seq_len, seq_len])