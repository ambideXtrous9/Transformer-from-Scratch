import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from Embedding import get_tokenizer, tokenize_batch, TokenEmbeddingModule
from MultiHeadSelfAttention import MultiHeadSelfAttention
from AddNorm import AddNorm


class PositionwiseFeedForward(LightningModule):
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1, activation: str = "relu"):
        """
        Position-wise Feed Forward Network used in Transformer blocks.
        
        Args:
            d_model (int): input/output hidden size
            d_ff (int): inner feed-forward dimension
            dropout (float): dropout probability
            activation (str): activation function ("relu" or "gelu")
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


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