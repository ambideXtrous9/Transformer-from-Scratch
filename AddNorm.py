# file: addnorm.py
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from Embedding import get_tokenizer, tokenize_batch, TokenEmbeddingModule
from MultiHeadSelfAttention import MultiHeadSelfAttention

class AddNorm(LightningModule):
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5):
        """
        Add & Norm module from Transformers.
        
        Args:
            d_model (int): hidden dimension of the model
            dropout (float): dropout probability
            eps (float): epsilon for LayerNorm
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input tensor (residual connection base)
            sublayer_out: (B, L, d_model) output from MHSA or FFN
        Returns:
            (B, L, d_model) after Add + Dropout + Norm
        """
        return self.layer_norm(x + self.dropout(sublayer_out))


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


