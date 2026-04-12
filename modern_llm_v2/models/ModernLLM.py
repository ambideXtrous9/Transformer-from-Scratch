"""
Modern Decoder-Only Language Model (200M Parameters)

Complete implementation of a production-level LLM with:
- Token embeddings with optional scaling
- RoPE (Rotary Position Embedding)
- Multiple Decoder Blocks with GQA
- SwiGLU Feed-Forward Networks
- RMSNorm (pre-norm architecture)
- Flash Attention support
- KV caching for efficient inference

Architecture matches modern LLMs like LLaMA-2, Mistral, etc.

Parameter Count: ~200M
- Embedding: 50257 * 768 = 38.6M
- 12x Decoder Blocks: ~115M
  - Attention (per layer): ~9.4M
  - FFN SwiGLU (per layer): ~9.4M
- Output projection: 768 * 50257 = 38.6M
- Total: ~200M
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple, List

from core.TokenEmbedding import TokenEmbedding
from models.DecoderBlock import DecoderBlock
from core.Normalization import RMSNorm


class ModernDecoderOnlyModel(pl.LightningModule):
    """
    Modern Decoder-Only Language Model
    
    Production-level implementation following the architecture of:
    - LLaMA / LLaMA-2
    - Mistral
    - GPT-Neo / GPT-J
    
    Features:
    - Pre-norm architecture for training stability
    - Grouped Query Attention (GQA) for efficiency
    - RoPE for positional encoding
    - SwiGLU FFN for better performance
    - RMSNorm for efficient normalization
    - Flash Attention support (if available)
    - KV caching for fast generation
    """
    
    def __init__(self, vocab_size: int = 50257, d_model: int = 768,
                 num_heads: int = 12, num_kv_heads: int = 4,
                 num_layers: int = 12, d_ff: int = 2048,
                 max_seq_length: int = 1024, dropout: float = 0.1,
                 use_flash_attention: bool = True, attention_bias: bool = False,
                 ffn_bias: bool = False, rope_theta: float = 10000.0,
                 norm_eps: float = 1e-5, scale_embeddings: bool = False,
                 tie_word_embeddings: bool = False, pad_token_id: int = 50256):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (hidden size)
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            num_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            use_flash_attention: Use Flash Attention if available
            attention_bias: Use bias in attention projections
            ffn_bias: Use bias in FFN projections
            rope_theta: RoPE base frequency
            norm_eps: RMSNorm epsilon
            scale_embeddings: Scale embeddings by sqrt(d_model)
            tie_word_embeddings: Share input/output embeddings
            pad_token_id: Padding token ID
        """
        super().__init__()
        self.save_hyperparameters(ignore=["pad_token_id"])
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            scale_embeddings=scale_embeddings
        )
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_length=max_seq_length,
                use_flash_attention=use_flash_attention,
                attention_bias=attention_bias,
                ffn_bias=ffn_bias,
                rope_theta=rope_theta,
                norm_eps=norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = RMSNorm(d_model, eps=norm_eps)
        
        # Output projection (language model head)
        if tie_word_embeddings:
            # Share weights with input embedding
            self.lm_head = self.token_embedding.embedding
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights.
        
        Uses truncated normal initialization for embeddings and
        small normal initialization for other weights.
        """
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        if not self.hparams.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (B, L)
            attention_mask: Padding mask of shape (B, L), 1 for valid, 0 for padding
            is_causal: Use causal masking
            past_key_values: Optional cached KV tuples from previous steps
            use_cache: Return KV cache for faster generation
        
        Returns:
            logits: Logits of shape (B, L, vocab_size)
            past_key_values: Optional cached KV tuples
        """
        B, L = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids)  # (B, L, d_model)
        x = self.dropout(x)
        
        # Prepare KV cache
        past_key_values = past_key_values or [None] * len(self.layers)
        new_past_key_values = [] if use_cache else None
        
        # Apply decoder blocks
        all_attention_weights = []
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            x, layer_past_kv, attn_weights = layer(
                x, mask=attention_mask, is_causal=is_causal,
                past_kv=past_kv, return_kv=use_cache
            )
            
            if use_cache:
                new_past_key_values.append(layer_past_kv)
            
            if attn_weights is not None:
                all_attention_weights.append(attn_weights)
        
        # Final normalization
        x = self.norm(x)  # (B, L, d_model)
        
        # Output projection
        logits = self.lm_head(x)  # (B, L, vocab_size)
        
        return logits, new_past_key_values
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                 repetition_penalty: float = 1.0, do_sample: bool = True,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using greedy/beam search or sampling.
        
        Args:
            input_ids: Input token IDs (B, L)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (0 to disable)
            top_p: Nucleus sampling (top-p) threshold
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            do_sample: Use sampling (False = greedy)
            eos_token_id: End-of-sequence token ID
        
        Returns:
            Generated token IDs (B, L + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            past_key_values = None
            generated = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Forward pass with KV caching
                logits, past_key_values = self(
                    generated if past_key_values is None else generated[:, -1:],
                    is_causal=True,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Get logits for last token
                next_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(generated.shape[0]):
                        for token_id in set(generated[i].tolist()):
                            if next_logits[i, token_id] < 0:
                                next_logits[i, token_id] *= repetition_penalty
                            else:
                                next_logits[i, token_id] /= repetition_penalty
                
                # Top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits.scatter_(-1, top_k_indices, top_k_values)
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Scatter back to original order
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample or greedy
                if do_sample:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break
            
            return generated
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        
        # Forward pass
        logits, _ = self(input_ids, attention_mask=attention_mask, is_causal=True)
        
        # Shift logits and labels for next token prediction
        # logits: (B, L, vocab_size) -> (B*L, vocab_size)
        # labels: (B, L) -> (B*L)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1)
        )
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        
        # Forward pass
        logits, _ = self(input_ids, attention_mask=attention_mask, is_causal=True)
        
        # Shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1)
        )
        
        # Compute perplexity
        # Proper token-weighted NLL calculation
        valid_mask = shift_labels != -100
        valid_tokens = valid_mask.sum()
        
        if valid_tokens > 0:
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            # Gather the log probabilities of the target tokens
            target_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Mask and sum
            total_nll = -(target_log_probs * valid_mask).sum()
        else:
            total_nll = torch.tensor(0.0, device=loss.device)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_total_nll", total_nll, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_valid_tokens", valid_tokens.float(), on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics."""
        # Get accumulated metrics
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_total_nll = self.trainer.callback_metrics.get("val_total_nll")
        val_valid_tokens = self.trainer.callback_metrics.get("val_valid_tokens")
        
        if val_valid_tokens is not None and val_valid_tokens > 0:
            perplexity = torch.exp(val_total_nll / val_valid_tokens)
            self.log("val_perplexity", perplexity, logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizer with warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.get("learning_rate", 3e-4),
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Learning rate scheduler with warmup
        # This will be configured in the trainer script
        return optimizer
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def print_model_info(self):
        """Print model information."""
        print("\n" + "=" * 80)
        print("Modern LLM Architecture (Decoder-Only)")
        print("=" * 80)
        print(f"Vocabulary size: {self.hparams.vocab_size:,}")
        print(f"Model dimension (d_model): {self.hparams.d_model}")
        print(f"Number of heads: {self.hparams.num_heads}")
        print(f"Number of KV heads: {self.hparams.num_kv_heads}")
        print(f"Number of layers: {self.hparams.num_layers}")
        print(f"FFN dimension: {self.hparams.d_ff:,}")
        print(f"Max sequence length: {self.hparams.max_seq_length:,}")
        print(f"Dropout: {self.hparams.dropout}")
        print(f"RoPE theta: {self.hparams.rope_theta}")
        print(f"\nTotal parameters: {self.get_num_params():,}")
        print(f"Embedding parameters: {self.token_embedding.embedding.weight.numel():,}")
        print(f"Output projection parameters: {self.lm_head.weight.numel():,}")
        print(f"Parameters per layer: {sum(p.numel() for p in self.layers[0].parameters()):,}")
        print(f"Total decoder layer parameters: {sum(p.numel() for p in self.layers.parameters()):,}")
        print("=" * 80)


if __name__ == "__main__":
    # Test the complete model
    print("Testing Modern Decoder-Only Model (200M Parameters)")
    print("=" * 80)
    
    # Import config
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config_200m as config
    
    # Create model with 200M parameters
    model = ModernDecoderOnlyModel(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_kv_heads=config.NUM_KV_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        max_seq_length=config.MAX_LENGTH,
        dropout=config.DROPOUT,
        scale_embeddings=True,
        tie_word_embeddings=config.TIE_WORD_EMBEDDINGS
    )
    
    # Print model info
    model.print_model_info()
    
    # Test forward pass
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_length))
    labels = input_ids.clone()
    
    print(f"\nForward Pass Test:")
    print(f"  Input shape: {input_ids.shape}")
    
    logits, past_key_values = model(input_ids, is_causal=True)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_length}, 50257)")
    
    # Test generation
    print(f"\nGeneration Test:")
    prompt = torch.randint(0, 50257, (1, 10))
    print(f"  Prompt shape: {prompt.shape}")
    
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    print(f"  Generated shape: {generated.shape}")
    print(f"  Expected shape: (1, {10 + 20})")
    
    # Test with attention mask
    print(f"\nAttention Mask Test:")
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    attention_mask[:, -10:] = False  # Mask last 10 tokens
    logits_masked, _ = model(input_ids, attention_mask=attention_mask, is_causal=True)
    print(f"  Masked logits shape: {logits_masked.shape}")
    
    # Flash Attention status
    from core.attention.GroupQueryAttention import FLASH_ATTN_AVAILABLE
    print(f"\nFlash Attention Available: {FLASH_ATTN_AVAILABLE}")
