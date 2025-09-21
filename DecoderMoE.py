import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

from Embedding import TokenEmbeddingModule
from MultiHeadSelfAttention import MultiHeadSelfAttention
from AddNorm import AddNorm

# Metrics
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
from bert_score import score as bertscore

pl.seed_everything(42)

# Expert MLP same as yours
class ExpertMLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# Router: returns full probs and topk (indices + probs)
class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (B, L, d_model)
        logits = self.linear(x)  # (B, L, num_experts)
        probs = F.softmax(logits, dim=-1)  # full probs
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)  # (B, L, top_k)
        # normalize topk probs so they sum to 1 across selected experts for each token
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return probs, topk_probs, topk_indices

class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, d_ff, dropout=dropout) for _ in range(num_experts)
        ])
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x):
        """
        x: (B, L, d_model)
        returns: out (B, L, d_model)
        """
        device = x.device
        dtype = x.dtype
        B, L, D = x.size()

        # 1) routing
        full_probs, topk_probs, topk_indices = self.router(x)   # shapes: (B,L,E), (B,L,top_k), (B,L,top_k)

        # flatten token axis for easier indexing: N = B*L
        N = B * L
        x_flat = x.reshape(N, D)                                 # (N, D)
        topk_indices_flat = topk_indices.reshape(N, self.top_k)  # (N, top_k)
        topk_probs_flat = topk_probs.reshape(N, self.top_k)      # (N, top_k)

        # 2) build a full-probs tensor (N, E) where only top-k positions are nonzero
        topk_probs_full = torch.zeros((N, self.num_experts), device=device, dtype=dtype)  # (N, E)
        # scatter the topk_probs_flat into the appropriate expert positions
        topk_probs_full.scatter_(1, topk_indices_flat, topk_probs_flat)

        # 3) compute expert outputs only for tokens that route to them (sparse compute)
        # We'll create a list of (N, D) tensors, one per expert, where rows for non-selected tokens are zero.
        expert_outputs = []
        for ei in range(self.num_experts):
            # mask of shape (N,) indicating tokens that selected this expert in their top-k
            selected_mask = (topk_indices_flat == ei).any(dim=-1)  # (N,)
            if selected_mask.any():
                x_sel = x_flat[selected_mask]           # (#sel, D)
                out_sel = self.experts[ei](x_sel)       # (#sel, D)
                temp = torch.zeros_like(x_flat)         # (N, D)
                temp[selected_mask] = out_sel
            else:
                temp = torch.zeros_like(x_flat)         # no tokens routed -> zeros
            expert_outputs.append(temp)                # list of (N, D)

        # 4) stack expert outputs -> (N, D, E)
        stacked = torch.stack(expert_outputs, dim=-1)   # (N, D, E)

        # 5) multiply by full gating probs broadcasted -> (N, 1, E)
        weighted = stacked * topk_probs_full.unsqueeze(1)  # (N, D, E)

        # 6) sum over experts -> (N, D) and reshape back to (B, L, D)
        combined = weighted.sum(dim=-1)                  # (N, D)
        out = combined.view(B, L, D)                     # (B, L, D)
        return out


# Now modified DecoderBlock that uses MoE instead of or in addition to FFN
class DecoderBlockMoE(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, dropout=0.1,
                 num_experts=4, top_k=2):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, causal=True)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        
        # Replace or augment the feedforward with MoE
        self.moef = MoEFeedForward(d_model, d_ff, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)
    
    def forward(self, x, tgt_mask=None):
        sa_out, self_attn = self.mhsa(x, tgt_mask)
        x2 = self.addnorm1(x, sa_out)
        
        # MoE instead of single FFN
        moe_out = self.moef(x2)
        x3 = self.addnorm2(x2, moe_out)
        
        return x3, self_attn

# Modified DecoderOnlyModel with MoE blocks
class DecoderOnlyMoEModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        tokenizer,
        d_model: int = 256,
        max_positions: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_token_id: Optional[int] = None,
        lr: float = 1e-4,
        use_sinusoidal_pos: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer

        self.embedding = TokenEmbeddingModule(
            vocab_size=vocab_size,
            d_model=d_model,
            max_positions=max_positions,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_sinusoidal_pos=use_sinusoidal_pos
        )

        self.layers = nn.ModuleList([
            DecoderBlockMoE(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                num_experts=num_experts,
                top_k=top_k
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # tracking and metrics as before
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.generated_texts = []
        self.reference_texts = []

    def forward(self, input_ids, tgt_mask=None):
        x = self.embedding(input_ids, tgt_mask)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, tgt_mask)
            attn_maps.append(attn)
        x = self.norm(x)
        logits = self.classifier(x)
        return logits, attn_maps

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        logits, _ = self(input_ids)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.train_epoch_losses.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_epoch_losses).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n----------------------------------------------\n \
                Training loss epoch: {avg_loss.item():.4f}\n \
                \n----------------------------------------------\n")

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        logits, _ = self(input_ids)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.val_epoch_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True)

        # Optional: gather generated / reference text for metrics
        preds = torch.argmax(logits, dim=-1)
        pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_texts = self.tokenizer.batch_decode(
            torch.where(labels != -100, labels, self.tokenizer.pad_token_id),
            skip_special_tokens=True
        )
        self.generated_texts.extend(pred_texts)
        self.reference_texts.extend(ref_texts)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_epoch_losses).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n----------------------------------------------\n \
                Validation loss epoch: {avg_loss.item():.4f}\n \
                \n----------------------------------------------\n")

        # Compute metrics like BLEU, ROUGE, etc., using same pattern as before
        # --- Compute Metrics ---
        if len(self.generated_texts) > 0:
            # BLEU
            bleu = sacrebleu.corpus_bleu(self.generated_texts, [self.reference_texts])
            self.log("val_bleu", bleu.score, prog_bar=True)

            # ROUGE
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_scores = [scorer.score(r, g) for r, g in zip(self.reference_texts, self.generated_texts)]

            avg_rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
            avg_rouge2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores)
            avg_rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)

            self.log("val_rouge1", avg_rouge1, prog_bar=True)
            self.log("val_rouge2", avg_rouge2, prog_bar=True)
            self.log("val_rougeL", avg_rougeL, prog_bar=True)

            # METEOR
            meteor_scores = [meteor_score([r.split()], g.split()) for r, g in zip(self.reference_texts, self.generated_texts)]
            avg_meteor = sum(meteor_scores) / len(meteor_scores)
            self.log("val_meteor", avg_meteor, prog_bar=True)

            # BERTScore (uses pre-trained RoBERTa-base by default)
            P, R, F1 = bertscore(self.generated_texts, self.reference_texts, lang="en", verbose=False)
            self.log("val_bertscore_p", P.mean().item(), prog_bar=True)
            self.log("val_bertscore_r", R.mean().item(), prog_bar=True)
            self.log("val_bertscore_f1", F1.mean().item(), prog_bar=True)

            print(f"\n----------------------------------------------\n \
            BLEU: {bleu.score:.2f}\n \
            ROUGE-1: {avg_rouge1:.4f}\n \
            ROUGE-2: {avg_rouge2:.4f}\n \
            ROUGE-L: {avg_rougeL:.4f}\n \
            METEOR: {avg_meteor:.4f}\n \
            BERTScore-F1: {F1.mean().item():.4f}\n \
            \n----------------------------------------------\n")

        # Reset accumulators
        self.val_epoch_losses = []
        self.generated_texts = []
        self.reference_texts = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
