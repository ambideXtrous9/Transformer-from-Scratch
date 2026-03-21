import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

from Embedding import TokenEmbeddingModule
from MultiHeadSelfAttention import MultiHeadSelfAttention
from AddNorm import AddNorm
from FFN import PositionwiseFeedForward

# Metrics
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
from bert_score import score as bertscore

pl.seed_everything(42)

# ---------------- Decoder Block (GPT-style) ----------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        # 1. Masked Self-Attention
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, causal=True)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)

        # 2. Feed Forward
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, activation="gelu")
        self.addnorm2 = AddNorm(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None):
        # 1. Masked Self-Attention + AddNorm
        mhsa_out, self_attn = self.mhsa(x, tgt_mask)
        x = self.addnorm1(x, mhsa_out)

        # 2. Feed Forward + AddNorm
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)

        return x, self_attn


# ---------------- Decoder-only Transformer ----------------
class DecoderOnlyModel(pl.LightningModule):
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer

        # 1. Embeddings
        self.embedding = TokenEmbeddingModule(
            vocab_size=vocab_size,
            d_model=d_model,
            max_positions=max_positions,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_sinusoidal_pos=use_sinusoidal_pos
        )

        # 2. Decoder Blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 3. Final LayerNorm + Linear
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Epoch losses
        self.train_epoch_losses = []
        self.val_epoch_losses = []

        # Metrics accumulators
        self.generated_texts = []
        self.reference_texts = []

    def forward(self, input_ids, tgt_mask=None):
        # Embedding
        x = self.embedding(input_ids, tgt_mask)

        # Pass through decoder blocks
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, tgt_mask)
            attn_maps.append(attn)

        # Final norm + logits
        x = self.norm(x)
        logits = self.classifier(x)  # (B, L, vocab_size)

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
        self.train_epoch_losses = []

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        logits, _ = self(input_ids)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.val_epoch_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True)

        # Store loss
        self.val_epoch_losses.append(loss.detach())

        # Decode predictions & references
        preds = torch.argmax(logits, dim=-1)  # greedy decode
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

        # Reset for next epoch
        self.val_epoch_losses = []
        self.generated_texts = []
        self.reference_texts = []


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

