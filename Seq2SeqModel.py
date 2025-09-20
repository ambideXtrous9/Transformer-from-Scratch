import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

from Encoder import Encoder
from Decoder import Decoder

# Metrics
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
from bert_score import score as bertscore

pl.seed_everything(42)

class Seq2SeqModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        tokenizer,
        d_model: int = 256,
        max_positions: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
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

        # Encoder & Decoder
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_positions=max_positions,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_sinusoidal_pos=use_sinusoidal_pos
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_positions=max_positions,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id,
            use_sinusoidal_pos=use_sinusoidal_pos
        )

        # Final classifier head
        self.classifier = nn.Linear(d_model, vocab_size)

        # Loss — ignore labels with -100
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Epoch losses
        self.train_epoch_losses = []
        self.val_epoch_losses = []

        # Metrics accumulators
        self.generated_texts = []
        self.reference_texts = []

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Encode
        enc_out, _ = self.encoder(src_ids, src_mask)

        # Decode
        dec_out, _, _ = self.decoder(
            tgt_ids,
            enc_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        # Project to vocab
        logits = self.classifier(dec_out)
        return logits

    def training_step(self, batch, batch_idx):
        src_ids, src_mask = batch["src_ids"], batch["src_mask"]
        tgt_ids, tgt_mask, labels = batch["tgt_ids"], batch["tgt_mask"], batch["labels"]

        logits = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        self.train_epoch_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_epoch_losses).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n----------------------------------------------\n \
                Training loss epoch: {avg_loss.item():.4f}\n \
                \n----------------------------------------------\n")
        self.train_epoch_losses = []

    def validation_step(self, batch, batch_idx):
        src_ids, src_mask = batch["src_ids"], batch["src_mask"]
        tgt_ids, tgt_mask, labels = batch["tgt_ids"], batch["tgt_mask"], batch["labels"]

        logits = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

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
