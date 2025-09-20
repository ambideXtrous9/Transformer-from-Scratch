import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

from Encoder import Encoder
from Decoder import Decoder
from Embedding import get_tokenizer

class Seq2SeqModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
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
        self.save_hyperparameters()

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

        # Loss — ignore labels with -100 (set in dataset)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Track epoch losses
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
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

        # Compute loss against labels (already shifted, padded, masked in dataset)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        self.train_epoch_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_epoch_losses).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n------ Training loss epoch: {avg_loss.item():.4f} ------\n")
        self.train_epoch_losses = []

    def validation_step(self, batch, batch_idx):
        src_ids, src_mask = batch["src_ids"], batch["src_mask"]
        tgt_ids, tgt_mask, labels = batch["tgt_ids"], batch["tgt_mask"], batch["labels"]

        logits = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        self.val_epoch_losses.append(loss.detach())
        self.log("val_loss_step", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_epoch_losses).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n------ Validation loss epoch: {avg_loss.item():.4f} ------\n")
        self.val_epoch_losses = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
