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

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        # Epoch loss tracking
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        enc_out, _ = self.encoder(src_ids, src_mask)
        dec_out, _, _ = self.decoder(
            tgt_ids,
            enc_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )
        logits = self.classifier(dec_out)
        return logits

    def training_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch["src_ids"], batch["tgt_ids"]
        src_mask, tgt_mask = batch.get("src_mask"), batch.get("tgt_mask")

        logits = self.forward(src_ids, tgt_ids[:, :-1], src_mask, tgt_mask[:, :-1] if tgt_mask is not None else None)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))

        self.train_epoch_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_epoch_losses).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n------Training loss epoch: {avg_loss.item()}------\n")
        self.train_epoch_losses = []  # reset for next epoch

    def validation_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch["src_ids"], batch["tgt_ids"]
        src_mask, tgt_mask = batch.get("src_mask"), batch.get("tgt_mask")

        logits = self.forward(src_ids, tgt_ids[:, :-1], src_mask, tgt_mask[:, :-1] if tgt_mask is not None else None)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))

        self.val_epoch_losses.append(loss.detach())
        self.log("val_loss_step", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_epoch_losses).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        print(f"\n------Validation loss epoch: {avg_loss.item()}------\n")
        self.val_epoch_losses = []  # reset for next epoch

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
