# CrossAttentionSeq2SeqModel.py

## Overview

The `CrossAttentionSeq2SeqModel.py` module implements a complete **Sequence-to-Sequence (Seq2Seq) Transformer Model** using **PyTorch Lightning**.

This model follows the standard **Encoder-Decoder** architecture, where:
1.  The **Encoder** processes the source sequence (e.g., input text) into a context representation.
2.  The **Decoder** generates the target sequence (e.g., translated text) auto-regressively, attending to the Encoder's output via Cross-Attention.

It also includes comprehensive metric tracking (BLEU, ROUGE, METEOR, BERTScore) during validation.

## Architecture

The model consists of:
1.  **Encoder**:  Extracts features from the source sequence.
2.  **Decoder**: Generates the target sequence using self-attention and cross-attention over the encoder output.
3.  **Classifier Head**: A linear layer projecting the decoder's output to the vocabulary size.
4.  **Loss Function**: Cross Entropy Loss, ignoring the padding token (`-100` or specified pad ID).

### Mermaid Diagram

```mermaid
graph TD
    subgraph "Encoder"
    SrcIDs[Source IDs] --> Enc[Encoder Module]
    end

    subgraph "Decoder"
    TgtIDs[Target IDs] --> Dec[Decoder Module]
    Enc --> |Cross Attention| Dec
    end

    Dec --> Classifier[Linear Classifier]
    Classifier --> Logits[Logits (Vocab Size)]
    Logits --> Loss[Cross Entropy Loss]
```

## Class Definition: `CrossAttentionSeq2SeqModel`

Inherits from `pl.LightningModule`.

### `__init__`

Initializes the model architecture and metrics.

-   **Key Parameters:**
    -   `vocab_size`: Size of the vocabulary.
    -   `tokenizer`: The tokenizer object (used for decoding predictions during validation).
    -   `d_model`, `num_encoder_layers`, `num_decoder_layers`, `num_heads`, `d_ff`: Transformer hyperparameters.
    -   `lr`: Learning rate for the optimizer.

-   **Components:**
    -   `self.encoder`: Instance of `Encoder` class.
    -   `self.decoder`: Instance of `Decoder` class.
    -   `self.classifier`: `nn.Linear(d_model, vocab_size)`.
    -   `self.loss_fn`: `nn.CrossEntropyLoss`.

### `forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None)`

Orchestrates the forward pass.

1.  **Encode**: Passes `src_ids` and `src_mask` to the **Encoder**.
    ```python
    enc_out, _ = self.encoder(src_ids, src_mask)
    ```
2.  **Decode**: Passes `tgt_ids`, `enc_out`, `tgt_mask`, and `src_mask` (as memory mask) to the **Decoder**.
    ```python
    dec_out, _, _ = self.decoder(tgt_ids, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask)
    ```
3.  **Project**: Projects output to vocabulary dimension.
    ```python
    logits = self.classifier(dec_out)
    ```

### `training_step(self, batch, batch_idx)`

-   Performs a forward pass.
-   Computes Cross Entropy Loss between `logits` and `labels`.
-   Logs `train_loss`.

### `validation_step(self, batch, batch_idx)`

-   Computes validation loss.
-   **Generates Predictions**: Uses greedy decoding (`torch.argmax`) on the logits to generate token IDs.
-   **Decodes Text**: Converts token IDs back to text strings using the `tokenizer`.
-   Stores predictions and references for epoch-end metric calculation.

### `on_validation_epoch_end(self)`

Aggregates predictions and computes NLP metrics:
-   **BLEU**: Using `sacrebleu`.
-   **ROUGE** (1, 2, L): Using `rouge_score`.
-   **METEOR**: Using `nltk`.
-   **BERTScore**: Using `bert_score`.

### `configure_optimizers(self)`

-   Uses `AdamW` optimizer with the specified learning rate.

## Example Usage

```python
from CrossAttentionSeq2SeqModel import CrossAttentionSeq2SeqModel
from transformers import AutoTokenizer

# 1. Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Initialize Model
model = CrossAttentionSeq2SeqModel(
    vocab_size=len(tokenizer),
    tokenizer=tokenizer,
    d_model=256,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=4
)

# 3. Dummy Inputs
import torch
src_ids = torch.randint(0, len(tokenizer), (2, 10)) # Batch 2, Seq 10
tgt_ids = torch.randint(0, len(tokenizer), (2, 10)) # Batch 2, Seq 10

# 4. Forward Pass
logits = model(src_ids, tgt_ids)
print("Logits shape:", logits.shape) 
# Expected: torch.Size([2, 10, vocab_size])
```
