# CrossAttentionSeq2SeqModel.py Module Documentation

## 1. Overview

The `CrossAttentionSeq2SeqModel` module implements a complete **Encoder-Decoder Transformer Architecture** for Sequence-to-Sequence (Seq2Seq) tasks such as machine translation or text summarization. It leverages `pytorch_lightning` to organize the training, validation, and optimization logic.

### Key Features
-   **Encoder-Decoder Structure**: Uses a bidirectional Encoder and an autoregressive Decoder.
-   **Cross-Attention**: The Decoder attends to the Encoder's output to generate the target sequence.
-   **Metric Integration**: Built-in support for calculating BLEU, ROUGE, METEOR, and BERTScore during validation.
-   **Greedy Decoding**: Implements a simple greedy decoding strategy for validation.

## 2. Modules Involved

-   **torch**: Core PyTorch library.
-   **torch.nn**: Neural network layers (`nn.Linear`, `nn.CrossEntropyLoss`).
-   **pytorch_lightning**: For `LightningModule` structure and training loop.
-   **Metrics Libraries**:
    -   `sacrebleu`: Standard BLEU score implementation.
    -   `rouge_score`: For ROUGE-1, ROUGE-2, ROUGE-L.
    -   `nltk`: For METEOR score.
    -   `bert_score`: For semantic similarity scoring.

### Dependencies
This module depends on the following custom modules:
-   `Encoder` (from `Encoder.py`)
-   `Decoder` (from `Decoder.py`)

## 3. Architecture

The high-level data flow involves processing source text through the Encoder and generating target text via the Decoder, conditioned on the Encoder's output.

### Architecture Diagram

```mermaid
graph TD
    subgraph "Encoder"
        SrcInput[Source IDs] --> Enc[Encoder Module]
        Enc --> EncOut[Encoder Output \n (Key/Value for Cross-Attn)]
    end

    subgraph "Decoder"
        TgtInput[Target IDs] --> Dec[Decoder Module]
        EncOut -.->|Cross-Attention| Dec
        Dec --> DecOut[Decoder Output]
    end

    DecOut --> Classifier[Linear Projection]
    Classifier --> Logits[Logits \n (Vocab Size)]
    
    Logits --> Loss{Cross Entropy Loss}
    Labels[Target Labels] --> Loss
```

## 4. Class Definition: `CrossAttentionSeq2SeqModel`

Inherits from `pl.LightningModule`.

### `__init__`

Initializes the full model hierarchy.

-   **Parameters**:
    -   `vocab_size` (int): Total size of the vocabulary.
    -   `tokenizer`: Tokenizer object (used for decoding text in validation).
    -   `d_model` (int): Hidden dimension size (default: 256).
    -   `max_positions` (int): Maximum sequence length (default: 512).
    -   `num_encoder_layers`, `num_decoder_layers`: Number of blocks in Encoder/Decoder.
    -   `num_heads`, `d_ff`, `dropout`: Transformer configurations.
    -   `lr`: Learning rate.

### `forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None)`

Orchestrates the forward pass through the entire network.

-   **Args**:
    -   `src_ids`: Source token indices `(Batch, Src_Len)`.
    -   `tgt_ids`: Target token indices `(Batch, Tgt_Len)`.
    -   `src_mask`: Padding mask for source.
    -   `tgt_mask`: Causal mask for target (usually handled internally by Decoder, but can be passed).
-   **Returns**:
    -   `logits`: `(Batch, Tgt_Len, Vocab_Size)`.

### Training & Validation Steps

-   **`training_step`**: Computes Cross Entropy Loss.
-   **`validation_step`**: 
    1.  Computes Loss.
    2.  Performs **Greedy Decoding** (`argmax` on logits) to generate text.
    3.  Decodes IDs to strings using the `tokenizer`.
-   **`on_validation_epoch_end`**: Aggregates all generated texts and computes NLP metrics (BLEU, ROUGE, etc.).

## 5. Step-by-Step Logic (Forward Pass)

1.  **Encoding**:
    -   The `src_ids` are passed to `self.encoder`.
    -   The Encoder embeds the tokens, adds positional info, and passes them through N Encoder Blocks.
    -   **Output**: `enc_out` tensor of shape `(Batch, Src_Len, d_model)`.

2.  **Decoding**:
    -   The `tgt_ids` (shifted right, i.e., starting with `[BOS]`) are passed to `self.decoder`.
    -   The `enc_out` is passed as the `enc_out` argument (acting as Memory).
    -   The Decoder embeds target tokens, adds positional info.
        -   **Self-Attention**: Attends to previous target tokens (causal).
        -   **Cross-Attention**: Queries from Decoder attend to Keys/Values from `enc_out`.
    -   **Output**: `dec_out` tensor of shape `(Batch, Tgt_Len, d_model)`.

3.  **Projection**:
    -   The `dec_out` is passed through `self.classifier` (Linear layer).
    -   **Output**: Logits of shape `(Batch, Tgt_Len, Vocab_Size)`.

## 6. Dry Run Trace

**Scenario**: 
-   Batch Size = 1
-   Source: `[10, 20]` (Length 2)
-   Target: `[30, 40]` (Length 2)
-   `d_model` = 4
-   `vocab_size` = 100

**Trace**:

1.  **Inputs**:
    -   `src_ids`: `[[10, 20]]` (1, 2)
    -   `tgt_ids`: `[[30, 40]]` (1, 2)

2.  **Encoder**:
    -   Input: `[[10, 20]]`
    -   Embedding: `(1, 2, 4)`
    -   Processing...
    -   `enc_out`: `(1, 2, 4)` (Contextualized representations of source)

3.  **Decoder**:
    -   Input `tgt_ids`: `[[30, 40]]`
    -   Input `enc_out`: `(1, 2, 4)`
    -   Embedding: `(1, 2, 4)`
    -   **Block 1**:
        -   Self-Attn (Masked): Looks at pos 0 only for pos 0; pos 0,1 for pos 1.
        -   Cross-Attn: Query `(1, 2, 4)` attends to Key/Value `(1, 2, 4)` from Encoder.
    -   Output `dec_out`: `(1, 2, 4)`

4.  **Classifier**:
    -   Input: `(1, 2, 4)`
    -   Linear(4 -> 100)
    -   Output `logits`: `(1, 2, 100)`

5.  **Loss Calculation** (if Training):
    -   Labels would be `tgt_ids` shifted or provided separately (e.g., `[40, 50]`).
    -   Cross Entropy between `logits` `(1, 2, 100)` and `labels` `(1, 2)`.

## 7. Metrics Logic

During implementation details for validation metrics:
-   **BLEU**: Compares n-grams.
-   **ROUGE**: Longest Common Subsequence and n-gram overlap.
-   **BERTScore**: Uses a pre-trained model (like RoBERTa) to calculate embedding similarity between generated and reference text. *Note: specific resource downloads occur on first run.*
