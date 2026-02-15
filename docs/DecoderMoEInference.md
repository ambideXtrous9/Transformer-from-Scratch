# DecoderMoEInference.py

## Overview

The `DecoderMoEInference.py` script demonstrates how to load a trained **Decoder-Only Mixture of Experts (MoE)** model and perform text generation using **greedy decoding**.

## Functions

### 1. `greedy_decode`

Generates text by iteratively selecting the token with the highest probability.

-   **Parameters:**
    -   `model`: The trained `DecoderOnlyMoEModel`.
    -   `tokenizer`: Tokenizer for encoding/decoding.
    -   `prompt` (str): Input text to start generation.
    -   `max_len` (int): Maximum length of total sequence.
    -   `device`: 'cuda' or 'cpu'.

-   **Logic:**
    1.  **Tokenize**: Convert prompt to input IDs.
    2.  **Loop**:
        -   Pass current sequence to model.
        -   Get logits for the last position.
        -   Select token with max logit (`argmax`).
        -   Append token to sequence.
        -   Break if EOS token is generated.
    3.  **Decode**: Convert full sequence of token IDs back to string.

### Mermaid Diagram: Greedy Decode Flow

```mermaid
graph TD
    Start([Start]) --> Tokenize[Tokenize Prompt]
    Tokenize --> InputIds[Input IDs]
    
    InputIds --> Model[Model Forward Pass]
    Model --> Logits[Logits]
    
    Logits --> ArgMax[ArgMax (Last Token)]
    ArgMax --> NextToken[Next Token ID]
    
    NextToken --> EOS{Is EOS?}
    EOS -- Yes --> Decode[Decode IDs to Text]
    EOS -- No --> Append[Append to Input IDs]
    
    Append --> Model
    
    Decode --> End([Return Text])
```

### 2. `load_latest_checkpoint`

Helper function to automate model loading.

-   **Logic**:
    -   Searches for `*.ckpt` files in the specified directory.
    -   Selects the file with the most recent modification time.
    -   Loads the model using `DecoderOnlyMoEModel.load_from_checkpoint`.

## Usage

This script is intended to be run as a standalone program.

```bash
python DecoderMoEInference.py
```

It will:
1.  Initialize the tokenizer (GPT-2).
2.  Find the latest checkpoint in `DecoderMoECheckpoints/`.
3.  Run inference on a sample prompt (e.g., "Artificial intelligence is").
4.  Print the generated output.
