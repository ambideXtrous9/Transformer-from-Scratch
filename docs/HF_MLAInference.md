# HF MLAInference.py Module Documentation

## 1. Overview

This script performs **text generation** using a `DecoderOnlyMLAModel` trained with the HuggingFace Trainer. It is the HF equivalent of `PLTrainerScripts/DecoderOnlyMLAInference.py`.

## 2. Dependencies
-   `DecoderOnlyMLAModel.py` -> `DecoderOnlyMLAModel`: The MLA model.
-   `HFTrainerScripts.hf_wrapper` -> `HFModelWrapper`, `load_wrapper_from_checkpoint`.

## 3. Functions

### `load_model(checkpoint_dir)`
Rebuilds `DecoderOnlyMLAModel` (num_heads=4, d_compress=64), wraps in `HFModelWrapper`, loads weights from `HF_MLACheckpoints/best/`.

### `greedy_decode(model, tokenizer, question, max_len=256)`
Formats question as GSM8K-style prompt (`"Question: {question}\nAnswer:"`), tokenizes, prepends BOS, calls `model.generate_greedy()` with sampling controls (temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2).

## 4. Usage

```bash
cd Transformer-from-Scratch
python HFTrainerScripts/MLAInference.py
```

Requires a trained checkpoint in `checkpoints/HF_MLACheckpoints/`.
