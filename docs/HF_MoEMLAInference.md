# HF MoEMLAInference.py Module Documentation

## 1. Overview

This script performs **text generation** using a `DecoderOnlyMoEMLAModel` trained with the HuggingFace Trainer. It is the HF equivalent of `PLTrainerScripts/DecoderMoEMLAInference.py`.

## 2. Dependencies
-   `DecoderMoEMLA.py` -> `DecoderOnlyMoEMLAModel`
-   `HFTrainerScripts.hf_wrapper` -> `HFModelWrapper`, `load_wrapper_from_checkpoint`

## 3. Functions

### `load_model(checkpoint_dir)`
Rebuilds `DecoderOnlyMoEMLAModel` (d_compress=64, num_experts=4, top_k=2), wraps in `HFModelWrapper`, loads weights from `HF_MoEMLACheckpoints/best/`.

### `greedy_decode(model, tokenizer, question, max_len=256)`
Formats question as GSM8K-style prompt, tokenizes, prepends BOS, calls `model.generate_greedy()` with sampling controls.

## 4. Usage

```bash
python HFTrainerScripts/MoEMLAInference.py
```

Requires a trained checkpoint in `checkpoints/HF_MoEMLACheckpoints/`.
