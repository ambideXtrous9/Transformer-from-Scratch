# HF MoEInference.py Module Documentation

## 1. Overview

This script performs **text generation** using a `DecoderOnlyMoEModel` trained with the HuggingFace Trainer. It is the HF equivalent of `PLTrainerScripts/DecoderMoEInference.py`.

## 2. Dependencies
-   `DecoderMoE.py` -> `DecoderOnlyMoEModel`: The MoE model.
-   `HFTrainerScripts.hf_wrapper` -> `HFModelWrapper`, `load_wrapper_from_checkpoint`.

## 3. Functions

### `load_model(checkpoint_dir)`
Rebuilds `DecoderOnlyMoEModel` (num_experts=4, top_k=2), wraps in `HFModelWrapper`, loads weights from `HF_MoECheckpoints/best/`.

### `greedy_decode(model, tokenizer, prompt, max_len)`
Tokenizes prompt, prepends BOS, calls `model.generate_greedy()` with sampling controls (temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2).

## 4. Usage

```bash
cd Transformer-from-Scratch
python HFTrainerScripts/MoEInference.py
```

Requires a trained checkpoint in `checkpoints/HF_MoECheckpoints/`.
