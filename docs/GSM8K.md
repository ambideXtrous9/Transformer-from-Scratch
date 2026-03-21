# GSM8K Training & Inference Pipeline

## Overview

This directory contains scripts for training and running inference with a **Decoder-Only Transformer** (`DecoderOnlyModel`) on the **GSM8K** (Grade School Math 8K) dataset — a benchmark for evaluating multi-step mathematical reasoning.

## Dataset: GSM8K

-   **Source**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
-   **Train**: ~7,473 samples
-   **Test**: ~1,319 samples
-   **Columns**: `question` (math word problem), `answer` (step-by-step solution + final number)

## Files

| File | Description |
|------|-------------|
| `GSM8KTrainer.py` | Training script — loads GSM8K, trains DecoderOnlyModel |
| `GSM8KInference.py` | Inference script — loads checkpoint, generates answers |
| `GSM8KCheckpoints/` | Directory for saved model checkpoints |

## Training

```bash
cd Transformer-from-Scratch
python GSM8K/GSM8KTrainer.py
```

### Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| num_layers | 4 |
| num_heads | 4 |
| d_ff | 512 |
| max_positions | 256 |
| batch_size | 4 |
| max_epochs | 100 |
| Learning Rate | 1e-3 |

## Inference

```bash
cd Transformer-from-Scratch
python GSM8K/GSM8KInference.py
```

The script formats questions as:
```
Question: {your question}
Answer:
```

And the model generates the step-by-step solution.

## Dependencies

-   `datasets` (HuggingFace): `pip install datasets`
-   All other dependencies from the main project (`torch`, `pytorch_lightning`, etc.)
