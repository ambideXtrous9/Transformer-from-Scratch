# Modern LLM V2 - 200M Parameter Language Model

Production-level implementation of a modern decoder-only language model built from scratch using PyTorch and PyTorch Lightning. This implementation follows the architecture of state-of-the-art LLMs like LLaMA-2, Mistral, and GPT-Neo.

## 🏗️ Architecture Overview

This implementation includes **~200M parameters** with modern LLM design choices:

| Component | Implementation |
|-----------|---------------|
| **Attention** | Grouped Query Attention (GQA) with RoPE |
| **Feed-Forward** | SwiGLU (Swish-Gated Linear Unit) |
| **Normalization** | RMSNorm (Pre-norm architecture) |
| **Positional Encoding** | Rotary Position Embedding (RoPE) |
| **Activations** | SiLU/Swish (for SwiGLU) |
| **Training** | Mixed precision (bf16) with gradient clipping |
| **Optimization** | AdamW with cosine warmup schedule |

### Model Configuration (200M)

```yaml
Vocabulary Size: 50,257 (GPT-2 tokenizer)
Model Dimension (d_model): 768
Attention Heads: 12
KV Heads: 4 (Grouped Query Attention)
Transformer Layers: 18
FFN Dimension: 2,048
Max Sequence Length: 1,024
Dropout: 0.1
RoPE Theta: 10,000
```

### Parameter Breakdown

```
Token Embedding:    38.6M (50,257 × 768)
Attention (×18):    ~42.4M (2.36M per layer)
SwiGLU FFN (×18):   ~84.9M (4.72M per layer)
Output Projection:  38.6M (768 × 50,257)
Normalization:      ~0.03M
────────────────────────────────────
Total:             ~190M parameters (close to 200M)
```

## 📁 Project Structure

```
modern_llm_v2/
├── config_200m.py              # Configuration (hyperparameters)
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── core/                       # Core building blocks
│   ├── __init__.py
│   ├── TokenEmbedding.py       # Token embedding with scaling
│   ├── PositionalEncoding.py   # RoPE, Learned, Sinusoidal
│   ├── Normalization.py        # RMSNorm, AddNorm, PreNorm
│   ├── FFN.py                  # SwiGLU, Standard FFN
│   └── attention/
│       ├── __init__.py
│       └── GroupQueryAttention.py  # GQA, MHA, MQA with RoPE
│
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── DecoderBlock.py         # Single transformer block
│   └── ModernLLM.py            # Complete 200M model
│
└── training/                   # Training & inference
    ├── __init__.py
    ├── train.py                # PyTorch Lightning trainer
    └── inference.py            # Generation utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for 2-3x speedup
pip install flash-attn --no-build-isolation
```

### 2. Training

```bash
# Basic training with TinyStories (small dataset, ~100MB)
cd modern_llm_v2
python training/train.py
```

**🚀 Quick Start Examples:**

```bash
# Quick test run (Tiny Shakespeare, 100 steps, ~2 minutes)
python examples/quick_start.py --mode test

# Small training run (TinyStories, 1000 steps, ~10 minutes)
python examples/quick_start.py --mode small
```

**Dataset Options:**

| Dataset | Size | Use Case |
|---------|------|----------|
| **TinyStories** (default) | ~100MB | Testing & experimentation ✅ |
| **Tiny Shakespeare** | ~1MB | Quick smoke tests |
| **WikiText-2** | ~200MB | Medium-scale testing |
| **OpenWebText** | ~40GB | Full production training |

To use a different dataset, edit `config_200m.py`:

```python
# Fast testing (recommended)
DATASET_NAME = "roneneldan/TinyStories"

# Super quick test
DATASET_NAME = "karpathy/tiny_shakespeare"

# Production training
DATASET_NAME = "Skylion007/openwebtext"
```

**Limit dataset size for quick testing:**

```python
# config_200m.py
DATASET_MAX_SAMPLES = 10000  # Use only 10k samples
```

**Training Configuration:**
- **Dataset**: TinyStories (default, ~100MB)
- **Optimizer**: AdamW (lr=3e-4, betas=(0.9, 0.95))
- **LR Schedule**: Cosine with 2000 steps warmup
- **Precision**: BF16 mixed
- **Batch Size**: Auto-detected based on GPU memory
- **Gradient Clipping**: 1.0
- **Validation**: Every 0.5 epochs

### 3. Inference

```python
from training.inference import LLMInference

# Load trained model
llm = LLMInference("checkpoints/modern-llm-200m/final_model")

# Generate text
result = llm.generate("Once upon a time,")
print(result["generated_text"])

# Interactive chat
llm.interactive_chat()
```

## 🧩 Core Modules

### 1. Token Embedding (`core/TokenEmbedding.py`)

Converts token IDs to dense vectors with optional sqrt(d_model) scaling (GPT-2 style).

```python
from core.TokenEmbedding import TokenEmbedding

embedding = TokenEmbedding(vocab_size=50257, d_model=768, scale_embeddings=True)
tokens = torch.randint(0, 50257, (2, 1024))
emb = embedding(tokens)  # (2, 1024, 768)
```

### 2. Positional Encoding (`core/PositionalEncoding.py`)

Supports three types of positional encoding:

```python
from core.PositionalEncoding import PositionalEncoding, apply_rope

# RoPE (recommended for modern LLMs)
rope = PositionalEncoding("rope", d_model=768, max_seq_length=1024)

# Apply to Q/K vectors
q, k = apply_rope(query, key, rope.freqs_cis)
```

### 3. Normalization (`core/Normalization.py`)

RMSNorm with pre-norm architecture (LLaMA style):

```python
from core.Normalization import RMSNorm, AddNorm

# RMSNorm
norm = RMSNorm(d_model=768, eps=1e-5)

# AddNorm (residual + dropout + norm)
add_norm = AddNorm(d_model=768, dropout=0.1, norm_type="rms")
```

### 4. Attention (`core/attention/GroupQueryAttention.py`)

Grouped Query Attention with RoPE and Flash Attention support:

```python
from core.attention.GroupQueryAttention import GroupQueryAttention

# GQA: 12 query heads, 4 KV heads (3:1 ratio)
attn = GroupQueryAttention(
    d_model=768,
    num_heads=12,
    num_kv_heads=4,
    dropout=0.1,
    use_flash_attention=True
)

output, past_kv, attn_weights = attn(x, is_causal=True)
```

**Attention Variants:**
- `MultiHeadAttention`: Standard MHA (num_kv_heads = num_heads)
- `GroupQueryAttention`: GQA (configurable KV heads)
- `MultiQueryAttention`: MQA (num_kv_heads = 1)

### 5. Feed-Forward Network (`core/FFN.py`)

SwiGLU activation (modern standard):

```python
from core.FFN import SwiGLU

ffn = SwiGLU(d_model=768, d_ff=2048, dropout=0.1)
output = ffn(x)  # (B, L, d_model)
```

### 6. Decoder Block (`models/DecoderBlock.py`)

Complete transformer block with pre-norm:

```python
from models.DecoderBlock import DecoderBlock

block = DecoderBlock(
    d_model=768,
    num_heads=12,
    num_kv_heads=4,
    d_ff=2048,
    dropout=0.1
)

output, past_kv, attn_weights = block(x, is_causal=True)
```

### 7. Complete Model (`models/ModernLLM.py`)

Full 200M parameter model:

```python
from models.ModernLLM import ModernDecoderOnlyModel

model = ModernDecoderOnlyModel(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    d_ff=2048,
    max_seq_length=1024
)

# Forward pass
logits, past_kv = model(input_ids)

# Generation
generated = model.generate(
    prompt_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

## 📊 Training with PyTorch Lightning

### Trainer Setup

The training script (`training/train.py`) includes:

- **W&B Logging**: Automatic experiment tracking
- **Model Checkpointing**: Save best models
- **LR Monitoring**: Track learning rate schedule
- **Mixed Precision**: BF16 training for speed
- **Multi-GPU**: DDP support for distributed training

```python
from training.train import create_trainer, create_model, create_datamodule

# Create components
model = create_model(vocab_size=50257)
train_dl, val_dl = create_datamodule(tokenizer, batch_size=16, max_length=1024)

# Create trainer with W&B
trainer = create_trainer(
    output_dir="checkpoints/",
    max_steps=100000,
    wandb_project="modern-llm-200m",
    wandb_entity="your-entity"  # Optional: your W&B username/team
)

# Train
trainer.fit(model, train_dl, val_dl)
```

### W&B Integration

The trainer automatically logs:
- Training/validation loss
- Learning rate schedule
- Gradient norms (if enabled)
- Generated text samples
- Model checkpoints (if entity specified)

```bash
# Set W&B entity in config_200m.py
WANDB_ENTITY = "your-username"

# Or set via environment variable
export WANDB_ENTITY="your-username"
```

### Learning Rate Schedule

Cosine decay with linear warmup:

```
LR
↑
│     /\
│    /  \
│   /    \______
│  /
│ /
│/
└──────────────────→ Steps
   2000   100000
  (warmup) (total)
```

## 🔧 Configuration

All hyperparameters are in `config_200m.py`:

```python
# Model architecture
D_MODEL = 768
NUM_HEADS = 12
NUM_KV_HEADS = 4
NUM_LAYERS = 12
D_FF = 2048

# Training
LEARNING_RATE = 3e-4
WARMUP_STEPS = 2000
MAX_STEPS = 100000
GRADIENT_CLIP_VAL = 1.0
PRECISION = "bf16-mixed"

# Dataset
DATASET_NAME = "openwebtext"
TOKENIZER_NAME = "gpt2"
MAX_LENGTH = 1024

# W&B
WANDB_PROJECT = "modern-llm-200m"
WANDB_ENTITY = None  # Set to your entity
```

## 🎯 Generation Strategies

The model supports multiple decoding strategies:

```python
# Greedy decoding
result = model.generate(prompt, do_sample=False)

# Temperature sampling
result = model.generate(prompt, temperature=0.7)

# Top-k filtering
result = model.generate(prompt, top_k=50)

# Nucleus (top-p) sampling
result = model.generate(prompt, top_p=0.95)

# With repetition penalty
result = model.generate(prompt, repetition_penalty=1.2)
```

## 📈 Performance Optimization

### Flash Attention

For 2-3x faster training, install Flash Attention:

```bash
pip install flash-attn --no-build-isolation
```

The model automatically uses it if available:

```python
model = ModernDecoderOnlyModel(
    use_flash_attention=True,  # Default
    ...
)
```

### GPU Memory Optimization

Batch size is auto-scaled based on GPU memory:

```python
from config_200m import get_batch_size_for_gpu, get_gpu_memory_gb

gpu_mem = get_gpu_memory_gb()  # Auto-detect
batch_size = get_batch_size_for_gpu(gpu_mem)
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # 4x effective batch size
    ...
)
```

## 📝 Custom Dataset

To use your own dataset:

```python
from training.train import ConstantLengthDataset, create_datamodule

# Load your dataset (HuggingFace format)
from datasets import load_dataset
dataset = load_dataset("your-dataset")

# Create dataloaders
train_dl, val_dl = create_datamodule(
    tokenizer=tokenizer,
    batch_size=16,
    max_length=1024,
    dataset_name="your-dataset"
)
```

## 🔬 Testing Individual Modules

Each module includes standalone tests:

```bash
# Test token embedding
python core/TokenEmbedding.py

# Test positional encoding
python core/PositionalEncoding.py

# Test normalization
python core/Normalization.py

# Test attention
python core/attention/GroupQueryAttention.py

# Test FFN
python core/FFN.py

# Test decoder block
python models/DecoderBlock.py

# Test complete model
python models/ModernLLM.py
```

## 📚 References

This implementation is based on:

1. **LLaMA/LLaMA-2**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
2. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
3. **SwiGLU**: "GLU Variants Improve Transformer" (Shazeer, 2020)
4. **GQA**: "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
5. **RMSNorm**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Experiment with different configurations

## 📄 License

This project is for educational and research purposes.

---

**Happy Training! 🚀**
