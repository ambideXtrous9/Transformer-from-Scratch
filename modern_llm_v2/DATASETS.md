# Dataset Configuration Guide

This guide explains the dataset options and how to configure them for different use cases.

## 📊 Available Datasets

### 1. **TinyStories** (Default - Recommended)
- **Size**: ~100MB
- **Samples**: ~2.8M short stories
- **Use Case**: Testing, experimentation, architecture validation
- **Download Time**: ~1-2 minutes
- **Training Time**: Hours to days (depending on configuration)

```python
# config_200m.py
DATASET_NAME = "roneneldan/TinyStories"
DATASET_MAX_SAMPLES = 10000  # Limit for quick testing
```

**Why TinyStories?**
✅ Small and fast to download  
✅ Diverse vocabulary  
✅ Good for testing model architecture  
✅ Can train to convergence in reasonable time  

---

### 2. **Tiny Shakespeare** (Quick Smoke Test)
- **Size**: ~1MB
- **Samples**: 1 (single file)
- **Use Case**: Quick validation, debugging
- **Download Time**: ~10 seconds
- **Training Time**: Minutes

```python
# config_200m.py
DATASET_NAME = "karpathy/tiny_shakespeare"
DATASET_MAX_SAMPLES = None  # Not needed (only 1 sample)
MAX_LENGTH = 128  # Short sequences
```

**Use when:**
- Testing if code runs
- Debugging issues
- Quick architecture validation

---

### 3. **WikiText-2** (Medium Testing)
- **Size**: ~200MB
- **Samples**: ~36k articles
- **Use Case**: Medium-scale testing
- **Download Time**: ~2-3 minutes
- **Training Time**: Hours

```python
# config_200m.py
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_MAX_SAMPLES = 5000  # Limit for faster testing
```

---

### 4. **OpenWebText** (Production)
- **Size**: ~40GB
- **Samples**: ~8M documents
- **Use Case**: Full production training
- **Download Time**: Hours
- **Training Time**: Days to weeks

```python
# config_200m.py
DATASET_NAME = "Skylion007/openwebtext"
DATASET_MAX_SAMPLES = None  # Use full dataset
```

**Use when:**
- Training production models
- You have GPU cluster access
- You want high-quality outputs

---

## 🔧 Configuration Options

### Limit Dataset Size

For quick testing, limit the number of samples:

```python
# config_200m.py

# Use only 10k samples (fast testing)
DATASET_MAX_SAMPLES = 10000

# Use 100k samples (medium testing)
DATASET_MAX_SAMPLES = 100000

# Use full dataset (production)
DATASET_MAX_SAMPLES = None
```

### Adjust Sequence Length

Shorter sequences = faster training, less memory:

```python
# config_200m.py

# Quick testing
MAX_LENGTH = 128

# Medium training
MAX_LENGTH = 256

# Production training
MAX_LENGTH = 1024
```

### Batch Size Configuration

Auto-detected based on GPU memory, or set manually:

```python
# config_200m.py

# Small GPU (8GB)
TRAIN_BATCH_SIZE = 4

# Medium GPU (16GB)
TRAIN_BATCH_SIZE = 8

# Large GPU (24GB+)
TRAIN_BATCH_SIZE = 16

# Gradient accumulation (simulate larger batches)
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = BATCH_SIZE * 4
```

---

## 🚀 Quick Start Scenarios

### Scenario 1: "Just test if it works" (2 minutes)

```python
# config_200m.py
DATASET_NAME = "karpathy/tiny_shakespeare"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 4
MAX_STEPS = 100
WARMUP_STEPS = 10

# Or run the example script:
python examples/quick_start.py --mode test
```

**Expected Results:**
- Downloads instantly
- Trains in ~2 minutes
- Loss decreases (validates architecture)
- Can generate text (will be nonsense, but works!)

---

### Scenario 2: "Experiment with architecture" (30 minutes)

```python
# config_200m.py
DATASET_NAME = "roneneldan/TinyStories"
DATASET_MAX_SAMPLES = 5000
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
MAX_STEPS = 1000
WARMUP_STEPS = 100

# Or run the example script:
python examples/quick_start.py --mode small
```

**Expected Results:**
- Downloads in ~1 minute
- Trains in ~30 minutes
- Generates coherent short stories
- Good for hyperparameter tuning

---

### Scenario 3: "Serious experimentation" (6-12 hours)

```python
# config_200m.py
DATASET_NAME = "roneneldan/TinyStories"
DATASET_MAX_SAMPLES = 100000
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 16
MAX_STEPS = 50000
WARMUP_STEPS = 2000
```

**Expected Results:**
- Downloads in ~2 minutes
- Trains in 6-12 hours
- Generates quality stories
- Can publish results

---

### Scenario 4: "Production training" (days)

```python
# config_200m.py
DATASET_NAME = "Skylion007/openwebtext"
DATASET_MAX_SAMPLES = None
MAX_LENGTH = 1024
TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 100000
WARMUP_STEPS = 2000
```

**Expected Results:**
- Downloads in hours
- Trains in days/weeks
- Production-quality model
- Ready for deployment

---

## 📝 Custom Datasets

To use your own dataset:

### 1. Prepare your data

```python
# Your dataset should be in HuggingFace format:
from datasets import Dataset

data = {
    "text": [
        "Your first document...",
        "Your second document...",
        # ...
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("my_dataset")
```

### 2. Load in config

```python
# config_200m.py
DATASET_NAME = "path/to/my_dataset"  # Local path
# OR
DATASET_NAME = "your-username/your-dataset"  # HuggingFace Hub
```

### 3. Update text column (if needed)

If your dataset uses a different column name:

```python
# In training/train.py, modify create_datamodule():
text_column = "your_column_name"  # Instead of "text"
```

---

## 💡 Tips

### Memory Optimization

If you get OOM errors:

```python
# Reduce sequence length
MAX_LENGTH = 256  # Instead of 1024

# Reduce batch size
TRAIN_BATCH_SIZE = 4  # Instead of 16

# Increase gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8  # Simulate larger batches

# Use mixed precision
PRECISION = "bf16-mixed"  # Or "16-mixed"
```

### Speed Optimization

```python
# Increase workers (if CPU allows)
NUM_WORKERS = 8  # Instead of 4

# Use Flash Attention
USE_FLASH_ATTENTION = True  # If available

# Reduce validation frequency
VAL_CHECK_INTERVAL = 1.0  # Once per epoch
```

---

## 📊 Dataset Comparison Table

| Dataset | Size | Download | Training (1000 steps) | Best For |
|---------|------|----------|----------------------|----------|
| Tiny Shakespeare | 1MB | 10s | 5 min | Smoke tests |
| TinyStories (5k) | 100MB | 1 min | 30 min | Experimentation |
| TinyStories (100k) | 100MB | 1 min | 6 hours | Research |
| WikiText-2 | 200MB | 2 min | 1 hour | Testing |
| OpenWebText | 40GB | 4 hours | 7 days | Production |

---

## 🎯 Recommended Starting Point

For **first-time users**:

```python
# config_200m.py
DATASET_NAME = "roneneldan/TinyStories"
DATASET_MAX_SAMPLES = 5000
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
MAX_STEPS = 1000
WARMUP_STEPS = 100

# Then run:
python examples/quick_start.py --mode small
```

This gives you:
- ✅ Fast download (~1 min)
- ✅ Quick training (~30 min)
- ✅ Visible results (generates stories)
- ✅ Good for learning the codebase

---

Happy training! 🚀
