"""
Configuration for Modern LLM Architecture (200M Parameters)
Production-level configuration matching GPT-2/LLaMA style architectures.

Parameter Count Calculation:
- Embedding: vocab_size * d_model = 50257 * 768 ≈ 38.6M
- Attention (per layer): 4 * d_model * d_model = 4 * 768 * 768 ≈ 2.36M
- FFN SwiGLU (per layer): 3 * d_model * d_ff = 3 * 768 * 2048 ≈ 4.72M
- Output layers: d_model * vocab_size = 768 * 50257 ≈ 38.6M
- Total with 12 layers: ≈ 200M parameters
"""

import os
import torch

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Smaller datasets for faster experimentation and testing
# Options:
#   - "roneneldan/TinyStories": Small synthetic stories (~100MB, recommended for testing)
#   - "Skylion007/openwebtext": Large web text (~40GB, production)
#   - "karpathy/tiny_shakespeare": Tiny Shakespeare (~1MB, quick testing)
#   - "Salesforce/wikitext": WikiText-103 (~200MB)
DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None  # Use default config
DATASET_SPLIT = "train"
DATASET_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
TOKENIZER_NAME = "gpt2"

# Dataset size control
DATASET_MAX_SAMPLES = None  # Set to int (e.g., 10000) to limit dataset size for testing

# ============================================================================
# SEQUENCE CONFIGURATION
# ============================================================================
MAX_LENGTH = 1024  # Context window size
PAD_TOKEN_ID = 50256  # GPT2 pad token

# ============================================================================
# MODEL CONFIGURATION (200M Parameters)
# ============================================================================
# Architecture matches modern LLM design (LLaMA/GPT-2 style)
# Parameter count: ~200M
D_MODEL = 768  # Hidden dimension
NUM_HEADS = 12  # Number of attention heads
NUM_KV_HEADS = 4  # Grouped Query Attention (3 query groups per KV head)
NUM_LAYERS = 18  # Number of transformer layers (for ~200M total)
D_FF = 2048  # Feed-forward dimension (SwiGLU uses 3x projection)
DROPOUT = 0.1  # Dropout rate
VOCAB_SIZE = 50257  # GPT2 vocabulary size

# Activation function
ACTIVATION = "swiglu"  # Options: "swiglu" (default), "gelu", "relu"

# Normalization
NORM_EPS = 1e-5  # RMSNorm epsilon
NORM_BEFORE_ATTENTION = True  # Pre-norm architecture (modern standard)

# Positional encoding
POSITIONAL_ENCODING = "rope"  # Options: "rope" ( Rotary), "learned", "sinusoidal"
ROPE_THETA = 10000.0  # Base frequency for RoPE
ROPE_SCALING = None  # Optional scaling for long context

# Attention configuration
USE_FLASH_ATTENTION = True  # Use Flash Attention 2 if available
ATTENTION_BIAS = False  # Don't use bias in attention projections (modern)
FFN_BIAS = False  # Don't use bias in FFN (modern)

# Tie embeddings (optional, saves parameters)
TIE_WORD_EMBEDDINGS = False  # Set True to share input/output embeddings

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
WARMUP_STEPS = 1000  # Learning rate warmup (reduced for small datasets)
MAX_STEPS = 50000  # Total training steps (reduced for small datasets)
GRADIENT_CLIP_VAL = 1.0

# Batch size (will be auto-scaled based on GPU memory)
# Set to None to use AUTO_BATCH_SIZE (recommended)
# Or set a specific value to override auto-detection
TRAIN_BATCH_SIZE = None  # Will use AUTO_BATCH_SIZE
VAL_BATCH_SIZE = None
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * ACCUM

# Mixed precision
PRECISION = "bf16-mixed"  # Options: "32", "16-mixed", "bf16-mixed"

# Memory optimization (for MPS or low-memory GPUs)
USE_GRADIENT_CHECKPOINTING = False  # Trade compute for memory (saves ~40% memory)
ENABLE_MEMORY_EFFICIENT_ATTENTION = True  # Use memory-efficient attention
PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0  # Set to 0.0 to disable MPS memory limit

# ============================================================================
# DATASET PROCESSING
# ============================================================================
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# Text processing
STRIP_TOKENS = True  # Strip whitespace tokens
TRUNCATE_LONG_SEQ = True  # Truncate sequences longer than MAX_LENGTH

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_MONITOR = "val_loss"  # Metric to monitor for checkpointing
CHECKPOINT_MODE = "min"  # Mode for checkpoint monitoring
SAVE_TOP_K = 3  # Number of best checkpoints to keep
CHECKPOINT_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================
VAL_CHECK_INTERVAL = 0.5  # Validate every 0.5 epochs (twice per epoch)
VAL_PERCENT_CHECKS = None  # Or use fraction: 0.1 for 10%
LOG_EVERY_N_STEPS = 50  # Log metrics every N steps

# ============================================================================
# WANDB CONFIGURATION
# ============================================================================
WANDB_PROJECT = "modern-llm-200m"
WANDB_ENTITY = None  # Set to your wandb entity
WANDB_LOG_MODEL = True  # Log model checkpoints to wandb
WANDB_WATCH_GRADIENTS = False  # Watch gradient norms

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_MAX_NEW_TOKENS = 256
INFERENCE_TEMPERATURE = 0.8
INFERENCE_TOP_K = 50
INFERENCE_TOP_P = 0.95
INFERENCE_REPETITION_PENALTY = 1.2
INFERENCE_DO_SAMPLE = True

# ============================================================================
# GPU-AWARE BATCH SIZING
# ============================================================================
def get_batch_size_for_gpu(gpu_mem_gb: int, is_mps: bool = False) -> int:
    """
    Get recommended batch size based on GPU memory.
    For 200M model with bf16:
    - MPS (Apple Silicon): More conservative due to memory limits
    - NVIDIA GPUs: Standard calculations
    """
    if is_mps:
        # MPS has stricter memory limits
        # For 190M model with seq_len=1024:
        # - 8GB: batch_size = 2
        # - 16GB: batch_size = 4
        # - 24GB+: batch_size = 6
        if gpu_mem_gb >= 24:
            return 6
        elif gpu_mem_gb >= 16:
            return 4
        else:
            return 2
    else:
        # Standard NVIDIA GPUs
        if gpu_mem_gb >= 40:
            return 32
        elif gpu_mem_gb >= 24:
            return 24
        elif gpu_mem_gb >= 16:
            return 16
        elif gpu_mem_gb >= 12:
            return 12
        else:
            return 8


def get_gpu_memory_gb() -> int:
    """Get GPU memory in GB (defaults to 16 if unable to detect)."""
    try:
        import torch
        if torch.cuda.is_available():
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            return int(mem_bytes / (1024 ** 3))
        elif torch.backends.mps.is_available():
            # MPS doesn't expose total memory, assume based on common configs
            # Apple M1/M2: 8-24GB unified memory
            return 16  # Conservative default
    except:
        pass
    return 16  # Default


def is_mps_device() -> bool:
    """Check if using Apple MPS backend."""
    try:
        import torch
        return not torch.cuda.is_available() and torch.backends.mps.is_available()
    except:
        return False


# Auto-detect batch size
GPU_MEM_GB = get_gpu_memory_gb()
USE_MPS = is_mps_device()
AUTO_BATCH_SIZE = get_batch_size_for_gpu(GPU_MEM_GB, is_mps=USE_MPS)

print(f"[CONFIG] Detected GPU memory: {GPU_MEM_GB}GB")
print(f"[CONFIG] Device type: {'MPS (Apple Silicon)' if USE_MPS else 'CUDA'}")
print(f"[CONFIG] Auto-selected batch size: {AUTO_BATCH_SIZE}")
print(f"[CONFIG] Model parameters: ~200M (d_model={D_MODEL}, layers={NUM_LAYERS}, heads={NUM_HEADS})")

if USE_MPS:
    print(f"[CONFIG] ⚠️  MPS detected - using conservative batch size to avoid OOM")
    print(f"[CONFIG] 💡 Tip: Consider reducing MAX_LENGTH to 512 for faster training")
