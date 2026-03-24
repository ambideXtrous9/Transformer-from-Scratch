"""
Centralized configuration for all trainers and inference scripts.

Edit values here to control hyperparameters across the entire project.
All trainer and inference scripts import from this file.
"""

# ==================== Dataset ====================
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
TOKENIZER_NAME = "gpt2"

# ==================== Sequence ====================
MAX_LENGTH = 256              # max sequence length for tokenization & positional embeddings

# ==================== Model (shared) ====================
D_MODEL = 512                 # model / embedding dimension
NUM_HEADS = 8                 # number of attention heads
NUM_LAYERS = 6                # number of decoder layers
D_FF = 2048                    # feed-forward inner dimension
DROPOUT = 0.1                 # dropout probability

# ==================== Seq2Seq (CrossAttention) ====================
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
SEQ2SEQ_D_FF = 128            # Seq2Seq uses a smaller FFN

# ==================== GQA ====================
NUM_KV_HEADS = 2              # number of key/value heads for Group Query Attention

# ==================== MLA ====================
D_COMPRESS = 64               # latent compression dimension for Multi-Head Latent Attention

# ==================== MoE ====================
NUM_EXPERTS = 4               # number of expert networks
TOP_K = 2                     # experts activated per token

# ==================== Training ====================
SEED = 42
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 100
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
NUM_WORKERS = 2
LOGGING_STEPS = 50            # HF Trainer: log every N steps

# ==================== Inference (HF generate_greedy) ====================
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_K_SAMPLING = 50           # top-k for sampling (0 = disabled)
TOP_P = 0.9                   # nucleus sampling threshold (1.0 = disabled)
REPETITION_PENALTY = 1.2      # penalise repeated tokens (1.0 = disabled)

# ==================== Checkpoint directories ====================
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))

CHECKPOINTS = {
    # PL
    "seq2seq":       os.path.join(_ROOT, "checkpoints", "Seq2SeqCheckpoints"),
    "decoder_only":  os.path.join(_ROOT, "checkpoints", "DecoderOnlyCheckpoints"),
    "moe":           os.path.join(_ROOT, "checkpoints", "DecoderMoECheckpoints"),
    "gqa":           os.path.join(_ROOT, "checkpoints", "GQACheckpoints"),
    "mqa":           os.path.join(_ROOT, "checkpoints", "MQACheckpoints"),
    "mla":           os.path.join(_ROOT, "checkpoints", "MLACheckpoints"),
    "gsm8k":         os.path.join(_ROOT, "checkpoints", "GSM8KCheckpoints"),
    # HF
    "hf_decoder_only": os.path.join(_ROOT, "checkpoints", "HF_DecoderOnlyCheckpoints"),
    "hf_moe":          os.path.join(_ROOT, "checkpoints", "HF_MoECheckpoints"),
    "hf_gqa":          os.path.join(_ROOT, "checkpoints", "HF_GQACheckpoints"),
    "hf_mqa":          os.path.join(_ROOT, "checkpoints", "HF_MQACheckpoints"),
    "hf_mla":          os.path.join(_ROOT, "checkpoints", "HF_MLACheckpoints"),
    "hf_gsm8k":        os.path.join(_ROOT, "checkpoints", "HF_GSM8KCheckpoints"),
}

# ==================== Sample questions for inference ====================
SAMPLE_QUESTIONS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
]
