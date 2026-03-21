"""
HF Trainer Inference — Decoder-Only with Mixture-of-Experts (MoE)

Original: TrainerScripts/DecoderMoEInference.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch

from core.Embedding import get_tokenizer
from models.DecoderMoE import DecoderOnlyMoEModel
from HFTrainerScripts.hf_wrapper import HFModelWrapper, load_wrapper_from_checkpoint

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "HF_MoECheckpoints")
MAX_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_dir=CHECKPOINT_DIR):
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    base_model = DecoderOnlyMoEModel(
        vocab_size=vocab_size, d_model=256, max_positions=MAX_LENGTH,
        num_layers=4, num_heads=4, d_ff=512, tokenizer=tokenizer,
        dropout=0.1, pad_token_id=tokenizer.pad_token_id, lr=1e-3,
        num_experts=4, top_k=2,
    )
    wrapper = HFModelWrapper(base_model)
    model = load_wrapper_from_checkpoint(wrapper, checkpoint_dir)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def greedy_decode(model, tokenizer, prompt, max_len=100):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    enc = tokenizer(prompt, truncation=True, max_length=max_len - 1, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    if bos_id is not None and input_ids[0, 0].item() != bos_id:
        input_ids = torch.cat([torch.tensor([[bos_id]], device=DEVICE), input_ids], dim=1)

    output_ids = model.generate_greedy(input_ids, max_new_tokens=max_len - input_ids.size(1), eos_token_id=eos_id)
    return tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()

    prompts = [
        "Artificial intelligence is",
        "The rise of renewable energy is changing global markets and Experts predict this shift will redefine economies",
        "Climate change poses significant challenges such as Researchers have pointed out that this shift is inevitable",
    ]

    for prompt in prompts:
        output = greedy_decode(model, tokenizer, prompt, max_len=100)
        print("\n----------------------------------------------")
        print("Input :", prompt)
        print("Output:", output)
        print("----------------------------------------------\n")
