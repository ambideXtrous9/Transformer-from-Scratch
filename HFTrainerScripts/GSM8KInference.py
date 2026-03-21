"""
HF Trainer Inference — GSM8K Math Problem Solving

Original: TrainerScripts/GSM8KInference.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch

from core.Embedding import get_tokenizer
from models.DecoderOnlySeq2SeqModel import DecoderOnlyModel
from HFTrainerScripts.hf_wrapper import HFModelWrapper, load_wrapper_from_checkpoint

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "HF_GSM8KCheckpoints")
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_dir=CHECKPOINT_DIR):
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    base_model = DecoderOnlyModel(
        vocab_size=vocab_size, d_model=256, max_positions=MAX_LENGTH,
        num_layers=4, num_heads=4, d_ff=512, tokenizer=tokenizer,
        dropout=0.1, pad_token_id=tokenizer.pad_token_id, lr=1e-3,
    )
    wrapper = HFModelWrapper(base_model)
    model = load_wrapper_from_checkpoint(wrapper, checkpoint_dir)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def greedy_decode(model, tokenizer, question, max_len=256):
    """Greedy decoding for math QA. Prompt: 'Question: ...\nAnswer:'"""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    prompt = f"Question: {question}\nAnswer:"
    enc = tokenizer(prompt, truncation=True, max_length=max_len - 1, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    if bos_id is not None and input_ids[0, 0].item() != bos_id:
        input_ids = torch.cat([torch.tensor([[bos_id]], device=DEVICE), input_ids], dim=1)

    output_ids = model.generate_greedy(
        input_ids, max_new_tokens=max_len - input_ids.size(1), eos_token_id=eos_id,
    )
    return tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()

    questions = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    ]

    for q in questions:
        output = greedy_decode(model, tokenizer, q, max_len=MAX_LENGTH)
        print("\n" + "=" * 60)
        print(f"Question: {q}")
        print(f"\nGenerated Answer:\n{output}")
        print("=" * 60)
