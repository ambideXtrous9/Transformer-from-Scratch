"""
HF Trainer Inference — GSM8K Math Problem Solving

Original: TrainerScripts/GSM8KInference.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import config

from core.Embedding import get_tokenizer
from models.DecoderOnlySeq2SeqModel import DecoderOnlyModel
from HFTrainerScripts.hf_wrapper import HFModelWrapper, load_wrapper_from_checkpoint, compute_batch_perplexity

CHECKPOINT_DIR = config.CHECKPOINTS["hf_gsm8k"]
MAX_LENGTH = config.MAX_LENGTH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_dir=CHECKPOINT_DIR):
    tokenizer = get_tokenizer(config.TOKENIZER_NAME, add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    base_model = DecoderOnlyModel(
        vocab_size=vocab_size, d_model=config.D_MODEL, max_positions=MAX_LENGTH,
        num_layers=config.NUM_LAYERS, num_heads=config.NUM_HEADS, d_ff=config.D_FF, tokenizer=tokenizer,
        dropout=config.DROPOUT, pad_token_id=tokenizer.pad_token_id, lr=config.LEARNING_RATE,
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

    questions = config.SAMPLE_QUESTIONS

    for q in questions:
        output = greedy_decode(model, tokenizer, q, max_len=MAX_LENGTH)
        print("\n" + "=" * 60)
        print(f"Question: {q}")
        print(f"\nGenerated Answer:\n{output}")
        print("=" * 60)

    # Evaluate perplexity on sample texts
    eval_texts = [f"Question: {q}\nAnswer:" for q in questions]
    ppl_result = compute_batch_perplexity(model, tokenizer, eval_texts, max_length=MAX_LENGTH, device=DEVICE)
    print("\n" + "=" * 60)
    print("Perplexity Evaluation")
    for text, ppl in zip(questions, ppl_result["perplexities"]):
        print(f"  {text[:60]}... → PPL: {ppl:.2f}")
    print(f"\n  Mean Perplexity: {ppl_result['mean_perplexity']:.2f}")
    print("=" * 60)
