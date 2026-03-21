import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import glob
import torch
from typing import Optional
from PLTrainerScripts.GSM8KTrainer import GSM8KModel
from core.Embedding import get_tokenizer


def greedy_decode(
    model: GSM8KModel,
    tokenizer,
    question: str,
    max_len: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """
    Greedy decoding for math question answering.

    The prompt is formatted as:
        "Question: {question}\nAnswer:"

    The model then generates the step-by-step solution.
    """
    model.eval()
    model.to(device)

    # Handle special tokens
    if bos_token_id is None:
        bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # Format prompt
    prompt = f"Question: {question}\nAnswer:"

    # Tokenize prompt
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len - 1,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)

    # Prepend BOS if needed
    if bos_token_id is not None:
        if input_ids[0, 0].item() != bos_token_id:
            input_ids = torch.cat(
                [torch.tensor([[bos_token_id]], device=device), input_ids], dim=1
            )

    # Iterative decoding
    for _ in range(max_len - input_ids.size(1)):
        with torch.no_grad():
            logits, _ = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1)  # greedy

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop if EOS generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    # Decode tokens into text
    decoded = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
    return decoded


def load_latest_checkpoint(checkpoint_dir, vocab_size, tokenizer):
    """Load the latest checkpoint from the GSM8K checkpoint directory."""
    ckpt_list = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_list:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    latest_ckpt = max(ckpt_list, key=os.path.getmtime)
    print(f"Loading latest checkpoint: {latest_ckpt}")

    model = GSM8KModel.load_from_checkpoint(
        latest_ckpt,
        vocab_size=vocab_size,
        tokenizer=tokenizer
    )
    return model


if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    model = load_latest_checkpoint(os.path.join(PROJECT_ROOT, 'checkpoints', 'GSM8KCheckpoints'), vocab_size, tokenizer)

    # Sample GSM8K-style questions
    questions = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    ]

    for q in questions:
        output = greedy_decode(model, tokenizer, q, max_len=256)
        print("\n" + "=" * 60)
        print(f"Question: {q}")
        print(f"\nGenerated Answer:\n{output}")
        print("=" * 60)
