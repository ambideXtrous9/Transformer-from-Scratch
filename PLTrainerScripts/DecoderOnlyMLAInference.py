import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import config
import glob
import torch
from typing import Optional
from models.DecoderOnlyMLAModel import DecoderOnlyMLAModel
from core.Embedding import get_tokenizer, tokenize_batch


def greedy_decode(
    model: DecoderOnlyMLAModel,
    tokenizer,
    prompt: str,
    max_len: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """
    Greedy decoding for decoder-only model (GPT-style).
    """
    model.eval()
    model.to(device)

    # Handle special tokens
    if bos_token_id is None:
        bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

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
    # Find all ckpt files in directory
    ckpt_list = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_list:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    # Get latest by modification time
    latest_ckpt = max(ckpt_list, key=os.path.getmtime)
    print(f"Loading latest checkpoint: {latest_ckpt}")

    model = DecoderOnlyMLAModel.load_from_checkpoint(
        latest_ckpt,
        vocab_size=vocab_size,
        tokenizer=tokenizer
    )
    return model


if __name__ == "__main__":
    tokenizer = get_tokenizer(config.TOKENIZER_NAME, add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    model = load_latest_checkpoint(config.CHECKPOINTS["mla"], vocab_size, tokenizer)

    questions = config.SAMPLE_QUESTIONS

    for q in questions:
        prompt = f"Question: {q}\nAnswer:"
        output = greedy_decode(model, tokenizer, prompt, max_len=config.MAX_LENGTH)
        print("\n" + "=" * 60)
        print(f"Question: {q}")
        print(f"\nGenerated Answer:\n{output}")
        print("=" * 60)
