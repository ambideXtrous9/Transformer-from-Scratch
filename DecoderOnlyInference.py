import os
import glob
import torch
from typing import Optional
from DecoderOnlySeq2SeqModel import DecoderOnlyModel
from Embedding import get_tokenizer, tokenize_batch


def greedy_decode(
    model: DecoderOnlyModel,
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
    
    model = DecoderOnlyModel.load_from_checkpoint(
        latest_ckpt,
        vocab_size=vocab_size,
        tokenizer=tokenizer
    )
    return model


if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    model = load_latest_checkpoint("DecoderOnlyCheckpoints", vocab_size, tokenizer)

    src_text = "Artificial intelligence is transforming"
    #src_text = "The rise of renewable energy is changing global markets and Experts predict this shift will redefine economies"
    # src_text = "Climate change poses significant challenges such as Researchers have pointed out that this shift is inevitable"
    output = greedy_decode(model, tokenizer, src_text, max_len=100)
    print("\n----------------------------------------------\n")
    print("Input :", src_text)
    print("Output:", output)
    print("\n----------------------------------------------\n")
