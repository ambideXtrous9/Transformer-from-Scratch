import torch
from typing import Optional
from Seq2SeqModel import Seq2SeqModel
from Embedding import get_tokenizer, tokenize_batch

def greedy_decode(
    model: Seq2SeqModel,
    tokenizer,
    src_text: str,
    max_len: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """
    Greedy decoding for inference.
    """
    model.eval()
    model.to(device)

    # Tokenize source text
    batch = tokenize_batch(tokenizer, [src_text], max_length=max_len)
    src_ids = batch["input_ids"].to(device)
    src_mask = batch["attention_mask"].to(device)

    # Encode
    with torch.no_grad():
        enc_out, _ = model.encoder(src_ids, src_mask)

    # Initialize decoder input with BOS token
    if bos_token_id is None:
        bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    tgt_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

    # Iterative decoding
    for _ in range(max_len):
        with torch.no_grad():
            dec_out, _, _ = model.decoder(tgt_ids, enc_out, tgt_mask=None, memory_mask=src_mask)
            logits = model.classifier(dec_out)  # (1, seq_len, vocab_size)
            next_token = logits[:, -1, :].argmax(dim=-1)  # greedy pick

        # Append predicted token
        tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(0)], dim=1)

        # Stop if EOS is generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    # Decode tokens into text
    decoded = tokenizer.decode(tgt_ids.squeeze().tolist(), skip_special_tokens=True)
    return decoded


if __name__ == "__main__":
    tokenizer = get_tokenizer("gpt2", add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)

    model = Seq2SeqModel.load_from_checkpoint("checkpoints/BestModel.ckpt", vocab_size=vocab_size)

    #src_text = "Artificial intelligence is transforming"
    src_text = "The rise of renewable energy is changing global markets and Experts predict this shift will redefine economies"
    #src_text = "Climate change poses significant challenges such as Researchers have pointed out that this shift is inevitable"
    output = greedy_decode(model, tokenizer, src_text, max_len=20)
    print("Input :", src_text)
    print("Output:", output)
