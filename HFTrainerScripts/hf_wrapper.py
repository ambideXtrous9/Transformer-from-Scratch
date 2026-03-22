"""
HF-compatible wrapper for custom decoder-only models.

Wraps any of the project's custom models (MHA, MQA, GQA, MLA, MoE) so they work
with HuggingFace Trainer.

The custom models' forward() returns (logits, attn_maps).
This wrapper adds loss computation and returns a CausalLMOutput that HF Trainer expects.
"""

import glob
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet', quiet=True)
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from transformers import PretrainedConfig, TrainerCallback
from transformers.modeling_outputs import CausalLMOutput


class CustomModelConfig(PretrainedConfig):
    """Config that satisfies HF Trainer callbacks (.to_json_string(), .get(), etc.)."""
    model_type = "custom_decoder_only"

    def __init__(self, vocab_size=50257, max_positions=64, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.is_encoder_decoder = False
        self.tie_word_embeddings = False


class HFModelWrapper(nn.Module):
    """
    Wrapper that makes custom decoder-only models compatible with HF Trainer.

    - forward() returns CausalLMOutput (loss, logits, attentions)
    - generate_greedy() supports temperature, top-k, top-p, repetition penalty
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        # Extract model hparams for config
        hp = getattr(model, "hparams", {})
        self.config = CustomModelConfig(
            vocab_size=hp.get("vocab_size", 50257),
            max_positions=hp.get("max_positions", 64),
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None, **kwargs):
        logits, attn_maps = self.model(input_ids)

        loss = None
        if labels is not None:
            # Shift: logits[..., :-1, :] predicts labels[..., 1:]
            # But our dataset already handles the shift:
            #   input_ids = [BOS] + tokens
            #   labels    = tokens + [EOS]
            # So logits[i] should predict labels[i] — no extra shift needed.
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            attentions=tuple(attn_maps) if attn_maps else None,
        )

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids,
        max_new_tokens=128,
        eos_token_id=None,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
    ):
        """
        Autoregressive decoding with sampling controls.

        Args:
            temperature: >1.0 = more random, <1.0 = more focused, 0.0 = pure greedy
            top_k: keep only top-k logits before sampling (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: penalise already-generated tokens (1.0 = disabled)
        """
        self.eval()
        max_positions = self.config.max_positions

        for _ in range(max_new_tokens):
            if input_ids.size(1) >= max_positions:
                break

            logits, _ = self.model(input_ids)
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # --- Repetition penalty ---
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            # --- Temperature ---
            if temperature <= 0 or temperature == 1e-9:
                # Pure greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

                # --- Top-k filtering ---
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float("-inf")

                # --- Top-p (nucleus) filtering ---
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative prob above threshold
                    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    # Scatter back
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids


# ==================== Metrics ====================

def preprocess_logits_for_metrics(logits, labels):
    """Convert logits to predicted token IDs before accumulation (saves memory)."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function for HF Trainer that computes
    BLEU, ROUGE-1/2/L, and METEOR — matching the PL Trainer metrics.
    """
    pad_token_id = tokenizer.pad_token_id

    def compute_metrics(eval_preds):
        pred_ids, label_ids = eval_preds

        # Decode predictions (already argmax'd by preprocess_logits_for_metrics)
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Replace -100 in labels with pad token before decoding
        label_ids = np.where(label_ids != -100, label_ids, pad_token_id)
        ref_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if len(pred_texts) == 0:
            return {}

        # BLEU
        bleu = sacrebleu.corpus_bleu(pred_texts, [ref_texts])

        # ROUGE
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = [scorer.score(r, g) for r, g in zip(ref_texts, pred_texts)]

        avg_rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
        avg_rouge2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)

        # METEOR
        meteor_scores = [
            meteor_score([r.split()], g.split())
            for r, g in zip(ref_texts, pred_texts)
        ]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        print(
            f"\n----------------------------------------------\n"
            f" BLEU: {bleu.score:.2f}\n"
            f" ROUGE-1: {avg_rouge1:.4f}\n"
            f" ROUGE-2: {avg_rouge2:.4f}\n"
            f" ROUGE-L: {avg_rougeL:.4f}\n"
            f" METEOR: {avg_meteor:.4f}\n"
            f"----------------------------------------------\n"
        )

        return {
            "bleu": bleu.score,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "meteor": avg_meteor,
        }

    return compute_metrics


# ==================== Save-best callback ====================

class SaveBestModelCallback(TrainerCallback):
    """
    Saves exactly ONE checkpoint — the model with the lowest eval_loss.

    Usage:
        callback = SaveBestModelCallback(save_dir="checkpoints/MyModel/best")
        trainer = Trainer(..., callbacks=[callback])

    Use save_strategy="no" in TrainingArguments to disable HF Trainer's
    own checkpoint saving, so only this callback writes to disk.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_eval_loss = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            print(f"\n  ** New best eval_loss: {eval_loss:.4f} (epoch {state.epoch:.0f}) — saving to {self.save_dir}")

            # Clear previous best
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)

            # Save model weights
            state_dict = model.state_dict()
            save_safetensors(state_dict, os.path.join(self.save_dir, "model.safetensors"))

            # Save a marker with the loss for reference
            with open(os.path.join(self.save_dir, "best_metric.txt"), "w") as f:
                f.write(f"eval_loss={eval_loss:.6f}\nepoch={state.epoch:.0f}\nglobal_step={state.global_step}\n")
        else:
            print(f"\n  eval_loss: {eval_loss:.4f} (best: {self.best_eval_loss:.4f}) — not saving")


# ==================== Checkpoint utilities ====================

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the best saved checkpoint directory.

    Priority:
      1. best/ subdirectory (saved by SaveBestModelCallback)
      2. checkpoint-XXXX/ subdirectories (HF Trainer pattern, latest by mtime)
      3. The directory itself
    """
    # 1. SaveBestModelCallback output
    best_path = os.path.join(checkpoint_dir, "best")
    if os.path.isfile(os.path.join(best_path, "model.safetensors")):
        return best_path

    # 2. HF Trainer checkpoint-XXXX/ directories
    ckpt_dirs = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=os.path.getmtime,
    )
    for d in reversed(ckpt_dirs):
        if (os.path.isfile(os.path.join(d, "model.safetensors"))
                or os.path.isfile(os.path.join(d, "pytorch_model.bin"))):
            return d

    # 3. The directory itself
    if (os.path.isfile(os.path.join(checkpoint_dir, "model.safetensors"))
            or os.path.isfile(os.path.join(checkpoint_dir, "pytorch_model.bin"))):
        return checkpoint_dir

    raise FileNotFoundError(
        f"No model checkpoint found in {checkpoint_dir}. Train a model first."
    )


def load_wrapper_from_checkpoint(wrapper, checkpoint_dir):
    """Load saved state_dict into an HFModelWrapper from HF Trainer checkpoint."""
    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    print(f"Loading checkpoint from: {ckpt_path}")

    safetensors_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        state_dict = load_safetensors(safetensors_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights found in {ckpt_path}")

    wrapper.load_state_dict(state_dict)
    return wrapper
