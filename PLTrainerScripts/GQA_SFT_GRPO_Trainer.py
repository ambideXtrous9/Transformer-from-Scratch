"""
PyTorch Lightning — GRPO on GSM8K with GQA model.

Loads a pre-trained PL GQA checkpoint and fine-tunes it with Group Relative Policy
Optimization (GRPO). For each question, samples G completions, scores them with a
reward function (exact-match on the final numerical answer), computes group-normalised
advantages, and updates the policy with a clipped surrogate objective + KL penalty
against the frozen reference.

Prerequisite: Train a GQA model first with PLTrainerScripts/DecoderOnlyGQATrainer.py

Reference: DeepSeek-R1 (Shao et al., 2025) — "DeepSeek-R1: Incentivizing Reasoning
           Capability in LLMs via Reinforcement Learning"
"""

import os, sys, re, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from core.Embedding import get_tokenizer
from models.DecoderOnlyGQAModel import DecoderOnlyGQAModel
from datasets import load_dataset
import config
import pytorch_lightning as pl
from dotenv import load_dotenv
import wandb

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

MAX_LENGTH = config.MAX_LENGTH


# ==================== Reward Function ====================

def extract_final_number(text: str):
    """
    Extract the numerical answer from GSM8K-style output.
    Looks for '#### <number>' first, then falls back to the last number in the text.
    """
    # Try GSM8K canonical format: #### <number>
    match = re.search(r'####\s*([\-\d,\.]+)', text)
    if match:
        num_str = match.group(1).replace(',', '')
        try:
            return float(num_str)
        except ValueError:
            pass

    # Fallback: last number in text
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def compute_reward(generated_text: str, reference_text: str) -> float:
    """
    Reward = 1.0 if the final number in the generated text matches the reference,
             0.0 otherwise.  Also gives +0.1 bonus if the '####' format is present.
    """
    gen_num = extract_final_number(generated_text)
    ref_num = extract_final_number(reference_text)

    if gen_num is None or ref_num is None:
        return 0.0

    reward = 0.0
    # Exact numerical match (within floating-point tolerance)
    if abs(gen_num - ref_num) < 1e-3:
        reward = 1.0

    # Format bonus: model learned to use #### notation
    if '####' in generated_text:
        reward += 0.1

    return reward


# ==================== Datasets ====================

class GSM8KDataset(Dataset):
    """SFT dataset: full question+answer pairs for next-token prediction."""
    def __init__(self, tokenizer, hf_dataset, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        for sample in hf_dataset:
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
            self.texts.append(text)

        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length - 2,
            return_tensors="pt", add_special_tokens=False
        )
        ids = enc["input_ids"].squeeze(0)

        input_ids = torch.cat([torch.tensor([self.bos_id]), ids], dim=0)
        labels = torch.cat([ids, torch.tensor([self.eos_id])], dim=0)

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_id)])
        else:
            input_ids = input_ids[:self.max_length]

        if len(labels) < self.max_length:
            pad_len = self.max_length - len(labels)
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            labels = labels[:self.max_length]

        return {"input_ids": input_ids.long(), "labels": labels.long()}


class GSM8KPromptDataset(Dataset):
    """GRPO dataset: only questions (prompts) + reference answers for reward."""
    def __init__(self, tokenizer, hf_dataset, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.questions = []
        self.answers = []
        for sample in hf_dataset:
            self.questions.append(sample['question'])
            self.answers.append(sample['answer'])

        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        prompt = f"Question: {self.questions[idx]}\nAnswer:"
        enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length - 1,
            return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = enc["input_ids"].squeeze(0)
        prompt_ids = torch.cat([torch.tensor([self.bos_id]), prompt_ids], dim=0)

        return {
            "prompt_ids": prompt_ids.long(),
            "prompt_len": len(prompt_ids),
            "reference_answer": self.answers[idx],
        }


def grpo_collate_fn(batch):
    """Collate prompts into a padded batch; keep reference answers as list."""
    max_len = max(item["prompt_len"] for item in batch)
    pad_id = batch[0]["prompt_ids"].new_full((1,), 0).item()  # will be overwritten

    prompt_ids_list = []
    prompt_lens = []
    ref_answers = []

    for item in batch:
        ids = item["prompt_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        prompt_ids_list.append(ids)
        prompt_lens.append(item["prompt_len"])
        ref_answers.append(item["reference_answer"])

    return {
        "prompt_ids": torch.stack(prompt_ids_list),
        "prompt_lens": prompt_lens,
        "reference_answers": ref_answers,
    }


# ==================== Sampling utility ====================

@torch.no_grad()
def sample_completions(model, prompt_ids, prompt_len, max_new_tokens, eos_token_id,
                       temperature=0.8, top_k=50, device="cuda"):
    """
    Autoregressive sampling from the policy model.
    Returns the full sequence (prompt + completion) and the completion-only token ids.
    """
    input_ids = prompt_ids.unsqueeze(0).to(device)  # (1, prompt_len)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= MAX_LENGTH:
            break
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    full_ids = input_ids.squeeze(0)          # (total_len,)
    completion_ids = full_ids[prompt_len:]    # only the generated part
    return full_ids, completion_ids


# ==================== GRPO per-token log-probs ====================

def get_per_token_logprobs(model, full_ids, prompt_len, device):
    """
    Compute log π(a_t | s_t) for each generated token.
    full_ids: (total_len,) — prompt + completion.
    Returns: log_probs tensor of shape (num_generated_tokens,).
    """
    input_ids = full_ids[:-1].unsqueeze(0).to(device)   # (1, T-1)
    target_ids = full_ids[1:].to(device)                 # (T-1,)

    logits, _ = model(input_ids)
    logits = logits.squeeze(0)  # (T-1, V)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(target_ids)), target_ids]  # (T-1,)

    # Only keep the completion part (after prompt)
    # prompt occupies positions 0..prompt_len-1 in input, so targets start at index prompt_len-1
    completion_log_probs = token_log_probs[prompt_len - 1:]
    return completion_log_probs


# ==================== Main ====================

if __name__ == "__main__":
    pl.seed_everything(config.SEED)
    print("Loading GSM8K dataset from HuggingFace...")
    gsm8k = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    train_data = gsm8k["train"]
    test_data = gsm8k["test"]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")

    tokenizer = get_tokenizer(config.TOKENIZER_NAME, add_pad_token_if_missing=True)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=config.WANDB_PROJECT,
        name="GQA-SFT-GRPO",
        config={
            "architecture": "GQA-SFT-GRPO",
            "d_model": config.D_MODEL,
            "num_layers": config.NUM_LAYERS,
            "num_heads": config.NUM_HEADS,
            "num_kv_heads": config.NUM_KV_HEADS,
            "d_ff": config.D_FF,
            "dropout": config.DROPOUT,
            "grpo_lr": config.GRPO_LR,
            "grpo_beta": config.GRPO_BETA,
            "grpo_clip_eps": config.GRPO_CLIP_EPS,
            "grpo_group_size": config.GRPO_GROUP_SIZE,
            "grpo_epochs": config.GRPO_EPOCHS,
            "max_length": MAX_LENGTH,
            "batch_size": config.TRAIN_BATCH_SIZE,
        }
    )

    # ================================================================
    # GRPO (Group Relative Policy Optimization)
    # Loads a pre-trained GQA checkpoint and fine-tunes with RL.
    # Train the base model first with DecoderOnlyGQATrainer.py
    # ================================================================
    print("\n" + "=" * 60)
    print("Group Relative Policy Optimization (GRPO)")
    print("=" * 60 + "\n")

    # Load pre-trained checkpoint as the starting policy
    import glob as glob_mod
    ckpt_dir = config.CHECKPOINTS["gqa"]
    ckpt_list = glob_mod.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_list:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. "
            "Train a GQA model first with: python PLTrainerScripts/DecoderOnlyGQATrainer.py"
        )

    best_ckpt = max(ckpt_list, key=os.path.getmtime)
    print(f"Loading pre-trained checkpoint: {best_ckpt}")
    policy_model = DecoderOnlyGQAModel.load_from_checkpoint(
        best_ckpt, vocab_size=vocab_size, tokenizer=tokenizer
    )

    policy_model = policy_model.to(device)
    policy_model.train()

    # Freeze a copy as reference policy
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # GRPO optimizer
    grpo_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.GRPO_LR,
                                        weight_decay=config.WEIGHT_DECAY)

    # GRPO dataset — prompts only
    grpo_dataset = GSM8KPromptDataset(tokenizer, train_data, max_length=MAX_LENGTH)
    grpo_loader = DataLoader(grpo_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                             shuffle=True, num_workers=0, collate_fn=grpo_collate_fn)

    G = config.GRPO_GROUP_SIZE
    beta = config.GRPO_BETA
    clip_eps = config.GRPO_CLIP_EPS
    max_new_tokens = config.GRPO_MAX_NEW_TOKENS

    global_step = 0
    best_avg_reward = -float("inf")

    for epoch in range(config.GRPO_EPOCHS):
        epoch_policy_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_reward = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(grpo_loader):
            prompt_ids_batch = batch["prompt_ids"]          # (B, max_prompt_len)
            prompt_lens = batch["prompt_lens"]              # list of int
            ref_answers = batch["reference_answers"]        # list of str
            B = len(prompt_lens)

            batch_policy_loss = torch.tensor(0.0, device=device)
            batch_kl_loss = torch.tensor(0.0, device=device)
            batch_rewards = []
            n_completions = 0

            for i in range(B):
                prompt = prompt_ids_batch[i][:prompt_lens[i]]

                # --- Sample G completions from current policy ---
                completions = []
                rewards = []
                for _ in range(G):
                    policy_model.eval()
                    full_ids, comp_ids = sample_completions(
                        policy_model, prompt, prompt_lens[i],
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_id, temperature=config.TEMPERATURE,
                        top_k=config.TOP_K_SAMPLING, device=device
                    )
                    policy_model.train()

                    # Decode and compute reward
                    gen_text = tokenizer.decode(comp_ids.tolist(), skip_special_tokens=True)
                    reward = compute_reward(gen_text, ref_answers[i])
                    completions.append(full_ids)
                    rewards.append(reward)

                batch_rewards.extend(rewards)

                # --- Group-relative advantage ---
                rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
                mean_r = rewards_tensor.mean()
                std_r = rewards_tensor.std() + 1e-8
                advantages = (rewards_tensor - mean_r) / std_r

                # --- Compute GRPO loss for each completion ---
                for j in range(G):
                    full_ids = completions[j]
                    advantage = advantages[j]

                    if len(full_ids) <= prompt_lens[i]:
                        continue  # no completion tokens generated

                    # Log-probs under current policy
                    policy_logprobs = get_per_token_logprobs(
                        policy_model, full_ids, prompt_lens[i], device
                    )

                    # Log-probs under reference policy
                    with torch.no_grad():
                        ref_logprobs = get_per_token_logprobs(
                            ref_model, full_ids, prompt_lens[i], device
                        )

                    # Importance ratio
                    ratio = torch.exp(policy_logprobs - ref_logprobs.detach())

                    # Clipped surrogate objective
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # KL penalty: KL(π_θ || π_ref) ≈ (π_ref/π_θ - 1) - log(π_ref/π_θ)
                    # This is the non-parametric approximation from Schulman (2020)
                    log_ratio = policy_logprobs - ref_logprobs.detach()
                    kl = torch.exp(-log_ratio) - 1.0 + log_ratio
                    kl_loss = kl.mean()

                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_kl_loss = batch_kl_loss + kl_loss
                    n_completions += 1

            if n_completions == 0:
                continue

            # Average over completions in this batch
            total_loss = (batch_policy_loss + beta * batch_kl_loss) / n_completions

            grpo_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            grpo_optimizer.step()

            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            epoch_policy_loss += (batch_policy_loss.item() / max(n_completions, 1))
            epoch_kl_loss += (batch_kl_loss.item() / max(n_completions, 1))
            epoch_reward += avg_reward
            epoch_steps += 1
            global_step += 1

            if global_step % 10 == 0:
                print(f"  [Step {global_step}] policy_loss={batch_policy_loss.item()/max(n_completions,1):.4f}  "
                      f"kl={batch_kl_loss.item()/max(n_completions,1):.4f}  "
                      f"avg_reward={avg_reward:.3f}")

            wandb.log({
                "grpo/policy_loss": batch_policy_loss.item() / max(n_completions, 1),
                "grpo/kl_loss": batch_kl_loss.item() / max(n_completions, 1),
                "grpo/avg_reward": avg_reward,
                "grpo/total_loss": total_loss.item(),
            }, step=global_step)

        # Epoch summary
        if epoch_steps > 0:
            avg_ep_reward = epoch_reward / epoch_steps
            print(f"\n----------------------------------------------\n"
                  f"  GRPO Epoch {epoch+1}/{config.GRPO_EPOCHS}\n"
                  f"  Avg Policy Loss: {epoch_policy_loss / epoch_steps:.4f}\n"
                  f"  Avg KL Loss:     {epoch_kl_loss / epoch_steps:.4f}\n"
                  f"  Avg Reward:      {avg_ep_reward:.4f}\n"
                  f"----------------------------------------------\n")

            wandb.log({
                "grpo/epoch_avg_policy_loss": epoch_policy_loss / epoch_steps,
                "grpo/epoch_avg_kl_loss": epoch_kl_loss / epoch_steps,
                "grpo/epoch_avg_reward": avg_ep_reward,
                "grpo/epoch": epoch + 1,
            }, step=global_step)

            # Save if best reward so far
            if avg_ep_reward > best_avg_reward:
                best_avg_reward = avg_ep_reward
                grpo_save_dir = config.CHECKPOINTS["gqa_sft_grpo"]
                os.makedirs(grpo_save_dir, exist_ok=True)
                save_path = os.path.join(grpo_save_dir, "GQA_GRPO_BestModel.ckpt")
                torch.save({
                    "state_dict": policy_model.state_dict(),
                    "hparams": policy_model.hparams,
                    "epoch": epoch,
                    "avg_reward": avg_ep_reward,
                }, save_path)
                print(f"  ** New best GRPO model saved (reward={avg_ep_reward:.4f})")

    print("\n" + "=" * 60)
    print("Training complete — SFT + GRPO")
    print("=" * 60)

    wandb.finish()
