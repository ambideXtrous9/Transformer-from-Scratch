"""
Inference and Generation Utilities

Production-level inference with:
- Model loading from checkpoints
- Interactive text generation
- Batch generation
- Streaming generation
- Safe generation with various decoding strategies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from typing import Optional, List, Union
import time

from models.ModernLLM import ModernDecoderOnlyModel
import config_200m as config


class LLMInference:
    """
    Production-level LLM inference interface.
    
    Provides:
    - Easy model loading
    - Text generation with various strategies
    - Batch generation
    - Interactive chat-like interface
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: Path to model checkpoint or directory
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        print(f"[INFERENCE] Loading tokenizer from {config.TOKENIZER_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"[INFERENCE] Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFERENCE] Model loaded on {self.device}")
        print(f"[INFERENCE] Parameters: {self.model.get_num_params():,}")
    
    def _load_model(self, model_path: str) -> ModernDecoderOnlyModel:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to checkpoint or model directory
        
        Returns:
            Loaded model
        """
        # Check if path is a directory with config
        if os.path.isdir(model_path):
            # Try to load from checkpoint
            checkpoint_path = os.path.join(model_path, "checkpoint.ckpt")
            if not os.path.exists(checkpoint_path):
                # Find .ckpt files in directory
                ckpt_files = [f for f in os.listdir(model_path) if f.endswith(".ckpt")]
                if ckpt_files:
                    checkpoint_path = os.path.join(model_path, ckpt_files[0])
                else:
                    raise FileNotFoundError(f"No checkpoint found in {model_path}")
        else:
            checkpoint_path = model_path
        
        # Load checkpoint
        print(f"[INFERENCE] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})
        
        # Create model with saved hyperparameters
        model = ModernDecoderOnlyModel(
            vocab_size=hparams.get("vocab_size", config.VOCAB_SIZE),
            d_model=hparams.get("d_model", config.D_MODEL),
            num_heads=hparams.get("num_heads", config.NUM_HEADS),
            num_kv_heads=hparams.get("num_kv_heads", config.NUM_KV_HEADS),
            num_layers=hparams.get("num_layers", config.NUM_LAYERS),
            d_ff=hparams.get("d_ff", config.D_FF),
            max_seq_length=hparams.get("max_seq_length", config.MAX_LENGTH),
            dropout=hparams.get("dropout", config.DROPOUT),
            use_flash_attention=hparams.get("use_flash_attention", config.USE_FLASH_ATTENTION),
            attention_bias=hparams.get("attention_bias", config.ATTENTION_BIAS),
            ffn_bias=hparams.get("ffn_bias", config.FFN_BIAS),
            rope_theta=hparams.get("rope_theta", config.ROPE_THETA),
            norm_eps=hparams.get("norm_eps", config.NORM_EPS),
            scale_embeddings=hparams.get("scale_embeddings", True),
            tie_word_embeddings=hparams.get("tie_word_embeddings", config.TIE_WORD_EMBEDDINGS)
        )
        
        # Load state dict
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
        return model
    
    def generate(self, prompt: str, max_new_tokens: int = None,
                 temperature: float = None, top_k: int = None,
                 top_p: float = None, repetition_penalty: float = None,
                 do_sample: bool = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty
            do_sample: Use sampling
        
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or config.INFERENCE_MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else config.INFERENCE_TEMPERATURE
        top_k = top_k if top_k is not None else config.INFERENCE_TOP_K
        top_p = top_p if top_p is not None else config.INFERENCE_TOP_P
        repetition_penalty = repetition_penalty if repetition_penalty is not None else config.INFERENCE_REPETITION_PENALTY
        do_sample = do_sample if do_sample is not None else config.INFERENCE_DO_SAMPLE
        
        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.hparams.max_seq_length
        ).to(self.device)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Calculate tokens per second
        num_new_tokens = generated_ids.shape[1] - input_ids.shape[1]
        tokens_per_second = num_new_tokens / generation_time
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens": num_new_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second
        }
    
    def generate_batch(self, prompts: List[str], max_new_tokens: int = None,
                       temperature: float = None, top_k: int = None,
                       top_p: float = None, do_sample: bool = None) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Use sampling
        
        Returns:
            List of generated texts
        """
        # Encode all prompts (pad to max length)
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.hparams.max_seq_length
        ).to(self.device)
        
        input_ids = encoded["input_ids"]
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens or config.INFERENCE_MAX_NEW_TOKENS,
                temperature=temperature if temperature is not None else config.INFERENCE_TEMPERATURE,
                top_k=top_k if top_k is not None else config.INFERENCE_TOP_K,
                top_p=top_p if top_p is not None else config.INFERENCE_TOP_P,
                do_sample=do_sample if do_sample is not None else config.INFERENCE_DO_SAMPLE,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode each generation
        results = []
        for i, prompt in enumerate(prompts):
            # Extract only the newly generated tokens
            prompt_length = encoded["attention_mask"][i].sum().item()
            generated_tokens = generated_ids[i, prompt_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text
            })
        
        return results
    
    def interactive_chat(self, system_prompt: str = None):
        """
        Start an interactive chat session.
        
        Args:
            system_prompt: Optional system prompt
        """
        print("\n" + "=" * 80)
        print("Interactive Chat (Press 'quit' or 'exit' to stop)")
        print("=" * 80)
        
        if system_prompt:
            print(f"[SYSTEM] {system_prompt}")
        
        history = ""
        if system_prompt:
            history = system_prompt + "\n"
        
        while True:
            try:
                # Get user input
                user_input = input("\n[You] ")
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n[SYSTEM] Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                # Generate response
                prompt = history + f"\n[You] {user_input}\n[Assistant] "
                result = self.generate(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.95
                )
                
                # Extract just the response
                response = result["generated_text"][len(prompt):]
                
                print(f"\n[Assistant] {response}")
                
                # Update history
                history = prompt + response
                
            except KeyboardInterrupt:
                print("\n\n[SYSTEM] Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of given text.
        
        Args:
            text: Text to evaluate
        
        Returns:
            Perplexity score (lower is better)
        """
        # Encode text
        input_ids = self.tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.hparams.max_seq_length
        ).to(self.device)
        
        # Compute loss
        with torch.no_grad():
            logits, _ = self.model(input_ids, is_causal=True)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1)
            )
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
        
        return perplexity


def main():
    """Main inference function."""
    import argparse
    
    print("\n" + "=" * 80)
    print("Modern LLM Inference (200M Parameters)")
    print("=" * 80)
    
    # Example usage (without loading actual model)
    print("\nThis is the inference module for the Modern LLM.")
    print("To use it, load a trained checkpoint:")
    print()
    print("Example:")
    print("-" * 60)
    print("from training.inference import LLMInference")
    print()
    print("# Load model")
    print("llm = LLMInference('checkpoints/modern-llm-200m/final_model')")
    print()
    print("# Generate text")
    print("result = llm.generate('Once upon a time')")
    print("print(result['generated_text'])")
    print()
    print("# Interactive chat")
    print("llm.interactive_chat()")
    print("-" * 60)
    
    # Demo with config values
    print(f"\nInference Configuration:")
    print(f"  Max new tokens: {config.INFERENCE_MAX_NEW_TOKENS}")
    print(f"  Temperature: {config.INFERENCE_TEMPERATURE}")
    print(f"  Top-k: {config.INFERENCE_TOP_K}")
    print(f"  Top-p: {config.INFERENCE_TOP_P}")
    print(f"  Repetition penalty: {config.INFERENCE_REPETITION_PENALTY}")
    print(f"  Do sample: {config.INFERENCE_DO_SAMPLE}")


if __name__ == "__main__":
    main()
