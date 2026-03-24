# 🚀 Transformer from Scratch

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white)](https://pytorch-lightning.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A complete, production-ready implementation of the Transformer architecture from "Attention Is All You Need"**

*Built with PyTorch Lightning & HuggingFace Trainer for scalable training and inference*

</div>

---

## ✨ What Makes This Special

🎯 **Complete Implementation** - Every component from the original paper, meticulously crafted
⚡ **Two Training Frameworks** - PyTorch Lightning (`PLTrainerScripts/`) and HuggingFace Trainer (`HFTrainerScripts/`)
🧠 **Production Ready** - Proper error handling, logging, and checkpointing
🔧 **Modular Design** - Each component is independently testable and reusable
🧪 **Independent Testing** - Run each module separately for debugging and learning
📚 **Educational** - Clean, well-documented code perfect for learning
🎨 **Modern Stack** - Uses GPT-2 tokenizer and state-of-the-art practices
🚀 **Multiple Architectures** - CrossAttention, DecoderOnly, MoE, GQA, MQA, and MLA
📊 **Comprehensive Metrics** - BLEU, ROUGE, METEOR, BERTScore, and Perplexity evaluation
🎛️ **Advanced Features** - Mixture of Experts, Group Query Attention, Multi-Query Attention, Multi-Head Latent Attention, SwiGLU FFN

---

## 🏗️ Architecture Deep Dive

### Core Components

| Component | Location | Description | Key Features |
|-----------|----------|-------------|--------------|
| **🔤 TokenEmbedding** | `core/Embedding.py` | Converts tokens to dense vectors | Scaling, padding handling, vocabulary mapping |
| **📍 PositionalEmbedding** | `core/Embedding.py` | Adds position information | Sinusoidal & learned encodings, flexible max positions |
| **🎯 MultiHeadSelfAttention** | `core/attention/MultiHeadSelfAttention.py` | Standard multi-head attention | Causal masking, cross-attention, scaled dot-product |
| **🔀 GroupQueryAttention** | `core/attention/GroupQueryAttention.py` | GQA — grouped KV heads | Tunable KV sharing, reduced KV cache |
| **⚡ MultiQueryAttention** | `core/attention/MultiQueryAttention.py` | MQA — single shared KV head | Maximum KV cache reduction |
| **🧬 MultiHeadLatentAttention** | `core/attention/MultiHeadLatentAttention.py` | MLA — compressed latent KV | Low-rank KV compression (DeepSeek-V2) |
| **🧠 PositionwiseFeedForward** | `core/FFN.py` | Non-linear transformations | SwiGLU (default), GELU, ReLU activations |
| **➕ AddNorm** | `core/AddNorm.py` | Residual connections + normalization | Layer normalization, dropout, gradient flow |

### Model Architectures

| Architecture | Location | Attention | Key Features |
|--------------|----------|-----------|--------------|
| **🔄 CrossAttentionSeq2Seq** | `models/CrossAttentionSeq2SeqModel.py` | MHA | Full encoder-decoder, cross-attention |
| **📝 DecoderOnly** | `models/DecoderOnlySeq2SeqModel.py` | MHA | GPT-style, causal masking |
| **🎛️ DecoderOnlyMoE** | `models/DecoderMoE.py` | MHA | Sparse MoE routing, expert specialization |
| **🔀 DecoderOnlyGQA** | `models/DecoderOnlyGQAModel.py` | GQA | Grouped KV heads (LLaMA 2, Mistral) |
| **⚡ DecoderOnlyMQA** | `models/DecoderOnlyMQAModel.py` | MQA | Single KV head, fastest inference |
| **🧬 DecoderOnlyMLA** | `models/DecoderOnlyMLAModel.py` | MLA | Compressed KV cache (DeepSeek-V2) |

### Attention Mechanism Comparison

| Mechanism | KV Heads | KV Cache Size | Params (d=256, h=8) | Used In |
|-----------|----------|---------------|---------------------|---------|
| **MHA** | `num_heads` (8) | 100% | 263K | GPT, BERT, T5 |
| **GQA** | `num_kv_heads` (2) | 25% | 164K | LLaMA 2 70B, Mistral |
| **MQA** | 1 | 12.5% | 148K | PaLM, Falcon |
| **MLA** | Compressed | `d_compress/d_model` | 149K | DeepSeek-V2 |

### Data Flow

```mermaid
graph TD
    A[Input Text] --> B[Tokenization]
    B --> C[Token Embedding]
    C --> D[Positional Encoding]
    D --> E[Encoder Stack]
    E --> F[Context Vectors]
    F --> G[Decoder Stack]
    G --> H[Output Logits]
    H --> I[Generated Text]
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

Two training frameworks are available — same models, different training loops:

#### Option A: PyTorch Lightning (`PLTrainerScripts/`)

```bash
python PLTrainerScripts/Trainer.py                  # CrossAttention Seq2Seq
python PLTrainerScripts/DecoderOnlyTrainer.py       # Decoder-Only (MHA)
python PLTrainerScripts/DecoderMoETrainer.py        # Mixture of Experts
python PLTrainerScripts/DecoderOnlyGQATrainer.py    # Group Query Attention
python PLTrainerScripts/DecoderOnlyMQATrainer.py    # Multi-Query Attention
python PLTrainerScripts/DecoderOnlyMLATrainer.py    # Multi-Head Latent Attention
python PLTrainerScripts/GSM8KTrainer.py             # GSM8K Math Reasoning
```

- Uses PyTorch Lightning `Trainer` with `ModelCheckpoint` callback
- Computes BLEU, ROUGE, METEOR, BERTScore, and Perplexity during validation
- Saves best model by `val_loss_epoch`

#### Option B: HuggingFace Trainer (`HFTrainerScripts/`)

```bash
python HFTrainerScripts/DecoderOnlyTrainer.py       # Decoder-Only (MHA)
python HFTrainerScripts/MoETrainer.py               # Mixture of Experts
python HFTrainerScripts/GQATrainer.py               # Group Query Attention
python HFTrainerScripts/MQATrainer.py               # Multi-Query Attention
python HFTrainerScripts/MLATrainer.py               # Multi-Head Latent Attention
python HFTrainerScripts/GSM8KTrainer.py             # GSM8K Math Reasoning
```

- Uses HuggingFace `Trainer` with `SaveBestModelCallback`
- Saves exactly one checkpoint — the model with the lowest `eval_loss`
- Generation with temperature, top-k, top-p, and repetition penalty

### 3. Inference

#### PyTorch Lightning models
```bash
python PLTrainerScripts/Inference.py                 # CrossAttention Seq2Seq
python PLTrainerScripts/DecoderOnlyInference.py      # Decoder-Only (MHA)
python PLTrainerScripts/DecoderMoEInference.py       # MoE
python PLTrainerScripts/DecoderOnlyGQAInference.py   # GQA
python PLTrainerScripts/DecoderOnlyMQAInference.py   # MQA
python PLTrainerScripts/DecoderOnlyMLAInference.py   # MLA
python PLTrainerScripts/GSM8KInference.py            # GSM8K
```

#### HuggingFace Trainer models
```bash
python HFTrainerScripts/DecoderOnlyInference.py      # Decoder-Only (MHA)
python HFTrainerScripts/MoEInference.py              # MoE
python HFTrainerScripts/GQAInference.py              # GQA
python HFTrainerScripts/MQAInference.py              # MQA
python HFTrainerScripts/MLAInference.py              # MLA
python HFTrainerScripts/GSM8KInference.py            # GSM8K
```

**Inference Features:**
- 🎲 **Greedy decoding** (PL) / **Sampling with controls** (HF) — temperature, top-k, top-p, repetition penalty
- ⚡ **Fast inference** - Optimized for production use
- 🎯 **Flexible input** - Handle variable length sequences
- 🔧 **Easy integration** - Simple API for your applications

### 4. Independent Module Testing

Each core component can be run independently for testing and experimentation:

```bash
# Test individual components
python core/Embedding.py                              # Token & positional embeddings
python core/attention/MultiHeadSelfAttention.py       # Standard multi-head attention
python core/attention/GroupQueryAttention.py           # Group Query Attention
python core/attention/MultiQueryAttention.py          # Multi-Query Attention
python core/attention/MultiHeadLatentAttention.py     # Multi-Head Latent Attention
python core/FFN.py                                    # Feed-forward network
python core/AddNorm.py                                # Residual connections & normalization

# Test model architectures
python models/Encoder.py                              # Encoder stack
python models/Decoder.py                              # Decoder stack
```

---

## 📊 Evaluation Metrics

The codebase includes comprehensive evaluation metrics for assessing model performance:

### Automatic Metrics

| Metric | Description | Range | Use Case |
|--------|-------------|-------|----------|
| **🎯 BLEU** | N-gram overlap with reference | 0-100 | Translation quality, text similarity |
| **📝 ROUGE-1** | Unigram overlap | 0-1 | Content coverage, summarization |
| **📝 ROUGE-2** | Bigram overlap | 0-1 | Phrase-level similarity |
| **📝 ROUGE-L** | Longest common subsequence | 0-1 | Structural similarity |
| **☄️ METEOR** | Semantic similarity with synonyms | 0-1 | Meaning preservation |
| **🧠 BERTScore** | Contextual embedding similarity | 0-1 | Semantic understanding |
| **📈 Perplexity** | Exponentiated avg negative log-likelihood | 1-∞ (lower = better) | Language modeling quality |

All metrics are automatically computed during training validation steps and logged to the progress bar and TensorBoard logs.

### Perplexity Implementation

Perplexity is computed using **log-softmax with proper token masking** — the gold-standard approach:

```python
# Per-batch: compute log probs, gather correct tokens, mask padding (-100)
log_probs = F.log_softmax(logits, dim=-1)
target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
target_log_probs = target_log_probs * mask.float()

# Accumulate across batches, then: PPL = exp(total_NLL / total_tokens)
```

- **PL models**: Token-weighted NLL accumulated in `validation_step`, perplexity computed in `on_validation_epoch_end`
- **HF models**: `PerplexityCallback` computes `exp(eval_loss)` after each evaluation; `compute_batch_perplexity()` utility available for inference

---

## 📊 Dataset & Task

**GSM8K Math Reasoning** *(used by all trainers)*
- 🧮 **7,473 training** + **1,319 test** grade school math problems
- 🎯 **Task**: Solve multi-step arithmetic word problems
- 📏 **Format**: `"Question: ..." → "Answer: step-by-step solution"`
- 📦 **Source**: `openai/gsm8k` via HuggingFace `datasets` library
- 🔧 **Max sequence length**: 256 tokens

---

## ⚙️ Configuration

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Model dimension (embedding size) |
| `num_heads` | 4-8 | Number of attention heads |
| `num_encoder_layers` | 2-6 | Encoder stack depth |
| `num_decoder_layers` | 2-6 | Decoder stack depth |
| `d_ff` | 128-1024 | Feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_positions` | 256 | Maximum sequence length |
| `use_sinusoidal_pos` | True | Use sinusoidal positional encoding |

### Attention Variant Configuration

| Parameter | Applies To | Default | Description |
|-----------|-----------|---------|-------------|
| `num_kv_heads` | GQA | 2 | Number of key/value heads |
| `d_compress` | MLA | 64 | Latent compression dimension |
| `num_experts` | MoE | 4 | Number of expert networks |
| `top_k` | MoE | 2 | Experts to activate per token |

### Training Configuration

| Parameter | PL | HF | Description |
|-----------|----|----|-------------|
| `batch_size` | 4 | 4 | Training batch size |
| `learning_rate` | 1e-3 | 1e-3 | Adam optimizer learning rate |
| `max_epochs` | 100 | 100 | Maximum training epochs |
| `checkpoint_monitor` | val_loss_epoch | eval_loss | Model selection metric |
| `checkpoint_strategy` | `ModelCheckpoint` (save_top_k=1) | `SaveBestModelCallback` (single best) | How the best model is saved |

### Training Framework Comparison

| Feature | PyTorch Lightning (`PLTrainerScripts/`) | HuggingFace Trainer (`HFTrainerScripts/`) |
|---------|----------------------------------------|-------------------------------------------|
| **Training loop** | `pl.Trainer` | `transformers.Trainer` |
| **Validation metrics** | BLEU, ROUGE, METEOR, BERTScore, Perplexity | BLEU, ROUGE, METEOR, Perplexity |
| **Checkpointing** | `ModelCheckpoint` callback | `SaveBestModelCallback` (exactly 1 best) |
| **Inference decoding** | Greedy (argmax) | Sampling (temperature, top-k, top-p, repetition penalty) |
| **Encoder-Decoder support** | Yes (CrossAttention Seq2Seq) | Decoder-only models only |
| **Model wrapper** | Native LightningModule | `HFModelWrapper` (returns `CausalLMOutput`) |

---

## 📁 Project Structure

```
transformer-from-scratch/
├── core/                                    # 🧠 Reusable Building Blocks
│   ├── Embedding.py                         #   Token & positional embeddings
│   ├── AddNorm.py                           #   Residual connections + normalization
│   ├── FFN.py                               #   Position-wise feed-forward network
│   └── attention/                           #   Attention Mechanisms
│       ├── MultiHeadSelfAttention.py        #     Standard multi-head attention (MHA)
│       ├── GroupQueryAttention.py            #     Group Query Attention (GQA)
│       ├── MultiQueryAttention.py           #     Multi-Query Attention (MQA)
│       └── MultiHeadLatentAttention.py      #     Multi-Head Latent Attention (MLA)
│
├── models/                                  # 🏗️ Complete Model Architectures
│   ├── Encoder.py                           #   Encoder stack
│   ├── Decoder.py                           #   Decoder stack (with cross-attention)
│   ├── CrossAttentionSeq2SeqModel.py        #   Full encoder-decoder Seq2Seq
│   ├── DecoderOnlySeq2SeqModel.py           #   GPT-style decoder-only (MHA)
│   ├── DecoderMoE.py                        #   Decoder-only with Mixture of Experts
│   ├── DecoderOnlyGQAModel.py               #   Decoder-only with GQA
│   ├── DecoderOnlyMQAModel.py               #   Decoder-only with MQA
│   └── DecoderOnlyMLAModel.py               #   Decoder-only with MLA
│
├── PLTrainerScripts/                        # ⚡ PyTorch Lightning Training & Inference
│   ├── Trainer.py                           #   CrossAttention Seq2Seq training
│   ├── Inference.py                         #   CrossAttention Seq2Seq inference
│   ├── DecoderOnlyTrainer.py                #   Decoder-only training
│   ├── DecoderOnlyInference.py              #   Decoder-only inference
│   ├── DecoderMoETrainer.py                 #   MoE training
│   ├── DecoderMoEInference.py               #   MoE inference
│   ├── DecoderOnlyGQATrainer.py             #   GQA training
│   ├── DecoderOnlyGQAInference.py           #   GQA inference
│   ├── DecoderOnlyMQATrainer.py             #   MQA training
│   ├── DecoderOnlyMQAInference.py           #   MQA inference
│   ├── DecoderOnlyMLATrainer.py             #   MLA training
│   ├── DecoderOnlyMLAInference.py           #   MLA inference
│   ├── GSM8KTrainer.py                      #   GSM8K math reasoning training
│   └── GSM8KInference.py                    #   GSM8K math reasoning inference
│
├── HFTrainerScripts/                        # 🤗 HuggingFace Trainer Training & Inference
│   ├── hf_wrapper.py                        #   HF-compatible model wrapper & utilities
│   ├── DecoderOnlyTrainer.py                #   Decoder-only (MHA) training
│   ├── DecoderOnlyInference.py              #   Decoder-only (MHA) inference
│   ├── MQATrainer.py                        #   Multi-Query Attention training
│   ├── MQAInference.py                      #   Multi-Query Attention inference
│   ├── GQATrainer.py                        #   Group Query Attention training
│   ├── GQAInference.py                      #   Group Query Attention inference
│   ├── MLATrainer.py                        #   Multi-Head Latent Attention training
│   ├── MLAInference.py                      #   Multi-Head Latent Attention inference
│   ├── MoETrainer.py                        #   Mixture of Experts training
│   ├── MoEInference.py                      #   Mixture of Experts inference
│   ├── GSM8KTrainer.py                      #   GSM8K math reasoning training
│   └── GSM8KInference.py                    #   GSM8K math reasoning inference
│
├── data/                                    # 📊 Datasets (GSM8K loaded via HuggingFace)
│   └── (downloaded automatically by `datasets` library)
│
├── checkpoints/                             # 💾 Model Checkpoints
│   ├── Seq2SeqCheckpoints/                  #   PL: CrossAttention model
│   ├── DecoderOnlyCheckpoints/              #   PL: Decoder-only model
│   ├── DecoderMoECheckpoints/               #   PL: MoE model
│   ├── GQACheckpoints/                      #   PL: GQA model
│   ├── MQACheckpoints/                      #   PL: MQA model
│   ├── MLACheckpoints/                      #   PL: MLA model
│   ├── GSM8KCheckpoints/                    #   PL: GSM8K model
│   ├── HF_DecoderOnlyCheckpoints/           #   HF: Decoder-only model
│   ├── HF_MQACheckpoints/                   #   HF: MQA model
│   ├── HF_GQACheckpoints/                   #   HF: GQA model
│   ├── HF_MLACheckpoints/                   #   HF: MLA model
│   ├── HF_MoECheckpoints/                   #   HF: MoE model
│   └── HF_GSM8KCheckpoints/                 #   HF: GSM8K model
│
├── docs/                                    # 📚 Documentation (38 files)
│   ├── Embedding.md, AddNorm.md, FFN.md     #   Core component docs
│   ├── MultiHeadSelfAttention.md            #   Attention mechanism docs
│   ├── GroupQueryAttention.md               #   GQA docs
│   ├── MultiQueryAttention.md               #   MQA docs
│   ├── MultiHeadLatentAttention.md          #   MLA docs
│   ├── DecoderOnlySeq2SeqModel.md           #   Model architecture docs
│   ├── DecoderOnly{GQA,MQA,MLA}Model.md    #   Variant model docs
│   └── ...                                  #   Trainer & inference docs
│
├── README.md
└── requirements.txt
```

---

## 🎯 Use Cases

### Perfect For:
- 📚 **Learning** - Understanding Transformer architecture and its variants
- 🔬 **Research** - Experimenting with attention mechanisms (MHA, GQA, MQA, MLA)
- 🚀 **Prototyping** - Quick seq2seq model development
- 🧪 **Component Testing** - Debug and validate individual modules

### Applications:

#### CrossAttention Seq2Seq Model
- 📄 **Summarization** - Generate concise summaries
- 🔄 **Translation** - Sequence-to-sequence translation
- 📝 **Question Answering** - Context-aware responses

#### Decoder-Only Models (MHA / GQA / MQA / MLA / MoE)
- 📝 **Text Completion** - Auto-complete sentences
- 💬 **Chatbots** - Conversational AI systems
- 🎨 **Creative Writing** - Story and content generation
- 🧮 **Math Reasoning** - GSM8K grade school math problems

---

## 🎛️ Mixture of Experts (MoE) Implementation

### Key Features

- **🔧 ExpertMLP** - Individual expert networks with SwiGLU activation
- **🎯 TopKRouter** - Intelligent routing mechanism for expert selection
- **⚡ Sparse Computation** - Only activate selected experts per token
- **📊 Load Balancing** - Automatic expert capacity management

### Usage Example

```python
# PyTorch Lightning
from models.DecoderMoE import DecoderOnlyMoEModel
model = DecoderOnlyMoEModel(
    vocab_size=vocab_size, d_model=256,
    num_experts=4, top_k=2, num_layers=6, tokenizer=tokenizer
)
trainer.fit(model, train_loader, val_loader)

# HuggingFace Trainer
from HFTrainerScripts.hf_wrapper import HFModelWrapper
wrapper = HFModelWrapper(model)
hf_trainer = Trainer(model=wrapper, args=training_args, ...)
hf_trainer.train()
```

---

## 🧠 SwiGLU Feed-Forward Network

All models use the **SwiGLU** activation (Shazeer, 2020) in their feed-forward layers, matching modern architectures like LLaMA, PaLM, and Mistral:

```python
# SwiGLU: gate = SiLU(x @ W1) ⊙ (x @ W2), output = gate @ W3
# hidden_dim = 2/3 * d_ff keeps param count comparable to standard FFN
hidden_dim = int(2 * d_ff / 3)
gate = F.silu(self.w1(x))       # gate projection + Swish activation
data = self.w2(x)               # data projection
return self.w3(gate * data)     # element-wise multiply + down projection
```

| Feature | Standard FFN | SwiGLU FFN |
|---------|-------------|------------|
| **Projections** | 2 (up + down) | 3 (gate + data + down) |
| **Activation** | ReLU/GELU | SiLU (Swish) with gating |
| **Hidden dim** | `d_ff` | `2/3 * d_ff` (param-matched) |
| **Bias** | Yes | No (`bias=False`) |
| **Used in** | Original Transformer | LLaMA, PaLM, Mistral |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌟 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 **Open** a Pull Request

### Areas for Contribution:
- 🚀 **Performance optimizations**
- 🧪 **Additional attention mechanisms**
- 📊 **More datasets and tasks**
- 📚 **Documentation improvements**
- 🐛 **Bug fixes and testing**

---

## 📚 References & Learning

### Papers
1. **Vaswani, A., et al.** (2017). "Attention is all you need." *NeurIPS 2017*
2. **Shazeer, N.** (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv*
3. **Ainslie, J., et al.** (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP 2023*
4. **DeepSeek-AI** (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." *arXiv*
5. **Shazeer, N.** (2020). "GLU Variants Improve Transformer." *arXiv* — SwiGLU activation used in FFN

### Resources
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- ⚡ [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- 🤗 [HuggingFace Trainer Documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- 🎓 [Attention Mechanism Explained](https://distill.pub/2016/augmented-rnns/)
- 🔥 [Transformer from Scratch](https://www.youtube.com/watch?v=ISNdQcPhsts)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and lots of ☕

[Report Bug](https://github.com/yourusername/transformer-from-scratch/issues) · [Request Feature](https://github.com/yourusername/transformer-from-scratch/issues) · [Documentation](https://github.com/yourusername/transformer-from-scratch/wiki)

</div>
