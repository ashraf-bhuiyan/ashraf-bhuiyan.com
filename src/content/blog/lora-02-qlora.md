---
title: "QLoRA: 4-bit Base + Full-Precision Adapters"
description: "NF4 quantization, double quantization, paged optimizers, and how QLoRA preserves quality while cutting memory 4×."
pubDate: 2026-05-03
series: "LoRA & QLoRA in vLLM"
part: 2
tags: ["QLoRA", "quantization", "NF4", "fine-tuning"]
---

## What Problem Does This Solve?

LoRA (Blog A1) reduces the number of *trainable* parameters from billions to millions. But the base model still needs to fit in GPU memory for the forward pass — every training step reads all 8 billion weights to compute activations and gradients through the frozen layers.

```
LoRA training memory breakdown (Llama 3.1-8B, FP16):

  Base model weights (frozen, FP16):     16 GB    ← still loaded!
  LoRA A & B matrices (trainable):        0.05 GB
  Optimizer states for LoRA (Adam):       0.10 GB
  Gradients for LoRA:                     0.05 GB
  Activations:                           ~4 GB
  ──────────────────────────────────────────────
  Total:                                 ~20 GB   → fits on 1× A100, barely

  For Llama 3.1-70B:
  Base model weights (frozen, FP16):    140 GB    ← this is the problem
  Total:                               ~150 GB    → needs 2× A100-80GB
```

LoRA makes training *cheap* (few parameters to update), but the frozen base model is still the memory bottleneck. QLoRA's key insight: **the base model is never updated — so quantize it aggressively.**

---

## The Core Idea: 4-Bit Base + 16-Bit LoRA

QLoRA combines three techniques to reduce the base model's memory footprint by 4×:

```
Standard LoRA:
  Base model: FP16 (16 bits per parameter)
  LoRA weights: FP16

QLoRA:
  Base model: NF4 (4 bits per parameter)     ← 4× smaller
  LoRA weights: BF16 (16 bits per parameter)  ← full precision

Memory comparison (Llama 3.1-70B):
  FP16 base:  70B × 2 bytes = 140 GB
  NF4 base:   70B × 0.5 bytes = 35 GB       ← fits on 1× A100-80GB!
```

The forward pass dequantizes NF4 weights to BF16 on-the-fly for each computation. The LoRA computation is always in full precision (BF16), so gradients remain clean and the quality of the fine-tuned adapter is unaffected.

---

## NF4: Normal Float 4-Bit Quantization

### Why Not Just Use INT4?

Standard INT4 quantization uses 16 equally spaced levels:

```
INT4 levels (symmetric, range [-1, 1]):
  -1.0, -0.867, -0.733, -0.6, -0.467, -0.333, -0.2, -0.067,
   0.067, 0.2, 0.333, 0.467, 0.6, 0.733, 0.867, 1.0

Problem: neural network weights are approximately normally distributed.
Most weights cluster near zero, but INT4 spaces levels uniformly.

  Weight distribution:           INT4 levels:
       ████                      |   |   |   |   |   |   |   |
      ██████                     |   |   |   |   |   |   |   |
     ████████                    ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕
    ██████████                   Equal spacing — wastes resolution
   ████████████                  where weights are dense (near 0)
  ██████████████
  ─────────────────
  -3σ   -σ  0  σ   3σ
```

Most weights are near zero, but INT4 gives the same number of levels to the dense center and the sparse tails. This wastes quantization resolution.

### NF4: Optimal for Normal Distributions

NF4 (Normal Float 4) places its 16 levels optimally for a standard normal distribution. Each level is chosen so that an equal proportion of the distribution falls between adjacent levels:

```
NF4 levels (approximately):
  -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
   0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0

NF4 spacing:
  -1.0                                               1.0
   |     |    |   |  | || || |  |   |    |     |      |
   ↕ ↕ ↕ ↕ ↕↕↕↕↕↕↕↕↕↕ ↕ ↕ ↕ ↕     ↕
   Dense near zero — matches the weight distribution!
```

The levels are computed from the quantile function of the normal distribution: divide the CDF into 16 equal-probability bins, and place levels at the bin centers.

### Quantization Process

For each block of weights (typically 64 values):

```
Step 1: Compute block scale factor
  absmax = max(|w₁|, |w₂|, ..., |w₆₄|)
  scale = absmax   (FP16 value, stored alongside the block)

Step 2: Normalize weights to [-1, 1]
  w_normalized = w / scale

Step 3: Map each normalized weight to nearest NF4 level
  w_quantized = argmin_level |w_normalized - level|

Step 4: Store as 4-bit indices (2 values per byte)
  64 weights → 32 bytes of NF4 data + 2 bytes for scale = 34 bytes
  vs. 64 × 2 = 128 bytes in FP16
  → 3.76× compression
```

### Dequantization (During Forward Pass)

```
Step 1: Read the 4-bit index
Step 2: Look up the NF4 level
Step 3: Multiply by the block scale factor
  w_approx = NF4_LEVELS[index] × scale

This happens on-the-fly during matrix multiplication — 
no full-size FP16 tensor is ever materialized.
```

---

## Double Quantization

### The Scale Factor Overhead

Each block of 64 values stores one FP16 scale factor (2 bytes):

```
Memory per parameter:
  4-bit NF4 data:     0.5 bytes
  FP16 scale per 64:  2 / 64 = 0.03125 bytes
  Total:              0.53125 bytes = 4.25 bits per parameter

For 70B parameters:
  NF4 overhead = 70B × 0.53125 bytes = 37.2 GB
  Pure 4-bit =   70B × 0.5 bytes = 35.0 GB
  Scale overhead = 2.2 GB
```

That 2.2 GB of scale factors is wasteful. Double quantization quantizes the scale factors themselves.

### How It Works

```
Step 1: Standard NF4 quantization with block_size=64
  → produces one FP16 scale per block

Step 2: Group 256 consecutive block scales together
  → quantize these 256 FP16 scales to FP8 (8-bit float)
  → store one FP32 "super-scale" per group of 256 scales

Memory per parameter with double quantization:
  4-bit NF4 data:             0.5 bytes
  FP8 scale per 64:           1 / 64 = 0.015625 bytes
  FP32 super-scale per 16384: 4 / 16384 ≈ 0.000244 bytes
  Total:                      0.515869 bytes ≈ 4.13 bits per parameter

Savings for 70B model:
  Without double quant: 37.2 GB
  With double quant:    36.1 GB
  Saved:                1.1 GB
```

1.1 GB doesn't sound like much, but on a 48 GB GPU, that's 2.3% of total memory — enough for a larger batch size or more context.

---

## Paged Optimizers

### The Gradient Spike Problem

During training with gradient checkpointing (re-computing activations during backward pass to save memory), optimizer states can temporarily spike:

```
Normal training:
  Memory: [model][LoRA grads][optimizer states]  → fits

With gradient checkpointing (backward pass):
  Memory: [model][recomputed activations][grads][optimizer]
                  ^^^^^^^^^^^^^^^^^^^^^^^^
                  Temporary spike!           → might OOM
```

The spike is brief — it happens during the backward pass when activations are recomputed — but it can cause out-of-memory errors.

### Paged Optimizers: CPU as Overflow

QLoRA uses NVIDIA's unified memory (managed by CUDA) to automatically page optimizer states between GPU and CPU:

```
Normal:   All optimizer states on GPU
  GPU: [model 35GB][optim 0.1GB][activations 4GB] = 39.1 GB

Paged:    Optimizer states spill to CPU when GPU is full
  GPU: [model 35GB][activations 4GB][hot optim pages]
  CPU: [cold optim pages]

During optimizer.step():
  - Pages needed for current update are on GPU (fast)
  - Pages for other parameters can be on CPU (slow, but not accessed)
  - CUDA unified memory handles the paging transparently
```

Since LoRA has very few trainable parameters (< 50M), the optimizer states are small (~0.1-0.2 GB), and paging is rarely triggered. But it provides a safety net against OOM during gradient spikes.

---

## QLoRA Training Walkthrough

### Setup with bitsandbytes + PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Configure NF4 quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # load base model in 4-bit
    bnb_4bit_quant_type="nf4",          # use NF4 (not INT4)
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16
    bnb_4bit_use_double_quant=True,     # enable double quantization
)

# Step 2: Load the base model in NF4
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto",     # spread across available GPUs
)
# Memory: ~36 GB (down from ~140 GB in FP16)

# Step 3: Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)
# This enables gradient checkpointing and fixes layer norms

# Step 4: Add LoRA adapters in BF16
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 81,920,000 || all params: 70,635,765,760 || trainable%: 0.116%
```

### What Happens During a Forward Pass

```
Input tokens → Embedding (NF4 → BF16 dequant) → Hidden states

For each transformer layer:
  1. Layer norm (BF16)
  2. Attention:
     a. Q = dequant(W_q_nf4) @ hidden + (α/r) · B_q(A_q(hidden))   ← LoRA in BF16
     b. K = dequant(W_k_nf4) @ hidden + (α/r) · B_k(A_k(hidden))
     c. V = dequant(W_v_nf4) @ hidden + (α/r) · B_v(A_v(hidden))
     d. Attention computation (BF16)
     e. O = dequant(W_o_nf4) @ attn_out + (α/r) · B_o(A_o(attn_out))
  3. Layer norm (BF16)
  4. MLP:
     a. Gate = dequant(W_gate_nf4) @ hidden + LoRA_gate(hidden)
     b. Up   = dequant(W_up_nf4) @ hidden + LoRA_up(hidden)
     c. Down = dequant(W_down_nf4) @ (gate*up) + LoRA_down(gate*up)

Backward pass:
  Gradients flow through the dequantized weights (BF16 computation)
  But only LoRA A and B matrices are updated
  Base NF4 weights are NEVER modified
```

The key point: dequantization (`dequant`) happens on-the-fly during the matrix multiply. The NF4 weights are read from memory in their compressed form, dequantized to BF16 inside the CUDA kernel, and used for the multiplication. No full-size FP16 copy of the weights is ever created.

### Memory Comparison

```
                        Full FT    LoRA (FP16)   QLoRA (NF4)
                        ────────   ───────────   ───────────
Base model weights:     140 GB     140 GB         35 GB
LoRA weights:           —           0.16 GB        0.16 GB
Optimizer states:       280 GB       0.32 GB        0.32 GB
Gradients:              140 GB       0.16 GB        0.16 GB
Activations:            ~20 GB     ~10 GB         ~10 GB
────────────────────────────────────────────────────────────
Total:                  ~580 GB    ~150 GB        ~46 GB
GPUs needed (A100-80):  8×         2×             1×
```

QLoRA puts Llama 3.1-70B training on a single 80 GB GPU. With gradient checkpointing and batch size 1, it can even fit on a 48 GB GPU (A6000 or consumer RTX 4090).

---

## Quality: Does 4-Bit Quantization Hurt?

This is the critical question. If quantizing to 4-bit degrades the adapter quality, the memory savings don't matter.

### Why QLoRA Preserves Quality

Three reasons the 4-bit base model doesn't hurt LoRA training:

**1. The base model is never updated.**
Quantization error in the base weights is a fixed noise source. It doesn't accumulate or amplify during training because the base weights are frozen.

**2. LoRA weights are full precision.**
The A and B matrices — the only parameters being trained — are in BF16. Gradients and optimizer states are full precision. The learning signal is clean.

**3. LoRA compensates for quantization noise.**
The LoRA update `B×A` can learn to compensate for any systematic bias introduced by quantization. In practice, the adapter partially "corrects" the quantization error in the directions that matter for the task.

### Empirical Results

From the QLoRA paper (Dettmers et al., 2023):

```
Benchmark comparison on Vicuna evaluation:

Method                          Score
──────────────────────────────────────
Full fine-tuning (FP16)         72.2
LoRA (FP16 base, r=64)         71.8
QLoRA (NF4 base, r=64)         71.5
QLoRA (INT4 base, r=64)        70.1    ← INT4 is worse than NF4

Key findings:
- QLoRA (NF4) matches full fine-tuning within ~1%
- NF4 is consistently better than INT4 (the normal-distribution-aware levels help)
- Double quantization doesn't degrade quality measurably
```

The practical consensus: QLoRA adapters are interchangeable with standard LoRA adapters in quality for most tasks.

---

## QLoRA vs. LoRA: When to Use Which

```
Use QLoRA when:
  ✓ The base model doesn't fit in GPU memory in FP16
  ✓ You're training on consumer hardware (RTX 3090, 4090, A6000)
  ✓ You want to train 70B+ models without multi-GPU setups
  ✓ Training throughput is not the primary concern

Use standard LoRA (FP16 base) when:
  ✓ The base model fits comfortably in GPU memory
  ✓ You need maximum training throughput (dequantization has overhead)
  ✓ You're doing many short training runs (startup cost of quantization matters)
  ✓ You want the simplest possible setup

Use full fine-tuning when:
  ✓ You have the compute budget (8+ GPUs)
  ✓ You need absolute maximum quality
  ✓ The task requires changing the model's behavior fundamentally
  ✓ You're training on millions of examples
```

### Training Speed

QLoRA is slower per step than standard LoRA due to the dequantization overhead:

```
Llama 3.1-8B, batch size 4, sequence length 512:

Full FT (FP16):       1.0× speed (baseline)
LoRA (FP16 base):     1.2× speed (fewer params to update)
QLoRA (NF4 base):     0.9× speed (dequant overhead)
```

QLoRA is ~25% slower than LoRA per step, but uses ~75% less memory. The memory savings often let you use a larger batch size, which can offset the per-step slowdown.

---

## The Adapter is the Same

A critical point for the rest of this series: **the adapter produced by QLoRA is identical in format to a standard LoRA adapter.** It's just A and B matrices in FP16/BF16.

```
QLoRA training:
  Input: NF4 base model + training data
  Output: LoRA adapter (A and B matrices in BF16)
          ↓
          Identical format

LoRA training:
  Input: FP16 base model + training data
  Output: LoRA adapter (A and B matrices in BF16)
```

When you serve the adapter in vLLM (Blog A3), vLLM doesn't know or care whether it was trained with QLoRA or standard LoRA. The base model can be loaded in FP16 or quantized with GPTQ/AWQ — the adapter works the same either way.

The NF4 quantization is a **training-time optimization only.** It reduces memory during training but doesn't affect the adapter that's produced or how it's served.

---

## Key Takeaways

1. **QLoRA = NF4 base model + BF16 LoRA.** The base model is quantized to 4-bit; the LoRA adapters stay in full precision.
2. **NF4 is optimal for normally distributed weights** — it places quantization levels where most weights are, unlike uniform INT4.
3. **Double quantization** compresses the scale factors themselves, saving an additional 1-2 GB.
4. **Quality is preserved** because the base model is never updated (fixed quantization error) and LoRA weights are full precision.
5. **QLoRA enables training 70B models on a single GPU** — 4× memory reduction vs. FP16.
6. **The adapter output is identical** to standard LoRA — vLLM can serve it the same way.

---

## What's Next

You've trained a LoRA adapter (with either LoRA or QLoRA). **Blog A3** shows how to serve it in vLLM — load the base model, attach the adapter, and make requests where each request can select which adapter to use.

---

## Further Reading

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — the QLoRA paper (Dettmers et al., 2023)
- [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes) — NF4 quantization implementation
- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339) — precursor work on quantized LLM inference
- **Next: [Blog A3 — Serving LoRA in vLLM](/blog/lora-03-serving/)** — load a LoRA adapter and serve it with vLLM
