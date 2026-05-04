---
title: "LoRA Fundamentals: The Math Behind Low-Rank Adaptation"
description: "B×A matrix decomposition, rank and alpha tradeoffs, target module selection, PEFT training, and adapter merging."
pubDate: 2026-05-03
series: "LoRA & QLoRA in vLLM"
part: 1
tags: ["LoRA", "fine-tuning", "low-rank", "PEFT"]
---

## What Problem Does This Solve?

You have a powerful pretrained LLM — Llama 3.1-8B, Mistral-7B, Qwen2-72B. It's great at general tasks, but you need it to excel at *your* specific task: medical Q&A, legal document analysis, code generation in your company's internal framework. The standard solution is **fine-tuning** — continue training the model on your task-specific data.

The problem is cost:

```
Full fine-tuning of Llama 3.1-8B:
  Model weights (FP16):     16 GB
  Optimizer states (Adam):  32 GB  (2 states × 16 GB)
  Gradients:                16 GB
  Activations:              ~8 GB  (depends on batch size)
  ────────────────────────────────
  Total:                    ~72 GB  → needs 2× A100-80GB

Full fine-tuning of Llama 3.1-70B:
  Model weights (FP16):    140 GB
  Optimizer states (Adam): 280 GB
  Gradients:               140 GB
  ────────────────────────────────
  Total:                   ~560 GB → needs 8× A100-80GB
```

And the storage problem: every fine-tuned variant is a full copy of the model. If you have 50 customers each with a custom model, that's 50 × 16 GB = 800 GB just for the 8B model. You can't efficiently serve 50 separate models from one GPU.

LoRA solves both problems.

---

## The Core Idea: Low-Rank Weight Updates

The key insight behind LoRA comes from a surprising empirical observation: **the weight updates during fine-tuning are low-rank.** That is, the change from the pretrained weights to the fine-tuned weights can be well-approximated by a much smaller matrix.

### What "Low-Rank" Means

A matrix has rank `r` if it can be expressed as the product of two smaller matrices. For a weight matrix W of shape `(d × k)`:

```
Full update:       ΔW has shape (d × k) → d × k parameters

Low-rank update:   ΔW ≈ B × A
                   where B has shape (d × r)
                   and   A has shape (r × k)
                   and   r << min(d, k)
                   → r × (d + k) parameters
```

Concrete example with Llama 3.1-8B's attention projection (d=4096, k=4096):

```
Full ΔW:   4096 × 4096 = 16,777,216 parameters

LoRA r=16: B is (4096 × 16), A is (16 × 4096)
           = 65,536 + 65,536 = 131,072 parameters
           = 128× fewer parameters

LoRA r=64: B is (4096 × 64), A is (64 × 4096)
           = 262,144 + 262,144 = 524,288 parameters
           = 32× fewer parameters
```

The claim: that 131K-parameter low-rank approximation captures most of what matters in the 16.7M-parameter full update. This has been validated empirically across many tasks and model sizes.

---

## How LoRA Works

### The Modified Forward Pass

In standard fine-tuning, you update the weight matrix directly:

```
Before:  y = W₀x                    (pretrained weights)
After:   y = (W₀ + ΔW)x             (fine-tuned weights)
```

In LoRA, you freeze `W₀` and decompose `ΔW` into two small matrices:

```
LoRA:    y = W₀x + (α/r) · B(Ax)
```

Where:
- `W₀` is the **frozen** pretrained weight matrix (d × k) — never updated
- `A` is the **down-projection** matrix (r × k) — trained
- `B` is the **up-projection** matrix (d × r) — trained
- `r` is the **rank** — a hyperparameter (typically 8, 16, 32, or 64)
- `α` is the **scaling factor** — controls the magnitude of the LoRA update
- `α/r` is the effective scaling — this keeps the update magnitude stable across different ranks

```
                    ┌────────────────────────────────┐
                    │         W₀ (frozen)            │
          x ──────►│       (d × k)                   │──────► y = W₀x + (α/r)·BAx
          │         │    pretrained weights           │  ▲
          │         └────────────────────────────────┘  │
          │                                             │  (add)
          │         ┌────────┐       ┌────────┐         │
          └────────►│  A     │──────►│  B     │─────────┘
                    │(r × k) │       │(d × r) │
                    │trained │       │trained │
                    └────────┘       └────────┘
                     down-proj        up-proj
                    (compress)       (expand)
```

### Initialization

The initialization of A and B matters:

- **A** is initialized with random Gaussian values (like Kaiming init)
- **B** is initialized to **zeros**

This means at the start of training, `BA = 0`, so the LoRA output is zero. The model starts behaving exactly like the pretrained model. As training progresses, A and B learn the task-specific update.

Why not initialize both randomly? Because then `BA` would be random noise at initialization, and the model would start from a random perturbation of the pretrained weights rather than from the pretrained weights themselves.

### The Scaling Factor: α and r

The `α/r` scaling factor is often confusing. Here's the intuition:

- If you increase `r` (higher rank), the LoRA matrices have more capacity, and their output magnitude increases
- The `α/r` ratio compensates: as `r` increases, the scaling decreases, keeping the magnitude of the LoRA update roughly constant
- This means you can change `r` without needing to retune the learning rate

Common settings:
- `r=16, α=32` (α/r = 2.0) — a safe default
- `r=16, α=16` (α/r = 1.0) — more conservative
- `r=64, α=128` (α/r = 2.0) — higher rank for complex tasks

The exact value of `α` is less important than keeping `α/r` consistent. Most practitioners set `α = 2r` and focus on choosing `r`.

---

## Where LoRA Layers Attach

### Target Modules in a Transformer

A standard transformer layer has these linear projections:

```
Transformer Layer:
  ┌─────────────────────────────────────┐
  │ Self-Attention:                     │
  │   Q projection:  (hidden → hidden)  │  ← LoRA target
  │   K projection:  (hidden → hidden)  │  ← LoRA target
  │   V projection:  (hidden → hidden)  │  ← LoRA target
  │   O projection:  (hidden → hidden)  │  ← LoRA target
  │                                     │
  │ MLP (Feed-Forward):                 │
  │   Gate projection: (hidden → ffn)   │  ← LoRA target
  │   Up projection:   (hidden → ffn)   │  ← LoRA target
  │   Down projection: (ffn → hidden)   │  ← LoRA target
  └─────────────────────────────────────┘
```

You can attach LoRA to any subset of these layers. Common configurations:

**Q and V only (original LoRA paper):**
```python
target_modules = ["q_proj", "v_proj"]
```
- 2 LoRA modules per layer
- Minimal parameter count
- Works well for many tasks

**All attention projections:**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```
- 4 LoRA modules per layer
- Better for tasks requiring strong attention pattern changes

**All linear layers:**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```
- 7 LoRA modules per layer
- Most expressive, highest parameter count
- Recommended when you have enough training data

### Parameter Count Comparison

For Llama 3.1-8B (32 layers, hidden_dim=4096, ffn_dim=14336):

```
                        LoRA params    % of model    Adapter size
Q+V only (r=16):        4.2M            0.05%         ~8 MB
All attention (r=16):    8.4M            0.10%        ~17 MB
All linear (r=16):      24.6M            0.30%        ~49 MB
All linear (r=64):      98.3M            1.21%       ~197 MB

Full model:           8,030M            100%       ~16,000 MB
```

Even the most aggressive LoRA configuration (all linear, r=64) uses just 1.2% of the full model's parameters and produces an adapter that's 80× smaller than the full model.

---

## Rank and Alpha Tradeoffs

### Choosing the Right Rank

The rank `r` controls the expressiveness of the LoRA update:

```
r=4:    Very constrained. Good for simple tasks (style transfer,
        language switch). Risk: underfitting complex tasks.

r=8:    Low. Good for tasks with clear patterns and moderate data.
        The minimum for most practical tasks.

r=16:   Default. Works well for most fine-tuning tasks. Good
        balance between capacity and efficiency.

r=32:   Medium. Better for complex tasks like instruction-following,
        multi-step reasoning, or code generation.

r=64:   High. For tasks requiring significant behavioral change
        from the base model. More data needed to avoid overfitting.

r=128+: Very high. Approaching full fine-tuning expressiveness.
        Rarely needed — consider full fine-tuning at this point.
```

### How to Choose

Rules of thumb:
1. **Start with r=16.** It works for the vast majority of tasks.
2. **If the model underfits** (training loss plateaus high), increase rank.
3. **If the model overfits** (validation loss diverges from training loss), decrease rank or add regularization.
4. **More data → can use higher rank** without overfitting.
5. **Simpler task → lower rank** is sufficient (e.g., translating style needs r=8, teaching a new programming language needs r=32+).

### The Efficiency Win

The parameter reduction translates directly to training efficiency:

```
Training Llama 3.1-8B on 4× A100-80GB:

Full fine-tuning:     ~72 GB memory, ~4 hours for 10K steps
LoRA (r=16, Q+V):     ~18 GB memory, ~1 hour for 10K steps
LoRA (r=16, all):     ~20 GB memory, ~1.5 hours for 10K steps

3.6-4× less memory, 2.5-4× faster training
```

---

## Training a LoRA Adapter

### Using HuggingFace PEFT

The standard way to train LoRA adapters is with the PEFT (Parameter-Efficient Fine-Tuning) library:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling factor
    target_modules=[         # which layers get LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # dropout on LoRA layers
    bias="none",             # don't train biases
    task_type="CAUSAL_LM",
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 24,576,000 || all params: 8,054,984,704 || trainable%: 0.305%
```

What `get_peft_model` does:
1. Freezes all base model parameters (`requires_grad=False`)
2. Inserts LoRA A and B matrices alongside each target module
3. Only the A and B matrices have `requires_grad=True`

### The Training Loop

The training loop is identical to standard fine-tuning — the only difference is that gradients flow through fewer parameters:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,        # higher LR than full FT (1e-5)
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

Note the learning rate: LoRA typically uses a higher learning rate (1e-4 to 3e-4) than full fine-tuning (1e-5 to 3e-5). This is because only a small number of parameters are being updated, so each update needs to be larger to have the same effect.

### Saving and Loading

Only the LoRA weights are saved — not the full model:

```python
# Save just the LoRA adapter (A and B matrices + config)
model.save_pretrained("./my-lora-adapter")

# What's saved:
# ./my-lora-adapter/
#   adapter_config.json    (LoRA hyperparameters)
#   adapter_model.safetensors   (A and B matrices, ~50 MB)
```

The adapter is tiny compared to the base model:

```
Base model:    16,000 MB (8B params × 2 bytes/param)
LoRA adapter:      49 MB (24.6M params × 2 bytes/param)
Ratio:         326:1
```

### Merging LoRA Back into the Base Model

For offline use (when you don't need multi-adapter serving), you can merge the LoRA weights into the base model:

```python
# Merge LoRA into base weights permanently
merged_model = model.merge_and_unload()

# Now merged_model is a standard model with LoRA baked in
# W = W₀ + (α/r) × BA
merged_model.save_pretrained("./merged-model")
```

After merging, the model behaves identically to one that was fully fine-tuned, with zero runtime overhead. But you lose the ability to swap adapters — it's just a regular model checkpoint.

---

## LoRA vs. Other PEFT Methods

LoRA isn't the only parameter-efficient fine-tuning method, but it's the most popular for good reasons:

```
Method          Params   Inference     Multi-adapter    Quality
                trained  overhead      serving?
─────────────────────────────────────────────────────────────────
Full FT         100%     None          No (full copy)   Best
LoRA            0.1-1%   2-5%          Yes              Very good
QLoRA           0.1-1%   2-5%          Yes              Very good
Prefix Tuning   0.01%    ~10%          Yes              Good
Adapters (Houlsby) 1-5%  5-10%         Possible         Very good
Prompt Tuning   0.001%   Minimal       Yes              Fair
```

LoRA's advantage is the combination of:
1. **High quality** (close to full fine-tuning on most benchmarks)
2. **Low inference overhead** (small extra matmul, can be batched — Blog A5)
3. **Native multi-adapter support** (swap adapters per-request in vLLM — Blog A4)
4. **Tiny adapter size** (MBs, not GBs)

---

## Key Takeaways

1. **LoRA decomposes weight updates** into two small matrices `B×A`, reducing trainable parameters by 100-1000×
2. **The rank `r`** controls expressiveness — `r=16` is a good default; increase for complex tasks, decrease for simple ones
3. **The scaling factor `α/r`** keeps update magnitudes stable across different ranks — typically set `α = 2r`
4. **Target modules** determine where LoRA attaches — Q+V for minimal overhead, all linear layers for maximum expressiveness
5. **Adapters are tiny** (MBs vs. GBs) and can be loaded/swapped at inference time — this is what enables multi-LoRA serving in vLLM (Blog A4)
6. **LoRA can be merged** into the base model for zero-overhead inference, but this sacrifices the ability to swap adapters

---

## What's Next

LoRA reduces trainable parameters but still requires the full base model in memory for the forward pass. **Blog A2** introduces QLoRA — which quantizes the base model to 4-bit, letting you fine-tune a 65B model on a single 48GB GPU.

---

## Further Reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — the original LoRA paper (Hu et al., 2021)
- [HuggingFace PEFT Library](https://github.com/huggingface/peft) — production LoRA training implementation
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka's guide
- **Next: [Blog A2 — QLoRA](/blog/lora-02-qlora/)** — fine-tune with a 4-bit base model to cut memory by 4×
