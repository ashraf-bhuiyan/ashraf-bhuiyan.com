---
title: "Serving a LoRA Adapter in vLLM"
description: "Enable LoRA serving, load adapters on the fly, and understand how the forward pass applies low-rank updates."
pubDate: 2026-05-03
series: "LoRA & QLoRA in vLLM"
part: 3
tags: ["LoRA", "vLLM", "serving", "inference"]
---

## What Problem Does This Solve?

You've trained a LoRA adapter — maybe with standard LoRA (Blog A1) or QLoRA (Blog A2). Now you need to serve it. You have two options:

**Option 1: Merge and serve.** Merge the LoRA weights into the base model (`W = W₀ + (α/r)·BA`), save the full model, and serve it like any other model. Simple, zero runtime overhead — but you lose the ability to swap adapters, and every variant is a full model copy.

**Option 2: Serve on-the-fly.** Load the base model once, load the LoRA adapter separately, and apply the LoRA computation during each forward pass. Small runtime overhead — but you can swap adapters per-request, and the base model is shared across all adapters.

vLLM uses Option 2. This is what makes multi-LoRA serving (Blog A4) possible — one base model, many adapters, adapter selected per-request.

---

## The On-the-Fly LoRA Forward Pass

When vLLM serves a request with a LoRA adapter, every linear layer in the model computes:

```
output = W₀x + (α/r) · B(A(x))
         ─────   ──────────────
         base      LoRA
         (frozen)  (adapter-specific)
```

This is two separate operations:
1. **Base computation:** `W₀x` — the standard matmul, identical for all requests regardless of adapter
2. **LoRA computation:** `B(A(x))` — two small matmuls, specific to the adapter

```
For Llama 3.1-8B Q projection (hidden_dim=4096, rank=16):

Base matmul:  (4096 × 4096) @ (4096 × 1) = 16.7M multiply-adds
LoRA A:       (16 × 4096) @ (4096 × 1)   = 65K multiply-adds
LoRA B:       (4096 × 16) @ (16 × 1)     = 65K multiply-adds
LoRA total:   130K multiply-adds

LoRA overhead: 130K / 16.7M = 0.78%
```

The LoRA computation is tiny compared to the base — less than 1% overhead per layer. Even with LoRA on all 7 linear layers per transformer block, the total overhead is under 5%.

---

## Launching vLLM with LoRA Support

### Basic Setup

```bash
# Start vLLM with LoRA enabled
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules my-adapter=/path/to/lora-adapter
```

Key flags:

| Flag | Purpose | Default |
|------|---------|---------|
| `--enable-lora` | Enable the LoRA infrastructure (weight loading, kernels) | Disabled |
| `--max-lora-rank` | Maximum rank of any adapter that can be loaded | 16 |
| `--max-loras` | Max adapters in GPU memory simultaneously | 1 |
| `--lora-modules` | Pre-register named adapters at startup | None |
| `--lora-extra-vocab-size` | Extra vocab capacity for adapters with new tokens | 256 |
| `--long-lora-scaling-factors` | RoPE scaling for long-context LoRA adapters | None |
| `--lora-dtype` | Data type for LoRA weights (auto, float16, bfloat16) | auto |
| `--max-cpu-loras` | Max adapters cached in CPU memory | None |

### What `--enable-lora` Does Internally

When you pass `--enable-lora`, vLLM:

1. **Allocates LoRA weight slots** on GPU — pre-sized to `max_loras × max_lora_rank × hidden_dim`
2. **Initializes PunicaWrapper** — the kernel dispatcher for batched LoRA computation (Blog A5)
3. **Creates LoRAModelManager** — manages adapter loading, caching, and eviction
4. **Modifies each linear layer** — wraps target modules so they compute `W₀x + BAx`

Without `--enable-lora`, none of this is loaded, and the model runs at standard speed with zero LoRA overhead.

### Pre-registering Adapters

You can register adapters at startup:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules \
    medical-qa=/data/adapters/medical-qa \
    legal-summarize=/data/adapters/legal-summarize \
    code-review=/data/adapters/code-review
```

Each adapter is assigned a name that clients use in the `model` field:

```json
{"model": "medical-qa", "messages": [...]}
```

The adapter path can be a local directory or a HuggingFace model ID (e.g., `my-org/my-lora-adapter`).

---

## Making Requests with a LoRA Adapter

### OpenAI-Compatible API

The simplest way — use the adapter name in the `model` field:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medical-qa",
    "messages": [
      {"role": "user", "content": "What are the symptoms of type 2 diabetes?"}
    ],
    "max_tokens": 200
  }'
```

If you use the base model name, the request runs without any adapter:

```bash
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [...]}'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Request with LoRA adapter
response = client.chat.completions.create(
    model="medical-qa",
    messages=[
        {"role": "user", "content": "What are the symptoms of type 2 diabetes?"}
    ],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### Programmatic API (vLLM Python)

For direct Python usage without the HTTP server:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_lora_rank=16,
)

sampling_params = SamplingParams(max_tokens=200, temperature=0.7)

# Create a LoRA request
lora_request = LoRARequest(
    lora_name="medical-qa",
    lora_int_id=1,                        # unique integer ID
    lora_local_path="/data/adapters/medical-qa",
)

# Generate with the LoRA adapter
outputs = llm.generate(
    ["What are the symptoms of type 2 diabetes?"],
    sampling_params,
    lora_request=lora_request,
)
print(outputs[0].outputs[0].text)
```

The `LoRARequest` object specifies:
- `lora_name`: human-readable name
- `lora_int_id`: integer ID for internal tracking (must be unique per adapter)
- `lora_local_path`: path to the adapter weights

---

## What Happens Internally

### Step-by-Step Request Flow

```
1. Request arrives: model="medical-qa"
   │
2. API server resolves "medical-qa" to a LoRARequest
   │
3. Scheduler checks: is the "medical-qa" adapter loaded in GPU memory?
   ├── Yes → proceed to step 4
   └── No  → load adapter weights to GPU (evict LRU adapter if at capacity)
   │
4. Request is scheduled into a batch
   │
5. Forward pass for each transformer layer:
   │  a. Base computation: y = W₀x   (same for all requests)
   │  b. LoRA computation: y += (α/r) · B_medical(A_medical(x))
   │     ↑ uses the medical-qa adapter's A and B matrices
   │
6. Sample next token
   │
7. Return token to client (streaming) or accumulate (non-streaming)
```

### Adapter Weight Storage

vLLM stores LoRA weights in a specific memory layout:

```
For each LoRA-wrapped linear layer:

  lora_a_stacked: [max_loras, 1, rank, in_features]
                   ↑           ↑  ↑     ↑
                   slot index  |  LoRA  input dim
                               |  rank
                               batch dim (for broadcasting)

  lora_b_stacked: [max_loras, 1, out_features, rank]

Example (max_loras=4, rank=16, Q projection of Llama-8B):
  lora_a: [4, 1, 16, 4096]  → 4 adapter slots
  lora_b: [4, 1, 4096, 16]
```

When an adapter is loaded, its A and B matrices are copied into the corresponding slot. When evicted, the slot is freed for reuse.

---

## LoRA with Quantized Base Models

One of the most powerful configurations: combine a quantized base model with LoRA adapters.

```bash
# Serve a GPTQ-quantized 70B model with LoRA
vllm serve TheBloke/Llama-2-70B-GPTQ \
  --enable-lora \
  --quantization gptq \
  --lora-modules my-adapter=/path/to/adapter
```

Memory breakdown:

```
Without quantization:
  Base model (FP16):  140 GB  → needs TP=4 on A100-80GB
  LoRA adapter:         0.1 GB

With GPTQ INT4:
  Base model (INT4):   35 GB  → fits on 1× A100-80GB!
  LoRA adapter:         0.1 GB
  Total:               35.1 GB
```

This is the serving equivalent of QLoRA training:
- Training: NF4 base + BF16 LoRA (Blog A2)
- Serving: GPTQ/AWQ base + FP16 LoRA

The LoRA computation always happens in full precision (FP16/BF16), regardless of the base model's quantization. The dequantized base output is added to the LoRA output in FP16.

---

## Performance: Merged vs. On-the-Fly

### When to Merge

If you only serve **one** adapter and never plan to swap:

```python
# Merge LoRA into base weights (one-time operation)
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("/path/to/adapter")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("/path/to/merged-model")

# Serve the merged model normally (no --enable-lora needed)
# vllm serve /path/to/merged-model
```

Merged advantages:
- Zero runtime overhead (no extra matmuls)
- No `--enable-lora` needed
- Slightly simpler deployment

### When to Keep Separate

If you serve **multiple** adapters or need to **swap** adapters:

- On-the-fly LoRA is the only option
- The overhead is small (2-5% for r=16)
- The memory savings are enormous (one base model instead of N copies)

### Benchmark Comparison

```
Llama 3.1-8B on A100-80GB, batch size 32, 512 output tokens:

Configuration              Throughput    Latency (P50)   Memory
                           (tok/s)       (ms/tok)        (GB)
──────────────────────────────────────────────────────────────────
Base model (no LoRA):      2,450         13.1            16.8
Merged LoRA:               2,445         13.1            16.8
On-the-fly LoRA (r=16):   2,380         13.5            17.1
On-the-fly LoRA (r=64):   2,290         14.0            17.6

Overhead (r=16):  ~3% throughput reduction
Overhead (r=64):  ~7% throughput reduction
```

For r=16, the overhead is negligible. Even for r=64, it's under 7%. The tradeoff is almost always worth it for the flexibility of adapter swapping.

---

## Adapter Validation and Error Handling

### What vLLM Checks When Loading an Adapter

```
1. File format: adapter_config.json + adapter_model.safetensors must exist
2. Base model compatibility: target modules must match the base model's layers
3. Rank check: adapter rank <= --max-lora-rank
4. Vocabulary: if adapter has extra embeddings, they must fit in --lora-extra-vocab-size
5. Dtype: adapter weights are cast to --lora-dtype if needed
```

### Common Errors

**Rank too high:**
```
Error: LoRA rank 64 exceeds max_lora_rank=16
Fix: restart with --max-lora-rank 64
```

**Target module mismatch:**
```
Error: LoRA target module "fc1" not found in model
Fix: the adapter was trained for a different model architecture
```

**Vocabulary mismatch:**
```
Error: LoRA adapter has 1000 extra embeddings, but lora_extra_vocab_size=256
Fix: restart with --lora-extra-vocab-size 1000
```

Failed adapter loads don't crash the server — the specific request gets an error, but other requests continue normally.

---

## Supported Model Architectures

Not all models support LoRA in vLLM. The model must have a `SupportsLoRA` mixin:

```
Supported (as of vLLM 0.8+):
  ✓ Llama family (Llama 2, Llama 3, Llama 3.1, Code Llama)
  ✓ Mistral / Mixtral
  ✓ Qwen / Qwen2 / Qwen2.5
  ✓ Gemma / Gemma 2
  ✓ Phi-3 / Phi-3.5
  ✓ Baichuan
  ✓ ChatGLM
  ✓ GPT-BigCode (StarCoder)

Not supported (missing LoRA layer mappings):
  ✗ Some very new model architectures (check vLLM docs)
  ✗ Models without standard QKV/MLP naming conventions
```

---

## Key Takeaways

1. **vLLM serves LoRA on-the-fly** — the base model is frozen, LoRA is computed as `W₀x + BAx` during each forward pass
2. **Use the `model` field** in the OpenAI API to select which adapter to use per-request
3. **Overhead is small**: ~3% for r=16, ~7% for r=64 — the LoRA matmuls are tiny compared to the base
4. **LoRA + quantized base**: serve a GPTQ/AWQ INT4 base model with FP16 LoRA adapters for maximum memory efficiency
5. **Merge when you can**: if you only serve one adapter, merge it for zero overhead
6. **Keep separate when you need flexibility**: on-the-fly LoRA enables multi-adapter serving (Blog A4)

---

## What's Next

Serving one adapter is useful, but the real power is serving *many* adapters from one base model. **Blog A4** covers multi-LoRA serving — how vLLM manages dozens of adapters in GPU memory with hot-loading, eviction, and per-request adapter selection.

---

## Further Reading

- [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/features/lora.html) — official guide
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) — the paper that influenced vLLM's multi-LoRA design
- **Next: [Blog A4 — Multi-LoRA Serving](/blog/lora-04-multi-lora/)** — one base model, many adapters
