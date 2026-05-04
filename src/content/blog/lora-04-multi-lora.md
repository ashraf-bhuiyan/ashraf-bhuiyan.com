---
title: "Multi-LoRA Serving: Many Adapters, One Base Model"
description: "Adapter caching, LRU eviction, memory budgeting, dynamic registration, and per-request routing patterns."
pubDate: 2026-05-03
series: "LoRA & QLoRA in vLLM"
part: 4
tags: ["multi-LoRA", "adapter cache", "serving", "vLLM"]
---

## What Problem Does This Solve?

You're a platform serving 100 customers. Each customer has a fine-tuned model for their specific domain — medical Q&A, legal contracts, customer support in French, etc. Without LoRA, you need 100 separate model deployments:

```
Without multi-LoRA (100 customers, Llama 3.1-8B):

  Customer 1: Full model copy → 16 GB GPU memory
  Customer 2: Full model copy → 16 GB GPU memory
  ...
  Customer 100: Full model copy → 16 GB GPU memory
  ──────────────────────────────────────────────
  Total: 1,600 GB → 20× A100-80GB just for weights

With multi-LoRA:

  Shared base model: 16 GB (one copy)
  Customer 1 adapter:  0.05 GB
  Customer 2 adapter:  0.05 GB
  ...
  Customer 100 adapters: 100 × 0.05 GB = 5 GB
  ──────────────────────────────────────────────
  Total: 21 GB → fits on 1× A100-80GB
```

The economics change from "1 GPU per customer" to "1 GPU for 100 customers." This is the feature that makes LoRA transformative for production serving.

---

## How Multi-LoRA Batching Works

### The Mixed-Adapter Batch

In a single vLLM batch, different requests can use different adapters — or no adapter at all:

```
Batch at step N:
  Request 0:  adapter="medical-qa"     → uses medical-qa LoRA weights
  Request 1:  adapter="legal-summary"  → uses legal-summary LoRA weights
  Request 2:  adapter="medical-qa"     → uses medical-qa LoRA weights
  Request 3:  (no adapter)             → base model only
  Request 4:  adapter="code-review"    → uses code-review LoRA weights
```

The forward pass splits into two parts:

**Part 1: Base computation (shared)**
```
All requests share the same base weights W₀:
  y_base = W₀ @ [x₀, x₁, x₂, x₃, x₄]ᵀ    ← one batched matmul
```

**Part 2: LoRA computation (per-adapter)**
```
Each request gets its adapter's LoRA applied:
  y₀ += (α/r) · B_medical(A_medical(x₀))
  y₁ += (α/r) · B_legal(A_legal(x₁))
  y₂ += (α/r) · B_medical(A_medical(x₂))
  y₃ += 0     (no adapter)
  y₄ += (α/r) · B_code(A_code(x₄))
```

The naive approach — loop over adapters, gather tokens per adapter, compute, scatter back — is slow. vLLM uses specialized SGMV/BGMV kernels (Blog A5) to compute all LoRA additions in a single fused kernel launch.

---

## Adapter Lifecycle Management

### The Adapter Cache

vLLM manages adapters like a CPU cache — hot adapters stay in GPU memory, cold ones get evicted:

```
GPU Memory Layout:
  [Base Model: 16 GB][KV Cache: ~40 GB][LoRA Slots: N × adapter_size]
                                        ↑
                                        max_loras slots

  Slot 0: medical-qa     (loaded, active)
  Slot 1: legal-summary  (loaded, active)
  Slot 2: code-review    (loaded, idle)
  Slot 3: <empty>        (available)
```

### Loading and Eviction

When a request arrives for an adapter:

```
Request arrives: model="french-support"

Is "french-support" in GPU memory?
├── Yes → Use it (fast path, microseconds)
│
└── No → Is there an empty LoRA slot?
    ├── Yes → Load "french-support" from disk/CPU to GPU
    │         (slow path: disk read + GPU copy, ~100ms)
    │
    └── No → Evict the least-recently-used adapter
             Load "french-support" into the freed slot
             (slow path: evict + load, ~200ms)
```

The eviction policy is **LRU** (Least Recently Used) — the adapter that hasn't been used for the longest time gets evicted first. This works well for typical multi-tenant workloads where a subset of customers are active at any given time.

### Adapter State Machine

```
                    ┌─────────┐
         load       │         │      evict
    ┌──────────────►│  GPU    │──────────────┐
    │               │  ACTIVE │              │
    │               └────┬────┘              ▼
    │                    │              ┌─────────┐
┌───┴─────┐              │              │  CPU    │
│  DISK   │◄─────────────┘              │  CACHED │
│  (cold) │    unload (manual)          └────┬────┘
└─────────┘                                  │
    ▲                                        │
    └────────────────────────────────────────┘
                  evict from CPU
```

With `--max-cpu-loras`, vLLM can also cache adapters in CPU memory — faster to reload than from disk:

```
Adapter load times (approximate):

  From disk (SSD):    50-200 ms  (depends on adapter size)
  From CPU memory:    5-20 ms    (GPU DMA copy)
  Already on GPU:     < 0.1 ms   (just index into the slot)
```

---

## Configuration and Memory Budgeting

### Key Parameters

**`--max-loras`**: Number of adapters that can be loaded in GPU memory simultaneously.

```
max_loras=1:  Only one adapter at a time. Every adapter switch triggers load/evict.
              Good for: single-customer deployment

max_loras=4:  Four adapters hot in GPU memory. Covers most active customers.
              Good for: small multi-tenant deployment

max_loras=16: Sixteen adapters hot. Rare evictions for moderate customer bases.
              Good for: medium multi-tenant deployment

max_loras=64: Sixty-four adapters hot. Each uses memory.
              Good for: large-scale multi-tenant (if memory allows)
```

**`--max-lora-rank`**: Maximum rank of any adapter. Must be >= the highest rank across all adapters.

```
max_lora_rank=16:  Pre-allocates for rank 16. Can serve r=4, r=8, r=16.
max_lora_rank=64:  Pre-allocates for rank 64. Uses more memory per slot.
max_lora_rank=128: Pre-allocates for rank 128. Significant memory per slot.
```

### Memory Budget Calculation

The LoRA slots consume GPU memory proportional to `max_loras × max_lora_rank`:

```
Per-adapter memory (Llama 3.1-8B, all 7 linear layers per block, 32 layers):

  Per LoRA module:  2 × rank × hidden_dim × 2 bytes (FP16)
                    = 2 × r × 4096 × 2

  Q projection:     2 × r × 4096 × 2 bytes
  K projection:     2 × r × 1024 × 2 bytes  (GQA: kv_heads=8, head_dim=128)
  V projection:     2 × r × 1024 × 2 bytes
  O projection:     2 × r × 4096 × 2 bytes
  Gate projection:  2 × r × 14336 × 2 bytes
  Up projection:    2 × r × 14336 × 2 bytes
  Down projection:  2 × r × 14336 × 2 bytes

  Total per layer:  2 × r × (4096+1024+1024+4096+14336+14336+14336) × 2
                  = 2 × r × 53,248 × 2 = r × 213,000 bytes

  All 32 layers:    32 × r × 213,000 ≈ r × 6.8 MB

  r=16:   6.8 × 16  = 109 MB per adapter
  r=64:   6.8 × 64  = 435 MB per adapter
  r=128:  6.8 × 128 = 870 MB per adapter

Total LoRA memory allocation:
  max_loras=4,  r=16:   4 × 109 MB  = 436 MB
  max_loras=16, r=16:  16 × 109 MB  = 1.7 GB
  max_loras=16, r=64:  16 × 435 MB  = 7.0 GB
  max_loras=64, r=16:  64 × 109 MB  = 7.0 GB
```

The memory is pre-allocated at startup (like the KV cache), so it reduces the memory available for the KV cache — meaning fewer concurrent requests. Choose `max_loras` based on your actual working set, not the total number of adapters.

---

## Dynamic Adapter Registration

### Adding Adapters at Runtime

You don't need to restart vLLM to add new adapters. Use the API:

```bash
# Register a new adapter dynamically
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "spanish-support",
    "lora_path": "/data/adapters/spanish-support"
  }'

# Now you can use it immediately
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "spanish-support", "messages": [...]}'

# Unload an adapter you no longer need
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -d '{"lora_name": "spanish-support"}'
```

Dynamic registration is useful for:
- **Continuous deployment**: upload new adapter versions without restarting
- **On-demand loading**: load adapters only when needed (saves memory)
- **A/B testing**: register two versions of an adapter, route traffic to both

### Validation on Load

When an adapter is registered, vLLM validates:

```
1. File exists and has correct format (adapter_config.json + weights)
2. Target modules match the base model's architecture
3. Rank <= max_lora_rank
4. Extra vocab size <= lora_extra_vocab_size
5. No name collision with existing adapters
```

Validation happens during registration, not during the first request — so errors surface immediately.

---

## Multi-LoRA Routing Patterns

### Pattern 1: Adapter per Customer (Multi-Tenant SaaS)

```
Customer A → adapter="customer-a"
Customer B → adapter="customer-b"
Customer C → adapter="customer-c"

All customers share the same base model, same vLLM instance,
same KV cache. Only the LoRA weights differ.
```

Best for: B2B platforms where each customer has custom fine-tuning.

### Pattern 2: Adapter per Task

```
User request: "Summarize this document"  → adapter="summarizer"
User request: "Write a Python function"  → adapter="coder"
User request: "Translate to French"      → adapter="translator"

Same user, different adapters based on the task.
```

Best for: multi-capability assistants.

### Pattern 3: Adapter per Language

```
Request in English  → adapter="en"
Request in French   → adapter="fr"
Request in Japanese → adapter="ja"

Language-specific adapters that share a multilingual base model.
```

Best for: multilingual deployments where per-language fine-tuning improves quality.

### Pattern 4: A/B Testing

```
90% of traffic → adapter="v2-stable"
10% of traffic → adapter="v3-candidate"

Client-side or load-balancer routing. vLLM serves both from one instance.
```

Best for: evaluating new adapter versions in production.

---

## Monitoring and Observability

### Key Metrics to Watch

vLLM exposes LoRA-related metrics via its `/metrics` endpoint:

```
# Adapter loading
vllm:lora_adapter_load_total          # Total adapter loads
vllm:lora_adapter_evict_total         # Total adapter evictions
vllm:lora_adapter_load_duration_ms    # Load latency

# Active adapters
vllm:lora_adapters_active             # Currently loaded adapter count
vllm:lora_requests_by_adapter         # Request count per adapter
```

### What to Monitor

```
High eviction rate (evictions/sec > 1):
  → max_loras is too low for the active adapter set
  → Increase max_loras or add CPU caching (--max-cpu-loras)

Adapter load latency > 500ms:
  → Adapter files are large or disk is slow
  → Use CPU caching or faster storage (NVMe)

Uneven adapter distribution:
  → Some adapters get 90% of traffic, others 0.1%
  → Consider merging the hot adapter for zero overhead
  → Consider evicting cold adapters proactively
```

---

## Practical Tips

### Right-sizing `--max-loras`

```
Rule of thumb:
  max_loras = number of adapters active in a 5-minute window

If you have 100 adapters but only 10 are active at any time:
  max_loras=10-12 (add 20% headroom for transient loads)

If your traffic pattern has bursts:
  max_loras = peak concurrent adapters + 2
```

### Adapter Organization on Disk

```
/data/adapters/
  ├── medical-qa/
  │   ├── adapter_config.json
  │   └── adapter_model.safetensors
  ├── legal-summary/
  │   ├── adapter_config.json
  │   └── adapter_model.safetensors
  └── code-review/
      ├── adapter_config.json
      └── adapter_model.safetensors
```

Keep adapters on fast storage (NVMe SSD). Loading from network storage (NFS, S3) adds latency on every cache miss.

### When Multi-LoRA Doesn't Make Sense

```
Single adapter, always active:
  → Merge the LoRA into the base model. Zero overhead.

Very high rank (r=256+):
  → LoRA overhead becomes significant (>10%)
  → Consider full fine-tuning with a separate deployment

Adapters trained for different base models:
  → Can't share a base model. Need separate vLLM instances.

Latency-critical, zero-tolerance:
  → Even 3% overhead matters. Merge the adapter.
```

---

## Key Takeaways

1. **Multi-LoRA turns one GPU into a multi-tenant serving platform** — 100 customers on 1 GPU instead of 100 GPUs
2. **Adapter management is a cache** — `max_loras` slots in GPU, LRU eviction, optional CPU caching
3. **Mixed-adapter batching** lets different requests in the same batch use different adapters — enabled by SGMV/BGMV kernels (Blog A5)
4. **Memory budget**: each slot costs `rank × 6.8 MB` (for 8B model, all linear layers). Budget accordingly.
5. **Dynamic registration**: add/remove adapters at runtime without server restart
6. **Monitor eviction rate**: high evictions mean `max_loras` is too low for your traffic pattern

---

## What's Next

Multi-LoRA batching requires computing different LoRA weights for different rows of the same batch. **Blog A5** explains the SGMV and BGMV GPU kernels that make this efficient — the single-kernel solution to the "different weights per row" problem.

---

## Further Reading

- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) — the paper behind vLLM's multi-LoRA design
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547) — the SGMV/BGMV kernels used by vLLM
- [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/features/lora.html)
- **Next: [Blog A5 — LoRA Kernel Internals](/blog/lora-05-kernels/)** — how SGMV and BGMV make batched multi-adapter inference fast
