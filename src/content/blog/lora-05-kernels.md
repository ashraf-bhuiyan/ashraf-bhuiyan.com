---
title: "LoRA Kernel Internals: SGMV and BGMV"
description: "How Punica SGMV/BGMV kernels batch multiple LoRA adapters in a single GPU operation."
pubDate: 2026-05-03
series: "LoRA & QLoRA in vLLM"
part: 5
tags: ["SGMV", "BGMV", "Punica", "GPU kernels"]
---

## What Problem Does This Solve?

In multi-LoRA serving, a single batch contains requests using *different* LoRA adapters. The standard batched matrix multiply (`Y = X @ W`) assumes all rows of `X` use the same weight matrix `W`. But in multi-LoRA, each row needs a different LoRA weight.

```
Standard batched GEMM:
  Row 0: y₀ = W @ x₀     ┐
  Row 1: y₁ = W @ x₁     ├── Same W for all rows → one cuBLAS call
  Row 2: y₂ = W @ x₂     │
  Row 3: y₃ = W @ x₃     ┘

Multi-LoRA batch:
  Row 0: y₀ = W₀ @ x₀ + B_medical(A_medical(x₀))     ← adapter "medical"
  Row 1: y₁ = W₀ @ x₁ + B_legal(A_legal(x₁))         ← adapter "legal"
  Row 2: y₂ = W₀ @ x₂ + B_medical(A_medical(x₂))     ← adapter "medical"
  Row 3: y₃ = W₀ @ x₃                                  ← no adapter
```

The naive solution: group tokens by adapter, run a separate matmul for each group, scatter results back. This means multiple kernel launches (expensive), poor GPU utilization (small groups don't saturate the GPU), and Python overhead (looping over adapters).

SGMV and BGMV solve this by computing all LoRA additions in a **single kernel launch**.

---

## The Naive Approach: Why Looping Is Slow

### Grouping by Adapter

```python
# Naive multi-LoRA computation
def naive_multi_lora(x, adapter_ids, lora_a_weights, lora_b_weights, alpha, rank):
    output = torch.zeros_like(x)
    
    for adapter_id in set(adapter_ids):
        # Find which tokens use this adapter
        mask = (adapter_ids == adapter_id)
        x_group = x[mask]                    # gather
        
        if adapter_id == -1:  # no adapter
            continue
        
        # LoRA computation for this adapter
        A = lora_a_weights[adapter_id]       # (rank, hidden_dim)
        B = lora_b_weights[adapter_id]       # (hidden_dim, rank)
        lora_out = (alpha / rank) * (x_group @ A.T @ B.T)
        
        output[mask] = lora_out              # scatter
    
    return output
```

Why this is slow:

```
Batch of 256 tokens, 10 different adapters:

Naive approach:
  10 × gather:           10 kernel launches
  10 × matmul (A):       10 kernel launches  
  10 × matmul (B):       10 kernel launches
  10 × scatter:          10 kernel launches
  Total: 40 kernel launches
  
  Average group size: 25 tokens → each matmul is tiny, 
  GPU is massively underutilized

SGMV approach:
  1 × fused kernel:      1 kernel launch
  Total: 1 kernel launch
  
  All 256 tokens computed together, GPU fully utilized
```

Each kernel launch has ~5-10 microseconds of overhead. 40 launches = 200-400 microseconds of pure overhead. More importantly, small matmuls (25 × hidden_dim) don't saturate the GPU's compute units — the GPU sits mostly idle.

---

## SGMV: Segmented Matrix-Vector Multiply

### The Key Idea

SGMV treats the multi-adapter batch as **segments** — contiguous groups of tokens that share the same adapter. A single kernel processes all segments, using an index array to select the correct weight matrix for each segment.

```
Input batch (sorted by adapter):
  [medical₀, medical₁, medical₂ | legal₃, legal₄ | code₅, code₆, code₇]
   ──────── segment 0 ─────────  ── segment 1 ──  ──── segment 2 ─────

Segment metadata:
  segment_starts:  [0, 3, 5]           ← where each segment begins
  segment_lengths: [3, 2, 3]           ← tokens in each segment
  adapter_ids:     [0, 1, 2]           ← which adapter for each segment

Stacked LoRA weights:
  lora_a[0] = A_medical   (rank × hidden_dim)
  lora_a[1] = A_legal     (rank × hidden_dim)
  lora_a[2] = A_code      (rank × hidden_dim)
```

### The SGMV Kernel

Each GPU thread block processes one segment:

```
Thread block 0 (segment 0, adapter="medical"):
  Read x[0:3]                         ← 3 tokens
  Read lora_a[0]                      ← medical's A matrix
  Compute x[0:3] @ A_medical.T        ← (3, rank) output
  Read lora_b[0]                      ← medical's B matrix
  Compute intermediate @ B_medical.T  ← (3, hidden_dim) output
  Write to output[0:3]

Thread block 1 (segment 1, adapter="legal"):
  Read x[3:5]                         ← 2 tokens
  Read lora_a[1]                      ← legal's A matrix
  Compute x[3:5] @ A_legal.T
  ...

Thread block 2 (segment 2, adapter="code"):
  Read x[5:8]                         ← 3 tokens
  Read lora_a[2]                      ← code's A matrix
  ...

All three thread blocks run simultaneously on the GPU!
```

### Why SGMV Is Fast

```
1. Single kernel launch (vs. 40+ in the naive approach)
   → eliminates launch overhead

2. All segments computed in parallel
   → GPU cores are distributed across segments proportionally

3. Coalesced memory access
   → tokens within a segment are contiguous in memory
   → LoRA weights are contiguous per adapter slot
   → both lead to efficient GPU memory bandwidth usage

4. No Python overhead
   → no Python loop over adapters
   → the kernel does all the indexing internally
```

### When SGMV Is Used

SGMV is most efficient during the **decode phase**, where each request contributes exactly one token:

```
Decode batch (32 requests, 5 adapters):
  Each request = 1 token = effectively a matrix-vector multiply
  SGMV: Segmented Matrix-Vector Multiply
  
  Small segments (1-10 tokens each) → SGMV handles this well
  because it's designed for thin (few-row) matrix operations
```

---

## BGMV: Batched Grouped Matrix-Vector Multiply

### When Segments Are Large

During the **prefill phase**, a single request can contribute hundreds or thousands of tokens. If multiple requests use the same adapter, the "segment" can be very large:

```
Prefill batch (4 requests):
  Request 0: 512 tokens, adapter="medical"
  Request 1: 256 tokens, adapter="medical"
  Request 2: 128 tokens, adapter="legal"
  Request 3: 384 tokens, adapter="medical"

Grouped by adapter:
  medical group: 512 + 256 + 384 = 1,152 tokens
  legal group:   128 tokens
```

BGMV (Batched Grouped Matrix-Vector Multiply) is optimized for this case:

```
BGMV groups tokens by adapter and runs a grouped GEMM:

Group 0 (medical, 1152 tokens):
  [x₀...x₅₁₁, x₅₁₂...x₇₆₇, x₈₉₆...x₁₂₇₉] @ A_medical.T
  → Large GEMM, good GPU utilization

Group 1 (legal, 128 tokens):
  [x₇₆₈...x₈₉₅] @ A_legal.T
  → Moderate GEMM
```

### BGMV vs. SGMV

```
                    SGMV                        BGMV
                    ────                        ────
Optimized for:      Small segments              Large groups
                    (decode: 1 token/request)   (prefill: many tokens/request)

Parallelism:        One thread block per        Groups tokens by adapter,
                    segment                     runs grouped GEMM per adapter

Best when:          Many adapters, few          Few adapters, many tokens
                    tokens per adapter          per adapter

GPU utilization:    Good for thin matrices      Good for tall matrices
```

### Automatic Selection

vLLM's `PunicaWrapper` automatically selects the right kernel based on the batch:

```
if all segments have 1 token (pure decode):
    use SGMV
elif any segment has > threshold tokens:
    use BGMV
else:
    use SGMV (default for mixed phases)
```

The threshold varies by GPU architecture and LoRA rank. The selection is per-step — a single vLLM run might use SGMV for decode steps and BGMV for prefill steps.

---

## PunicaWrapper: The Dispatch Layer

### Architecture

`PunicaWrapper` is the abstraction layer between vLLM's model code and the SGMV/BGMV kernels:

```
                  Model forward pass
                        │
                  ┌─────▼─────┐
                  │  Linear    │
                  │  Layer     │
                  │            │
                  │  y = W₀x  │ ← base computation (standard cuBLAS)
                  │     +     │
                  │  LoRA(x)  │ ← LoRA computation (dispatched below)
                  └─────┬─────┘
                        │
                  ┌─────▼──────────┐
                  │ PunicaWrapper  │
                  │                │
                  │  1. Build      │ ← which token → which adapter
                  │     index      │
                  │                │
                  │  2. Select     │ ← SGMV or BGMV?
                  │     kernel     │
                  │                │
                  │  3. Dispatch   │ ← launch the kernel
                  └────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼                   ▼
          ┌───────┐          ┌────────┐
          │ SGMV  │          │  BGMV  │
          │kernel │          │ kernel │
          └───────┘          └────────┘
```

### Building the Index

On each forward pass, `PunicaWrapper` builds an index array mapping tokens to adapters:

```python
# Conceptual index building (simplified)
token_adapter_map = []
for request in batch:
    adapter_id = request.lora_id  # -1 if no adapter
    for token in request.tokens:
        token_adapter_map.append(adapter_id)

# Sort or segment by adapter_id for the kernel
# token_adapter_map: [0, 0, 0, 1, 1, 2, 2, 2, -1, -1]
# segments:          [0:3→adapter0, 3:5→adapter1, 5:8→adapter2, 8:10→none]
```

This indexing happens on every step because the batch composition changes (requests join and leave with continuous batching).

---

## Weight Stacking and Memory Layout

### How LoRA Weights Are Stored

vLLM pre-allocates a contiguous tensor for all LoRA adapter slots:

```
lora_a_stacked: shape [max_loras, 1, rank, in_features]
lora_b_stacked: shape [max_loras, 1, out_features, rank]

Example (max_loras=4, rank=16, Llama-8B Q projection):

lora_a_stacked: [4, 1, 16, 4096]    ← 4 slots × 16 × 4096 × 2B = 512 KB
lora_b_stacked: [4, 1, 4096, 16]    ← 4 slots × 4096 × 16 × 2B = 512 KB

Adapter in slot 0: lora_a_stacked[0] = A_medical   (16 × 4096)
Adapter in slot 1: lora_a_stacked[1] = A_legal     (16 × 4096)
Adapter in slot 2: lora_a_stacked[2] = A_code      (16 × 4096)
Adapter in slot 3: lora_a_stacked[3] = <empty>     (available)
```

### Why Contiguous Stacking Matters

The SGMV/BGMV kernels index into the stacked tensor using the adapter slot ID:

```
To compute LoRA for token with adapter_slot=1:
  A = lora_a_stacked[1]    ← single indexed memory access
  B = lora_b_stacked[1]    ← single indexed memory access
  output += (α/r) × (x @ A.T) @ B.T
```

Contiguous stacking enables:
- **Coalesced GPU memory access**: adjacent adapters are adjacent in memory
- **Simple indexing**: adapter_slot maps directly to the first dimension
- **Pre-allocated sizing**: no dynamic allocation during inference

### Per-Layer Storage

Each LoRA-wrapped linear layer has its own stacked tensors:

```
Layer 0, Q projection:  lora_a[4, 1, 16, 4096],  lora_b[4, 1, 4096, 16]
Layer 0, K projection:  lora_a[4, 1, 16, 1024],  lora_b[4, 1, 1024, 16]
Layer 0, V projection:  lora_a[4, 1, 16, 1024],  lora_b[4, 1, 1024, 16]
...
Layer 31, Down proj:    lora_a[4, 1, 16, 14336], lora_b[4, 1, 4096, 16]
```

Each layer stores its own A and B stacks. When an adapter is loaded into slot `i`, its weights are written to `lora_a_stacked[i]` and `lora_b_stacked[i]` across all layers.

---

## Performance Characteristics

### Overhead vs. Batch Size

```
LoRA overhead (% of total forward pass time) for Llama 3.1-8B, rank=16:

Batch size   No LoRA    With LoRA    LoRA overhead
           (ms/step)    (ms/step)    (%)
──────────────────────────────────────────────────
1            12.1        12.5          3.3%
8            12.8        13.2          3.1%
32           14.5        15.0          3.4%
128          21.3        22.0          3.3%
256          35.2        36.4          3.4%

LoRA overhead is roughly constant as a percentage (~3-4% for r=16)
because the LoRA matmuls scale proportionally with batch size
```

### Overhead vs. Rank

```
LoRA overhead by rank (batch size 32, Llama 3.1-8B):

Rank    LoRA overhead    LoRA matmul time
        (% of total)     (ms/step)
──────────────────────────────────────────
8        1.8%             0.26
16       3.4%             0.50
32       6.2%             0.93
64       11.5%            1.73
128      20.1%            3.02
```

At rank=128, the LoRA computation becomes non-trivial — one-fifth of the total forward pass time. This is the practical limit for on-the-fly LoRA; beyond this, merging is usually better.

### Scaling with Number of Adapters

```
LoRA overhead by number of active adapters (batch 32, rank 16):

Active adapters    LoRA overhead    Notes
─────────────────────────────────────────────────────────────
1                  3.2%             All tokens use same adapter
4                  3.4%             Same kernel, different segments
16                 3.5%             Slightly more index overhead
32                 3.6%             Marginal increase
64                 3.8%             Still negligible scaling

Key insight: going from 1 to 64 adapters adds only 0.6% overhead.
The kernel does the same amount of LoRA computation regardless
of how many different adapters are involved — it's the total 
number of tokens × rank that determines cost, not the adapter count.
```

---

## Profiling LoRA Kernels

### Using torch.profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model.generate(inputs, lora_request=lora_request)

# Look for Punica/SGMV/BGMV kernels in the trace
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### What to Look For

```
Kernel Name                         CUDA Time    Count
──────────────────────────────────────────────────────
ampere_sgemm_128x32_tn              45.2 ms      640     ← base model matmuls
sgmv_shrink                          0.8 ms      640     ← LoRA A (x → rank)
sgmv_expand                          0.9 ms      640     ← LoRA B (rank → hidden)
flash_fwd_kernel                    12.3 ms       32     ← attention
elementwise_kernel                   1.1 ms      320     ← activations

LoRA kernels (sgmv_shrink + sgmv_expand) = 1.7 ms out of ~60 ms total = 2.8%
```

The `sgmv_shrink` kernel computes `x @ A.T` (project down to rank), and `sgmv_expand` computes `intermediate @ B.T` (project back up to hidden_dim).

---

## Key Takeaways

1. **SGMV processes all LoRA adapters in a single kernel launch** — no Python loops, no per-adapter kernel launches
2. **BGMV handles large groups** during prefill when many tokens share an adapter
3. **PunicaWrapper** auto-selects SGMV or BGMV based on batch composition
4. **Weight stacking** pre-allocates contiguous tensors for all adapter slots — the kernel indexes by slot ID
5. **Overhead scales with rank, not adapter count** — going from 1 to 64 adapters adds < 1% overhead
6. **Practical limits**: rank 16 → ~3% overhead (negligible), rank 128 → ~20% overhead (consider merging)

---

## What's Next

You now understand how vLLM serves multi-LoRA efficiently at the kernel level. **Blog A6** takes this to production scale: S-LoRA's paged adapter memory, composing LoRA with tensor/data parallelism, and tuning for real workloads.

---

## Further Reading

- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547) — the SGMV/BGMV kernel paper
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) — adapter memory management
- [CUTLASS Grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped) — NVIDIA's grouped GEMM primitives
- **Next: [Blog A6 — Production Multi-LoRA](/blog/lora-06-production/)** — scaling multi-LoRA to production
