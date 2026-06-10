---
title: "Part 11: Expert Parallelism"
description: "Mixture-of-Experts architecture with AllToAll communication to dispatch tokens across GPUs holding different experts."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 11
tags: ["MoE", "expert parallelism", "AllToAll", "DeepSeek"]
---

## What Problem Does This Solve?

Mixture-of-Experts (MoE) models like DeepSeek-V3 have **256 expert FFN networks** per layer, but each token only activates 8 of them. The total parameter count is huge (671B), but active parameters per token are small (~37B). This is what makes MoE efficient — sparse activation.

The problem: these 256 experts need to live somewhere. With pure tensor parallelism (TP=8), each expert's tiny 2048-dim intermediate gets split across 8 GPUs (256 dims per GPU). That's too small to saturate GPU compute — you're doing 256-dim matrix multiplies on a chip designed for thousands.

```
Pure TP=8 (bad for MoE):
  Each expert: [hidden=7168, inter=2048]
  Per GPU:     [hidden=7168, inter=256]   ← too small!
  GPU utilization: ~5%

Expert parallelism (EP=8):
  GPU 0: experts 0-31   (32 complete experts)
  GPU 1: experts 32-63  (32 complete experts)
  ...
  GPU 7: experts 224-255 (32 complete experts)
  
  Each expert stays whole: [hidden=7168, inter=2048]  ← full-size GEMMs!
  All-to-All routes tokens to the right GPU
```

EP keeps expert matrices whole so each GPU does meaningful-sized matrix multiplies. The cost: All-to-All communication to route tokens to the GPU that owns their assigned expert.

---

## The Core Idea: Route Tokens, Not Weights

In TP, we split weights and replicate data. In EP, we split experts and route data:

```
TP approach (split each expert):
  Every GPU gets a slice of every expert
  Tokens stay on their GPU
  AllReduce after each layer

EP approach (assign whole experts):
  Each GPU gets complete experts
  Tokens travel to the GPU with their expert
  All-to-All to dispatch and collect tokens
```

### MoE Architecture Recap

A MoE layer replaces the dense FFN with a router + multiple expert FFNs:

```
Dense FFN:    output = FFN(input)                      (all tokens → one FFN)

MoE FFN:     expert_ids = router(input)                (which experts?)
             output = Σ weight_k × expert_k(input)     (top-K experts)

Router:      logits = input @ W_gate   → softmax → top-K
             Selects K experts per token (typically K=2)

Example (8 experts, top-2):
  Token "The":    → experts [2, 5]  weights [0.6, 0.4]
  Token "weather": → experts [0, 3]  weights [0.55, 0.45]
  Token "is":     → experts [1, 7]  weights [0.7, 0.3]
```

### The All-to-All Communication Pattern

All-to-All is a collective where each GPU sends different data to each other GPU:

```
Before All-to-All:
  GPU 0 has tokens for:  experts 0,1 (local) + experts 2,3 (GPU 1)
  GPU 1 has tokens for:  experts 0,1 (GPU 0) + experts 2,3 (local)

All-to-All (dispatch):
  GPU 0 sends "tokens for experts 2,3" → GPU 1
  GPU 1 sends "tokens for experts 0,1" → GPU 0

After All-to-All:
  GPU 0 has: all tokens assigned to experts 0,1 (from all GPUs)
  GPU 1 has: all tokens assigned to experts 2,3 (from all GPUs)

Each GPU runs its local experts, then All-to-All sends results back.
```

---

## How It Works

### EP Forward Pass

```
Input: x [num_tokens, hidden_size]
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
   Router (replicated)     Router (replicated)
   GPU 0                   GPU 1
   indices, weights        indices, weights
       │                         │
       ▼                         ▼
   Group tokens by          Group tokens by
   destination GPU          destination GPU
       │                         │
       └────── All-to-All ───────┘
              (dispatch)
       │                         │
       ▼                         ▼
   Run experts 0,1          Run experts 2,3
   on received tokens       on received tokens
       │                         │
       └────── All-to-All ───────┘
              (combine)
       │                         │
       ▼                         ▼
   Weighted sum             Weighted sum
   output                   output
```

### Load Balancing

If the router sends all tokens to expert 0, GPU 0 is overloaded while other GPUs are idle. Load balancing is critical:

```
Balanced routing (good):
  Expert 0: 32 tokens
  Expert 1: 28 tokens
  Expert 2: 31 tokens
  Expert 3: 33 tokens
  Imbalance: 1.2x (max/min)

Imbalanced routing (bad):
  Expert 0: 95 tokens  ← hot expert, GPU stalls
  Expert 1: 10 tokens
  Expert 2: 12 tokens
  Expert 3: 8 tokens
  Imbalance: 11.9x
```

DeepSeek-V3 uses an **auxiliary loss** during training that penalizes load imbalance, plus a **bias term** in the router that dynamically adjusts to equalize expert load.

### Expert Capacity and Token Dropping

Some systems set a maximum **capacity** per expert: if an expert receives more tokens than its capacity, excess tokens are dropped (processed by a residual connection instead). This bounds the worst-case compute per GPU but can hurt quality.

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `Router` (linear + top-k) | `FusedMoE.forward()` | `FusedMoE` |
| `EPMoELayer` | `FusedMoEMethodBase` with EP | EP in `FusedMoE` |
| `All-to-All` (dispatch) | `AllToAllManager` | All-to-All via NCCL |
| `All-to-All` (combine) | `AllToAllManager` | All-to-All via NCCL |
| Python loop over experts | Fused Triton/CUTLASS kernels | Fused MoE kernel |
| Manual token grouping | `moe_align_block_size()` kernel | Similar alignment |

Key details:

**Fused MoE kernels:** Real systems don't loop over experts in Python. They use fused Triton or CUTLASS kernels that process all experts in one GPU kernel launch. The kernel groups tokens by expert, runs all expert GEMMs, and combines results — all without returning to Python.

**All-to-All implementations:** vLLM supports multiple All-to-All backends:
- **NCCL All-to-All:** Standard, works everywhere
- **AllGather + ReduceScatter:** An alternative decomposition that can be faster for small messages
- **Custom pipelining:** Overlap communication with expert computation

**EP + TP composition:** For DeepSeek-V3 on 8 GPUs, the recommended config is TP=4, EP=2:
- Attention layers: TP=4 AllReduce within each group
- MoE layers: EP=2 All-to-All between the two groups
- Each expert stays at full 2048-dim intermediate

---

## The Implementation

The complete implementation is in [11_expert_parallelism.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/11_expert_parallelism.py) (~350 lines).

### Router

```python
class Router(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return indices, weights
```

### EP Forward with All-to-All

```python
class EPMoELayer(nn.Module):
    def forward(self, x):
        # 1. Route tokens
        indices, weights = self.router(x)
        
        # 2. Group tokens by destination GPU
        send_buf = group_by_destination(x, indices)
        
        # 3. All-to-All dispatch
        recv_buf = all_to_all(send_buf)
        
        # 4. Run local experts
        recv_out = run_local_experts(recv_buf)
        
        # 5. All-to-All combine
        result_buf = all_to_all(recv_out)
        
        # 6. Weighted sum
        return weighted_sum(result_buf, weights)
```

---

## Running the Code

**Demo with 2 GPUs:**
```bash
torchrun --nproc_per_node=2 11_expert_parallelism.py --demo
```

**Demo with 4 GPUs:**
```bash
torchrun --nproc_per_node=4 11_expert_parallelism.py --demo
```

Expected output (EP=4 on H100):
```
8 experts, top-2, hidden=512, inter=1024
2 experts per GPU

Correctness: Max difference 8.20e-08 ✓

Expert distribution:
  GPU 0: experts 0-1
  GPU 1: experts 2-3
  GPU 2: experts 4-5
  GPU 3: experts 6-7

Load balancing (128 tokens, top-2):
  Expert 0:  33 tokens (12.9%)
  Expert 1:  28 tokens (10.9%)
  ...
  Imbalance: 1.5x
```

---

## Benchmarks

| Metric | Single GPU (all experts) | EP=4 |
|--------|------------------------|------|
| Expert memory/GPU | 100% | 25% |
| Expert GEMM size | Full (2048-dim) | Full (2048-dim) |
| Communication | None | 2 All-to-All/layer |
| GPU utilization | High (full GEMMs) | High (full GEMMs) |

| Config | Pros | Cons |
|--------|------|------|
| Pure TP=8 | Simple, well-tested | Tiny expert GEMMs (256-dim) |
| TP=4, EP=2 | Good expert GEMMs, TP for attention | More complex |
| TP=2, EP=4 | Best expert efficiency | Attention only TP=2 |
| Pure EP=8 | Full-size experts | Replicated attention (memory waste) |

For DeepSeek-V3 inference, TP=4 × EP=2 is the recommended configuration on 8 GPUs.

---

## Key Takeaways

1. **MoE models** have many expert FFNs but each token activates only a few (sparse activation)
2. **Expert parallelism** assigns complete experts to GPUs — each GPU owns a subset of experts
3. **All-to-All** routes tokens to the GPU owning their assigned expert (dispatch) and routes results back (combine)
4. **EP keeps expert GEMMs full-sized** — unlike TP which splits them into inefficiently small matrices
5. **Load balancing** is critical — imbalanced routing creates hot-spot GPUs
6. **EP + TP composition** is standard: TP for attention layers, EP for MoE layers

---

## Further Reading

- [Switch Transformer](https://arxiv.org/abs/2101.03961) — foundational MoE paper
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — 256 experts, auxiliary-loss-free load balancing
- [Megablocks: Efficient Sparse Training with MoE](https://arxiv.org/abs/2211.15841) — fused MoE kernel design
- **Next: [Blog 12 — KV CPU Offloading](12_kv_cpu_offloading.md)** — swap KV cache to CPU when GPU memory runs out
