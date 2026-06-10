---
title: "Part 9: Tensor Parallelism"
description: "Column-parallel and row-parallel linear layers (the Megatron pattern), AllReduce for combining results, and weight sharding across GPUs."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 9
tags: ["tensor parallelism", "Megatron", "multi-GPU", "AllReduce"]
---

## What Problem Does This Solve?

Models are getting larger than GPUs. Llama 3.1 405B at FP16 needs 810GB of memory — no single GPU exists with that much. Even a 70B model at FP16 needs 140GB, exceeding the 80GB of an H100.

The solution: **split the model across multiple GPUs.** Tensor parallelism (TP) splits individual weight matrices across GPUs. Each GPU holds a slice of every layer, and they cooperate to produce the same output as a single large GPU would.

```
Single GPU (impossible — doesn't fit):
┌──────────────────────────────────────────────┐
│ GPU 0: Full 70B model (140GB)                │  ✗ Doesn't fit in 80GB
└──────────────────────────────────────────────┘

Tensor Parallelism (TP=2):
┌───────────────────────┐  ┌───────────────────────┐
│ GPU 0: Left half of   │  │ GPU 1: Right half of  │
│ every weight matrix   │  │ every weight matrix    │
│ (70GB)                │  │ (70GB)                 │
└───────────────────────┘  └───────────────────────┘
         │                          │
         └────── AllReduce ─────────┘
            (sync after each layer)
```

TP=2 cuts memory per GPU in half. TP=4 cuts it to a quarter. TP=8 puts a 405B model across 8 GPUs.

---

## The Core Idea: Column-Parallel + Row-Parallel

The Megatron-LM paper introduced a pattern for parallelizing transformer layers using two types of linear layers: **column-parallel** and **row-parallel**. They're always used in pairs, and the pair requires only ONE AllReduce communication.

### Column-Parallel Linear

Split the weight matrix **by columns** (output dimension). Each GPU computes a slice of the output independently — no communication needed.

```
Standard:              Y = X @ W           W: [in, out]

Column-parallel:       Y_0 = X @ W_0      W_0: [in, out/2]   GPU 0
                       Y_1 = X @ W_1      W_1: [in, out/2]   GPU 1

Full W = [W_0 | W_1]  (concatenated by columns)
Full Y = [Y_0 | Y_1]  (each GPU has a different slice of Y)

Example with W: [4, 8], TP=2:
  GPU 0 gets W[:, 0:4] → computes Y[:, 0:4]
  GPU 1 gets W[:, 4:8] → computes Y[:, 4:8]
  No communication — each GPU works independently!
```

### Row-Parallel Linear

Split the weight matrix **by rows** (input dimension). Each GPU multiplies its slice of the input by its slice of the weight, producing a **partial** output. An AllReduce sums the partials to get the final result.

```
Standard:              Y = X @ W           W: [in, out]

Row-parallel:          Y_0 = X_0 @ W_0    W_0: [in/2, out]   GPU 0
                       Y_1 = X_1 @ W_1    W_1: [in/2, out]   GPU 1
                       Y = Y_0 + Y_1      ← AllReduce(SUM)!

Full W = [W_0]   (stacked by rows)
         [W_1]

Example with W: [8, 4], TP=2:
  GPU 0 gets W[0:4, :] and input X[:, 0:4] → partial Y
  GPU 1 gets W[4:8, :] and input X[:, 4:8] → partial Y
  AllReduce(SUM) → full Y on all GPUs
```

### The Megatron-LM Pair

In a transformer FFN, there are two linear layers in sequence. The first is column-parallel, the second is row-parallel:

```
┌──────────────────────────────────────────────────────────┐
│                  Tensor-Parallel FFN                      │
│                                                          │
│  Input X ──► [Column-Parallel W1] ──► GeLU ──► [Row-Parallel W2] ──► AllReduce ──► Output
│                                                                                    │
│  GPU 0:  X @ W1_0 → GeLU → hidden_0 @ W2_0 → partial_0 ─┐                       │
│  GPU 1:  X @ W1_1 → GeLU → hidden_1 @ W2_1 → partial_1 ─┤─► AllReduce → Y       │
│  GPU 2:  X @ W1_2 → GeLU → hidden_2 @ W2_2 → partial_2 ─┤                       │
│  GPU 3:  X @ W1_3 → GeLU → hidden_3 @ W2_3 → partial_3 ─┘                       │
│                                                                                    │
│  Communication: ZERO between W1 and W2 (the slices line up!)                       │
│                 ONE AllReduce after W2 (sum the partials)                           │
└──────────────────────────────────────────────────────────┘
```

The key insight: the output of column-parallel W1 on each GPU is exactly the input that row-parallel W2 needs on that GPU. The dimensions line up, so no communication is needed between the two layers. The only communication is the AllReduce at the end.

---

## How It Works

### AllReduce: The Cost of TP

AllReduce is the collective operation where every GPU contributes data, all data is summed element-wise, and every GPU gets the full result:

```
Before AllReduce:
  GPU 0: [1, 2, 3, 4]
  GPU 1: [5, 6, 7, 8]

AllReduce(SUM):
  GPU 0: [6, 8, 10, 12]   (element-wise sum)
  GPU 1: [6, 8, 10, 12]   (same result on both!)
```

AllReduce is typically implemented as a **ring** — each GPU sends and receives from its neighbors in a pipeline:

```
Ring AllReduce (4 GPUs):

  Step 1: Reduce-Scatter                Step 2: All-Gather
  (each GPU gets 1/4 of final sum)      (broadcast the partial sums)

       GPU 0                                 GPU 0
      ╱     ╲                               ╱     ╲
   GPU 3   GPU 1          →             GPU 3   GPU 1
      ╲     ╱                               ╲     ╱
       GPU 2                                 GPU 2

  GPU 0 → sends chunk to GPU 1         GPU 0 → sends result to GPU 1
  GPU 1 → sends chunk to GPU 2         GPU 1 → sends result to GPU 2
  GPU 2 → sends chunk to GPU 3         GPU 2 → sends result to GPU 3
  GPU 3 → sends chunk to GPU 0         GPU 3 → sends result to GPU 0

  Total data sent per GPU: 2 × (N-1)/N × data_size
  With NVLink (900 GB/s): microseconds for typical hidden sizes
```

For each transformer layer, TP requires **2 AllReduces**: one after the attention output projection, one after the FFN down projection. A 32-layer model does 64 AllReduces per forward pass.

AllReduce communication cost: each GPU sends `2 × (N-1)/N × data_size` bytes, where N is the TP degree. For a hidden size of 4096 with batch×seq = 2048 tokens at FP16:

```
data_size = 2048 × 4096 × 2 bytes = 16 MB
AllReduce cost (TP=8): 2 × 7/8 × 16 MB = 28 MB sent per AllReduce
64 AllReduces × 28 MB = 1.8 GB total communication per forward pass

H100 NVLink: 900 GB/s → 1.8 GB / 900 = 2 ms of communication
```

### Weight Matrix Slicing: Visual

Here's exactly how the weight matrices are sliced for TP=2:

```
Column-Parallel (gate_proj, up_proj, Q, K, V projections):

  Full weight: [out=8, in=4]          GPU 0 gets:        GPU 1 gets:
  ┌─────────────────────────┐        ┌────────────┐     ┌────────────┐
  │ a  b  c  d  │  e  f  g  h │      │ a  b  c  d │     │ e  f  g  h │
  │ i  j  k  l  │  m  n  o  p │  →   │ i  j  k  l │     │ m  n  o  p │
  │ q  r  s  t  │  u  v  w  x │      │ q  r  s  t │     │ u  v  w  x │
  │ 1  2  3  4  │  5  6  7  8 │      │ 1  2  3  4 │     │ 5  6  7  8 │
  └─────────────┴─────────────┘      └────────────┘     └────────────┘
   columns 0-3    columns 4-7         [4, 4]             [4, 4]

  X @ W_0 = Y_0 (partial output)     X @ W_1 = Y_1 (partial output)
  No communication needed — Y_0 and Y_1 are independent slices of Y

Row-Parallel (down_proj, output projection):

  Full weight: [out=4, in=8]          GPU 0 gets:        GPU 1 gets:
  ┌─────────────────────────┐        ┌────────────┐     ┌────────────┐
  │ a  b  c  d  e  f  g  h  │      │ a  b  c  d │     │ e  f  g  h │
  │ i  j  k  l  m  n  o  p  │  →   │ i  j  k  l │     │ m  n  o  p │
  │ q  r  s  t  u  v  w  x  │      │ q  r  s  t │     │ u  v  w  x │
  │ 1  2  3  4  5  6  7  8  │      │ 1  2  3  4 │     │ 5  6  7  8 │
  └─────────────────────────┘      └────────────┘     └────────────┘
                                     rows 0-3           rows 4-7
                                     [4, 4]             [4, 4]

  X_0 @ W_0 = partial_0             X_1 @ W_1 = partial_1
  AllReduce(partial_0 + partial_1) = Y   ← communication here
```

### TP for Attention

Attention heads are naturally parallel — each head is independent. TP splits heads across GPUs:

```
Llama with 32 attention heads, TP=4:
  GPU 0: heads 0-7    (QKV projection: column-parallel)
  GPU 1: heads 8-15
  GPU 2: heads 16-23
  GPU 3: heads 24-31

Output projection: row-parallel (AllReduce after)
→ 1 AllReduce per attention sublayer
```

For GQA (grouped-query attention) with fewer KV heads, TP requires the number of KV heads to be divisible by the TP degree. Llama 3.1 70B has 8 KV heads, so TP≤8.

### TP for FFN (SwiGLU)

Modern LLMs use SwiGLU FFN with gate and up projections:

```
SwiGLU FFN:   Y = (SiLU(X @ W_gate) ⊙ (X @ W_up)) @ W_down

TP split:
  W_gate, W_up  → column-parallel (split output dimension)
  W_down         → row-parallel (AllReduce after)

GPU 0: gate_0 = X @ W_gate_0,  up_0 = X @ W_up_0
        hidden_0 = SiLU(gate_0) ⊙ up_0
        partial_0 = hidden_0 @ W_down_0
GPU 1: ... same with shard 1 ...
AllReduce(partial_0 + partial_1 + ...) → full output
```

### Weight Loading: Sharding Full Weights

When loading a model, the full checkpoint contains the complete weight matrices. Each GPU extracts its shard:

```python
# Column-parallel: slice columns
full_weight: [intermediate, hidden]  = [11008, 4096]
gpu_0_shard: full_weight[0:5504, :]  = [5504, 4096]    # first half of outputs
gpu_1_shard: full_weight[5504:, :]   = [5504, 4096]    # second half

# Row-parallel: slice rows (input dimension)
full_weight: [hidden, intermediate]  = [4096, 11008]
gpu_0_shard: full_weight[:, 0:5504]  = [4096, 5504]    # first half of inputs
gpu_1_shard: full_weight[:, 5504:]   = [4096, 5504]    # second half
```

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `ColumnParallelLinear` | `ColumnParallelLinear` | `ColumnParallelLinear` |
| `RowParallelLinear` | `RowParallelLinear` | `RowParallelLinear` |
| `AllReduce(SUM)` in row-parallel | `tensor_model_parallel_all_reduce()` | Same (via PyTorch dist) |
| `load_weight_shard()` | `weight_loader()` per layer | `load_weights()` with sharding |
| `torchrun --nproc_per_node` | Ray or multiprocessing with NCCL | multiprocessing with NCCL |
| Manual weight slicing | `WeightTiledLoader` + `QuantizedWeightsLoader` | Similar weight loading |

Key differences:

**vLLM's weight loading:** vLLM doesn't load the full checkpoint and then slice. Instead, each GPU loads only its shard directly from the checkpoint using the `weight_loader` function attached to each parameter. This avoids the peak memory of having all weights in CPU memory.

**Custom AllReduce:** vLLM implements a custom AllReduce kernel (`custom_ar`) that's faster than NCCL for small message sizes. For hidden sizes under ~16K tokens, the latency of launching NCCL's ring algorithm dominates. vLLM's custom kernel uses shared memory and direct GPU-to-GPU copies via NVLink, reducing launch overhead.

**Quantization + TP:** When combining TP with quantization (e.g., INT4 weight-only), the weight shards are stored in quantized format. Each GPU holds quantized weights for its shard, dequantizing on-the-fly during the matmul. This compounds the memory savings: TP=4 with INT4 reduces memory by 16x vs FP16 on one GPU.

---

## The Implementation

The complete implementation is in [09_tensor_parallelism.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/09_tensor_parallelism.py) (~280 lines).

### Column-Parallel Linear

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        self.out_per_rank = out_features // world_size
        self.linear = nn.Linear(in_features, self.out_per_rank)

    def load_weight_shard(self, full_weight):
        start = self.rank * self.out_per_rank
        end = start + self.out_per_rank
        self.linear.weight.data.copy_(full_weight[start:end, :])

    def forward(self, x):
        return self.linear(x)  # no communication!
```

### Row-Parallel Linear

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        self.in_per_rank = in_features // world_size
        self.linear = nn.Linear(self.in_per_rank, out_features)

    def load_weight_shard(self, full_weight):
        start = self.rank * self.in_per_rank
        end = start + self.in_per_rank
        self.linear.weight.data.copy_(full_weight[:, start:end])

    def forward(self, x):
        y = self.linear(x)
        dist.all_reduce(y, op=dist.ReduceOp.SUM)  # THE communication
        return y
```

### TP MLP (Megatron-LM Pattern)

```python
class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size, rank):
        self.gate_proj = ColumnParallelLinear(...)   # no comm
        self.down_proj = RowParallelLinear(...)       # AllReduce
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.gate_proj(x))   # each GPU: independent
        return self.down_proj(h)           # AllReduce inside
```

---

## Running the Code

**Demo with 2 GPUs:**
```bash
torchrun --nproc_per_node=2 09_tensor_parallelism.py --demo
```

**Demo with 4 GPUs:**
```bash
torchrun --nproc_per_node=4 09_tensor_parallelism.py --demo
```

Expected output (TP=2 on H100):
```
World size (TP degree): 2
GPUs: 2x NVIDIA H100 80GB HBM3

--- Correctness Test ---
  Max absolute difference: 3.91e-03
  Match: YES (fp32 rounding from different op order)

--- Weight Sharding ---
  Full gate_proj weight: [4096, 1024]
    GPU 0: gate_proj shard [2048, 1024] (columns 0:2048)
    GPU 1: gate_proj shard [2048, 1024] (columns 2048:4096)

--- Performance ---
  Config: hidden=4096, inter=11008, batch=16, seq=128
  Single GPU:  7.59 ms/forward
  TP=2:       3.99 ms/forward
  Speedup:     1.90x

  Weight memory per GPU:
    Single GPU: 360.7 MB (full weights)
    TP=2:      180.4 MB (1/2 of weights)
```

---

## Benchmarks

| TP Degree | Weight Memory/GPU | Compute Speedup | Communication Overhead |
|-----------|------------------|-----------------|----------------------|
| TP=1 | 100% | 1.0x | None |
| TP=2 | 50% | ~1.9x | 2 AllReduces/layer |
| TP=4 | 25% | ~3.5x | 2 AllReduces/layer |
| TP=8 | 12.5% | ~5-6x | 2 AllReduces/layer |

Why isn't TP=8 8x faster? AllReduce communication grows with TP degree (more GPUs to synchronize). At TP=8, the communication overhead eats into the compute savings. The sweet spot is usually TP=2 or TP=4 for inference.

| Model | FP16 Size | Min GPUs (80GB) | Recommended TP |
|-------|----------|----------------|---------------|
| Llama 3.1 8B | 16 GB | 1 | TP=1 |
| Llama 3.1 70B | 140 GB | 2 | TP=2 or TP=4 |
| Llama 3.1 405B | 810 GB | 11 | TP=8 |
| DeepSeek V3 (671B) | 1.3 TB | 17 | TP=8 + EP |

---

## Key Takeaways

1. **Tensor parallelism** splits weight matrices across GPUs so each GPU holds a fraction of every layer
2. **Column-parallel** for the first linear (no communication), **row-parallel** for the second (AllReduce)
3. The Megatron-LM pattern requires **2 AllReduces per transformer layer** — that's all the communication
4. TP reduces both memory and compute per GPU, but AllReduce overhead limits scaling beyond TP=8
5. Attention heads are split across GPUs — TP degree must divide the number of KV heads
6. Real systems like vLLM use custom AllReduce kernels that beat NCCL for small messages

---

## Further Reading

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — the paper that introduced column/row-parallel TP
- [vLLM distributed inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) — production TP configuration
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — the AllReduce implementation
- **Next: [Blog 10 — Data Parallelism](10_data_parallelism.md)** — replicate the model on multiple GPUs for higher throughput
