---
title: "Part 12: KV Cache CPU Offloading"
description: "Two-tier GPU/CPU memory with swap-out and swap-in using pinned memory to handle burst traffic beyond GPU capacity."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 12
tags: ["KV cache", "CPU offloading", "memory management"]
---

## What Problem Does This Solve?

GPU memory is the bottleneck for concurrent LLM serving. An H100 has 80GB of HBM. A 70B model's weights consume ~35GB in FP16, leaving ~45GB for KV cache. With a 4096-token context window, each request's KV cache at FP16 consumes roughly 1.3GB (80 layers x 8 KV heads x 128 head_dim x 4096 positions x 2 bytes x 2 for K+V). That means you can serve about 35 concurrent requests before the GPU runs out of KV cache memory.

Meanwhile, the host machine typically has 512GB to 2TB of CPU DRAM sitting mostly idle. That is 6-25x more memory than GPU HBM. What if we could use it?

```
Without CPU offloading:
  GPU HBM: [model weights 35GB][KV cache 45GB]
  CPU RAM: [                512GB idle                ]
  Max concurrent requests: ~35
  New request arrives: REJECTED (no GPU memory)

With CPU offloading:
  GPU HBM: [model weights 35GB][KV cache 45GB (hot)]
  CPU RAM: [      KV cache overflow (cold) 100GB     ]
  Completed sequence KV → swapped to CPU
  New request with matching prefix → swapped back from CPU
  No recomputation needed!
```

The insight: when a request finishes, its KV blocks are freed from GPU. But if a future request shares the same prefix, we would need to recompute those KV values from scratch. CPU offloading saves completed KV blocks to CPU memory, so a future prefix match can restore them with a memory copy instead of a full forward pass.

This is especially valuable for:
- **System prompts** shared across thousands of requests
- **Multi-turn conversations** where the history is repeated each turn
- **RAG pipelines** where the same retrieved documents appear repeatedly
- **High-throughput servers** under memory pressure

---

## The Core Idea: Two-Tier Cache

Think of GPU and CPU memory as two levels of a cache hierarchy, just like L1/L2 caches in a CPU, or RAM/disk in an operating system:

```
                    Two-Tier KV Cache
  ┌──────────────────────────────────────────┐
  │           GPU Tier (Hot)                 │
  │   Fast access (~2 TB/s bandwidth)       │
  │   Limited capacity (e.g., 64 blocks)    │
  │                                          │
  │   Active sequences live here             │
  │   Prefix cache for recent prompts        │
  │                                          │
  │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
  │   │Blk 0 │ │Blk 1 │ │Blk 2 │ │ ...  │  │
  │   └──────┘ └──────┘ └──────┘ └──────┘  │
  └──────────────────┬───────────────────────┘
                     │ swap-out (GPU → CPU)
                     │ swap-in  (CPU → GPU)
  ┌──────────────────┴───────────────────────┐
  │           CPU Tier (Cold)                │
  │   Slower (~50 GB/s with pinned memory)  │
  │   Much larger capacity (e.g., 256 blocks)│
  │                                          │
  │   Completed sequences' KV blocks stored  │
  │   here after GPU blocks are freed        │
  │                                          │
  │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
  │   │Blk 0 │ │Blk 1 │ │Blk 2 │ │ ...  │  │
  │   └──────┘ └──────┘ └──────┘ └──────┘  │
  └──────────────────────────────────────────┘
```

The lifecycle of a KV block:

1. **Allocate on GPU** -- new request arrives, blocks are allocated from GPU free list
2. **Compute KV** -- prefill/decode fills blocks with key-value data
3. **Register in GPU hash table** -- prefix caching records block hashes for future reuse
4. **Swap-out to CPU** -- when the sequence completes, KV blocks are copied to CPU pinned memory
5. **Free GPU blocks** -- GPU blocks return to the free list for new requests
6. **Swap-in from CPU** -- when a new request matches a CPU-cached prefix, blocks are copied back to GPU

The key difference from Blog 7's prefix caching: there, cached blocks stayed on GPU. Here, they move to CPU when GPU pressure is high, and come back when needed. The GPU hash table is the L1 cache; the CPU hash table is the L2.

---

## How It Works

### Two-Tier Lookup on Allocation

When a new request arrives, the allocator checks both tiers in order:

```
New request: "System prompt... What is gravity?"
Token IDs: [101, 202, 303, ..., 404, 505]

1. Compute block hashes:
   Block 0: hash(None, tokens[0:16])    = 0xABCD
   Block 1: hash(0xABCD, tokens[16:32]) = 0x1234

2. Two-tier lookup:
   0xABCD → check GPU hash table → MISS
           → check CPU hash table → HIT! (cpu_block 7)
           → swap-in: copy cpu_block 7 → gpu_block 3
           → record in GPU hash table
           → source = "cpu_hit"

   0x1234 → check GPU hash table → MISS
           → check CPU hash table → HIT! (cpu_block 12)
           → swap-in: copy cpu_block 12 → gpu_block 5
           → source = "cpu_hit"

3. Allocate fresh GPU blocks for remaining tokens

Result: 32 cached tokens restored from CPU, skip prefill for them
```

The lookup order matters: GPU first (fast, no copy needed), then CPU (slower, requires a copy), then miss (allocate fresh, full computation). This mirrors how CPU cache hierarchies work -- check L1 before L2 before main memory.

### Swap-Out: GPU to CPU

When a sequence finishes generation, its KV blocks contain valuable prefix data that future requests might reuse. Instead of just freeing the GPU blocks, we copy them to CPU first:

```
Sequence "abc123" finishes generating
Token IDs: [101, 202, ..., 404]

For each full block:
  1. Compute hash for the block
  2. If hash already exists in GPU or CPU cache → skip (already cached)
  3. If hash is new:
     a. Allocate a free CPU block
     b. Copy GPU K tensor → CPU K tensor (gpu_k[block] → cpu_k[block])
     c. Copy GPU V tensor → CPU V tensor (gpu_v[block] → cpu_v[block])
     d. Record hash → cpu_block in CPU hash table
     e. Increment swap_out counter

Then free all GPU blocks for this sequence
```

The copy is a `tensor.copy_()` call. On a real GPU setup, this is a device-to-host transfer over PCIe or NVLink. With pinned memory (explained below), this transfer can be asynchronous -- the GPU can continue processing other requests while the DMA engine handles the copy.

### Swap-In: CPU to GPU

When the two-tier lookup finds a hash in the CPU tier, the block's KV data must be copied back to GPU before the model can use it:

```
def _swap_in(cpu_block, gpu_block):
    gpu_k[gpu_block].copy_(cpu_k[cpu_block])
    gpu_v[gpu_block].copy_(cpu_v[cpu_block])
```

Each block contains KV data for all layers at all positions within the block:

```
Block shape: [num_layers, block_size, num_kv_heads, head_dim]

For TinyLlama (22 layers, 4 KV heads, 64 head_dim, block_size=16):
  Per block: 22 * 16 * 4 * 64 * 4 bytes (float32) = 352 KB
  Swap-in: 352 KB copy per block (CPU → GPU)
  Swap-out: 352 KB copy per block (GPU → CPU)
```

For a 32-token prefix (2 blocks), swap-in costs ~700KB of memory transfer. Compare that to recomputing 32 tokens through a 1.1B parameter model -- the copy is dramatically cheaper.

### Pinned Memory

Regular CPU memory is pageable -- the OS can swap it to disk at any time. When a GPU needs to read from pageable CPU memory, it must first copy the data to a pinned (page-locked) staging buffer, then transfer it to GPU memory. This double-copy is slow.

Pinned memory is allocated with `pin_memory=True` and is locked in physical RAM -- the OS guarantees it will never be swapped to disk. The GPU can DMA directly from pinned memory without the intermediate copy:

```
Pageable memory (slow):
  CPU pageable → [OS copies to] → CPU pinned staging → [DMA] → GPU
  Two copies, CPU involved in first copy

Pinned memory (fast):
  CPU pinned → [DMA directly] → GPU
  One copy, CPU is free during transfer
```

In our implementation, the CPU tier tensors are allocated with `pin_memory=True` when a CUDA device is available:

```python
self.cpu_k = torch.zeros(
    num_cpu_blocks, num_layers, block_size, num_kv_heads, head_dim,
    pin_memory=(self.device.type == "cuda"),
)
```

The trade-off: pinned memory cannot be swapped to disk by the OS, so it permanently consumes physical RAM. Allocating too much pinned memory can starve other processes. In practice, systems allocate a fixed CPU block budget and stick to it.

### GPU Block Eviction

When the GPU free list is empty and a new block is needed, the allocator evicts the least-recently-used cached block from the GPU hash table:

```python
def _alloc_gpu_block(self):
    if self.gpu_free:
        return self.gpu_free.pop()
    # Evict LRU cached block from GPU
    for h, block in list(self.gpu_hash_to_block.items()):
        del self.gpu_hash_to_block[h]
        return block
    return None
```

The `OrderedDict` maintains insertion/access order. The first entry is the least recently used. This is the same LRU strategy from Blog 7, but now evicted blocks might still be available in the CPU tier -- they are not lost entirely, just demoted to a slower tier.

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Notes |
|----------|-----------|-------|
| `TwoTierKVCache` | `CpuGpuBlockAllocator` | Manages blocks across both devices |
| `swap_out_sequence()` | `CacheEngine.swap_out()` via `KVConnector.send_kv_caches_and_hidden_states()` | Batch swap with CUDA streams |
| `_swap_in()` | `CacheEngine.swap_in()` via `KVConnector.recv_kv_caches()` | Async copy with events |
| `gpu_hash_to_block` | `BlockPool` + `BlockHashToBlockMap` | Per-device hash tables |
| `cpu_hash_to_block` | CPU block allocator within `CpuGpuBlockAllocator` | Separate free list |
| `pin_memory=True` | `torch.cuda.pin_memory()` on CPU tensors | Same mechanism |
| LRU eviction (OrderedDict) | `FreeKVCacheBlockQueue` (doubly-linked list) | O(1) eviction |
| Sync `copy_()` | `torch.cuda.Stream` + `copy_(non_blocking=True)` | Overlap with compute |

Key details in production systems:

**Async transfers with CUDA streams:** Our implementation uses synchronous `copy_()` calls. Real vLLM uses dedicated CUDA streams for swap-in and swap-out, allowing these transfers to overlap with model computation on the default stream. The GPU can run the next batch's forward pass while simultaneously copying completed KV blocks to CPU in the background.

```
Our code (synchronous):
  [forward pass] → [swap-out copy] → [forward pass]
                    ^^^^ GPU idle during copy

vLLM (async with streams):
  Default stream: [forward pass] [forward pass] [forward pass]
  Swap stream:         [swap-out copy]    [swap-in copy]
                       ^^^^ overlapped with compute
```

**HMA (Hybrid Memory Allocation):** For hybrid models that mix attention-heavy layers with MoE layers, vLLM supports allocating KV cache blocks for different layers on different devices. Attention-heavy layers keep their KV on GPU; less-critical layers can store KV on CPU from the start.

**KVConnector abstraction:** vLLM's `KVConnector` is a general interface for moving KV cache between locations. CPU offloading is one implementation. The same interface supports disaggregated serving (Blog 13), where KV is transferred between machines over the network.

**Block-level granularity:** Both our code and vLLM operate at block granularity. A 16-token block is the atomic unit of swap. This keeps metadata overhead low (one hash per block, not per token) and enables efficient bulk transfers.

---

## The Implementation

The complete implementation is in [12_kv_cpu_offloading.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/12_kv_cpu_offloading.py) (~600 lines).

### Two-Tier Cache Setup

The cache pre-allocates tensors for both GPU and CPU tiers:

```python
class TwoTierKVCache:
    def __init__(self, num_gpu_blocks, num_cpu_blocks,
                 block_size, num_layers, num_kv_heads, head_dim,
                 device=None):
        # GPU tier
        self.gpu_k = torch.zeros(
            num_gpu_blocks, num_layers, block_size, num_kv_heads, head_dim,
            device=self.device,
        )
        self.gpu_v = torch.zeros(
            num_gpu_blocks, num_layers, block_size, num_kv_heads, head_dim,
            device=self.device,
        )
        self.gpu_free = list(range(num_gpu_blocks))

        # CPU tier (pinned memory for fast GPU<->CPU transfer)
        self.cpu_k = torch.zeros(
            num_cpu_blocks, num_layers, block_size, num_kv_heads, head_dim,
            pin_memory=(self.device.type == "cuda"),
        )
        self.cpu_v = torch.zeros(
            num_cpu_blocks, num_layers, block_size, num_kv_heads, head_dim,
            pin_memory=(self.device.type == "cuda"),
        )
        self.cpu_free = list(range(num_cpu_blocks))

        # Hash-based cache lookup (prefix caching)
        self.gpu_hash_to_block = OrderedDict()
        self.cpu_hash_to_block = OrderedDict()
```

Both tiers have independent free lists and hash tables. The GPU tier is small (64 blocks by default), the CPU tier is large (256 blocks). The ratio reflects the memory capacity difference between GPU and CPU.

### Two-Tier Allocation with Swap-In

```python
def allocate(self, seq_id, token_ids):
    hashes = self.compute_block_hashes(token_ids)
    cached_blocks = 0
    source = "miss"

    for h in hashes:
        if h in self.gpu_hash_to_block:
            # GPU cache hit -- reuse directly, no copy needed
            gpu_block = self.gpu_hash_to_block[h]
            self.gpu_hash_to_block.move_to_end(h)  # mark as recently used
            self.block_tables[seq_id].append(gpu_block)
            cached_blocks += 1
            self.gpu_hits += 1
            source = "gpu_hit"

        elif h in self.cpu_hash_to_block:
            # CPU cache hit -- swap in to GPU
            cpu_block = self.cpu_hash_to_block[h]
            gpu_block = self._alloc_gpu_block()
            if gpu_block is None:
                break
            self._swap_in(cpu_block, gpu_block)
            self.gpu_hash_to_block[h] = gpu_block
            self.block_tables[seq_id].append(gpu_block)
            cached_blocks += 1
            self.cpu_hits += 1
            self.swap_ins += 1
            source = "cpu_hit"
        else:
            break  # first miss -- contiguous prefix only

    # Allocate fresh blocks for remaining tokens
    remaining = len(token_ids) - cached_blocks * self.block_size
    new_blocks = math.ceil(remaining / self.block_size)
    for _ in range(new_blocks):
        gpu_block = self._alloc_gpu_block()
        if gpu_block is None:
            raise RuntimeError("Out of GPU KV cache blocks")
        self.block_tables[seq_id].append(gpu_block)
        self.misses += 1

    return cached_tokens, remaining, source
```

Notice the three-way return: `gpu_hit` (free, instant), `cpu_hit` (cheap copy), or `miss` (expensive recomputation). The caller uses this to decide whether to run a full prefill or a partial one.

### Swap-Out on Completion

```python
def swap_out_sequence(self, seq_id, token_ids):
    hashes = self.compute_block_hashes(token_ids)
    block_table = self.block_tables[seq_id]

    for i, h in enumerate(hashes):
        if i >= len(block_table):
            break
        if h in self.gpu_hash_to_block or h in self.cpu_hash_to_block:
            continue  # already cached somewhere, skip

        gpu_block = block_table[i]
        if self.cpu_free:
            cpu_block = self.cpu_free.pop()
            # Copy KV data: GPU → CPU
            self.cpu_k[cpu_block].copy_(self.gpu_k[gpu_block])
            self.cpu_v[cpu_block].copy_(self.gpu_v[gpu_block])
            self.cpu_hash_to_block[h] = cpu_block
            self.swap_outs += 1
```

The swap-out only copies blocks whose hash is not already present in either tier. This avoids redundant copies when multiple sequences share the same prefix -- the first completion caches the shared blocks, subsequent completions skip them.

### Engine Integration

The `OffloadingEngine` wraps the two-tier cache with a TinyLlama model. The key addition compared to Blog 7's engine is the `offload_after` parameter in `generate()`:

```python
def generate(self, prompt, max_tokens=64, temperature=0.0,
             offload_after=True):
    # ... prefill and decode as usual ...

    if offload_after:
        self.cache.swap_out_sequence(seq_id, token_ids)

    self.cache.free_gpu(seq_id)
    return result
```

After generation completes, `swap_out_sequence` saves the KV blocks to CPU, then `free_gpu` returns the GPU blocks to the free list. Future requests with the same prefix will find the blocks in the CPU tier.

---

## Running the Code

**Demo mode** (shows cold cache, GPU hits, and CPU restore):

```bash
python 12_kv_cpu_offloading.py --demo
```

**Server mode:**

```bash
python 12_kv_cpu_offloading.py --port 5000

# First request (cold cache):
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a helpful assistant. What is 2+2?", "max_tokens": 20}'

# Second request (GPU hit from prefix caching):
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a helpful assistant. Name 3 colors.", "max_tokens": 20}'

# Check cache stats:
curl http://localhost:5000/health
```

**Custom block budgets:**

```bash
# Small GPU, large CPU -- forces more swap-in/swap-out
python 12_kv_cpu_offloading.py --demo --gpu-blocks 8 --cpu-blocks 64
```

Expected demo output:

```
DEMO: KV CPU Offloading
============================================================

--- Phase 1: Initial requests (cold cache) ---
  Request 1 [miss]: What is the capital of France?
    Cached: 0 tokens, Prefill: 285ms
  Request 2 [gpu_hit]: Explain gravity in one sentence.
    Cached: 16 tokens, Prefill: 190ms
  Request 3 [gpu_hit]: What is 2+2?
    Cached: 16 tokens, Prefill: 175ms

  After Phase 1:
    GPU cached blocks: 5
    CPU cached blocks: 9
    Swap-outs: 9

--- Phase 2: Repeat with same prefix (GPU cache hit) ---
  Request 4 [gpu_hit]: Name three colors.
    Cached: 16 tokens, Prefill: 172ms
  Request 5 [gpu_hit]: What is Python?
    Cached: 16 tokens, Prefill: 168ms

--- Phase 3: Simulate GPU pressure + CPU restore ---
  Request A [miss]: 0 cached
  Request B [miss]: 0 cached
    CPU cached: 4 blocks (from swap-out)
  Request A again [cpu_hit]: 16 cached
    Swap-ins: 1 (restored from CPU)
```

Phase 3 is the interesting part: with only 8 GPU blocks, Request B evicts Request A's cached prefix from GPU. When Request A repeats, the prefix is found in the CPU tier and swapped back in -- no recomputation needed.

---

## Benchmarks

| Metric | No Offloading (GPU-only cache) | With CPU Offloading |
|--------|-------------------------------|---------------------|
| Effective cache capacity | 64 blocks (GPU only) | 64 + 256 = 320 blocks |
| Cache hit after GPU eviction | MISS (recompute) | CPU HIT (swap-in) |
| Prefill on GPU hit | ~170ms | ~170ms (same) |
| Prefill on CPU hit | N/A | ~180ms (copy + partial prefill) |
| Prefill on miss | ~285ms | ~285ms (same) |
| Swap-out cost per block | N/A | ~0.1ms (async possible) |
| Swap-in cost per block | N/A | ~0.1ms (sync copy) |
| Memory usage (CPU) | Minimal | +256 blocks of pinned RAM |

The cost-benefit analysis:

```
Recomputing 32 tokens through TinyLlama: ~100ms (full forward pass)
Swap-in 2 blocks from CPU:              ~0.2ms (memory copy)

Speedup from CPU hit vs miss: ~500x for the cached portion
```

On larger models, the gap widens. A 70B model at 4096 context needs ~1.3GB of KV per request. Recomputing that from scratch takes hundreds of milliseconds on an H100. Copying it from pinned CPU memory over PCIe 5.0 (64 GB/s) takes ~20ms. That is a 10-50x win.

| Scenario | GPU-only Cache | Two-Tier Cache |
|----------|---------------|----------------|
| Chatbot (shared system prompt) | First request cached, rest recompute after eviction | CPU tier preserves prefix indefinitely |
| RAG (same docs, different queries) | Only fits recent docs in GPU cache | CPU tier holds hundreds of document prefixes |
| Multi-turn conversation | Re-prefill entire history after GPU eviction | Restore from CPU, only compute new turn |
| High concurrency (100+ requests) | Constant eviction, low hit rate | CPU absorbs overflow, high overall hit rate |

---

## Key Takeaways

1. **GPU memory limits concurrent requests.** CPU DRAM is 10-100x larger and can serve as overflow storage for KV cache blocks.

2. **Two-tier cache = GPU hot tier + CPU cold tier.** The GPU hash table is checked first (free hit), then the CPU hash table (cheap copy), then miss (expensive recomputation).

3. **Swap-out saves KV blocks to CPU when a sequence completes.** This preserves prefix data for future requests without consuming GPU memory.

4. **Swap-in restores KV blocks from CPU when a prefix match is found.** A memory copy is orders of magnitude cheaper than recomputing through the model.

5. **Pinned memory enables fast GPU-CPU transfers.** The OS guarantees pinned pages stay in physical RAM, allowing the GPU DMA engine to transfer directly without an intermediate copy.

6. **Async transfers overlap with compute in production.** Real systems use dedicated CUDA streams so swap-in and swap-out happen concurrently with model inference on the default stream.

---

## Further Reading

- [vLLM CPU Offloading](https://docs.vllm.ai/en/latest/features/cpu-offload.html) -- production CPU offloading documentation
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) -- prefix caching that integrates with CPU tier
- [FlexGen: High-Throughput Generative Inference](https://arxiv.org/abs/2303.06865) -- systematic GPU-CPU-disk offloading for throughput-oriented serving
- [CUDA Pinned Memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) -- NVIDIA guide on optimizing host-device transfers
- **Next: [Blog 13 -- Disaggregated Serving](13_disaggregated_serving.md)** -- separate prefill and decode across machines
