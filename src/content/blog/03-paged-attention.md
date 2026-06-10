---
title: "Part 3: Paged Attention"
description: "How paged attention borrows the OS virtual memory concept to store KV cache in fixed-size blocks, enabling 2-4x more concurrent requests."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 3
tags: ["paged attention", "KV cache", "memory management", "vLLM"]
---

## What Problem Does This Solve?

In Blog 1, we used HuggingFace's built-in `DynamicCache` — a contiguous tensor that grows as the sequence gets longer. This works fine for a single request, but it has a fatal flaw when serving many requests simultaneously: **memory fragmentation**.

Here's the problem. Suppose you pre-allocate KV cache for a maximum sequence length of 2048 tokens per request, and you can fit 10 such allocations in memory:

```
Memory with contiguous allocation (max_seq_len=2048):

Request A (actual: 200 tokens):  [████░░░░░░░░░░░░░░░░░░░░░░░░░░]  ← 90% wasted
Request B (actual: 800 tokens):  [████████████████░░░░░░░░░░░░░░]  ← 60% wasted
Request C (actual: 50 tokens):   [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ← 97% wasted
Request D: REJECTED (no space)   [                                ]

Total memory used: 30%
But no room for Request D because each slot reserves max_seq_len!
```

You could try allocating exactly what each request needs, but you don't know how many tokens a request will generate until it's done. You could allocate small and grow, but growing a contiguous tensor means copying the entire cache — expensive, and you might not find a big enough contiguous region.

This is the same problem that operating systems faced in the 1960s with memory management. The solution there was **virtual memory with paging** — split memory into fixed-size pages, give each process a page table mapping virtual pages to physical frames. This eliminated external fragmentation entirely.

vLLM applies the same idea to KV cache. This is the core contribution of the [vLLM paper](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023), and it's what enables serving 2-4x more concurrent requests than naive memory management.

---

## The Core Idea: Blocks Instead of Contiguous Tensors

Instead of one big contiguous tensor per sequence, split the KV cache into fixed-size **blocks** (typically 16 tokens each). Pre-allocate a pool of blocks. When a sequence needs more space, hand it a free block from the pool. When a sequence finishes, return its blocks.

```
Contiguous (Blog 1):
  Seq A: [████████████████████████████████████████]  one big allocation
  Seq B: [██████████████████████]                    another big allocation
         ↑ can't grow A without copying

Paged (Blog 3):
  Block pool: [0][1][2][3][4][5][6][7][8][9][10][11]...

  Seq A's block table: [3, 7, 1, 9]     → 4 blocks = up to 64 tokens
  Seq B's block table: [0, 5, 11]       → 3 blocks = up to 48 tokens
  Free blocks:         [2, 4, 6, 8, 10] → available for new sequences

  Seq A needs more space? Take block 2 from the free list.
  Seq B done? Return blocks [0, 5, 11] to the free list.
  No copying, no fragmentation.
```

The key insight: **blocks don't need to be physically contiguous.** A sequence's KV cache can be scattered across any blocks in the pool. The block table maps logical positions to physical locations, just like an OS page table maps virtual addresses to physical frames.

---

## How It Works

### The Block Table

Each sequence gets a **block table** — an array where entry `i` is the physical block index for logical block `i`. To find where token `t` is stored:

```
logical_block = t // block_size        (which block?)
offset        = t % block_size         (where in that block?)
physical_block = block_table[logical_block]

KV data = cache[physical_block][offset]
```

Here's a concrete example with `block_size=4`:

```
Sequence: "The capital of France is Paris , a city"
Tokens:    T₀   T₁     T₂  T₃     T₄ T₅   T₆ T₇  T₈

Block table: [5, 2, 8]    (3 blocks for 9 tokens)

Physical block 5:  [T₀ T₁ T₂ T₃]     ← logical block 0
Physical block 2:  [T₄ T₅ T₆ T₇]     ← logical block 1
Physical block 8:  [T₈ _  _  _ ]     ← logical block 2 (partially filled)

Where is T₆?
  logical_block = 6 // 4 = 1
  offset        = 6 %  4 = 2
  physical_block = block_table[1] = 2
  → cache[2][2] = T₆ ✓
```

### Allocation and Freeing

Block allocation works like a **free list allocator**:

```
Initial state:  free = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Seq A arrives (prompt = 6 tokens, block_size = 4):
  Needs ceil(6/4) = 2 blocks
  Allocate blocks 9, 8 from free list
  Seq A block table: [9, 8]
  free = [0, 1, 2, 3, 4, 5, 6, 7]

Seq A generates 3 more tokens (total = 9):
  Needs ceil(9/4) = 3 blocks, has 2 → allocate 1 more
  Allocate block 7 from free list
  Seq A block table: [9, 8, 7]
  free = [0, 1, 2, 3, 4, 5, 6]

Seq A finishes:
  Return blocks [9, 8, 7] to free list
  free = [0, 1, 2, 3, 4, 5, 6, 9, 8, 7]
```

The beauty: **waste is at most `block_size - 1` tokens per sequence** (the last block's unused slots). With block_size=16, that's less than 4% waste at the median. Compare this to contiguous allocation where waste can be 50-90%.

### Scatter and Gather

Two operations move data between the model's contiguous tensors and our paged blocks:

**Scatter (write):** After the model computes new K/V tensors, scatter them into the correct block positions:

```
Model output (contiguous): [K₀ K₁ K₂ K₃ K₄ K₅]
                              ↓  ↓  ↓  ↓  ↓  ↓
Block table: [5, 2]          ↓  ↓  ↓  ↓  ↓  ↓
Physical block 5:           [K₀ K₁ K₂ K₃]
Physical block 2:           [K₄ K₅  _   _ ]
```

**Gather (read):** Before calling the model, gather KV from scattered blocks into a contiguous tensor:

```
Block table: [5, 2]
Physical block 5: [K₀ K₁ K₂ K₃]    →  Contiguous: [K₀ K₁ K₂ K₃ K₄ K₅]
Physical block 2: [K₄ K₅  _   _ ]   ↗
```

In our CPU implementation, we do this gather explicitly before each decode step. In real vLLM, the **paged attention CUDA kernel** reads directly from the blocks using the block table — no gather needed. This kernel is the performance-critical piece: it computes attention while reading K/V from non-contiguous memory locations, avoiding the copy entirely.

### Memory Savings

Let's quantify. With TinyLlama (22 layers, 4 KV heads, head_dim=64):

```
KV per token per layer = 2 × 4 × 64 × 4 bytes = 2,048 bytes
KV per token (all layers) = 2,048 × 22 = 45,056 bytes ≈ 44 KB

With block_size=16:
  Block = 16 tokens × 44 KB = 704 KB per block

Contiguous (max_seq_len=2048):
  Per-sequence allocation = 2048 × 44 KB = 88 MB
  10 sequences = 880 MB
  Actual usage (avg 200 tokens) = 88 MB → 90% wasted

Paged (block_size=16):
  Avg 200 tokens → ceil(200/16) = 13 blocks = 9.1 MB
  10 sequences = 91 MB
  Max waste = 15 tokens × 44 KB = 0.66 MB per sequence
  → ~10x less memory than contiguous allocation
```

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `PagedKVCache` (Python blocks) | `BlockSpaceManager` + GPU tensor pool | `TokenToKVPool` + `ReqToTokenPool` |
| `allocate()` | `BlockSpaceManager.allocate()` + `append_slots()` | Pool allocator with `alloc()` |
| `update()` (scatter) | Custom CUDA `cache_ops` kernel | `store_kv_cache()` kernel |
| `get_kv_for_model()` (gather) | Not needed — paged attention kernel reads blocks directly | Not needed — FlashInfer reads from pool |
| `free()` | `BlockSpaceManager.free()` | Pool `free()` |
| `block_tables` (Python dict) | GPU tensor of block table indices | Mapped via `ReqToTokenPool` |
| `free_blocks` (Python list) | `BlockAllocator` free list | Prefix-aware pool allocator |

Key architectural differences:

**Our gather vs. vLLM's kernel:** Our biggest inefficiency is the `get_kv_for_model()` gather step — we copy all cached KV into a contiguous tensor before every decode step. In real vLLM, the [paged attention CUDA kernel](https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu) reads directly from the block table during attention computation, avoiding any data movement.

**SGLang's approach:** SGLang uses a different but equivalent abstraction. Instead of blocks, it uses two pools: `ReqToTokenPool` (maps requests to token indices) and `TokenToKVPool` (maps token indices to KV storage). The result is the same: non-contiguous KV storage with O(1) allocation/free.

**Block size choice:** vLLM defaults to block_size=16. Larger blocks mean less metadata overhead but more internal fragmentation. Smaller blocks mean finer-grained allocation but more block table entries. 16 is a sweet spot for GPU memory access patterns.

---

## The Implementation

The complete implementation is in [03_paged_attention.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/03_paged_attention.py) (~290 lines). Here are the key parts:

### The Paged KV Cache

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_kv_heads, head_dim):
        # Pre-allocate ALL blocks up front
        self.k_blocks = torch.zeros(
            num_blocks, num_layers, block_size, num_kv_heads, head_dim
        )
        self.v_blocks = torch.zeros(
            num_blocks, num_layers, block_size, num_kv_heads, head_dim
        )
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}   # seq_id -> [physical_block_idx, ...]
```

The pool is allocated once at startup. `k_blocks` and `v_blocks` are 5D tensors: `[block_idx, layer, slot, head, dim]`. This layout lets us write/read individual slots without touching the rest of the block.

### Scatter: Writing KV to Blocks

```python
def update(self, seq_id, layer_idx, new_keys, new_values, start_pos):
    block_table = self.block_tables[seq_id]
    for i in range(new_keys.shape[0]):
        pos = start_pos + i
        logical_block = pos // self.block_size
        offset = pos % self.block_size
        physical_block = block_table[logical_block]
        self.k_blocks[physical_block, layer_idx, offset] = new_keys[i]
        self.v_blocks[physical_block, layer_idx, offset] = new_values[i]
```

During prefill, `start_pos=0` and `new_keys.shape[0]` is the prompt length — we scatter the entire prompt's KV into blocks. During decode, `start_pos=current_length` and we scatter just 1 token.

### Gather: Assembling KV for the Model

```python
def get_kv_for_model(self, seq_id, max_pos=None):
    cache = DynamicCache()
    for layer_idx in range(self.num_layers):
        k = torch.zeros(seq_len, self.num_kv_heads, self.head_dim)
        for pos in range(seq_len):
            physical_block = block_table[pos // self.block_size]
            offset = pos % self.block_size
            k[pos] = self.k_blocks[physical_block, layer_idx, offset]
        # ... same for v ...
        cache.update(k.permute(1, 0, 2).unsqueeze(0), ...)
    return cache
```

This is the expensive part — O(seq_len × num_layers) copy operations. Real vLLM avoids this entirely with its custom CUDA kernel.

### The Engine: Prefill and Decode with Paged KV

The engine's `prefill()` and `decode_step()` methods follow the same pattern as Blog 1, but with an important difference: instead of passing `past_key_values` between calls, we scatter KV into paged blocks after each forward pass and gather them before the next one.

```python
# In prefill: scatter model output into paged blocks
for layer_idx in range(self.num_layers):
    k, v = outputs.past_key_values[layer_idx]
    k_sq = k.squeeze(0).permute(1, 0, 2)  # [seq_len, heads, dim]
    self.kv_cache.update(seq_id, layer_idx, k_sq, v_sq, start_pos=0)

# In decode_step: gather from paged blocks for the model
past_cache = self.kv_cache.get_kv_for_model(seq_id, max_pos=position)
outputs = self.model(input_ids=..., past_key_values=past_cache, ...)
```

---

## Running the Code

**Demo mode:**
```bash
python 03_paged_attention.py --demo
```

**Server mode:**
```bash
python 03_paged_attention.py --port 5000

# Generate text:
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'

# Check memory usage:
curl http://localhost:5000/health
```

**Custom block configuration:**
```bash
# More blocks (serve more concurrent requests):
python 03_paged_attention.py --num-blocks 512 --port 5000

# Smaller blocks (less waste per sequence):
python 03_paged_attention.py --block-size 8 --port 5000
```

Expected demo output:
```
Block pool: 256 blocks x 16 tokens/block
Free blocks before: 256

--- Request 1: Short prompt ---
Prompt: 'The capital of France is'
Output: Paris. ...
  Prefill: 111ms (6 tokens)
  Decode:  7229ms (30 tokens, 249ms/tok)
  Blocks used: 3
  Free blocks after: 256

--- Block Table Visualization ---
  Sequence with 42 tokens (block_size=16):
  Logical blocks needed: 3
  Block table: [252, 255, 254]
  Token 0  -> block 252, offset 0
  Token 15 -> block 252, offset 15
  Token 16 -> block 255, offset 0
  Token 41 -> block 254, offset 9
  After free: 256 blocks available
```

---

## Benchmarks

Paged attention doesn't make individual requests faster — the gather/scatter overhead actually makes each decode step slightly slower. The win is **memory efficiency**, which lets you serve more requests:

| Metric | Contiguous (Blog 1) | Paged (Blog 3) |
|--------|---------------------|-----------------|
| Decode latency | ~240ms/token | ~250ms/token (+4% overhead) |
| Memory per sequence (200 tokens) | 88 MB (pre-allocated for 2048) | 9.1 MB (13 blocks) |
| Max concurrent sequences (4 GB budget) | 45 | 440 |
| Memory waste | 50-90% | < 4% |
| Can grow dynamically? | No (pre-allocated) | Yes (allocate blocks on demand) |

The 4% overhead comes from the gather step on CPU. On GPU with the paged attention CUDA kernel, this overhead drops to near zero because the kernel reads blocks directly during attention computation.

The real impact: **~10x more concurrent sequences** in the same memory budget. This is why paged attention was the breakthrough that made vLLM practical for production serving.

---

## Key Takeaways

1. **Contiguous KV cache wastes memory** — you must pre-allocate for max_seq_len, wasting 50-90% on average
2. **Paged attention** splits the KV cache into fixed-size blocks, allocated on demand and freed when done
3. The **block table** maps logical token positions to physical block locations — just like an OS page table
4. **Scatter** writes new KV into blocks; **gather** assembles contiguous KV from blocks for the model
5. In real vLLM, the paged attention CUDA kernel avoids the gather step entirely by reading blocks directly during attention
6. The result: ~10x more concurrent sequences in the same memory, with < 4% internal fragmentation

---

## Further Reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the vLLM paper
- [vLLM block manager source](https://github.com/vllm-project/vllm/tree/main/vllm/core) — production block allocation
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — efficient paged attention kernels used by SGLang
- **Next: [Blog 4 — Continuous Batching](04_continuous_batching.md)** — serve multiple requests simultaneously by scheduling at the iteration level
