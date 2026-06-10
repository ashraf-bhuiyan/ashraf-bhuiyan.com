---
title: "Part 7: Prefix Caching"
description: "Content-addressable KV cache using chained hashing, LRU eviction, and reference counting to skip recomputing shared prefixes."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 7
tags: ["prefix caching", "KV cache", "hashing", "RadixAttention"]
---

## What Problem Does This Solve?

In production LLM serving, many requests share the same prefix. Every ChatGPT conversation starts with a system prompt like "You are a helpful assistant." Every RAG pipeline prepends the same retrieved documents. Every multi-turn conversation repeats the entire history.

Without prefix caching, each request re-prefills these shared tokens from scratch:

```
Request 1: [System prompt: 500 tokens] [User: "What is 2+2?"]  → prefill 510 tokens
Request 2: [System prompt: 500 tokens] [User: "Name 3 colors"]  → prefill 510 tokens
Request 3: [System prompt: 500 tokens] [User: "Explain gravity"] → prefill 512 tokens

Total: 1532 tokens of prefill
Wasted: 1000 tokens (same system prompt computed 3 times)
```

With prefix caching, the system prompt's KV blocks are computed once and reused:

```
Request 1: [System prompt: 500 tokens ← MISS] [User: 10 tokens]  → prefill 510 tokens
Request 2: [System prompt: 500 tokens ← HIT]  [User: 10 tokens]  → prefill 10 tokens!
Request 3: [System prompt: 500 tokens ← HIT]  [User: 12 tokens]  → prefill 12 tokens!

Total: 532 tokens of prefill
Saved: 1000 tokens (66% reduction)
```

For long system prompts or RAG documents, this can reduce prefill time by 90%+. In multi-turn conversations where the history grows with each turn, prefix caching avoids re-prefilling the entire conversation every time.

---

## The Core Idea: Hash Blocks, Cache Blocks, Reuse Blocks

Prefix caching works by treating KV cache blocks like a content-addressable cache. Each block of tokens gets a hash. When a new request arrives, we hash its prompt blocks and check if matching KV blocks already exist in the cache.

```
Block size: 16 tokens
System prompt: "You are a helpful AI assistant..." (32 tokens = 2 blocks)

Request 1 (cold cache):
  Block 0: hash("You are a helpful") = 0xABCD  → MISS → allocate, compute KV
  Block 1: hash("AI assistant...")    = 0x1234  → MISS → allocate, compute KV
  
  After completion: blocks 0xABCD and 0x1234 stay in cache (ref count = 0)

Request 2 (warm cache):
  Block 0: hash("You are a helpful") = 0xABCD  → HIT → reuse block!
  Block 1: hash("AI assistant...")    = 0x1234  → HIT → reuse block!
  Block 2: hash("What is 2+2?")      = 0x5678  → MISS → allocate, compute KV
  
  Skip prefill for 32 cached tokens, only compute 10 new tokens!
```

### Chained Hashing

The hash for each block depends on the hash of the **previous** block — a hash chain. This ensures that the same tokens at different positions produce different hashes:

```
hash(block_0) = SHA256(None, token_ids[0:16])
hash(block_1) = SHA256(hash(block_0), token_ids[16:32])
hash(block_2) = SHA256(hash(block_1), token_ids[32:48])
```

If token_ids[0:16] changes, all subsequent hashes change too. This prevents false cache hits where the same tokens appear at different positions in different prompts.

### LRU Eviction

When all blocks are in use and a new request needs blocks, we evict the **least recently used** cached block — the one that hasn't been accessed in the longest time. Blocks that are actively in use by a running sequence can't be evicted (they have ref count > 0).

```
Cache state:
  Block A (ref=0, last used 10s ago)  ← evict this first
  Block B (ref=0, last used 5s ago)
  Block C (ref=2, in use)             ← can't evict
  Block D (ref=0, last used 1s ago)
```

---

## How It Works

### Request Lifecycle with Prefix Caching

```
New request: "System prompt... What is 2+2?"
                                                           
1. Tokenize: [101, 202, 303, ..., 404, 505, 606]          
                                                           
2. Compute block hashes:                                   
   Block 0: hash(None, tokens[0:16])  = 0xABCD            
   Block 1: hash(0xABCD, tokens[16:32]) = 0x1234          
   Block 2: hash(0x1234, tokens[32:38]) = partial, no hash
                                                           
3. Cache lookup:                                           
   0xABCD → HIT  (reuse physical block 7)                 
   0x1234 → HIT  (reuse physical block 3)                 
   → 32 cached tokens, skip their computation!             
                                                           
4. Allocate new blocks for remaining tokens:               
   tokens[32:38] → allocate physical block 12              
                                                           
5. Prefill only tokens[32:38]:                             
   model(input_ids=[404, 505, 606, ...],                   
         past_key_values=cached_kv_for_tokens[0:32])       
   → Only 6 tokens of compute instead of 38!              
                                                           
6. After completion:                                       
   Free sequence → decrement ref counts                    
   Blocks stay in cache for future reuse                   
```

### Cache Hit vs Miss: Block-Level View

```
Request: "You are a helpful AI assistant. What is 2+2?"
         ├── block 0 (16 tokens) ──┤├── block 1 (16 tokens) ──┤├─ block 2 (6 tok) ─┤

Block 0: hash = 0xABCD                Block 1: hash = 0x1234      Block 2: partial
         │                                      │                            │
         ▼                                      ▼                            ▼
  ┌──────────────┐                      ┌──────────────┐             ┌──────────────┐
  │ Cache lookup  │                      │ Cache lookup  │             │ Not hashable  │
  │ 0xABCD → ?   │                      │ 0x1234 → ?   │             │ (incomplete)  │
  └──────┬───────┘                      └──────┬───────┘             └──────┬───────┘
         │                                      │                            │
    ┌────▼────┐                           ┌────▼────┐                  ┌────▼────┐
    │   HIT   │                           │   HIT   │                  │  MISS   │
    │ phys: 7 │                           │ phys: 3 │                  │ alloc 12│
    └─────────┘                           └─────────┘                  └─────────┘
         │                                      │                            │
    Reuse block 7                          Reuse block 3                Compute KV
    (skip prefill)                         (skip prefill)              (run forward)
         │                                      │                            │
         └──────────────────┬───────────────────┘                            │
                            ▼                                                │
                   past_key_values                                           │
                   (from cached blocks)                                      │
                            │                                                │
                            └─────────── model(tokens[32:38], past_kv) ──────┘
                                         Only 6 tokens computed!
```

### Why Blocks Must Be Full

We only cache **complete blocks** (all `block_size` tokens filled). Partial blocks (like the last block of a prompt) aren't cached because they could match false positives — two prompts that diverge in the middle of a block would share the same partial hash but have different KV values.

### Reference Counting

When a sequence uses a cached block, its reference count increments. When the sequence finishes, the ref count decrements. Blocks with ref count 0 are eligible for eviction but stay in the cache for future reuse. This means:

- **ref > 0**: Block is in use, can't be evicted
- **ref = 0**: Block is cached, can be reused or evicted
- **Evicted**: Block is freed, its hash is removed from the cache

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `PrefixCachingAllocator` | `KVCacheManager` + `BlockPool` | `RadixCache` |
| `_hash_block()` (SHA256) | `hash_block_tokens()` (SHA256 or xxhash) | Trie-based prefix matching |
| `hash_to_block` (OrderedDict) | `BlockHashToBlockMap` | Radix tree nodes |
| LRU eviction | `FreeKVCacheBlockQueue` (doubly-linked list) | LRU eviction on tree nodes |
| `allocate_with_prefix()` | `find_longest_cache_hit()` + `allocate_slots()` | `match_prefix()` |
| `register_computed_blocks()` | Implicit on block fill | Automatic on insert |
| Reference counting | `ref_cnt` per `KVCacheBlock` | Tree node reference counting |

Key differences:

**SGLang's RadixAttention:** Instead of hashing individual blocks, SGLang uses a **radix tree** (prefix tree) where each node stores a block of tokens. Prefix lookup is a tree traversal — follow the path that matches the prompt's tokens. This naturally handles shared prefixes of any length without hash collisions.

```
RadixAttention tree:
         [root]
           |
    [System prompt block 0]
           |
    [System prompt block 1]
         /     \
  [User A Q]  [User B Q]  ← different suffixes, shared prefix
```

**vLLM's hash algorithm:** vLLM supports both SHA256 (default, no collisions) and xxhash (faster, extremely low collision probability). The hash chain ensures position-dependent uniqueness.

**GPU prefix caching:** vLLM can also cache KV blocks that have been swapped to CPU, enabling a two-tier cache (GPU hot cache + CPU cold cache). Blocks evicted from GPU can be restored from CPU without recomputation.

---

## The Implementation

The complete implementation is in [07_prefix_caching.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/07_prefix_caching.py) (~400 lines).

### Hash Chain Computation

```python
def _hash_block(self, parent_hash, token_ids):
    data = f"{parent_hash}:{token_ids}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def compute_block_hashes(self, token_ids):
    hashes = []
    parent_hash = None
    for i in range(0, len(token_ids), self.block_size):
        block_tokens = tuple(token_ids[i:i + self.block_size])
        if len(block_tokens) < self.block_size:
            break  # don't hash partial blocks
        h = self._hash_block(parent_hash, block_tokens)
        hashes.append(h)
        parent_hash = h
    return hashes
```

### Prefix-Aware Allocation

```python
def allocate_with_prefix(self, seq_id, token_ids):
    block_hashes = self.compute_block_hashes(token_ids)
    
    cached_blocks = 0
    for h in block_hashes:
        if h in self.hash_to_block:
            phys = self.hash_to_block[h]
            self.block_tables[seq_id].append(phys)
            self.block_ref_count[phys] += 1
            cached_blocks += 1
        else:
            break  # first miss → stop (contiguous prefix only)
    
    # Allocate new blocks for the rest
    remaining = len(token_ids) - cached_blocks * self.block_size
    new_blocks = math.ceil(remaining / self.block_size)
    for _ in range(new_blocks):
        self.block_tables[seq_id].append(self.free_blocks.pop())
    
    return cached_blocks * self.block_size, remaining
```

### Partial Prefill with Cached KV

```python
# In generate():
cached_tokens, new_tokens = cache.allocate_with_prefix(seq_id, token_ids)

if cached_tokens > 0:
    # Get cached KV for prefix
    past_cache = cache.get_kv_for_model(seq_id, max_pos=cached_tokens)
    # Only forward the new (uncached) tokens
    outputs = model(input_ids=token_ids[cached_tokens:],
                    past_key_values=past_cache)
```

---

## Running the Code

**Demo mode (5 requests with shared system prompt):**
```bash
python 07_prefix_caching.py --demo
```

**Server mode:**
```bash
python 07_prefix_caching.py --port 5000

# First request (cold cache):
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are helpful. What is 2+2?", "max_tokens": 20}'

# Second request (warm cache — shared prefix!):
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are helpful. Name 3 colors.", "max_tokens": 20}'

# Check cache stats:
curl http://localhost:5000/health
```

Expected demo output:
```
System prompt: 23 tokens (2 blocks)

Request 1 [MISS]: What is the capital of France?
  Cached: 0 tokens, New: 29 tokens, Prefill: 282ms

Request 2 [HIT]: Explain gravity in one sentence.
  Cached: 16 tokens, New: 13 tokens, Prefill: 185ms  ← 34% faster!

Request 5 [HIT]: What is Python?
  Cached: 16 tokens, New: 10 tokens, Prefill: 115ms  ← 59% faster!

Overall hit rate: 40%
```

---

## Benchmarks

| Metric | No Caching | With Prefix Caching |
|--------|-----------|-------------------|
| Prefill (cold, first request) | 282ms | 282ms (miss) |
| Prefill (warm, cached prefix) | 282ms | 115-185ms (hit) |
| Prefill speedup | 1x | 1.5-2.5x (depends on prefix length) |
| Prefill with 500-token system prompt | ~500ms | ~50ms (90% hit) |
| Memory overhead | None | Hash table (~1KB per cached block) |
| Cache hit rate (shared system prompt) | N/A | 40-80% |

The benefit scales with prefix length:
- 16-token prefix: ~35% prefill reduction
- 500-token system prompt: ~90% prefill reduction
- 4000-token RAG document: ~95% prefill reduction

---

## Key Takeaways

1. **Prefix caching** reuses KV blocks for shared prompt prefixes across requests
2. **Chained hashing** ensures position-dependent block identity — same tokens at different positions get different hashes
3. **Only full blocks are cached** — partial blocks at the end of a prompt aren't eligible
4. **LRU eviction** frees least-recently-used blocks when memory is tight
5. **Reference counting** prevents eviction of blocks in active use
6. SGLang uses a **radix tree** instead of hashing — same concept, different data structure
7. The first request is always a miss; subsequent requests with shared prefixes hit the cache

---

## Further Reading

- [vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) — production documentation
- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104) — radix tree approach to prefix caching
- **Next: [Blog 8 — Speculative Decoding](08_speculative_decoding.md)** — use a small draft model to generate multiple tokens per step
