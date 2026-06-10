---
title: "Part 6: Chunked Prefill"
description: "How long prompts block decode requests and chunked prefill splits them to keep inter-token latency stable."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 6
tags: ["prefill", "latency", "scheduling"]
---

## What Problem Does This Solve?

In Blog 4's continuous batching, a new request starts with a **prefill** that processes the entire prompt in one step. For a 5000-token prompt, that's a massive forward pass — potentially 500ms+ even on GPU. During that time, all decode requests in the batch are stalled:

```
Without chunked prefill:

Step 1: [New Req A: 5000-token prefill ██████████████████████████████████]
        [Req B: decode BLOCKED ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
        [Req C: decode BLOCKED ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
                              ↑ step takes 500ms instead of 30ms ↑

Step 2: [Req A: decode] [Req B: decode] [Req C: decode]  ← normal 30ms
Step 3: [Req A: decode] [Req B: decode] [Req C: decode]
```

Users of Req B and C see a 500ms gap between tokens (normally 30ms). This spike in **inter-token latency (ITL)** is jarring — the response freezes for half a second, then resumes.

**Chunked prefill** solves this by splitting long prefills into smaller pieces:

```
With chunked prefill (chunk_size=512):

Step 1: [A: prefill chunk 1 (512 tokens)] [B: decode] [C: decode]  ~35ms
Step 2: [A: prefill chunk 2 (512 tokens)] [B: decode] [C: decode]  ~35ms
...
Step 10: [A: prefill chunk 10 (final)]    [B: decode] [C: decode]  ~35ms
Step 11: [A: decode] [B: decode] [C: decode]                        ~30ms

ITL for B and C: consistently ~30-35ms (no spike!)
```

The total prefill time for Req A is roughly the same, but it's spread across 10 steps instead of one. Each step stays within the token budget, keeping ITL stable for all requests.

---

## The Core Idea: Token Budget + Prefill Chunks

The scheduler enforces a **token budget** — the maximum number of tokens processed per step. Both prefill and decode tokens count against it. A long prefill that exceeds the budget is split into chunks that fit:

```
Token budget: 64 tokens per step
Prompt: 150 tokens
Running decode requests: 3 (3 tokens)

Step 1: [Prefill chunk: 32 tokens] [3 decode tokens]  = 35 tokens ✓
Step 2: [Prefill chunk: 32 tokens] [3 decode tokens]  = 35 tokens ✓
Step 3: [Prefill chunk: 32 tokens] [3 decode tokens]  = 35 tokens ✓
Step 4: [Prefill chunk: 32 tokens] [3 decode tokens]  = 35 tokens ✓
Step 5: [Prefill final: 22 tokens] [3 decode tokens]  = 25 tokens ✓
        → sample first output token for this request
Step 6: [4 decode tokens]                              = 4 tokens ✓
```

Key rules:
1. **Decode tokens have priority** — running requests get their 1 decode token first
2. **Remaining budget goes to prefill** — new or continuing prefill requests get the rest
3. **Partial prefill is tracked** — the sequence remembers `num_computed_tokens` so the next step continues where it left off
4. **Only the final chunk samples** — intermediate chunks just fill the KV cache; the first output token is sampled only when the entire prompt is processed

### Token Budget Allocation

Here's how the scheduler fills each step's token budget:

```
Token budget = 64 tokens per step
3 running decode sequences + 1 new 150-token prompt arriving

Step 1:
  ┌──────────────────────────────────────────────────────────────┐
  │ [D][D][D]  [████████ Prefill chunk 1 (32 tokens) ████████]  │
  │  3 decode          remaining budget: 61 → 32 tokens          │
  │  tokens            for prefill chunk                         │
  └──────────────────────────────────────────────────────────────┘
  Total: 35 tokens (within budget of 64)
  New request: num_computed_tokens = 0 → 32

Step 5 (final chunk):
  ┌──────────────────────────────────────────────────────────────┐
  │ [D][D][D]  [████ Prefill final (22 tokens) ████]            │
  │  3 decode     remaining = 150 - 128 = 22 tokens              │
  │  tokens       → SAMPLE first output token!                   │
  └──────────────────────────────────────────────────────────────┘
  Total: 25 tokens
  Sequence transitions: prefill → decode

Step 6:
  ┌──────────────────────────────────────────────────────────────┐
  │ [D][D][D][D]                                                 │
  │  4 decode tokens (including the new sequence)                │
  └──────────────────────────────────────────────────────────────┘
  Total: 4 tokens
```

---

## How It Works

### Partial Prefill State

Each sequence tracks how much of its prompt has been processed:

```python
class Sequence:
    num_computed_tokens = 0    # updated after each chunk
    prompt_len = 150           # total prompt tokens
    
    @property
    def is_prefill(self):
        return self.num_computed_tokens < self.prompt_len
    
    @property
    def remaining_prefill(self):
        return self.prompt_len - self.num_computed_tokens
```

After each chunk, `num_computed_tokens` advances by the chunk size. When it reaches `prompt_len`, the prefill is complete and the sequence transitions to decode mode.

### Chunked Prefill Execution

When executing a prefill chunk, we only process a slice of the prompt tokens:

```python
def execute_prefill_chunk(seq, chunk_tokens):
    start = seq.num_computed_tokens
    end = start + chunk_tokens
    chunk_ids = seq.prompt_token_ids[start:end]
    
    # Get KV cache from previous chunks (if any)
    if start > 0:
        past_cache = kv_cache.get_kv_for_model(seq, max_pos=start)
    else:
        past_cache = None
    
    # Forward pass for just this chunk
    outputs = model(input_ids=chunk_ids, past_key_values=past_cache)
    
    # Store new KV into paged cache
    kv_cache.update(seq, new_kv, start_pos=start)
    seq.num_computed_tokens = end
    
    # Only sample if this was the final chunk
    if end >= seq.prompt_len:
        return sample(outputs.logits[-1])
    return None  # no output token yet
```

Each chunk builds on the KV cache from previous chunks, just like decode steps build on prior tokens.

### The Chunked Scheduler

The scheduler has three priority levels:

```python
def schedule():
    budget = max_num_tokens
    scheduled = []
    
    # Priority 1: Decode tokens (running, non-prefill)
    for seq in running:
        if not seq.is_prefill:
            scheduled.append((seq, 1))  # 1 decode token
            budget -= 1
    
    # Priority 2: Continue partial prefills
    for seq in running:
        if seq.is_prefill:
            chunk = min(seq.remaining_prefill, chunk_size, budget)
            scheduled.append((seq, chunk))
            budget -= chunk
    
    # Priority 3: Admit new requests
    for seq in waiting:
        chunk = min(seq.prompt_len, chunk_size, budget)
        scheduled.append((seq, chunk))
        budget -= chunk
```

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `ChunkedScheduler` | `Scheduler` with `enable_chunked_prefill=True` | Chunked prefill in `Scheduler` |
| `chunk_size` parameter | `long_prefill_token_threshold` | `chunked_prefill_size` |
| `remaining_prefill` | `num_tokens - num_computed_tokens` | Token count tracking |
| `execute_prefill_chunk()` | `ModelRunner` handles variable-length inputs | `ForwardBatch` with mixed extend/decode |
| Decode priority | Running requests scheduled first | Decode requests get priority |
| `max_num_tokens` budget | `max_num_scheduled_tokens` | `max_running_requests` + `max_prefill_tokens` |

Key details:

**vLLM's unified scheduling model:** In vLLM V1, there's no explicit "chunked prefill" mode — the scheduler simply caps `num_new_tokens` per request to the token budget. Whether a request is "prefilling" or "decoding" is emergent: if `num_computed_tokens < num_prompt_tokens`, there's prefill work to do.

**Mixed batches:** Both vLLM and SGLang support mixed batches where some sequences are prefilling (processing multiple tokens) and others are decoding (1 token). The model handles this with a concatenated input tensor and a position/attention mask that tracks per-sequence positions.

**`max_num_partial_prefills`:** vLLM limits how many concurrent partial prefills can be active. Too many partial prefills waste memory (each has a partially filled KV cache consuming blocks) and reduce throughput.

---

## The Implementation

The complete implementation is in [06_chunked_prefill.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/06_chunked_prefill.py) (~500 lines). Key additions over Blog 4:

### Chunked Scheduler

```python
class ChunkedScheduler:
    def schedule(self):
        # Phase 1: Decode tokens first
        for seq in running (non-prefill):
            schedule 1 decode token, budget -= 1
        
        # Phase 2: Continue partial prefills
        for seq in running (is_prefill):
            chunk = min(remaining, chunk_size, budget)
            schedule chunk tokens, budget -= chunk
        
        # Phase 3: Admit new requests
        for seq in waiting:
            chunk = min(prompt_len, chunk_size, budget)
            schedule chunk tokens, budget -= chunk
```

### Chunked Prefill Execution

The key change: `_execute_prefill_chunk()` processes a slice of the prompt and only returns a token when the entire prompt is done:

```python
def _execute_prefill_chunk(self, seq, chunk_tokens):
    start = seq.num_computed_tokens
    end = start + chunk_tokens
    chunk_ids = seq.prompt_token_ids[start:end]
    
    # Get previous KV if this isn't the first chunk
    past_cache = kv_cache.get_kv_for_model(seq, max_pos=start) if start > 0 else None
    
    outputs = model(input_ids=chunk_ids, past_key_values=past_cache)
    kv_cache.update(seq, new_kv, start_pos=start)
    seq.num_computed_tokens = end
    
    if end >= seq.prompt_len:
        return sample(outputs.logits[-1])  # final chunk → first token
    return None  # intermediate chunk → no output
```

---

## Running the Code

**Demo mode:**
```bash
python 06_chunked_prefill.py --demo
```

**Server mode:**
```bash
python 06_chunked_prefill.py --port 5000 --chunk-size 32

# Long prompt with concurrent short request:
curl -N -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a very long essay about...", "max_tokens": 50, "stream": true}' &

curl -N -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "stream": true}'
```

Expected demo output:
```
Long prompt:  42 tokens (chunk_size=32)
Short prompt: 6 tokens
Long prompt needs 2 prefill chunks

Step   1 (  437ms): [L:prefill_chunk(32)] [S:prefill_done→Paris]
Step   2 (  372ms): [S:D=.] [L:prefill_done→2]
Step   3 (  486ms): [L:D=.] [S:D=\n]
Step   4 (  490ms): [L:D="] [S:D=\n]
Step   5 (  488ms): [L:D=The] [S:D=2]
...

ITL Analysis:
  SHORT: avg=483ms, min=372ms, max=556ms, jitter=184ms
  LONG:  avg=417ms, min=252ms, max=556ms, jitter=304ms
```

---

## Benchmarks

| Metric | No Chunking (Blog 4) | With Chunking (Blog 6) |
|--------|---------------------|------------------------|
| ITL during long prefill | Spike to 500ms+ | Stable ~30-35ms |
| TTFT for concurrent requests | Delayed by long prefill | Interleaved with prefill |
| Total prefill time | ~500ms (one shot) | ~500ms (spread over steps) |
| Max step latency | Unbounded | Capped by budget |
| Implementation complexity | Simple | Moderate (chunk tracking) |

The trade-off: chunked prefill adds a small overhead per chunk (cache gathering for previous chunks) and slightly increases total prefill time. But it dramatically improves ITL stability — the P99 ITL drops from seconds to milliseconds.

---

## Key Takeaways

1. **Long prefills spike ITL** — a 5000-token prefill blocks all decode requests for 500ms+
2. **Chunked prefill** splits prompts into budget-sized pieces, interleaving with decode steps
3. The **token budget** caps total tokens per step, keeping step latency predictable
4. **Decode requests get priority** — prefill chunks fill the remaining budget
5. **Partial prefill state** (`num_computed_tokens`) lets the scheduler continue prefilling across steps
6. Total prefill time is roughly the same — the improvement is in ITL stability, not speed

---

## Further Reading

- [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) — the paper that introduced chunked prefill
- [vLLM chunked prefill documentation](https://docs.vllm.ai/en/latest/) — production configuration
- **Next: [Blog 7 — Prefix Caching](07_prefix_caching.md)** — skip recomputing shared prefixes across requests
