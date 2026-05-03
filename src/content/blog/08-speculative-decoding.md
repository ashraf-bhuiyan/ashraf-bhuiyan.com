---
title: "Part 8: Speculative Decoding"
description: "The draft-verify paradigm: guess K tokens cheaply, verify all at once, with rejection sampling to preserve exact output distribution."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 8
tags: ["speculative decoding", "draft model", "rejection sampling"]
---

## What Problem Does This Solve?

LLM decode is **memory-bandwidth bound**. Each token requires reading the entire model from GPU memory — billions of parameters — but only does one token's worth of matrix multiplies. The GPU's compute units sit mostly idle, waiting for data to arrive from memory.

```
Generating 4 tokens (standard decode):

Step 1: Read 7B params → multiply by 1 token → write 1 token    [GPU: 5% utilized]
Step 2: Read 7B params → multiply by 1 token → write 1 token    [GPU: 5% utilized]
Step 3: Read 7B params → multiply by 1 token → write 1 token    [GPU: 5% utilized]
Step 4: Read 7B params → multiply by 1 token → write 1 token    [GPU: 5% utilized]

Total: read model 4 times, generate 4 tokens
Bottleneck: memory bandwidth, not compute
```

Each forward pass reads the full model from GPU HBM. With a 7B model at FP16, that's 14GB of memory reads per token. On an A100 with 2TB/s bandwidth, that's ~7ms per token — and the math takes a fraction of that.

Speculative decoding exploits this waste: **what if we could verify 5 tokens in a single forward pass, for roughly the same cost as generating 1?**

```
Generating 4 tokens (speculative decode):

Draft:  Cheap method generates [t1', t2', t3', t4'] very quickly
Verify: Read 7B params → multiply by 5 tokens → verify all at once  [GPU: 25% utilized]
Result: accept t1', t2', reject t3' → use target's t3'' instead

Total: read model 1 time + cheap draft, generate 3 tokens
Speedup: ~3x fewer expensive forward passes
```

Since memory bandwidth is the bottleneck, processing 5 tokens costs nearly the same wall-clock time as processing 1. The extra compute (5x more matrix multiplies) fits within the time the GPU would otherwise spend idle, waiting for memory.

---

## The Core Idea: Draft, Then Verify

Speculative decoding has two phases:

### Phase 1: Draft — Generate K Tokens Cheaply

Use a **cheap method** to guess the next K tokens. This could be:
- A small model (1B parameters drafting for a 70B target)
- N-gram pattern matching against the prompt (zero cost, no model needed)
- The target model's own extra prediction heads (MTP, EAGLE)

The draft doesn't need to be perfect — it just needs to be *good enough* that some guesses match what the target model would generate.

### Phase 2: Verify — Check All K Tokens in One Pass

Feed all K draft tokens to the target model **at once**, as if they were a prefill. The target model produces logits for every position, and we compare each draft token against what the target would have generated.

```
Draft tokens:  [sunny] [and] [warm] [today]
                  ↓       ↓     ↓      ↓
Target logits: [sunny✓] [and✓] [cold✗] [---]
                                 ↑
                          First mismatch → use target's "cold"
                          Everything after is discarded

Result: "sunny and cold" — 3 tokens from 1 expensive forward pass
```

### The Key Insight: Sequential Draft, Parallel Verify

Drafting is sequential (each draft token depends on the previous one), but verification is parallel — all draft tokens are checked in one forward pass. Since verification dominates the cost and verification is parallel, we get multiple tokens per expensive step:

```
Standard:     [Fwd1] → t1, [Fwd2] → t2, [Fwd3] → t3    3 expensive passes
Speculative:  [Draft: t1',t2',t3'] [Verify all] → t1,t2,t3    1 expensive pass
```

### Speculative Decoding Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Speculative Decode Loop                     │
│                                                               │
│  Context: "The weather in Tokyo is"                          │
│                                                               │
│  ┌─────────────────────┐                                     │
│  │  DRAFT (cheap)      │  N-gram: match "is" in context      │
│  │  "sunny and warm    │  Small model: 1B params              │
│  │   today"            │  EAGLE: target's own hidden states   │
│  └────────┬────────────┘                                     │
│           │ K=4 draft tokens                                  │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  VERIFY (expensive) │  Target model: 70B params            │
│  │  Forward pass on    │  Process all K tokens at once         │
│  │  all draft tokens   │  (parallel, like prefill)            │
│  └────────┬────────────┘                                     │
│           │ K logit vectors                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  ACCEPT / REJECT    │  Compare draft vs target:            │
│  │  [sunny ✓]          │  Accept matching prefix              │
│  │  [and    ✓]         │  Reject at first mismatch            │
│  │  [warm   ✗ → cold]  │  Use target's token instead          │
│  │  [today  ─ discard] │  Truncate KV cache to accepted len   │
│  └────────┬────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  Context: "The weather in Tokyo is sunny and cold"            │
│  (3 new tokens from 1 target forward pass)                    │
│                                                               │
│  └───────── repeat until max_tokens or EOS ──────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## How It Works

### The Draft-Verify Loop

```
Token sequence so far: "The weather in Tokyo is"

Iteration 1:
  Draft (K=4): ["sunny", "and", "warm", "today"]
  Verify:      target model forward([sunny, and, warm, today])
               → logits at each position
  Compare:
    pos 0: target says "sunny"  → draft "sunny" ✓ ACCEPT
    pos 1: target says "and"    → draft "and"   ✓ ACCEPT
    pos 2: target says "cold"   → draft "warm"  ✗ REJECT
    pos 3: (depends on rejected pos 2, discard)
  Result: accept "sunny and", use target's "cold"
  Sequence: "The weather in Tokyo is sunny and cold"

Iteration 2:
  Draft (K=4): ["in", "the", "winter", "months"]
  Verify:      target model forward([in, the, winter, months])
  Compare:
    pos 0: target says "in"     → draft "in"    ✓ ACCEPT
    pos 1: target says "the"    → draft "the"   ✓ ACCEPT
    pos 2: target says "winter" → draft "winter" ✓ ACCEPT
    pos 3: target says "."     → draft "months" ✗ REJECT
  Result: accept "in the winter", use target's "."
  Sequence: "The weather in Tokyo is sunny and cold in the winter."

Total: 9 tokens generated in 2 target forward passes (+ 2 cheap drafts)
Standard: would need 9 target forward passes
Speedup: ~4.5x fewer target forward passes
```

### Greedy vs. Probabilistic Acceptance

**Greedy acceptance** (temperature=0): Accept if the target model's argmax matches the draft token. Simple, but only works with greedy sampling.

**Probabilistic acceptance** (temperature>0): Uses rejection sampling to guarantee the output distribution is *mathematically identical* to the target model alone. No quality loss.

```
For each draft token t with draft probability q(t) and target probability p(t):

  if p(t) >= q(t):
    ACCEPT  (target agrees or assigns even higher probability)
  
  else:
    Accept with probability p(t) / q(t)
    Reject with probability 1 - p(t) / q(t)
    
    If rejected: sample a "recovered" token from:
      recovered ~ normalize(max(0, p(t) - q(t)))
    This ensures the output distribution exactly matches p(t)

Stop at first rejection. Discard all subsequent draft tokens.
```

The math guarantees that whether you use the target model alone or speculative decoding with rejection sampling, the probability of generating any specific sequence is identical.

### Acceptance Rate and Speedup

The **acceptance rate** (α) is the fraction of draft tokens that match the target. It determines the speedup:

```
K = number of draft tokens
α = acceptance rate (0 to 1)

Expected accepted tokens per step = α × K
Total tokens per target forward pass = α × K + 1  (accepted + bonus/recovered)

Speedup in forward passes ≈ α × K + 1

Example with K=5:
  α=0%   → 1.0 tokens/pass → no benefit
  α=50%  → 3.5 tokens/pass → 3.5x fewer passes
  α=75%  → 4.75 tokens/pass → 4.75x fewer passes  
  α=100% → 6.0 tokens/pass → 6x fewer passes
```

Wall-clock speedup is lower than the forward pass reduction because drafting itself has some cost. But on GPU, the draft cost is tiny compared to the verification cost for large models.

### N-Gram Drafting

The simplest draft method requires no model at all — just pattern matching. Look at the last N tokens and search for the same N-gram earlier in the context. If found, use whatever followed that earlier occurrence as the draft:

```
Context: "The quick brown fox jumps over the lazy dog. The quick brown fox"
Last 3 tokens: "brown fox"

Search context for "brown fox":
  Found at position 3! After "brown fox" came "jumps over the lazy dog"

Draft: ["jumps", "over", "the", "lazy", "dog"]

If the model generates similar text, these drafts will be accepted!
```

N-gram drafting works best for repetitive text: code with repeated patterns, structured data, conversations that reference earlier context. For diverse, creative text, acceptance rates drop because the model generates novel tokens that don't match any prior n-gram.

### KV Cache Truncation on Rejection

When draft tokens are rejected, their KV entries must be removed from the cache. The verification forward pass computed KV for all draft tokens, but rejected positions have invalid context (they assumed previous draft tokens that were wrong).

```
Before verify:   KV cache has positions [0..N-1]
After verify:    KV cache has positions [0..N-1, N, N+1, N+2, N+3, N+4]
                                                  ↑    ↑     ↑     ↑
                                               next  d[0]  d[1]  d[2]
If d[1] rejected:
  Truncate KV to [0..N-1, N, N+1]  (keep next_token and d[0])
  d[2]'s KV is invalid (computed assuming d[1] was correct)
```

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `NGramDrafter` | `NGramProposer` (GPU-vectorized) | N-gram proposer |
| `SpeculativeDecoder.generate()` | `SpecDecWorker.execute_model()` | Speculation in `ModelRunner` |
| Greedy comparison | `RejectionSampler` (strict + probabilistic) | Rejection sampling |
| `_truncate_kv()` | KV cache rollback in `ModelRunner` | Cache revert |
| `num_speculative_tokens` | `SpeculativeConfig.num_speculative_tokens` | `--speculative-num-steps` |
| N/A | `DraftModelRunner` (separate small model) | Draft model support |
| N/A | `EagleProposer` (EAGLE/EAGLE-3) | EAGLE support |
| N/A | `MTPProposer` (DeepSeek's MTP heads) | MTP support |
| N/A | `DFlashProposer` (parallel bidirectional) | N/A |

### Speculation Methods in Production

Real systems support multiple drafting strategies:

**Draft Model:** A separate small model (e.g., 1B parameters) generates draft tokens. The draft model shares the target model's tokenizer but has its own weights. ~50% TPOT reduction at batch size 1.

**EAGLE/EAGLE-3:** A lightweight "head" model that uses the target model's hidden states to predict future tokens. Trained specifically for each target model. Good balance of speed and acceptance rate.

**MTP (Multi-Token Prediction):** Built into models like DeepSeek V3 — extra prediction heads that generate tokens N+2, N+3, etc. alongside the standard N+1 prediction. No extra model to load, highest acceptance rates.

**DFlash:** Generates all draft tokens in a single forward pass using bidirectional attention. Achieved 4.6x speedup on HumanEval coding tasks.

### Zero-Bubble Async Scheduling

A key challenge in production: speculative decoding requires knowing draft token IDs before scheduling the next step, but async scheduling (Blog 5) overlaps scheduling with GPU execution. vLLM solves this with **optimistic scheduling** — assume all draft tokens will be accepted, schedule accordingly, then correct when rejection results arrive.

---

## The Implementation

The complete implementation is in [08_speculative_decoding.py](../code-blog-simple-vllm/08_speculative_decoding.py) (~350 lines).

### N-Gram Drafter

```python
class NGramDrafter:
    def __init__(self, n: int = 3):
        self.n = n

    def draft(self, token_ids: list[int], num_draft: int) -> list[int]:
        suffix = tuple(token_ids[-self.n:])
        for i in range(len(token_ids) - self.n - 1):
            window = tuple(token_ids[i:i + self.n])
            if window == suffix:
                start = i + self.n
                end = min(start + num_draft, len(token_ids))
                return token_ids[start:end]
        return []  # no match → fall back to standard decode
```

### Verify and Accept

```python
# Feed next_token + all draft tokens to the target model
verify_input = torch.tensor([[next_token] + draft_tokens])
out = model(input_ids=verify_input, past_key_values=past, use_cache=True)
verify_logits = out.logits[0]  # [K+1, vocab_size]

# Compare each draft token against target's prediction
num_accepted = 0
for i in range(K):
    target_token = argmax(verify_logits[i])  # what target would generate
    if target_token == draft_tokens[i]:
        num_accepted += 1
        gen_tokens.append(draft_tokens[i])
    else:
        gen_tokens.append(target_token)  # use target's token
        break  # stop at first mismatch

if num_accepted == K:  # all accepted → bonus token
    bonus = argmax(verify_logits[K])
    gen_tokens.append(bonus)
```

### KV Cache Truncation

```python
def _truncate_kv(self, past_key_values, keep_len):
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_key_values):
        cache.update(
            k[:, :, :keep_len, :],
            v[:, :, :keep_len, :],
            layer_idx,
        )
    return cache

# After rejection at position i:
if num_accepted < K:
    keep_len = kv_len - (K - num_accepted)
    past = truncate_kv(out.past_key_values, keep_len)
```

---

## Running the Code

**Demo mode (standard vs speculative comparison):**
```bash
python 08_speculative_decoding.py --demo
```

**Server mode:**
```bash
python 08_speculative_decoding.py --port 5000

# Speculative generation:
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A B C D A B C D A B C", "max_tokens": 30}'

# Standard generation (for comparison):
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 30, "speculative": false}'

# Check acceptance stats:
curl http://localhost:5000/health
```

Expected demo output:
```
Prompt 1 (repetitive colors):
  Standard:    30 passes, 7489ms, 1.0 tok/pass
  Speculative: 6 passes, 700ms, 5.2 tok/pass
  Accepted: 5.0 avg, rate=100%
  Forward pass reduction: 5.0x fewer passes

Prompt 2 (counting):
  Standard:    30 passes, 7754ms, 1.0 tok/pass
  Speculative: 17 passes, 3260ms, 1.8 tok/pass
  Accepted: 3.2 avg, rate=65%
  Forward pass reduction: 1.8x fewer passes

Prompt 3 (repeated sentence):
  Standard:    30 passes, 6535ms, 1.0 tok/pass
  Speculative: 6 passes, 685ms, 5.2 tok/pass
  Accepted: 5.0 avg, rate=100%
  Forward pass reduction: 5.0x fewer passes
```

Repetitive text achieves 100% acceptance (n-gram matches perfectly). Novel text has lower acceptance but still benefits.

---

## Benchmarks

| Metric | Standard Decode | Speculative (α=100%) | Speculative (α=50%) |
|--------|----------------|---------------------|---------------------|
| Tokens per forward pass | 1.0 | 6.0 | 3.5 |
| Forward passes (30 tokens) | 30 | 6 | ~10 |
| Wall-clock time | Baseline | ~10x faster | ~2x faster |
| Memory overhead | None | ~0 (n-gram) | ~0 (n-gram) |

With a draft model instead of n-gram:

| Metric | Standard Decode | Draft Model (α=75%) |
|--------|----------------|---------------------|
| Tokens per forward pass | 1.0 | 4.75 |
| GPU memory | Target only | Target + Draft model |
| TPOT reduction | Baseline | ~50% at BS=1 |

Production results (from vLLM with DFlash on B200):

| Dataset | Speedup | Tokens/sec |
|---------|---------|-----------|
| HumanEval (code) | 4.63x | 1,018 |
| GSM8k (math) | 3.51x | 743 |
| MT-Bench (chat) | 2.82x | 484 |

### When Does Speculative Decoding Help?

**Helps most:**
- Low batch sizes (BS=1-4) — GPU is heavily underutilized
- Repetitive or predictable text — high acceptance rates
- Large target models — more memory bandwidth waste to exploit

**Helps least:**
- High batch sizes — GPU already utilized by batching many requests
- Creative, diverse text — low acceptance rates
- Small target models — less memory bandwidth waste

---

## Key Takeaways

1. **Decode is memory-bandwidth bound** — the GPU reads the entire model per token but barely uses its compute. Verifying K tokens costs roughly the same as generating 1
2. **Draft tokens are verified in parallel** — one target forward pass checks all K drafts at once
3. **Accept the longest matching prefix** — on first mismatch, use the target's token and discard the rest
4. **Rejection sampling guarantees identical output** — probabilistic acceptance preserves the exact target distribution
5. **N-gram drafting is free** — no model needed, just pattern matching against the context
6. Production systems support multiple draft methods: separate draft models, EAGLE, MTP, DFlash (parallel drafting), and n-gram

---

## Further Reading

- [Speculative Decoding paper](https://arxiv.org/abs/2211.17192) — the original "Fast Inference from Transformers via Speculative Decoding"
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — using target model hidden states for drafting
- [Medusa: Simple Framework for Accelerating LLM Generation](https://arxiv.org/abs/2401.10774) — tree-structured speculation with multiple heads
- [DFlash](https://arxiv.org/abs/2502.12048) — parallel bidirectional attention drafting
- **Next: [Blog 9 — Tensor Parallelism](09_tensor_parallelism.md)** — split a model across multiple GPUs
