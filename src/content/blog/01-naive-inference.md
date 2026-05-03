---
title: "Part 1: The Simplest LLM Server"
description: "Build a working LLM inference server in ~130 lines of Python. Covers autoregressive generation, the KV cache, prefill vs. decode phases, and basic sampling."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 1
tags: ["LLM", "inference", "KV cache", "sampling"]
---

## What Problem Does This Solve?

When you type a message into ChatGPT, Claude, or any LLM-powered app, something has to load the model into memory, feed it your prompt, generate tokens one by one, and send them back over the network. That "something" is an **inference server**.

Production inference servers like [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) are 300K+ lines of code with dozens of optimizations. But at their core, they do exactly what we'll build in this blog: load a model, run a generation loop, and serve text over HTTP.

This blog builds the simplest possible version of that. By the end, you'll have a working LLM server in ~130 lines of Python that you can query with `curl`. More importantly, you'll understand the two phases of inference — **prefill** and **decode** — and why the KV cache is the single most important optimization in LLM serving.

---

## The Core Idea: Autoregressive Generation

LLMs generate text **one token at a time**. Given the prompt "The capital of France is", the model predicts the next token ("Paris"), then given "The capital of France is Paris", it predicts the next token (","), and so on until it decides to stop.

This one-at-a-time process is called **autoregressive generation**, and it's why LLM inference is fundamentally different from, say, image classification (which runs the model once and returns a label).

```
Input:  "The capital of France is"
Step 1: Model("The capital of France is")        → " Paris"
Step 2: Model("The capital of France is Paris")   → ","
Step 3: Model("The capital of France is Paris,")  → " a"
Step 4: Model("The capital of France is Paris, a") → " city"
...until the model generates a stop token or hits max_tokens
```

But there's a huge problem with this naive approach: **we're reprocessing every token on every step.** Step 4 re-reads "The capital of France is Paris, a" even though the model already processed those tokens in Steps 1-3. For a 1000-token prompt generating 500 tokens, that's 500 × 1000 = 500,000 redundant token computations.

The solution is the **KV cache**.

---

## How It Works: Prefill, KV Cache, and Decode

### What Is the KV Cache?

Inside every transformer layer, the attention mechanism computes three matrices from the input:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I carry?"

Attention scores are computed as Q·K^T — the query of the current token is compared against the keys of ALL previous tokens. The result is used to weight the values, producing the attention output.

The critical insight: **the K and V for a given token depend only on that token's position, not on future tokens.** Once computed, they never change. So we can save them and reuse them on every subsequent step.

```
KV Cache Structure (per layer, per sequence):

  Layer 0:  K: [num_heads, seq_len, head_dim]    V: [num_heads, seq_len, head_dim]
  Layer 1:  K: [num_heads, seq_len, head_dim]    V: [num_heads, seq_len, head_dim]
  ...
  Layer 21: K: [num_heads, seq_len, head_dim]    V: [num_heads, seq_len, head_dim]

  TinyLlama: 22 layers × 4 heads × 64 head_dim × 2 (K+V)
  At 50 tokens: 22 × 4 × 50 × 64 × 2 × 4 bytes ≈ 22 MB

How attention uses the cache (decode step for token N):
  ┌─────────────────────────────────────────────────────────────┐
  │  New token → compute Q_N, K_N, V_N                         │
  │                                                             │
  │  K_N appended to cache: [K₁ K₂ K₃ ... K_{N-1} K_N]        │
  │  V_N appended to cache: [V₁ V₂ V₃ ... V_{N-1} V_N]        │
  │                                                             │
  │  Attention:                                                 │
  │    scores = Q_N · [K₁ K₂ ... K_N]ᵀ    ← read all K from cache
  │    weights = softmax(scores / √d)                           │
  │    output = weights · [V₁ V₂ ... V_N]  ← read all V from cache
  └─────────────────────────────────────────────────────────────┘
```

Without the KV cache, every step recomputes K and V for all previous tokens. With it, each step only computes K and V for the **one new token**, then reads the cached values for attention. This turns a quadratic operation into a linear one.

### The Two Phases of Inference

Every LLM request goes through two distinct phases:

```
┌─────────────────────────────────────────────────────────────┐
│ REQUEST: "What is 2+2?"                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PREFILL (compute-bound)           DECODE (memory-bound)    │
│  ┌─────────────────────┐          ┌──────────────────────┐  │
│  │ Process ALL prompt   │          │ Generate ONE token    │  │
│  │ tokens at once       │          │ at a time             │  │
│  │                     │          │                      │  │
│  │ Input: 8 tokens     │          │ Input: 1 token       │  │
│  │ Output: first token │    ×N    │ Output: next token   │  │
│  │ + KV cache filled   │  ──────→ │ + KV cache grows     │  │
│  │                     │          │                      │  │
│  │ ~100ms              │          │ ~240ms/token          │  │
│  │ (big matrix multiply│          │ (tiny compute, but    │  │
│  │  across all tokens) │          │  reads entire cache   │  │
│  └─────────────────────┘          │  from memory)         │  │
│                                    └──────────────────────┘  │
│                                                             │
│  Time: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│        ^prefill          ^decode (each █ = one token)       │
└─────────────────────────────────────────────────────────────┘
```

**Prefill** processes the entire prompt in one forward pass. All tokens are fed into the model simultaneously, and the self-attention is computed as a dense matrix multiply. This is **compute-bound** — the CPU/GPU is doing useful math the whole time. Typical: 5-20ms per token in the prompt.

**Decode** generates tokens one at a time. Each step feeds only the latest token, but the model must read the entire KV cache from memory to compute attention. This is **memory-bound** — the tiny compute for 1 token finishes quickly, but loading all the cached K/V tensors is slow. Typical: 30-250ms per token.

Why the asymmetry? During prefill, you process N tokens in one pass — the matrix multiply is [N, d] × [d, d], which keeps the GPU's compute units busy. During decode, you process 1 token — the multiply is [1, d] × [d, d], which finishes in microseconds, but you still have to read the full KV cache from DRAM. The ratio of compute to memory access (arithmetic intensity) drops dramatically.

### Sampling: Picking the Next Token

After each forward pass, the model outputs a vector of **logits** — one score for every token in the vocabulary (e.g., 32,000 scores for LLaMA). We need to pick one token from these scores.

### The Generation Pipeline

Here's the full end-to-end flow of generating a response:

```
┌───────────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                            │
│                                                                   │
│  "What is 2+2?"                                                  │
│        │                                                          │
│        ▼                                                          │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐ │
│  │ Tokenize │───►│ [1, 1724, 338, 29871, 29906, 29974, 29906]  │ │
│  └──────────┘    └──────────────────────┬───────────────────────┘ │
│                                         │                         │
│                                         ▼                         │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  PREFILL: forward(all 7 tokens)                         │      │
│  │    → logits[vocab_size]  → sample → "The" (first token) │      │
│  │    → KV cache filled (7 entries per layer)              │      │
│  └────────────────────────────────┬────────────────────────┘      │
│                                   │                               │
│                                   ▼                               │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  DECODE LOOP:                                           │      │
│  │    forward("The", past_kv) → sample → " answer"        │      │
│  │    forward(" answer", past_kv) → sample → " is"        │      │
│  │    forward(" is", past_kv) → sample → " "              │      │
│  │    forward(" ", past_kv) → sample → "4"                │      │
│  │    forward("4", past_kv) → sample → "." (or EOS)      │      │
│  └────────────────────────────────┬────────────────────────┘      │
│                                   │                               │
│                                   ▼                               │
│  ┌──────────┐    ┌────────────────────────────────────────┐       │
│  │ Detokenize│◄──│ [450, 1234, 338, 29871, 29946, 29889] │       │
│  └────┬─────┘    └────────────────────────────────────────┘       │
│       │                                                           │
│       ▼                                                           │
│  "The answer is 4."                                              │
└───────────────────────────────────────────────────────────────────┘
```

### Sampling: Picking the Next Token

After each forward pass, the model outputs a vector of **logits** — one score for every token in the vocabulary (e.g., 32,000 scores for LLaMA). We need to pick one token from these scores.

**Greedy decoding** (temperature = 0): Always pick the highest-scoring token. Deterministic — same input always gives same output. Fast but can be repetitive.

**Temperature sampling** (temperature > 0): Divide logits by temperature, apply softmax to get probabilities, then sample randomly. Temperature < 1 sharpens the distribution (more deterministic), temperature > 1 flattens it (more random/creative).

```
Logits:      [1.0, 5.0, 0.5, 2.0]     (raw model scores)

temp=0:      argmax → token 1          (always "Paris")
temp=0.5:    softmax([2, 10, 1, 4])    → [0.00, 0.99, 0.00, 0.01]  (almost greedy)
temp=1.0:    softmax([1, 5, 0.5, 2])   → [0.02, 0.88, 0.01, 0.04]  (moderate variety)
temp=2.0:    softmax([0.5, 2.5, 0.25, 1]) → [0.08, 0.64, 0.06, 0.14] (more random)
```

---

## How vLLM/SGLang Implements This

Even this simple inference loop maps directly to real vLLM components:

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `Engine.__init__` (load model) | `LLMEngine.__init__` | `ModelRunner.__init__` |
| `Engine.generate` (prefill path) | `ModelRunner.execute_model` (prefill) | `ForwardBatch` (extend mode) |
| `Engine.generate` (decode path) | `ModelRunner.execute_model` (decode) | `ForwardBatch` (decode mode) |
| `Engine._sample` | `Sampler.forward` | `Sampler` |
| HuggingFace `DynamicCache` | Custom paged KV cache (Blog 3) | Radix attention cache (Blog 7) |
| Flask `/generate` | FastAPI `/v1/chat/completions` | FastAPI + `sgl-router` |

The key difference: vLLM and SGLang replace HuggingFace's simple `DynamicCache` with sophisticated memory management systems (paged attention, radix attention) that can serve hundreds of requests simultaneously. We'll build those in Blogs 3-7.

---

## The Implementation

The complete implementation is in [01_naive_inference.py](../code-blog-simple-vllm/01_naive_inference.py) (~130 lines). Here are the key parts:

### The Engine

```python
class Engine:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
```

`model.eval()` disables dropout and switches batch normalization to inference mode. Without it, you'd get different outputs on every run even with temperature=0.

### Prefill

```python
out = self.model(input_ids=input_ids, use_cache=True)
past = out.past_key_values      # KV cache for all layers
next_token = self._sample(out.logits[0, -1, :], temperature)
```

`use_cache=True` tells HuggingFace to return `past_key_values` — a `DynamicCache` object containing K and V tensors for every layer. For TinyLlama with a 6-token prompt, this is 22 layers × 2 (K+V) × [1, 4, 6, 64] tensors.

### Decode Loop

```python
for _ in range(max_tokens - 1):
    if next_token == self.tokenizer.eos_token_id:
        break
    out = self.model(
        input_ids=torch.tensor([[next_token]]),
        past_key_values=past,
        use_cache=True,
    )
    past = out.past_key_values
    next_token = self._sample(out.logits[0, -1, :], temperature)
```

Notice we feed `torch.tensor([[next_token]])` — a [1, 1] tensor (batch=1, seq_len=1). The model uses the KV cache for attention over all previous tokens, but only computes new K/V for this one token.

### The HTTP Server

```python
@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json()
    result = engine.generate(data["prompt"], ...)
    return jsonify(result)
```

A minimal Flask server. One endpoint, blocking, single-threaded. In Blog 2, we'll replace this with async FastAPI and streaming.

---

## Running the Code

**Demo mode** (no server, just runs three prompts):
```bash
python 01_naive_inference.py --demo
```

**Server mode:**
```bash
python 01_naive_inference.py --port 5000

# In another terminal:
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "temperature": 0.0}'
```

Expected output:
```
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Parameters: 1.1B
  Layers: 22
  Hidden dim: 2048
  Vocab size: 32000

DEMO: Naive LLM Inference
Prompt: 'The capital of France is'
Output: Paris.
  Prefill: 110ms (6 tokens)
  Decode:  11818ms (50 tokens, 241ms/tok)
```

---

## Benchmarks

On CPU (TinyLlama-1.1B, float32):

| Metric | Value |
|--------|-------|
| Prefill latency | ~15ms/token (compute-bound) |
| Decode latency | ~240ms/token (memory-bound) |
| Prefill/decode ratio | ~16x faster per token during prefill |
| Memory (model weights) | ~4.4 GB |
| Memory (KV cache, 50 tokens) | ~28 MB |

The 16x difference between prefill and decode per-token latency is the fundamental asymmetry that drives almost every optimization in this blog series: continuous batching (Blog 4), chunked prefill (Blog 6), speculative decoding (Blog 8), and disaggregated serving (Blog 13) all exist because prefill and decode have radically different performance characteristics.

---

## Key Takeaways

1. **LLMs generate one token at a time** (autoregressive generation), not all at once
2. **The KV cache** stores Key/Value tensors from previous tokens so they don't need to be recomputed — this is the most important optimization in LLM inference
3. **Prefill is compute-bound** (dense matrix multiply over all prompt tokens), **decode is memory-bound** (tiny compute, large cache read)
4. **Sampling** turns logits into a token — temperature controls the randomness
5. This simple loop is the foundation of every production inference server

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the transformer paper
- [KV Cache Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249) — visual walkthrough
- **Next: [Blog 2 — Async Streaming](02_async_streaming.md)** — replace blocking Flask with async FastAPI and stream tokens to the client in real-time
