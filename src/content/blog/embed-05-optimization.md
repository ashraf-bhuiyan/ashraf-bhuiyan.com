---
title: "Embedding Throughput Optimization"
description: "Saturation curves, prefix caching for embeddings, benchmarking methodology, and deployment patterns at scale."
pubDate: 2026-05-03
series: "Embeddings, Pooling & Rerankers in vLLM"
part: 5
tags: ["throughput", "optimization", "benchmarking", "prefix caching"]
---

## What Problem Does This Solve?

You can serve embeddings and rerankers with vLLM (Blogs B3-B4). But how fast? If you need to embed a million documents for indexing, or serve real-time search at 10,000 QPS, the default configuration probably isn't optimal. This blog covers how to maximize throughput and minimize latency for embedding and scoring workloads.

---

## Throughput Metrics

### What to Measure

```
For batch/offline workloads (indexing a document corpus):
  Tokens/sec:     how many input tokens processed per second
  Documents/sec:  how many documents embedded per second
  
  Goal: maximize throughput, latency doesn't matter much.

For online workloads (real-time search API):
  QPS:            queries per second at acceptable latency
  P50 latency:    median per-request latency
  P99 latency:    tail latency (what the slowest 1% experience)
  
  Goal: maximize QPS while keeping P99 under SLA (e.g., <50ms).
```

### Embedding vs. Generation Throughput

Embedding workloads have fundamentally different performance characteristics:

```
Generation (Blog 4 of inference series):
  Throughput limited by: decode speed (memory bandwidth)
  Bottleneck: reading model weights once per output token
  Scaling: more output tokens = linearly more time

Embedding:
  Throughput limited by: prefill speed (compute)
  Bottleneck: processing all input tokens in one pass
  Scaling: longer inputs = more compute per request,
           but no decode overhead
  
  Key insight: embedding has NO decode phase.
  Every request is a single prefill + pool.
  This makes embeddings much more GPU-compute-efficient.
```

---

## Continuous Batching for Embeddings

### How It Works

Embedding requests are all prefill-only — no decode phase. But continuous batching still helps because requests have variable lengths:

```
Without continuous batching (static padding):
  Batch: ["Hello" (2 tok), "How do I reset my password?" (7 tok)]
  Padded to max: both sequences padded to 7 tokens
  Wasted compute: 5 padding tokens × hidden_dim multiplies = ~42% waste

With continuous batching:
  Step 1: Process both sequences (2 + 7 = 9 tokens total)
  "Hello" completes first → return embedding, free memory
  New request joins the batch immediately
  
  No padding waste. GPU always processing real tokens.
```

### Token Budget Tuning

The scheduler fills each batch up to a token budget:

```bash
vllm serve BAAI/bge-large-en-v1.5 \
  --task embed \
  --max-num-seqs 256 \           # max requests per batch
  --max-model-len 512            # max input length
```

For embedding workloads with short inputs (typical: 50-200 tokens), increase `--max-num-seqs` to pack more requests into each batch:

```
Input length ~50 tokens, batch of 256 → 12,800 tokens per batch
  → Good GPU utilization for a 335M model

Input length ~50 tokens, batch of 32 → 1,600 tokens per batch
  → Poor GPU utilization, GPU mostly idle between batches
```

---

## Prefix Caching for Embeddings

### The Opportunity

Many embedding models use instruction prefixes (Blog B3):

```
E5: "query: How do I reset my password?"
     ↑↑↑↑↑↑
     Same prefix for every query

Instructor: "Represent this sentence for retrieval: ..."
             ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
             Same prefix for every sentence
```

The instruction prefix is identical across all requests. Without prefix caching, vLLM re-encodes it on every request. With prefix caching (from Inference Blog 7), the KV cache for the prefix is computed once and reused.

### Impact

```
E5-large-v2 with "query: " prefix (2 tokens):
  Minimal savings — the prefix is only 2 tokens

Instructor-large with full instruction (15 tokens):
  Prefix: "Represent this sentence for retrieval: "
  Average query: 30 tokens
  Prefix = 15/45 = 33% of tokens → 33% prefill savings

Custom model with long system prompt (100 tokens):
  Prefix: 100-token task description
  Average input: 200 tokens  
  Prefix = 100/300 = 33% → 33% prefill savings
```

For instruction-prefixed models with long instructions, prefix caching can reduce prefill time by 30-40%.

### Enabling Prefix Caching

Prefix caching is enabled by default in vLLM V1. The hash-based caching mechanism from Inference Blog 7 automatically detects shared prefixes:

```
Request 1: "query: How do I reset my password?"
  → Compute KV for "query: " → cache with hash H1
  → Compute KV for "How do I reset my password?"

Request 2: "query: What is the refund policy?"
  → Look up hash H1 → cache HIT for "query: "
  → Compute KV for "What is the refund policy?" only

Savings: skipped re-encoding the prefix tokens
```

---

## Optimal Batch Sizes and Concurrency

### The Saturation Curve

As batch size increases, throughput rises until the GPU saturates:

```
BGE-large-en-v1.5 (335M params) on A100-80GB, input_len=128:

Batch size    Throughput       GPU Util    Latency/req
              (docs/sec)       (%)         (ms)
──────────────────────────────────────────────────────
1             500              5%          2.0
4             1,900            18%         2.1
16            6,500            55%         2.5
64            18,000           82%         3.6
128           26,000           90%         4.9
256           30,000           93%         8.5
512           31,000           94%         16.5   ← saturated

Saturation point: ~256 for this model/GPU combination
Beyond 256: throughput barely increases, latency rises fast

GTE-Qwen2-7B on A100-80GB, input_len=128:

Batch size    Throughput       GPU Util    Latency/req
──────────────────────────────────────────────────────
1             60               3%          16.7
4             230              12%         17.4
16            800              42%         20.0
32            1,400            65%         22.9
64            2,200            82%         29.1
128           2,800            90%         45.7   ← saturated
```

### Finding Your Saturation Point

```
Rule of thumb:
  Small model (<500M):    saturation at batch 128-512
  Medium model (1-3B):    saturation at batch 32-128
  Large model (7B+):      saturation at batch 16-64

To find yours:
  1. Sweep batch sizes: 1, 4, 16, 64, 128, 256, 512
  2. Plot throughput vs. batch size
  3. Saturation point = where throughput gains < 5% per doubling
  4. Set max_num_seqs to this value
```

### Client-Side Concurrency

For online workloads, use async requests with a concurrency limiter:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")

async def embed_with_concurrency(texts, max_concurrent=32):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_one(text):
        async with semaphore:
            response = await client.embeddings.create(
                model="BAAI/bge-large-en-v1.5",
                input=text,
            )
            return response.data[0].embedding
    
    return await asyncio.gather(*[embed_one(t) for t in texts])
```

Setting `max_concurrent` to match the GPU's saturation point gives optimal throughput without causing excessive queuing.

---

## Benchmarking Methodology

### Variables to Sweep

```
1. Batch size:        1, 4, 16, 64, 128, 256
2. Input length:      32, 64, 128, 256, 512 tokens
3. Concurrency:       1, 4, 16, 32, 64 concurrent requests
4. Prefix length:     0, 16, 64 tokens (for prefix caching impact)
```

### What to Report

```
For each configuration:
  - Throughput: docs/sec and tokens/sec
  - Latency: P50, P95, P99 (ms)
  - GPU utilization (%)
  - GPU memory usage (GB)

Include:
  - Model name and parameter count
  - GPU model and memory
  - vLLM version
  - Input length distribution (mean, min, max)
```

### Common Benchmarking Pitfalls

```
Pitfall 1: Uniform input lengths
  Real traffic has variable lengths (10-500 tokens).
  Benchmark with a realistic distribution, not all-same-length.

Pitfall 2: No warmup
  The first few requests trigger CUDA warmup, model loading, JIT.
  Run 100+ warmup requests before measuring.

Pitfall 3: Single-threaded client
  A single-threaded client can't saturate the GPU.
  Use async requests or multiple threads.

Pitfall 4: Measuring wall time only
  Wall time includes network latency, serialization, etc.
  Measure GPU kernel time separately (torch.profiler) for
  model-level benchmarks.
```

### Sample Benchmark Script

```python
import time
import asyncio
import numpy as np
from openai import AsyncOpenAI

async def benchmark(
    model: str,
    texts: list[str],
    concurrency: int = 32,
    warmup: int = 100,
):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    
    async def embed_one(text):
        async with sem:
            start = time.perf_counter()
            await client.embeddings.create(model=model, input=text)
            latencies.append(time.perf_counter() - start)
    
    # Warmup
    await asyncio.gather(*[embed_one(texts[i % len(texts)]) for i in range(warmup)])
    latencies.clear()
    
    # Benchmark
    start = time.perf_counter()
    await asyncio.gather(*[embed_one(t) for t in texts])
    elapsed = time.perf_counter() - start
    
    print(f"Throughput: {len(texts) / elapsed:.0f} docs/sec")
    print(f"P50 latency: {np.percentile(latencies, 50)*1000:.1f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95)*1000:.1f} ms")
    print(f"P99 latency: {np.percentile(latencies, 99)*1000:.1f} ms")
```

---

## Serving Classification and Reward Models

### Classification Models

Beyond embeddings and reranking, vLLM supports classification models:

```bash
vllm serve cardiffnlp/twitter-roberta-base-sentiment-latest --task classify
```

Classification follows the same pipeline: prefill → pool → classification head → class probabilities. The output is a vector of probabilities over classes instead of a single score.

### Reward Models

Used in RLHF pipelines to score (prompt, response) quality:

```bash
vllm serve OpenAssistant/reward-model-deberta-v3-large --task score
```

Reward models are essentially cross-encoders where text_1 is the prompt and text_2 is the response. The score indicates response quality.

### Optimization

All the same techniques apply:
- Continuous batching for variable-length inputs
- Prefix caching for shared prompts (reward models often evaluate multiple responses to the same prompt)
- Batch sizes tuned to GPU saturation

---

## Production Deployment Patterns

### Pattern 1: Dedicated Embedding Instance

```
vLLM Instance 1: Embedding model (port 8001)
  --task embed
  --max-num-seqs 256 (high batch for throughput)

vLLM Instance 2: Generative LLM (port 8002)
  (default task)
  --max-num-seqs 32 (lower batch, longer sequences)

Separate instances allow independent scaling:
  - Scale embeddings during bulk indexing jobs
  - Scale generation during peak user traffic
```

### Pattern 2: Embedding + Reranker Co-located

```
Same GPU, two model instances:
  vLLM (embed):  BGE-large (335M) → uses ~1 GB
  vLLM (score):  BGE-reranker-v2 (568M) → uses ~2 GB
  
  Remaining 77 GB for KV cache and overhead
  Both models are small enough to share a GPU
```

### Pattern 3: Offline Batch Embedding

```
For indexing a document corpus:
  1. Split corpus into chunks of 10K documents
  2. Run batch embedding with max concurrency
  3. Store embeddings in vector database
  4. Shutdown the embedding vLLM instance (save cost)

  No need for a persistent server — run as a batch job.
```

---

## Key Takeaways

1. **Find your saturation point**: sweep batch sizes to find where throughput plateaus — set `max_num_seqs` there
2. **Prefix caching saves 30-40%** for instruction-prefixed models — enabled by default in vLLM V1
3. **Embedding workloads are compute-bound** (no decode phase), so GPU utilization can be much higher than generation
4. **Benchmark realistically**: variable input lengths, warmup rounds, async clients, realistic distributions
5. **Use vLLM for large models** (7B+), consider TEI for small BERT-class models
6. **All non-generative tasks** (embed, score, classify, reward) follow the same optimization principles

---

## Series Summary

This completes the Embeddings, Pooling & Rerankers series:

```
B1: Embedding fundamentals (bi-encoders, contrastive training, similarity metrics)
B2: Pooling strategies (CLS, mean, last-token — and why getting it wrong is silent)
B3: Serving embeddings in vLLM (--task embed, /v1/embeddings, internal pipeline)
B4: Rerankers and scoring (cross-encoders, retrieve-then-rerank, /v1/score)
B5: Throughput optimization (batching, prefix caching, benchmarking, deployment)
```

---

## Further Reading

- [vLLM Benchmarking Guide](https://docs.vllm.ai/en/latest/performance/benchmarks.html)
- [TEI Performance Benchmarks](https://huggingface.co/docs/text-embeddings-inference/en/performance)
- [FAISS: Efficient Similarity Search](https://github.com/facebookresearch/faiss) — vector database for embedding search
- [Milvus](https://milvus.io/) — scalable vector database
