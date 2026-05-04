---
title: "Optimizing Multi-Modal Serving"
description: "Prefix caching for repeated images, chunked prefill with visual tokens, TP for VLMs, resolution tuning, and production checklist."
pubDate: 2026-05-03
series: "Vision, Language & Audio Models in vLLM"
part: 6
tags: ["optimization", "prefix caching", "chunked prefill", "production"]
---

## What Problem Does This Solve?

Multi-modal requests are expensive. A single image adds 576+ tokens to the KV cache. A video clip can add 18,000+. Without optimization, multi-modal serving has lower throughput, higher latency, and serves fewer concurrent requests than text-only workloads.

This blog covers the techniques that close the gap: prefix caching for shared images, chunked prefill for large visual inputs, tensor parallelism for vision encoders, and configuration tuning for production workloads.

---

## Image Caching and Prefix Caching

### The Opportunity

Many real workloads reuse the same images across requests:

```
E-commerce:
  100 queries about the same product photo
  → same image encoded 100 times without caching

Document analysis:
  10 questions about the same scanned page
  → same page image encoded 10 times

System monitoring:
  Dashboard screenshot analyzed hourly
  → same chart layout encoded repeatedly
```

### How Prefix Caching Works for Images

From Inference Blog 7, prefix caching reuses KV cache blocks when the token content matches. For VLMs, this applies to visual tokens too:

```
Request 1: "What color is the car?" + [image_A]
  → Compute KV for all tokens including 576 visual tokens
  → Cache blocks with hash H(visual_tokens)

Request 2: "How many wheels are visible?" + [image_A]
  → Check hash of visual tokens → MATCH! (same image_A)
  → Reuse KV cache blocks for the 576 visual tokens
  → Only compute KV for the different text tokens

Savings:
  Without caching: 576 visual tokens × compute_per_token = full prefill
  With caching:    576 visual tokens × 0 (reused) = skip entirely
  
  If visual tokens are 75% of the total → 75% prefill reduction!
```

### Requirements for Cache Hits

The visual tokens must be **identical** in content and **at the same positions** for the prefix to match:

```
Cache HIT:
  Request 1: "Describe:" [IMG₁..IMG₅₇₆] "\n" "What color?"
  Request 2: "Describe:" [IMG₁..IMG₅₇₆] "\n" "How many?"
              ↑ same prefix through visual tokens ↑

Cache MISS:
  Request 1: "Describe this:" [IMG₁..IMG₅₇₆]
  Request 2: "What is:" [IMG₁..IMG₅₇₆]
              ↑ text differs before image ↑ → different positions!
```

For maximum cache hits: put the image **before** the varying text, so the shared prefix is as long as possible.

### Impact

```
Workload: 100 questions about the same product photo
Model: Qwen2-VL-7B, image produces 576 tokens

Without prefix caching:
  Each request: full prefill (576 image + ~20 text = 596 tokens)
  100 requests: 100 × 596 = 59,600 tokens processed

With prefix caching:
  Request 1: full prefill (596 tokens) → cache visual prefix
  Requests 2-100: only new text tokens (~20 each)
  Total: 596 + 99 × 20 = 2,576 tokens processed
  
  Speedup: 23× fewer tokens to process!
  TTFT reduction: 90%+ for requests after the first
```

---

## Chunked Prefill with Multi-Modal Inputs

### The Latency Spike Problem

Without chunked prefill, a multi-image request blocks all decode requests:

```
Batch before the multi-image request arrives:
  32 decode requests running, each producing 1 token per step
  ITL (inter-token latency): ~30ms per step

Multi-image request arrives (5 images, 2,880 visual tokens + 200 text):
  Prefill 3,080 tokens → ~800ms

Without chunked prefill:
  Step N:   decode 32 requests (30ms)
  Step N+1: prefill 3,080 tokens (800ms) ← ALL decode requests blocked!
  Step N+2: decode 33 requests (30ms)
  
  P99 ITL spikes from 30ms to 800ms!
  Users see an 800ms pause in their streaming output.

With chunked prefill (chunk_size=512):
  Step N:   decode 32 + prefill chunk 1 (512 tokens) → ~80ms
  Step N+1: decode 32 + prefill chunk 2 (512 tokens) → ~80ms
  ...
  Step N+6: decode 32 + prefill chunk 6 (final chunk) → ~80ms
  Step N+7: decode 33 → ~30ms
  
  P99 ITL: ~80ms (controlled by chunk size)
```

### Vision Encoder vs. LLM Prefill

An important nuance: chunked prefill only applies to the **LLM's processing** of visual tokens, not the vision encoder itself:

```
Vision encoder: runs once, processes all images at once
  → NOT chunked (it's a single GPU operation, ~50-100ms for 5 images)
  → This is a fixed overhead

LLM prefill: processes the combined text + visual token sequence
  → THIS is what gets chunked
  → 3,080 tokens → 6 chunks of ~512 tokens each
  → Each chunk interleaved with decode steps

Total overhead: encoder (100ms) + chunked prefill (6 × ~80ms)
  = ~580ms spread across 7 steps instead of one 800ms block
```

### Configuration

Chunked prefill is enabled by default in vLLM V1. The key parameter:

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --max-num-batched-tokens 2048   # token budget per step

# Larger budget → faster prefill, but more ITL impact
# Smaller budget → slower prefill, but smoother ITL
# Default: auto-tuned based on model and GPU
```

---

## Tensor Parallelism for VLMs

### How VLM Components Are Distributed

When using tensor parallelism (TP) for a VLM, the three components are handled differently:

```
TP=4, Qwen2-VL-72B:

  Vision Encoder (SigLIP, ~400M params):
    → REPLICATED on all 4 GPUs
    → Each GPU runs the full encoder independently
    → Small enough that replication is fine (400M << 72B)
    → No communication needed

  Projection Layer (~10M params):
    → REPLICATED on all 4 GPUs
    → Trivially small

  LLM Backbone (72B params):
    → SHARDED across 4 GPUs (standard TP)
    → Each GPU has 1/4 of each linear layer
    → AllReduce communication between GPUs
```

### Why Replicate the Vision Encoder?

```
Option A: Replicate (each GPU has full encoder)
  Memory: 4 × 400M × 2B = 3.2 GB total (0.8 GB per GPU)
  Communication: none (each GPU encodes independently)
  Speed: each GPU processes the image in parallel → fast

Option B: Shard (split encoder across GPUs)
  Memory: 400M × 2B = 0.8 GB total (0.2 GB per GPU)
  Communication: AllReduce within the encoder → adds latency
  Speed: slower due to communication overhead

Replication wins:
  - 0.6 GB extra per GPU is negligible when the LLM uses 18 GB per GPU
  - No communication overhead
  - Simpler implementation
```

### Exception: Very Large Vision Encoders

InternVL2 uses InternViT-6B (6 billion parameter vision encoder):

```
InternViT-6B:
  Replicated (TP=4): 4 × 6B × 2B = 48 GB total (12 GB per GPU)
  → Significant memory overhead!

  Sharded (TP=4): 6B × 2B = 12 GB total (3 GB per GPU)
  → Much more memory-efficient, but requires communication

InternVL2 shards the vision encoder across TP ranks.
Most other VLMs replicate it.
```

### Throughput Impact

```
Qwen2-VL-7B, image request, A100-80GB:

TP=1: TTFT = 120ms, decode = 13ms/token
TP=2: TTFT = 80ms,  decode = 9ms/token
TP=4: TTFT = 55ms,  decode = 7ms/token

The vision encoder overhead (~40ms) doesn't decrease with TP
because it's replicated. Only the LLM backbone gets faster.

At TP=4: vision encoder = 40ms, LLM prefill = 15ms
  → Vision encoder becomes 73% of TTFT!
  → Further TP scaling gives diminishing returns
```

---

## Resolution Tuning: Quality vs. Throughput

### The Core Tradeoff

```
Higher resolution → more visual tokens → better understanding → slower, more memory
Lower resolution  → fewer visual tokens → worse understanding → faster, less memory
```

### Finding the Right Resolution

**For chatbot applications** (casual image questions):

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 401408}'

# max_pixels=401408 → up to ~20×20 patches after merge → ~400 tokens per image
# Good enough for "What's in this image?" type questions
# Saves 30% tokens vs. default
```

**For document understanding / OCR** (need every detail):

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 2073600}'

# max_pixels=2073600 → up to ~36×36 patches after merge → ~1296 tokens per image
# Preserves text detail in documents, charts, diagrams
# Uses 2× more memory per image
```

**For video** (frame quality matters less):

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 200704, "fps": 0.5}'

# Lower resolution per frame + fewer frames per second
# Each frame ~196 tokens × 15 frames for 30s = 2,940 tokens
# vs. default: ~576 × 30 = 17,280 tokens (6× reduction!)
```

---

## Production Tuning Checklist

### 1. Set `--limit-mm-per-prompt`

```bash
# Prevent single requests from consuming all memory
--limit-mm-per-prompt image=5,video=1

# For document analysis (many pages):
--limit-mm-per-prompt image=10
```

### 2. Tune `max_pixels` / `min_pixels`

```bash
# For your specific quality-throughput tradeoff
# Profile with actual user queries to find the sweet spot
--mm-processor-kwargs '{"max_pixels": 602112}'
```

### 3. Set Appropriate `--max-model-len`

```
Account for worst-case token count:
  max_images × tokens_per_image + max_text_tokens + max_output_tokens

Example:
  5 images × 576 tokens + 500 text + 1000 output = 4,380 tokens
  → --max-model-len 4096 is too low!
  → --max-model-len 8192 provides headroom
```

### 4. Enable Prefix Caching (Default in V1)

```
Prefix caching is on by default. Verify it's working:
  - Monitor cache hit rate in metrics
  - If hit rate is low, check that shared images appear at the
    same position in the token sequence across requests
```

### 5. Chunked Prefill (Default in V1)

```
If P99 ITL spikes when multi-image requests arrive:
  - Reduce max_num_batched_tokens (smaller chunks)
  - Or reduce max_pixels (fewer visual tokens to prefill)
```

### 6. Separate Multi-Modal and Text-Only Traffic

```
If you serve both VLM and text-only requests:
  Consider separate model instances:
  
  Instance 1: VLM (handles image/video/audio requests)
    → Tuned for multi-modal: lower max_num_seqs, higher max_model_len
    
  Instance 2: Text-only (handles text requests)
    → Tuned for text: higher max_num_seqs, lower max_model_len
    
  Why: multi-modal requests use much more memory per request,
  reducing concurrent text-only capacity
```

### 7. Monitor GPU Memory

```
Multi-modal requests have high variance in memory usage:
  Text request:     25 MB of KV cache
  Image request:    100 MB of KV cache
  Video request:    1,200 MB of KV cache

Monitor:
  - KV cache utilization (should stay under 90%)
  - Request queue depth (growing queue = memory pressure)
  - OOM events (should be zero with proper limits)
```

---

## Benchmarking Multi-Modal Throughput

### What to Measure

```
Metrics:
  - Images/sec: throughput at the request level
  - Tokens/sec: throughput including visual tokens
  - TTFT (time to first token): dominated by vision encoder + prefill
  - ITL (inter-token latency): should match text-only (decode is the same)
  - P50/P99 of TTFT and ITL

Variables to sweep:
  - Image resolution: low, medium, high
  - Image count: 1, 3, 5, 10
  - Concurrency: 1, 4, 16, 32
  - Prefix caching: on/off
```

### Expected Results

```
Prefix caching ON vs. OFF (repeated-image workload):
  TTFT: 3-5× faster with caching
  Throughput: 2-4× higher with caching

Chunked prefill ON vs. OFF (mixed VLM + text workload):
  P99 ITL: 10-50× lower with chunking
  Throughput: similar (chunking doesn't change total compute)

TP=2 vs. TP=1 (single-image requests):
  TTFT: ~1.5× faster (LLM prefill is sharded, encoder is replicated)
  Decode: ~1.8× faster (standard TP speedup)
  Note: sublinear speedup because the vision encoder is replicated
```

---

## Key Takeaways

1. **Prefix caching for repeated images** can reduce TTFT by 90%+ — put images before varying text for maximum cache hit rate
2. **Chunked prefill prevents ITL spikes** from multi-image/video requests — the vision encoder runs once, then LLM prefill is chunked
3. **Vision encoders are typically replicated** across TP ranks (they're small) — only the LLM backbone is sharded
4. **Resolution is the primary quality-throughput knob** — `max_pixels` controls tokens per image
5. **Set limits**: `--limit-mm-per-prompt` and `--max-model-len` prevent memory exhaustion
6. **Separate VLM and text traffic** in production for independent scaling and tuning

---

## Series Summary

This completes the Vision, Language & Audio series:

```
C1: VLM architecture (vision encoder → projection → LLM, token count tradeoffs)
C2: Serving VLMs (launching, API format, preprocessing pipeline)
C3: Multi-image & video (token explosion, frame sampling, memory management)
C4: Audio models (Whisper encoder, mel spectrogram, speech understanding)
C5: Multi-modal internals (registry, plugins, processor, placeholder replacement)
C6: Optimization (prefix caching, chunked prefill, TP, resolution tuning)
```

---

## Further Reading

- [vLLM Multi-Modal Documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html)
- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) — multi-image and video VLM
- [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191) — dynamic resolution serving
- [InternVL2 Technical Report](https://arxiv.org/abs/2404.16821) — dynamic tiling and large vision encoders
