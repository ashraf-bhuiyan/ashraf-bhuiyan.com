---
title: "Part 15: The Full Architecture"
description: "Complete architecture diagram with all 14 techniques, request lifecycle, performance attribution, and deployment configurations."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 15
tags: ["architecture", "vLLM", "SGLang", "system design"]
---

## What This Blog Covers

We've spent 14 blogs building an LLM inference engine piece by piece. Each blog introduced one technique in isolation — paged attention here, continuous batching there, tensor parallelism over there. But production inference engines like vLLM and SGLang use all of these simultaneously.

This capstone blog does three things:

1. **Shows the full architecture** — how all 14 techniques compose into one system
2. **Walks through a request lifecycle** — from HTTP arrival to token delivery, touching every technique
3. **Gives configuration recommendations** — which techniques to enable for common deployment scenarios

No new code. Just the big picture.

---

## The Full Architecture

Here's what a production LLM inference engine looks like with all techniques enabled:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         API Server (Blog 2)                         │
│                 FastAPI + SSE Streaming + Request Queue              │
└──────────┬───────────────────────────────────────────────┬──────────┘
           │                                               │
           │            Load Balancer (Blog 10)            │
           │         round-robin / least-pending /         │
           │              cache-aware routing               │
           │                                               │
    ┌──────▼──────┐                              ┌────────▼────────┐
    │  DP Replica 0│                              │  DP Replica 1   │
    │              │                              │                 │
    │ ┌──────────┐ │                              │ ┌─────────────┐ │
    │ │ Async    │ │                              │ │ Async       │ │
    │ │Scheduler │ │                              │ │ Scheduler   │ │
    │ │(Blog 5)  │ │                              │ │ (Blog 5)    │ │
    │ └────┬─────┘ │                              │ └──────┬──────┘ │
    │      │       │                              │        │        │
    │ ┌────▼─────┐ │                              │ ┌──────▼──────┐ │
    │ │Continuous│ │                              │ │ Continuous  │ │
    │ │ Batching │ │                              │ │  Batching   │ │
    │ │+ Chunked │ │                              │ │ + Chunked   │ │
    │ │ Prefill  │ │                              │ │  Prefill    │ │
    │ │(Blog 4+6)│ │                              │ │ (Blog 4+6)  │ │
    │ └────┬─────┘ │                              │ └──────┬──────┘ │
    │      │       │                              │        │        │
    │ ┌────▼─────┐ │                              │ ┌──────▼──────┐ │
    │ │ Prefix   │ │                              │ │  Prefix     │ │
    │ │ Cache    │ │                              │ │  Cache      │ │
    │ │(Blog 7)  │ │                              │ │ (Blog 7)    │ │
    │ └────┬─────┘ │                              │ └──────┬──────┘ │
    │      │       │                              │        │        │
    │ ┌────▼─────────────────────────────────┐    │ ┌──────▼──────┐ │
    │ │       Model Runner                   │    │ │Model Runner │ │
    │ │  ┌───────────────────────────────┐   │    │ │             │ │
    │ │  │ Quantized Weights (Blog 14)  │   │    │ │ (mirror)    │ │
    │ │  │ INT4/INT8/FP8 + dequantize   │   │    │ │             │ │
    │ │  └───────────────────────────────┘   │    │ └─────────────┘ │
    │ │  ┌───────────────────────────────┐   │    │                 │
    │ │  │ TP across GPUs (Blog 9)      │   │    │                 │
    │ │  │ Column/Row parallel + AllRed. │   │    │                 │
    │ │  └───────────────────────────────┘   │    │                 │
    │ │  ┌───────────────────────────────┐   │    │                 │
    │ │  │ EP for MoE layers (Blog 11)  │   │    │                 │
    │ │  │ All-to-All dispatch/combine   │   │    │                 │
    │ │  └───────────────────────────────┘   │    │                 │
    │ │  ┌───────────────────────────────┐   │    │                 │
    │ │  │ Spec. Decoding (Blog 8)      │   │    │                 │
    │ │  │ Draft K → verify → accept    │   │    │                 │
    │ │  └───────────────────────────────┘   │    │                 │
    │ └──────────────────────────────────────┘    │                 │
    │                                             │                 │
    │ ┌──────────────────────────────────────┐    │                 │
    │ │     Paged KV Cache (Blog 3)          │    │                 │
    │ │  ┌─────────┐    ┌──────────┐         │    │                 │
    │ │  │GPU Blocks│    │CPU Blocks│         │    │                 │
    │ │  │(hot tier)│◄──►│(cold tier)│        │    │                 │
    │ │  └─────────┘    └──────────┘         │    │                 │
    │ │     CPU Offloading (Blog 12)          │    │                 │
    │ └──────────────────────────────────────┘    │                 │
    └─────────────────────────────────────────────┘─────────────────┘

Optional: Disaggregated P/D (Blog 13)
  Prefill and decode run on separate GPU pools
  Connected by KV Connector (RDMA/NVLink)
```

### Technique Dependency Map

Some techniques are independent (can be enabled/disabled without affecting others). Some depend on or interact with each other:

```
Independent (orthogonal):
  Blog 2  (Async Streaming)     — always needed
  Blog 7  (Prefix Caching)      — toggle on/off
  Blog 8  (Speculative Decoding) — toggle on/off
  Blog 12 (CPU Offloading)      — toggle on/off
  Blog 14 (Quantization)        — choose precision at load time

Dependent chain:
  Blog 3  (Paged Attention) ← required by everything below
    └─ Blog 4  (Continuous Batching) ← requires paged attention
        └─ Blog 5  (Async Scheduling) ← requires batching
        └─ Blog 6  (Chunked Prefill) ← requires batching

Parallelism (choose based on hardware):
  Blog 9  (Tensor Parallelism) — model doesn't fit on 1 GPU
  Blog 10 (Data Parallelism)   — need more throughput
  Blog 11 (Expert Parallelism) — MoE models only
  DP × TP × EP = total GPUs

System architecture:
  Blog 13 (Disaggregated P/D) — replaces colocated architecture
```

---

## Request Lifecycle: End-to-End

Here's the timeline showing which blog's technique handles each phase of a request:

```
Time ──────────────────────────────────────────────────────────────────►

│ HTTP POST    │ Route to  │ Prefix   │ Schedule │ Prefill  │ KV     │
│ arrives      │ DP replica│ cache    │ into     │ forward  │ stored │
│              │           │ lookup   │ batch    │ pass     │ paged  │
│ Blog 2       │ Blog 10   │ Blog 7   │ Blog 4+5 │ Blog 9+14│ Blog 3 │
│ (FastAPI+SSE)│ (DP route)│ (hash)   │ (async)  │ (TP+Quant)│(blocks)│
├──────────────┼───────────┼──────────┼──────────┼──────────┼────────┤
│   ~0.1ms     │  ~0.01ms  │  ~0.1ms  │  ~0.5ms  │  ~50ms   │ ~0.01ms│
              
              continues...

│ Chunked if   │ Decode    │ Spec.    │ CPU      │ Stream   │
│ prompt long  │ loop      │ decoding │ offload  │ tokens   │
│              │           │ (draft+  │ (if mem  │ via SSE  │
│ Blog 6       │ Blog 1    │  verify) │ pressure)│          │
│ (chunk=2048) │ (autoregr)│ Blog 8   │ Blog 12  │ Blog 2   │
├──────────────┼───────────┼──────────┼──────────┼──────────┤
│  if needed   │ per token │ optional │ optional │ per token│
│              │  ~20ms    │  3x fewer│          │  ~0.01ms │
│              │           │  passes  │          │          │

Optional: Blog 13 (Disaggregated) — prefill on P-worker, KV transfer, decode on D-worker
Optional: Blog 11 (Expert Parallel) — All-to-All in MoE layers during forward pass
```

Let's trace a single request through the entire system. The user sends:

```json
POST /generate
{"prompt": "Explain quantum computing in simple terms", "max_tokens": 200}
```

### Step 1: API Server Receives Request (Blog 2)

The FastAPI server accepts the HTTP request and creates a `StreamingResponse`. The connection stays open for Server-Sent Events (SSE) — tokens will be streamed as they're generated.

```
Client ──HTTP POST──► FastAPI ──creates──► StreamingResponse
                                           (SSE connection open)
```

### Step 2: Load Balancer Routes to a Replica (Blog 10)

With DP=4, the load balancer picks the best replica. SGLang's cache-aware router hashes the prompt prefix — if this system prompt was seen before, route to the replica that has it cached.

```
Request ──► Load Balancer
              │ Strategy: cache-aware
              │ Hash("Explain quantum") → Replica 2
              └──► Replica 2 (GPU 4+5, TP=2)
```

### Step 3: Prefix Cache Lookup (Blog 7)

The scheduler checks if any prefix of this prompt is already cached. The prompt is tokenized and block hashes are computed. If the system prompt "You are a helpful assistant..." was in a previous request, those KV blocks are already computed.

```
Token blocks:  [block_0: "Explain quantum"]  [block_1: "computing in"]  [block_2: "simple terms"]
Hash lookup:   block_0 → MISS    block_1 → MISS    block_2 → MISS
Result: no cache hit, full prefill needed
```

If there had been a cache hit, the cached blocks would be reused and only the new suffix would need prefill — saving compute proportional to the shared prefix length.

### Step 4: Scheduler Adds to Batch (Blogs 4 + 5)

The async scheduler runs on a separate thread, overlapping scheduling with GPU execution. It checks the waiting queue and decides whether this request can join the current batch:

```
Scheduler check:
  Token budget: 2048 tokens/step (Blog 6: chunked prefill)
  Current decode tokens: 15 sequences × 1 = 15 tokens
  Remaining budget: 2048 - 15 = 2033 tokens
  This request's prompt: 12 tokens → fits entirely
  Decision: add to batch, full prefill
```

If the prompt were 5000 tokens, chunked prefill (Blog 6) would split it across multiple steps, interleaving with ongoing decode requests to keep ITL stable.

### Step 5: Prefill Forward Pass (Blogs 3, 9, 11, 14)

The model processes all 12 prompt tokens in one forward pass. Multiple techniques operate simultaneously:

**Quantized weights (Blog 14):** Each linear layer stores weights as INT4 (packed). During the forward pass, weights are dequantized on-the-fly to FP16 for the matrix multiply. The model uses 4x less memory.

**Tensor parallelism (Blog 9):** Each attention and FFN layer is split across 2 GPUs. Column-parallel linear layers split the weight by output dimension. Row-parallel layers split by input dimension. AllReduce synchronizes partial results after each layer.

**Expert parallelism (Blog 11):** If this is a MoE model (e.g., DeepSeek-V3), the MoE FFN layers use All-to-All to route tokens to the GPU owning their assigned expert. Non-MoE layers use standard TP.

**Paged attention (Blog 3):** The KV cache is allocated in fixed-size blocks from a pre-allocated pool. The block table maps this sequence's logical blocks to physical GPU memory locations.

```
GPU 0 (TP rank 0):                    GPU 1 (TP rank 1):
  W_q[:, 0:d/2] × x → q_partial       W_q[:, d/2:d] × x → q_partial
  W_k[:, 0:d/2] × x → k_partial       W_k[:, d/2:d] × x → k_partial
  Attention on partial                  Attention on partial
  AllReduce(partial_out)         ↔      AllReduce(partial_out)
  FFN column-parallel                   FFN column-parallel
  FFN row-parallel + AllReduce   ↔      FFN row-parallel + AllReduce
```

### Step 6: First Token Sampled (Blog 1)

The prefill produces logits for the last position. The sampling strategy (greedy, top-p, temperature) selects the first token. This token is immediately streamed to the client via SSE.

```
Logits[last_pos] → softmax → "Quantum" (token 15991)
SSE: data: {"token": "Quantum"}\n\n
```

### Step 7: Decode Loop with Speculative Decoding (Blog 8)

Instead of generating one token at a time, the draft model (or n-gram drafter) guesses multiple tokens ahead:

```
Iteration 1:
  Draft:  ["Quantum", " computing", " is", " a", " type"]  (5 tokens)
  Verify: target model forward on all 5
  Accept: all 5 match → generated 5 tokens in 1 target forward pass
  Stream: " computing is a type"

Iteration 2:
  Draft:  [" of", " computation", " that", " uses", " quantum"]
  Verify: " of" ✓, " computation" ✗ (target says " computing")
  Accept: 1 token (" of") + target's correction (" computing")
  Stream: " of computing"

...repeat until max_tokens or EOS...
```

Each decode step also updates the paged KV cache — one new KV entry per generated token. If the current block is full, a new block is allocated from the pool.

### Step 8: Memory Pressure Handling (Blog 12)

Midway through generation, GPU memory fills up. The scheduler detects this and triggers CPU offloading for a lower-priority sequence:

```
GPU blocks: 950/1000 used ← pressure threshold
Scheduler: swap_out(sequence_42)  ← oldest waiting sequence
  Block 0,1,2 → CPU pinned memory (async copy)
  3 GPU blocks freed → sequence_42 paused
Current request continues uninterrupted
```

When sequence_42 is needed again, its KV blocks are swapped back in from CPU memory.

### Step 9: Response Completion (Blog 2)

The model generates an EOS token or hits max_tokens. The final SSE event is sent and the connection closes:

```
SSE: data: {"token": "."}\n\n
SSE: data: [DONE]\n\n
Connection closed
```

The sequence's KV blocks are freed back to the block pool (or kept in the prefix cache for future reuse).

---

## Performance Attribution

Each technique contributes differently depending on the workload:

```
┌────────────────────────┬─────────────────┬──────────────────────┬────────────┐
│ Technique              │ Throughput Gain  │ Latency Impact       │ Memory     │
├────────────────────────┼─────────────────┼──────────────────────┼────────────┤
│ Paged Attention        │ 2-4x            │ Neutral              │ 90%+ util  │
│ Continuous Batching    │ 5-20x           │ Neutral              │ Dynamic    │
│ Async Scheduling       │ 5-15%           │ -5-15% latency       │ Neutral    │
│ Chunked Prefill        │ Neutral         │ Stable ITL           │ Neutral    │
│ Prefix Caching         │ 2-8x (on hits)  │ -50-90% TTFT (hits) │ Saves      │
│ Speculative Decoding   │ 1.5-3x          │ -30-60% latency      │ +draft mem │
│ Tensor Parallelism     │ Near-linear/TP   │ +comm overhead       │ 1/TP       │
│ Data Parallelism       │ Linear with DP  │ Neutral              │ DP copies  │
│ Expert Parallelism     │ Required for MoE│ +All-to-All          │ 1/EP       │
│ CPU Offloading         │ +capacity       │ +swap latency        │ +CPU pool  │
│ Disaggregated P/D      │ +20-40%         │ Stable ITL           │ Transfer   │
│ Quantization (INT4)    │ 2-3x decode     │ -50% decode          │ 25% of FP16│
└────────────────────────┴─────────────────┴──────────────────────┴────────────┘
```

### The Biggest Wins

The techniques with the largest impact, in order:

1. **Continuous Batching** (Blog 4): 5-20x throughput. The single biggest architectural decision — going from static to continuous batching.

2. **Paged Attention** (Blog 3): 2-4x throughput. Enables continuous batching by eliminating memory fragmentation. Without it, you can't fit as many sequences concurrently.

3. **Quantization** (Blog 14): 2-4x capacity. Fitting the model in less memory means more room for KV cache, which means more concurrent sequences.

4. **Prefix Caching** (Blog 7): 2-8x TTFT reduction. If 90% of requests share a system prompt, 90% of prefill compute is saved.

5. **Tensor/Data Parallelism** (Blogs 9-10): Linear scaling with hardware. More GPUs = proportionally more throughput.

---

## Configuration Recommendations

### Scenario 1: Single-GPU Chatbot (Llama 8B on 1x H100)

```
Model: Llama-3-8B-Instruct
Hardware: 1x H100 80GB
Expected: 50-100 requests/s

Configuration:
  TP=1, DP=1
  Quantization: FP8 or INT8 (saves memory for KV cache)
  Continuous batching: ON
  Chunked prefill: ON (chunk_size=512)
  Prefix caching: ON (shared system prompt)
  Speculative decoding: Optional (EAGLE if latency-sensitive)
  CPU offloading: OFF (enough GPU memory)
  Disaggregated: OFF (single GPU)
```

### Scenario 2: Multi-GPU Production API (Llama 70B on 8x H100)

```
Model: Llama-3-70B-Instruct
Hardware: 8x H100 80GB (NVLink)
Expected: 200-500 requests/s

Configuration:
  TP=4, DP=2 (2 replicas, each across 4 GPUs)
  Quantization: FP8 (nearly lossless, 2x memory savings)
  Continuous batching: ON
  Chunked prefill: ON (chunk_size=2048)
  Prefix caching: ON
  Speculative decoding: ON (draft model on same GPU)
  CPU offloading: ON (for burst traffic)
  Disaggregated: Consider for strict ITL SLAs

  Alternative: INT4 (GPTQ/AWQ) allows TP=2, DP=4
               → 2x more replicas, 2x throughput
               → small quality tradeoff
```

### Scenario 3: Long-Context Document Processing (128K+ tokens)

```
Model: Llama-3-8B-Instruct (128K context)
Hardware: 4x H100 80GB
Expected: Low QPS, very long prompts

Configuration:
  TP=2, DP=2
  Quantization: INT8 (save memory for huge KV cache)
  Continuous batching: ON (but small batch sizes)
  Chunked prefill: ON (chunk_size=4096, critical for long prompts)
  Prefix caching: ON (document prefix reuse)
  Speculative decoding: OFF (decode is short relative to prefill)
  CPU offloading: ON (128K tokens × 32 layers = huge KV cache)
  Disaggregated: ON (long prefills would stall decode)
```

### Scenario 4: MoE Model (DeepSeek-V3 on 8x H100)

```
Model: DeepSeek-V3 (671B total, ~37B active)
Hardware: 8x H100 80GB
Expected: 50-200 requests/s

Configuration:
  TP=4, EP=2 (attention: TP=4, MoE: EP=2 between groups)
  Quantization: FP8 (fit 671B params across 8 GPUs)
  Continuous batching: ON
  Chunked prefill: ON
  Prefix caching: ON
  Speculative decoding: Optional
  CPU offloading: ON (experts are sparse, many params idle)
  Disaggregated: Optional
```

---

## Mapping to Production Codebases

### vLLM V1 Architecture

```
vllm/
├── entrypoints/
│   └── openai/api_server.py          ← Blog 2: FastAPI server
├── v1/
│   ├── engine/
│   │   ├── async_llm.py              ← Blog 5: async engine
│   │   ├── core.py                   ← Blog 4: scheduler + core loop
│   │   └── dp_request_router.py      ← Blog 10: DP load balancer
│   ├── core/
│   │   ├── scheduler.py              ← Blog 4+6: batching + chunked prefill
│   │   └── kv_cache_manager.py       ← Blog 3+7: paged attention + prefix cache
│   ├── worker/
│   │   └── gpu_model_runner.py       ← Blog 9+11+14: TP + EP + quantized weights
│   └── spec_decode/                  ← Blog 8: speculative decoding
├── distributed/
│   └── parallel_state.py             ← Blog 9+10+11: TP/DP/EP groups
└── kv_transfer/
    └── kv_connector/                 ← Blog 12+13: CPU offload + disaggregated
```

### SGLang Architecture

```
sglang/
├── srt/
│   ├── server.py                     ← Blog 2: FastAPI server
│   ├── managers/
│   │   ├── scheduler.py              ← Blog 4+5+6: zero-overhead scheduler
│   │   └── data_parallel_controller.py ← Blog 10: DP controller
│   ├── mem_cache/
│   │   ├── radix_cache.py            ← Blog 7: RadixAttention prefix cache
│   │   └── memory_pool.py            ← Blog 3: token-level paged memory
│   ├── model_executor/
│   │   └── model_runner.py           ← Blog 9+11+14: model execution
│   └── speculative/                  ← Blog 8: EAGLE speculative decoding
```

### Our Code ↔ Production Mapping

| Blog | Our File | vLLM Module | SGLang Module |
|------|----------|-------------|---------------|
| 1 | `01_naive_inference.py` | `ModelRunner.forward()` | `ModelRunner.forward()` |
| 2 | `02_async_streaming.py` | `entrypoints/openai/` | `server.py` |
| 3 | `03_paged_attention.py` | `KVCacheManager` | `memory_pool.py` |
| 4 | `04_continuous_batching.py` | `Scheduler` | `Scheduler` |
| 5 | `05_async_scheduling.py` | `AsyncLLM` + `EngineCoreProc` | Zero-overhead scheduler |
| 6 | `06_chunked_prefill.py` | `Scheduler._schedule_prefills()` | Chunked prefill in scheduler |
| 7 | `07_prefix_caching.py` | `PrefixCachingBlockAllocator` | `RadixCache` |
| 8 | `08_speculative_decoding.py` | `SpecDecWorker` | `EAGLEWorker` |
| 9 | `09_tensor_parallelism.py` | `ColumnParallelLinear` / `RowParallelLinear` | Same (shared Megatron pattern) |
| 10 | `10_data_parallelism.py` | `DPRequestRouter` | `DataParallelController` |
| 11 | `11_expert_parallelism.py` | `FusedMoE` + All-to-All | `FusedMoE` |
| 12 | `12_kv_cpu_offloading.py` | `CpuGpuBlockAllocator` + `KVConnector` | N/A |
| 13 | `13_disaggregated_serving.py` | `KVConnectorBase` + P/D workers | N/A |
| 14 | `14_quantization.py` | `Fp8LinearMethod`, `GPTQLinearMethod` | Quantization methods |

---

## The Technique Composition Matrix

Not all techniques can be combined freely. Here's what composes with what:

```
                  Paged  Cont.  Async  Chunk  Prefix Spec   TP    DP    EP    CPU   P/D   Quant
                  Attn   Batch  Sched  Pref   Cache  Dec                      Off   Dis
Paged Attn          —     req    ✓      ✓      ✓      ✓     ✓     ✓     ✓     ✓     ✓     ✓
Cont. Batching    req      —     ✓      ✓      ✓      ✓     ✓     ✓     ✓     ✓     ✓     ✓
Async Scheduling   ✓      ✓      —      ✓      ✓      ✓     ✓     ✓     ✓     ✓     ✓     ✓
Chunked Prefill    ✓      ✓      ✓      —      ✓      ✓     ✓     ✓     ✓     ✓     ✓     ✓
Prefix Caching     ✓      ✓      ✓      ✓      —      ✓     ✓     ✓     ✓     ✓     ~     ✓
Spec. Decoding     ✓      ✓      ✓      ✓      ✓      —     ✓     ⚠     ✓     ✓     ~     ✓
TP                 ✓      ✓      ✓      ✓      ✓      ✓     —     ✓     ✓     ✓     ✓     ✓
DP                 ✓      ✓      ✓      ✓      ✓      ⚠     ✓     —     ✓     ✓     ✓     ✓
EP                 ✓      ✓      ✓      ✓      ✓      ✓     ✓     ✓     —     ✓     ✓     ✓
CPU Offloading     ✓      ✓      ✓      ✓      ✓      ✓     ✓     ✓     ✓     —     ~     ✓
Disagg. P/D        ✓      ✓      ✓      ✓      ~      ~     ✓     ✓     ✓     ~     —     ✓
Quantization       ✓      ✓      ✓      ✓      ✓      ✓     ✓     ✓     ✓     ✓     ✓     —

✓ = composes cleanly    ⚠ = needs coordination    ~ = partial/complex    req = required by
```

Key interactions:
- **Speculative decoding + DP**: All DP ranks must agree on whether to draft in a given step (otherwise TP within each DP group deadlocks)
- **Prefix caching + disaggregated**: Cache lives on specific workers, routing must be cache-aware
- **CPU offloading + disaggregated**: Both use the KV Connector abstraction, but serve different purposes

---

## Key Takeaways

1. **The foundation is paged attention + continuous batching.** Everything else builds on top of these two. Without them, you're leaving 5-20x throughput on the table.

2. **Orthogonal techniques compose freely.** Prefix caching, speculative decoding, quantization, and CPU offloading can each be toggled independently. Enable what your workload needs.

3. **Parallelism is about hardware fitting.** Use TP when the model doesn't fit on one GPU. Use DP to scale throughput. Use EP for MoE models. The formula is simple: DP x TP x EP = total GPUs.

4. **The async boundary matters.** Overlapping CPU scheduling with GPU execution (Blog 5) gives 5-15% free throughput. SGLang's zero-overhead scheduler takes this furthest.

5. **Quantization is the highest-leverage single change.** Going from FP16 to INT4 gives 4x memory savings, fitting larger models on fewer GPUs, with manageable quality loss when using GPTQ/AWQ.

6. **Disaggregation is the future for large deployments.** Separating prefill and decode lets each phase use optimal hardware and eliminates interference. The main barrier is KV transfer bandwidth.

7. **Production is about composition.** No single technique is a silver bullet. The art is choosing the right combination for your model, hardware, and traffic pattern.

---

## The Complete Blog Series

| # | Blog | Technique | Key Insight |
|---|------|-----------|-------------|
| 1 | [Naive Inference](01_naive_inference.md) | Autoregressive generation | KV cache avoids recomputation |
| 2 | [Async Streaming](02_async_streaming.md) | SSE + FastAPI | Don't block on full generation |
| 3 | [Paged Attention](03_paged_attention.md) | Virtual memory for KV | Eliminate memory fragmentation |
| 4 | [Continuous Batching](04_continuous_batching.md) | Dynamic batching | Let sequences join/leave freely |
| 5 | [Async Scheduling](05_async_scheduling.md) | CPU-GPU overlap | Schedule next while GPU runs current |
| 6 | [Chunked Prefill](06_chunked_prefill.md) | Split long prefills | Keep decode latency stable |
| 7 | [Prefix Caching](07_prefix_caching.md) | Hash-based KV reuse | Skip redundant prefill compute |
| 8 | [Speculative Decoding](08_speculative_decoding.md) | Draft-verify loop | Multiple tokens per forward pass |
| 9 | [Tensor Parallelism](09_tensor_parallelism.md) | Split weights across GPUs | Fit large models |
| 10 | [Data Parallelism](10_data_parallelism.md) | Replicate model | Scale throughput linearly |
| 11 | [Expert Parallelism](11_expert_parallelism.md) | Route tokens to experts | Efficient MoE inference |
| 12 | [KV CPU Offloading](12_kv_cpu_offloading.md) | Two-tier cache | Trade bandwidth for capacity |
| 13 | [Disaggregated Serving](13_disaggregated_serving.md) | Split prefill/decode | Optimize each phase independently |
| 14 | [Quantization](14_quantization.md) | Lower precision | 2-4x memory savings |
| 15 | The Full Architecture (this blog) | Everything together | Composition is the real technique |

---

## Further Reading

**Production implementations:**
- [vLLM](https://github.com/vllm-project/vllm) — the most widely deployed open-source inference engine
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention, zero-overhead scheduling
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang) — 5K-line educational implementation by LMSYS

**Key papers:**
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM (Blog 3)
- [Orca: A Distributed Serving System for Transformer-Based Models](https://www.usenix.org/conference/osdi22/presentation/yu) — continuous batching (Blog 4)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — RadixAttention (Blog 7)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding (Blog 8)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — TP (Blog 9)
- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670) — disaggregated serving (Blog 13)

**This series:** [All code on GitHub](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm)
