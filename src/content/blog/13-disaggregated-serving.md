---
title: "Part 13: Disaggregated Prefill-Decode"
description: "Separate prefill and decode worker pools with KV transfer via the KVConnector abstraction for hardware-optimized serving."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 13
tags: ["disaggregated serving", "prefill-decode", "KVConnector"]
---

## What Problem Does This Solve?

LLM inference has two phases with opposite hardware profiles. **Prefill** processes the entire prompt in one forward pass — it's compute-bound, saturating GPU FLOPs with large matrix multiplies across many tokens. **Decode** generates one token at a time — it's memory-bandwidth bound, reading the full model weights for a single token's worth of computation.

Running both on the same GPU means neither is optimally served:

```
Same GPU (today's default):
  Time 0-50ms:   Prefill (1000 tokens) — GPU compute: 95%, memory BW: 40%
  Time 50-500ms: Decode (50 tokens)    — GPU compute: 5%,  memory BW: 95%
                                         ↑ GPU mostly idle during decode

Problem: a new prefill request arriving during decode
  must wait (stalls TTFT) — or interrupts decode (stalls ITL)
```

Disaggregated serving splits them into separate worker pools. Prefill workers handle prompt processing, then transfer the computed KV cache to decode workers that handle token generation. Each pool can be independently sized, scheduled, and even run on different hardware.

```
Disaggregated:
  Prefill Pool (compute-optimized GPUs):
    [GPU 0] [GPU 1]  ← handle all prompts, high FLOPs utilization
         │       │
         │  KV Transfer (RDMA/NVLink)
         ▼       ▼
  Decode Pool (bandwidth-optimized GPUs):
    [GPU 2] [GPU 3] [GPU 4]  ← handle all generation, high memory BW
```

This is the architecture behind systems like DistServe, Splitwise, and vLLM's disaggregated prefill feature.

---

## The Core Idea: Split by Hardware Profile

The fundamental insight is that prefill and decode want different things from the GPU:

```
┌──────────────┬─────────────────┬───────────────────┐
│              │     Prefill      │      Decode       │
├──────────────┼─────────────────┼───────────────────┤
│ Bottleneck   │ Compute (FLOPs) │ Memory bandwidth  │
│ Tokens/step  │ Many (prompt)   │ One               │
│ GPU util     │ High (80-100%)  │ Low (5-20%)       │
│ Latency goal │ Low TTFT        │ Low ITL           │
│ Batch size   │ Large OK        │ Large helps       │
│ Ideal GPU    │ High compute    │ High bandwidth    │
└──────────────┴─────────────────┴───────────────────┘
```

In a colocated system, these two workloads fight for the same resources. Long prefills cause **prefill stalls** — decode requests are blocked while the GPU processes a 4000-token prompt. This shows up as ITL spikes (inter-token latency jumps) that users perceive as the model "stuttering."

Disaggregation eliminates this interference entirely. Prefill workers never do decode. Decode workers never do prefill. Each pool is independently sized:

- **Prefill pool**: sized for peak prompt processing throughput. Can use fewer, more compute-dense GPUs.
- **Decode pool**: sized for the number of concurrent active sequences. Can use more GPUs optimized for memory bandwidth.

The cost is a KV cache transfer between pools — which, with modern interconnects, is surprisingly small.

---

## How It Works

### The Request Lifecycle

A request flows through three phases in disaggregated serving:

```
┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ Client  │───►│ API Router   │───►│ Prefill Pool │───►│ Response │
└─────────┘    └──────────────┘    └──────┬───────┘    └──────────┘
                                          │                  ▲
                                   KV Transfer               │
                                   (first token)             │
                                          │                  │
                                   ┌──────▼───────┐          │
                                   │ Decode Pool  │──────────┘
                                   └──────────────┘
```

**Phase 1 — Prefill:** A prefill worker receives the prompt, runs the full forward pass, and produces:
- The KV cache for all prompt tokens (the expensive part to compute)
- The first generated token (sampled from the last position's logits)

**Phase 2 — KV Transfer:** The prefill worker sends the KV cache to a decode worker. This is the key operation — moving cached attention state across processes or machines.

**Phase 3 — Decode:** A decode worker receives the KV cache, reconstructs its local cache, and continues autoregressive generation token by token until the sequence is complete.

### KV Transfer: The Critical Path

The KV cache contains the key and value projections for every layer and every prompt token. For a 70B model:

```
KV size per token per layer:
  K: [num_kv_heads × head_dim] = [8 × 128] = 1024 elements
  V: [num_kv_heads × head_dim] = [8 × 128] = 1024 elements
  Total: 2048 elements × 2 bytes (FP16) = 4 KB

Total KV for a 1000-token prompt:
  4 KB × 80 layers × 1000 tokens = 320 MB

Transfer time:
  At 100 GB/s (NVLink):       3.2 ms
  At 25 GB/s (InfiniBand):   12.8 ms
  At 12.5 GB/s (RDMA):       25.6 ms
  At 1 GB/s (TCP):          320.0 ms
```

With NVLink or InfiniBand, the transfer is a few milliseconds — negligible compared to the decode time of hundreds of milliseconds. With TCP, it becomes a bottleneck. This is why production disaggregated serving requires high-bandwidth interconnects.

### System Architecture: Two Independent Pools

Here's the complete disaggregated serving architecture with multiple workers per pool:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Router                                │
│     ┌──────────────────────────────────────────────────────┐     │
│     │  1. Receive request                                  │     │
│     │  2. Pick prefill worker (least loaded)               │     │
│     │  3. Prefill worker → computes KV → picks decode worker│    │
│     │  4. Decode worker → generates tokens → streams back  │     │
│     └──────────────────────────────────────────────────────┘     │
└────────────┬──────────────────────────────────┬─────────────────┘
             │                                  │
    ┌────────▼────────┐                ┌────────▼────────┐
    │  PREFILL POOL   │                │  DECODE POOL    │
    │  (compute-opt.) │                │  (bandwidth-opt.)│
    │                 │                │                  │
    │ ┌─────────────┐ │   KV Transfer  │ ┌──────────────┐ │
    │ │ P-Worker 0  │─┼───────────────►│ │ D-Worker 0   │ │
    │ │ GPU: H100   │ │   (NVLink/     │ │ GPU: H100    │ │
    │ │ High FLOPs  │ │    RDMA/       │ │ High BW      │ │
    │ └─────────────┘ │    Shared mem) │ └──────────────┘ │
    │ ┌─────────────┐ │                │ ┌──────────────┐ │
    │ │ P-Worker 1  │─┼───────────────►│ │ D-Worker 1   │ │
    │ └─────────────┘ │                │ └──────────────┘ │
    │ ┌─────────────┐ │                │ ┌──────────────┐ │
    │ │ P-Worker 2  │─┼───────────────►│ │ D-Worker 2   │ │
    │ └─────────────┘ │                │ └──────────────┘ │
    │                 │                │ ┌──────────────┐ │
    │  Scale for peak │                │ │ D-Worker 3   │ │
    │  prompt rate    │                │ └──────────────┘ │
    │                 │                │                  │
    │                 │                │  Scale for max   │
    │                 │                │  concurrent seqs │
    └─────────────────┘                └──────────────────┘
```

### The KV Connector Abstraction

The transport mechanism is abstracted behind a **KV Connector** interface. This is a pluggable component that handles the actual data movement:

```
KVConnector interface:
  send_kv(request_id, token_ids, kv_data, first_token)
  recv_kv(timeout) → {request_id, token_ids, kv_data, first_token}
  send_result(request_id, response, stats)
  recv_result(timeout) → {request_id, response, stats}
```

In production, the connector implementation depends on the hardware topology:

```
Same node, same GPU:    Shared GPU memory (fastest, no copy)
Same node, diff GPU:    NVLink P2P transfer
Same node, CPU:         Shared memory (mmap)
Cross-node:             RDMA / InfiniBand
Fallback:               TCP sockets
```

vLLM's `KVConnector` base class defines this interface, with `PyNcclConnector` and `MooncakeConnector` as concrete implementations.

### Orchestrating P and D Workers

The orchestrator (API router) manages the assignment of requests to workers:

```
API Router decisions:
  1. Which prefill worker gets this request?
     → Least loaded, or one with a prefix cache hit

  2. Which decode worker receives the KV?
     → Least active sequences, or best memory availability

  3. What if prefill finishes faster than decode can accept?
     → Buffer KV in CPU memory, apply backpressure
```

In a balanced system, the prefill pool and decode pool are sized so neither is the bottleneck. If prompts are long on average, you need more prefill workers. If generation is long, you need more decode workers.

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `KVConnector` | `KVConnectorBase` | N/A (not yet in SGLang) |
| `send_kv()` / `recv_kv()` | `send_kv_caches_and_hidden_states()` | — |
| `PrefillWorker` | `EngineCoreProc` (prefill role) | — |
| `DecodeWorker` | `EngineCoreProc` (decode role) | — |
| `DisaggregatedEngine` | `AsyncLLM` with P/D config | — |
| `multiprocessing.Queue` | `PyNcclConnector` / `MooncakeConnector` | — |
| Sequential P→D | Async pipeline | — |

Key details:

**vLLM's KV Connector system:** vLLM defines `KVConnectorBase` with methods for sending and receiving KV caches. The two main implementations are:
- **`PyNcclConnector`**: Uses NCCL for GPU-to-GPU transfer. Fast but requires GPUs to be in the same NCCL communicator group.
- **`MooncakeConnector`**: Uses the Mooncake transfer engine for cross-node RDMA transfers. Designed for disaggregated deployments where prefill and decode are on different machines.

**Role-based engine configuration:** In vLLM, the same `EngineCoreProc` code runs on both prefill and decode workers. The behavior changes based on configuration — prefill workers skip the decode loop, decode workers skip the prefill forward pass. This is simpler than maintaining two separate codebases.

**Async pipeline:** Our implementation runs prefill and decode sequentially (prefill finishes, then decode starts). Real systems pipeline this — while decode worker A is generating tokens, prefill worker B is already processing the next prompt. The orchestrator maintains queues of pending KV transfers.

**Hidden state transfer:** In addition to KV caches, vLLM transfers the sampled token and model hidden states. Some architectures need intermediate hidden states (not just KV) to continue generation correctly.

---

## The Implementation

The complete implementation is in [13_disaggregated_serving.py](../code-blog-simple-vllm/13_disaggregated_serving.py) (~440 lines).

### KV Connector

```python
class KVConnector:
    def __init__(self):
        self.kv_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()

    def send_kv(self, request_id, token_ids, kv_data, next_token):
        kv_cpu = [(k.cpu(), v.cpu()) for k, v in kv_data]
        self.kv_queue.put({
            "request_id": request_id,
            "token_ids": token_ids,
            "kv_data": kv_cpu,
            "next_token": next_token,
        })
```

The connector moves tensors to CPU before putting them on the queue — this simulates a cross-device transfer. In production, NCCL or RDMA would handle this without going through CPU.

### Prefill Worker

```python
class PrefillWorker:
    def process(self, request_id, prompt, temperature=0.0):
        token_ids = self.tokenizer.encode(prompt)
        outputs = self.model(input_ids=torch.tensor([token_ids]),
                            use_cache=True)
        next_token = self._sample(outputs.logits[0, -1, :], temperature)

        # Extract KV from DynamicCache
        kv_data = []
        past = outputs.past_key_values
        for layer_idx in range(len(past)):
            k, v = past[layer_idx]
            kv_data.append((k.squeeze(0), v.squeeze(0)))

        self.connector.send_kv(request_id, token_ids, kv_data, next_token)
```

The prefill worker runs a single forward pass, extracts the KV cache from the model's `past_key_values`, and sends it through the connector along with the first sampled token.

### Decode Worker

```python
class DecodeWorker:
    def process_one(self, max_tokens=64, temperature=0.0):
        msg = self.connector.recv_kv(timeout=30.0)

        # Reconstruct KV cache
        past = DynamicCache()
        for layer_idx, (k, v) in enumerate(msg["kv_data"]):
            past.update(k.unsqueeze(0), v.unsqueeze(0), layer_idx)

        # Decode loop
        gen_tokens = [msg["next_token"]]
        next_token = msg["next_token"]
        for _ in range(max_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                break
            out = self.model(input_ids=torch.tensor([[next_token]]),
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = self._sample(out.logits[0, -1, :], temperature)
            gen_tokens.append(next_token)
```

The decode worker reconstructs a `DynamicCache` from the received tensors and continues the standard autoregressive decode loop from where prefill left off.

---

## Running the Code

**Demo mode:**
```bash
python 13_disaggregated_serving.py --demo
```

**Server mode:**
```bash
python 13_disaggregated_serving.py --port 5000

curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 20}'
```

Expected demo output:
```
Request 1: "The capital of France is"
  Phase 1 (Prefill):  91ms (6 tokens)
  KV Transfer:        1ms
  KV Restore:         0ms
  Phase 2 (Decode):   4559ms (20 tokens)
  Response: "Paris. ..."

Latency breakdown:
  Prefill:    151.0ms (6%)
  Transfer:     0.1ms (0%)
  Restore:      0.7ms (0%)
  Decode:    2261.8ms (94%)
```

The KV transfer is under 1ms for this small model. For larger models with longer prompts, transfer time grows but remains small relative to decode time with fast interconnects.

---

## Benchmarks

| Metric | Colocated | Disaggregated (NVLink) | Disaggregated (InfiniBand) |
|--------|-----------|----------------------|--------------------------|
| TTFT | Varies (prefill + queue wait) | Low (dedicated prefill pool) | Low |
| ITL stability | Spikes during prefill | Stable (no prefill interference) | Stable |
| KV transfer overhead | None | ~3ms per 1K tokens (70B) | ~13ms per 1K tokens |
| GPU utilization (prefill) | Mixed | 80-100% (compute-saturated) | 80-100% |
| GPU utilization (decode) | Mixed | Higher (batched decode only) | Higher |
| Operational complexity | Simple | Higher (two pools, transfer infra) | Higher |

### When Disaggregation Helps

| Scenario | Benefit |
|----------|---------|
| Mixed prompt lengths (short + very long) | Eliminates prefill-decode interference |
| Strict ITL SLA (chatbots) | Stable decode latency, no prefill stalls |
| Different hardware available | Use compute GPUs for prefill, bandwidth GPUs for decode |
| High request rate with long prompts | Scale prefill and decode pools independently |

### When to Avoid

| Scenario | Reason |
|----------|--------|
| Low traffic | Overhead of two pools not justified |
| Uniform short prompts | Little interference to eliminate |
| TCP-only interconnect | KV transfer too slow (320ms for 1K tokens at 70B) |
| Single GPU deployment | No pool to split |

---

## Key Takeaways

1. **Prefill is compute-bound, decode is memory-bandwidth bound** — running both on the same GPU means neither workload is optimal
2. **Disaggregated serving** splits them into separate worker pools connected by KV cache transfer
3. **KV transfer cost** is small with fast interconnects (3ms for 1K tokens at NVLink speeds) but prohibitive over TCP
4. **The KV Connector** abstracts the transport — implementations range from shared memory to RDMA to NCCL
5. **Each pool scales independently** — add prefill workers for prompt throughput, decode workers for concurrent sequences
6. **ITL stability** is the main user-facing benefit — no more latency spikes from prefill interrupting decode

---

## Further Reading

- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670) — foundational paper on P/D disaggregation
- [Splitwise: Efficient Generative LLM Inference](https://arxiv.org/abs/2311.18677) — Microsoft's approach to phase splitting
- [vLLM Disaggregated Prefill](https://docs.vllm.ai/en/latest/serving/disagg_prefill.html) — production configuration
- [Mooncake: KV Cache-Centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079) — transfer engine used by vLLM
- **Next: [Blog 14 — Quantization](14_quantization.md)** — reduce model memory with lower precision
