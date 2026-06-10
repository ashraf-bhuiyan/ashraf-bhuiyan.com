---
title: "Part 10: Data Parallelism"
description: "Independent model replicas with request routing strategies: round-robin, least-pending, and cache-aware routing for throughput scaling."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 10
tags: ["data parallelism", "load balancing", "scaling"]
---

## What Problem Does This Solve?

Tensor parallelism (Blog 9) lets you split one model across GPUs to fit larger models. But what if your model already fits on one GPU and you just need to serve more requests?

A single Llama 8B replica on an H100 might handle 100 requests/second. If you need 400 requests/second, you need 4 replicas:

```
Single replica (100 req/s):
  All requests → [GPU 0: Llama 8B] → responses
                     ↑ bottleneck

Data parallelism, DP=4 (400 req/s):
  ┌──────────────────┐
  │   Load Balancer   │
  └──┬───┬───┬───┬───┘
     ↓   ↓   ↓   ↓
  [GPU 0] [GPU 1] [GPU 2] [GPU 3]
  Llama8B Llama8B Llama8B Llama8B
  100r/s  100r/s  100r/s  100r/s
```

Each replica is a complete, independent copy of the model. They don't communicate with each other — no AllReduce, no synchronization. The load balancer just distributes requests across them.

---

## The Core Idea: Independent Replicas + Load Balancing

Data parallelism for inference is conceptually simple:

1. **Load the model onto N GPUs** — each GPU gets a complete copy
2. **Distribute requests** — a load balancer assigns each incoming request to a replica
3. **Process independently** — each replica generates tokens without talking to others
4. **Return results** — responses go back through the load balancer to the client

```
Client requests:  R1  R2  R3  R4  R5  R6  R7  R8

Load balancer (round-robin):
  GPU 0: R1, R3, R5, R7
  GPU 1: R2, R4, R6, R8

Each GPU processes its queue independently.
No communication between GPUs.
Throughput: ~2x single GPU.
```

### DP vs TP

| | Data Parallelism (DP) | Tensor Parallelism (TP) |
|---|---|---|
| **Purpose** | More throughput | Fit larger models |
| **Model on each GPU** | Full copy | 1/N of each layer |
| **Communication** | None between replicas | AllReduce every layer |
| **Memory per GPU** | Full model | 1/N of model |
| **When to use** | Model fits on 1 GPU | Model too large for 1 GPU |
| **Scaling** | Near-linear throughput | Sub-linear (comm overhead) |

### Engine Process Architecture

Each DP replica is a fully independent engine with its own model, scheduler, and KV cache. They share nothing:

```
┌─────────────────────────────────────────────────────────────┐
│                     API Server Process                       │
│  ┌────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ HTTP       │  │ Request Router │  │ Response Collector │  │
│  │ Listener   │─►│ (load balance) │  │ (gather results)  │  │
│  └────────────┘  └──┬────┬────┬──┘  └───────────────────┘  │
│                     │    │    │                              │
└─────────────────────┼────┼────┼──────────────────────────────┘
                      │    │    │
          ┌───────────┘    │    └───────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Engine 0     │  │ Engine 1     │  │ Engine 2     │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │ Scheduler│ │  │ │ Scheduler│ │  │ │ Scheduler│ │
│ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │
│ │ KV Cache │ │  │ │ KV Cache │ │  │ │ KV Cache │ │
│ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │
│ │ Model    │ │  │ │ Model    │ │  │ │ Model    │ │
│ │ (GPU 0)  │ │  │ │ (GPU 1)  │ │  │ │ (GPU 2)  │ │
│ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │
└──────────────┘  └──────────────┘  └──────────────┘
  No communication between engines — fully independent
```

### Load Balancing Strategies

**Round-robin:** Cycle through workers in order. Simple, works well when requests have similar latency.

```
Request 1 → Worker 0
Request 2 → Worker 1
Request 3 → Worker 2
Request 4 → Worker 0  (wrap around)
```

**Least-pending:** Send to the worker with the fewest in-flight requests. Better when request latency varies (long prompts vs short prompts).

```
Worker 0: 3 pending
Worker 1: 1 pending  ← next request goes here
Worker 2: 5 pending
```

**Cache-aware (SGLang):** Hash the prompt prefix and route to the worker most likely to have a cache hit. Requests with the same system prompt always go to the same replica, maximizing prefix cache reuse.

```
System prompt A → always Worker 0 (prefix cached)
System prompt B → always Worker 1 (prefix cached)
Random prompt   → least-pending fallback
```

### Routing Decision Flowchart

```
Incoming request
      │
      ▼
┌─────────────────┐    YES    ┌──────────────────────────┐
│ Cache-aware mode?│─────────►│ Hash prompt prefix       │
└────────┬────────┘           │ prefix_hash % num_workers│
         │ NO                 │ → target_worker          │
         ▼                    │                          │
┌─────────────────┐           │ Is target overloaded?    │
│ Strategy?       │           │ (pending > threshold)    │
├─────────────────┤           └───┬──────────────┬───────┘
│                 │               │ NO           │ YES
│ round_robin:    │               ▼              ▼
│   next = (counter++) % N        target      fall through
│                 │               worker      to least_pending
│ least_pending:  │                              │
│   min(pending_counts) ◄────────────────────────┘
│                 │
└─────────────────┘
```

---

## How It Works

### DP × TP Composition

With 8 GPUs, you choose how to divide them between DP and TP:

```
DP × TP = Total GPUs

DP=8, TP=1: 8 independent replicas (small model)
  [GPU0: full] [GPU1: full] ... [GPU7: full]
  Throughput: ~8x, Memory: full model per GPU

DP=4, TP=2: 4 replicas, each split across 2 GPUs
  [GPU0+1: half+half] [GPU2+3] [GPU4+5] [GPU6+7]
  Throughput: ~4x, Memory: half model per GPU

DP=2, TP=4: 2 replicas, each across 4 GPUs
  [GPU0+1+2+3: quarter each] [GPU4+5+6+7]
  Throughput: ~2x, Memory: quarter model per GPU

DP=1, TP=8: 1 replica across all 8 GPUs (huge model)
  [GPU0+1+2+3+4+5+6+7]
  Throughput: 1x, Memory: 1/8 model per GPU
```

The choice depends on the model size:
- **Llama 8B (16GB FP16):** Fits on 1 GPU → maximize DP (DP=8)
- **Llama 70B (140GB FP16):** Needs 2+ GPUs → DP=4, TP=2
- **Llama 405B (810GB FP16):** Needs 8+ GPUs → DP=1, TP=8

### How vLLM Handles DP

In vLLM V1, DP is implemented by running multiple engine processes. Each process is a complete vLLM engine with its own scheduler, KV cache, and model runner. A front-end router distributes requests:

```
┌─────────────────────────────────────────────────┐
│                   API Server                     │
│          (FastAPI + request router)              │
└──────────┬──────────┬──────────┬────────────────┘
           │          │          │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼─────┐
    │ Engine 0 │ │ Engine 1 │ │ Engine 2 │   DP=3, TP=2 each
    │ GPU 0+1  │ │ GPU 2+3  │ │ GPU 4+5  │
    │ Sched+KV │ │ Sched+KV │ │ Sched+KV │
    └──────────┘ └──────────┘ └──────────┘
```

Each engine is independent — it has its own KV cache, its own block allocator, its own scheduler. The API server's router picks which engine gets each request.

### The GIL Problem

In Python, the Global Interpreter Lock (GIL) prevents true CPU parallelism between threads. This is why production systems use separate **processes**, not threads, for DP replicas:

```
Threads (limited by GIL):
  Thread 0: [Python ██████] [GPU ▓▓▓] [Python ██████]
  Thread 1: [wait...........] [Python ██████] [GPU ▓▓▓]
  ↑ Only one thread runs Python at a time

Processes (true parallelism):
  Process 0: [Python ██████] [GPU ▓▓▓] [Python ██████]
  Process 1: [Python ██████] [GPU ▓▓▓] [Python ██████]
  ↑ Both run Python simultaneously
```

vLLM uses Ray or multiprocessing to spawn separate engine processes, each with its own Python interpreter.

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `InferenceWorker` | `EngineCoreProc` (separate process) | Engine process |
| `LoadBalancer` | `DPRequestRouter` | `DataParallelController` |
| `round_robin` | Round-robin router | Configurable strategy |
| `least_pending` | Load-aware routing | Cache-aware routing |
| Threading | Ray / multiprocessing | multiprocessing |
| N/A | DP coordination for spec decode | DP group NCCL |

Key details:

**vLLM's DP coordination:** Even though DP replicas are independent, they need minimal coordination for speculative decoding. All DP ranks must agree on whether to run the drafter in a given step, otherwise collective operations inside the drafter (if using TP within each DP group) will deadlock.

**SGLang's cache-aware routing:** SGLang hashes the first few tokens of each request to determine which DP replica to route to. This ensures requests with the same prefix (e.g., same system prompt) hit the same replica, maximizing prefix cache hits. The hash is consistent — same tokens always route to the same replica.

---

## The Implementation

The complete implementation is in [10_data_parallelism.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/10_data_parallelism.py) (~300 lines).

### Inference Worker

```python
class InferenceWorker:
    def __init__(self, worker_id, model_name, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(device)
        self.device = device

    def generate(self, prompt, max_tokens=64):
        # Standard autoregressive generation on self.device
        ...
```

### Load Balancer

```python
class LoadBalancer:
    def __init__(self, workers, strategy="round_robin"):
        self.workers = workers
        self._rr_counter = 0
        self._pending = {i: 0 for i in range(len(workers))}

    def select_worker(self):
        if self.strategy == "round_robin":
            idx = self._rr_counter % len(self.workers)
            self._rr_counter += 1
        elif self.strategy == "least_pending":
            idx = min(self._pending, key=self._pending.get)
        self._pending[idx] += 1
        return self.workers[idx]
```

### Data-Parallel Engine

```python
class DataParallelEngine:
    def __init__(self, model_name, dp_degree):
        self.workers = []
        for i in range(dp_degree):
            device = torch.device(f"cuda:{i}")
            worker = InferenceWorker(i, model_name, device)
            self.workers.append(worker)
        self.lb = LoadBalancer(self.workers)
```

---

## Running the Code

**Demo mode:**
```bash
python 10_data_parallelism.py --demo --dp 4
```

**Server mode:**
```bash
python 10_data_parallelism.py --port 5000 --dp 4 --strategy least_pending

# Send requests:
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 20}'

# Check worker stats:
curl http://localhost:5000/health
```

Expected demo output:
```
DP degree: 4
GPUs: 4x NVIDIA H100 80GB HBM3

--- Sequential (1 worker) ---
  8 requests, 1 GPU
  Total time: 2.1s, Throughput: 75.1 tok/s

--- Concurrent (4 workers) ---
  8 requests, 4 GPUs
  Request distribution:
    Request 0 → Worker 0 (cuda:0)
    Request 1 → Worker 1 (cuda:1)
    Request 2 → Worker 2 (cuda:2)
    Request 3 → Worker 3 (cuda:3)
    Request 4 → Worker 0 (cuda:0)
    ...
```

---

## Benchmarks

| Metric | DP=1 | DP=2 | DP=4 | DP=8 |
|--------|------|------|------|------|
| Max throughput | 1x | ~2x | ~4x | ~8x |
| Memory per GPU | 100% | 100% | 100% | 100% |
| Communication | None | None | None | None |
| Latency | Baseline | Same | Same | Same |

DP scales throughput **near-linearly** because replicas are independent. The only overhead is the load balancer (negligible) and model loading time (one-time cost at startup).

### When to Use DP vs TP

```
Model fits on 1 GPU?
  YES → Use DP only (maximize replicas)
  NO  → Must use TP (minimum TP = ceil(model_size / GPU_memory))
        Then fill remaining GPUs with DP replicas

Example: 8 H100s (80GB each), Llama 70B (140GB FP16)
  Minimum TP = ceil(140/80) = 2
  DP = 8 / 2 = 4 replicas
  → DP=4, TP=2: 4 replicas each split across 2 GPUs
```

---

## Key Takeaways

1. **Data parallelism** replicates the full model on each GPU — no communication between replicas
2. **Throughput scales linearly** with the number of replicas (unlike TP which has AllReduce overhead)
3. **Load balancing** distributes requests: round-robin for uniform load, least-pending for variable load, cache-aware for prefix reuse
4. **DP × TP = total GPUs** — use TP for model size, DP for throughput
5. Production systems use **separate processes** (not threads) to avoid GIL contention
6. SGLang's **cache-aware routing** hashes prompt prefixes to maximize cache hits per replica

---

## Further Reading

- [vLLM data parallelism](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) — `--data-parallel-size` configuration
- [SGLang Data Parallelism](https://docs.sglang.ai/) — cache-aware routing design
- **Next: [Blog 11 — Expert Parallelism](11_expert_parallelism.md)** — split Mixture-of-Experts across GPUs
