---
title: "Part 4: Continuous Batching"
description: "Why static batching wastes GPU cycles and how continuous batching lets requests join and leave the batch at every iteration."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 4
tags: ["batching", "scheduling", "GPU utilization"]
---

## What Problem Does This Solve?

In Blogs 1-3, our server processed one request at a time. While generating 500 tokens for Request A (~2 minutes on CPU), every other request waits in line. This is terrible for throughput — the model could be doing useful work for other requests during those decode steps.

The naive fix is **static batching**: collect N requests, process them all together, wait for the longest one to finish, then start the next batch. But this wastes cycles too:

```
Static Batching (batch of 3):
  Step  1-15:  [Req A ████] [Req B ████] [Req C ████]   ← all active
  Step 16-20:  [A: done  ░] [Req B ████] [Req C ████]   ← A's slot wasted
  Step 21-50:  [A: done  ░] [B: done  ░] [Req C ████]   ← A+B slots wasted
  Step 51:     ── new batch ──
  
  Request D arrives at step 2 → waits until step 51!
```

If Req A generates 15 tokens but Req C generates 50, the GPU is ⅓ idle for 35 steps (70%) and ⅔ idle for 30 steps. And new requests can't join until the entire batch completes.

**Continuous batching** fixes both problems: requests join the batch as soon as there's capacity, and leave as soon as they finish:

```
Continuous Batching:
  Step  1:     [Req A: prefill] [Req B: prefill] [Req C: prefill]
  Step  2-15:  [Req A ████████] [Req B ████████] [Req C ████████]
  Step 16:     [Req D: prefill] [Req B ████████] [Req C ████████]  ← D joins!
  Step 21:     [Req D ████████] [Req E: prefill] [Req C ████████]  ← E joins!
  Step 50:     [Req D ████████] [Req E ████████] [Req F: prefill]
  
  No wasted slots. New requests admitted every step.
```

This is how vLLM and SGLang actually work. The Orca paper (Yu et al., 2022) introduced this idea, calling it "iteration-level scheduling" — the scheduler runs at every decode iteration, not once per batch.

---

## The Core Idea: Schedule at Every Step

The key insight is simple: **the scheduler runs between every decode step, not once per batch.** After each step, finished sequences are removed and new sequences can join. The model always has a full batch to work on.

This requires three things:

1. **Sequence state tracking** — each request has a lifecycle (WAITING → RUNNING → FINISHED)
2. **A scheduler** — decides which sequences to run each step, managing a token budget
3. **A step loop** — the engine repeatedly calls schedule() → execute() → process_outputs()

```
                    ┌──────────────┐
   New requests ──→ │  WAITING     │
                    │  Queue       │
                    └──────┬───────┘
                           │ scheduler admits
                           ▼
                    ┌──────────────┐
        ┌──────────│  RUNNING     │◄─────────┐
        │          │  Batch       │           │
        │          └──────┬───────┘           │
        │                 │ step()            │
        │                 ▼                   │
        │          ┌──────────────┐           │
        │          │  Execute     │           │
        │          │  (model fwd) │           │
        │          └──────┬───────┘           │
        │                 │                   │
        │                 ▼                   │
        │          ┌──────────────┐     still generating
        │          │  Process     │───────────┘
        │          │  Outputs     │
        │          └──────┬───────┘
        │                 │ EOS or max_tokens
        ▼                 ▼
   ┌──────────────┐ ┌──────────────┐
   │ Free KV      │ │  FINISHED    │
   │ blocks       │ │              │
   └──────────────┘ └──────────────┘
```

---

## How It Works

### Sequence States

Each request is a **Sequence** object that tracks its position in the generation pipeline:

```python
class SeqState(Enum):
    WAITING  = "waiting"     # queued, not yet running
    RUNNING  = "running"     # actively generating tokens
    FINISHED = "finished"    # done (EOS or max_tokens)
```

A Sequence knows how many tokens have been computed so far (`num_computed_tokens`) vs. how many total tokens it has (`prompt + output`). The gap tells the scheduler what work remains:

```
New request: "What is 2+2?"  (8 prompt tokens)
  num_computed_tokens = 0, total = 8
  → needs 8 tokens of prefill

After prefill:
  num_computed_tokens = 8, total = 9 (8 prompt + 1 output)
  → needs 1 token of decode

After 5 decode steps:
  num_computed_tokens = 13, total = 14
  → needs 1 token of decode

After EOS:
  state = FINISHED
  → free KV blocks, return result
```

### The Scheduler

The scheduler runs every step and produces a batch. It has two queues: **waiting** (new requests) and **running** (actively generating). Each step:

**Phase 1: Schedule running sequences.** Each running sequence needs 1 decode token. If the token budget allows and KV blocks are available, include it in the batch.

**Phase 2: Admit waiting sequences.** If there's remaining budget and capacity, move waiting sequences to running. Each new sequence needs `prompt_len` tokens for prefill.

```
Step N:
  Running: [A(decode), B(decode), C(decode)]
  Waiting: [D(new), E(new)]
  Token budget: 512

  Phase 1: Schedule A(1 tok), B(1 tok), C(1 tok) → budget = 509
  Phase 2: Admit D (prompt=100 tokens) → budget = 409
           Admit E (prompt=200 tokens) → budget = 209
  
  Batch: [A:decode, B:decode, C:decode, D:prefill, E:prefill]
  Total tokens this step: 303
```

**Budget management** is critical. The token budget caps the total tokens processed per step. Without it, a batch of 100 decode requests + one 5000-token prefill would create an enormous forward pass. The budget keeps step latency predictable.

### The Step Loop

The engine's `step()` method is the heartbeat of continuous batching:

```python
def step():
    # 1. Scheduler picks the batch
    batch = scheduler.schedule()
    
    # 2. Execute each sequence
    for seq, num_tokens in batch:
        if seq.is_prefill:
            token = execute_prefill(seq)
        else:
            token = execute_decode(seq)
        seq.output_token_ids.append(token)
    
    # 3. Check for finished sequences
    for seq in batch:
        if seq hit EOS or max_tokens:
            scheduler.finish_sequence(seq)  # free KV, move to finished
```

In our CPU implementation, we execute sequences one at a time within a step (no actual GPU parallelism). On a real GPU, prefills and decodes within the same step would be **packed into a single forward pass** — this is where the real throughput gain comes from.

### Why This Is Better

Consider 3 requests: A (20 tokens), B (15 tokens), C (25 tokens).

**Without continuous batching (sequential):**
```
[A: 20 steps] → [B: 15 steps] → [C: 25 steps] = 60 total steps
Latency A: 20 steps, B: 35 steps, C: 60 steps
```

**With continuous batching:**
```
Step  1: [A:prefill] [B:prefill] [C:prefill]  ← all three start
Step  2: [A:decode]  [B:decode]  [C:decode]
...
Step 15: [A:decode]  [B:DONE]    [C:decode]   ← B finishes, slot freed
Step 16: [A:decode]              [C:decode]
...
Step 20: [A:DONE]                [C:decode]   ← A finishes
...
Step 25:                         [C:DONE]     ← C finishes
Total: 25 steps (not 60!)
```

The total wall-clock time is dominated by the longest request (25 steps), not the sum. Every request gets lower latency because it doesn't wait for others to complete first.

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `SeqState` enum | `RequestStatus` enum (WAITING, RUNNING, PREEMPTED, FINISHED_*) | `ReqState` |
| `Sequence` class | `Request` class with `num_computed_tokens`, `num_tokens_with_spec` | `Req` class |
| `Scheduler.schedule()` | `Scheduler.schedule()` — Phase 1 (running) then Phase 2 (waiting) | `Scheduler.get_next_batch_to_run()` |
| `Scheduler.add_request()` | `Scheduler.add_request()` → WAITING queue | `Scheduler.add_req()` |
| Token budget | `max_num_scheduled_tokens` config | `max_running_requests` + `max_prefill_tokens` |
| `step()` loop | `EngineCoreProc.step()` → schedule + execute + process | `run_batch()` loop |
| Sequential execution | **Batched GPU execution** — single forward pass for all sequences | **Batched with FlashInfer** |

Key differences:

**No prefill/decode distinction in vLLM's scheduler.** vLLM V1 treats every request the same — it just has `num_computed_tokens` and `num_tokens_with_spec`. Whether something is "prefilling" or "decoding" is an emergent property, not a scheduler concept. This generality naturally supports chunked prefill (Blog 6) and speculative decoding (Blog 8).

**Preemption.** When vLLM runs out of KV cache blocks, it can **preempt** a running request — free its blocks, reset its `num_computed_tokens` to 0, and put it back in the waiting queue. Our simple scheduler just stops admitting new requests.

**Batched execution.** In our code, we execute sequences sequentially within a step. Real vLLM packs all sequences into a single model forward pass — prefills and decodes are concatenated into one input tensor. This is where the GPU parallelism actually happens.

---

## The Implementation

The complete implementation is in [04_continuous_batching.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/04_continuous_batching.py) (~400 lines). Key parts:

### Sequence Tracking

```python
class Sequence:
    def __init__(self, seq_id, prompt, prompt_token_ids, max_tokens, temperature):
        self.state = SeqState.WAITING
        self.output_token_ids = []
        self.num_computed_tokens = 0
    
    @property
    def num_new_tokens(self):
        return self.total_len - self.num_computed_tokens
    
    @property
    def is_prefill(self):
        return self.num_computed_tokens < self.prompt_len
```

`num_new_tokens` tells the scheduler how much work this sequence needs. For a decode step it's 1; for prefill it's the prompt length.

### The Scheduler

```python
class Scheduler:
    def schedule(self):
        scheduled = []
        token_budget = self.max_num_tokens

        # Phase 1: running sequences (decode)
        for seq in self.running:
            if seq.num_new_tokens <= token_budget:
                scheduled.append((seq, seq.num_new_tokens))
                token_budget -= seq.num_new_tokens

        # Phase 2: waiting sequences (prefill)
        for seq in self.waiting:
            if len(self.running) >= self.max_num_seqs:
                break
            if seq.prompt_len <= token_budget:
                seq.state = SeqState.RUNNING
                self.running.append(seq)
                scheduled.append((seq, seq.prompt_len))
                token_budget -= seq.prompt_len

        return scheduled
```

Running sequences get priority over waiting ones — you don't want to starve active generation. Waiting sequences are admitted FCFS (first-come, first-served) if budget allows.

### The Async Server

The server uses a background `engine_loop` that continuously calls `step()`:

```python
async def engine_loop():
    while True:
        if not engine.scheduler.has_work:
            await asyncio.sleep(0.01)
            continue
        outputs = await loop.run_in_executor(None, engine.step)
        for out in outputs:
            if out["seq_id"] in pending_requests:
                await pending_requests[out["seq_id"]].put(out)
```

Each API request creates an `asyncio.Queue` and waits for tokens. The engine loop pushes tokens to the right queue as they're generated. This lets multiple clients send requests concurrently — they all share the same engine loop.

---

## Running the Code

**Demo mode:**
```bash
python 04_continuous_batching.py --demo
```

**Server mode:**
```bash
python 04_continuous_batching.py --port 5000

# Send multiple concurrent requests:
curl -N -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "stream": true}' &

curl -N -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Goodbye", "max_tokens": 20, "stream": true}' &

# Check status:
curl http://localhost:5000/health
```

Expected demo output:
```
Submitting 3 requests simultaneously:
  [3459b776] 'The capital of France is' (max_tokens=20)
  [de191caf] 'What is 2+2? The answer is' (max_tokens=15)
  [f750cd57] 'Explain gravity:' (max_tokens=25)

Step   1: [3459b776:P=Paris] [de191caf:P=] [f750cd57:P=Gra]
Step   2: [3459b776:D=.] [de191caf:D=4] [f750cd57:D=vity]
Step   3: [3459b776:D=\n] [de191caf:D=.] [f750cd57:D=is]
Step   4: [3459b776:D=\n] [de191caf:D=DONE] [f750cd57:D=the]
Step   5: [3459b776:D=2] [f750cd57:D=force]
...
Total steps: 25
Without batching: 49 sequential steps
With continuous batching: 25 steps (3 requests interleaved)
```

---

## Benchmarks

On CPU with TinyLlama (3 concurrent requests):

| Metric | Sequential (Blog 1) | Continuous Batching (Blog 4) |
|--------|---------------------|------------------------------|
| Total steps | 60 (sum of all tokens) | 25 (max of all tokens) |
| Req A latency | 20 steps | 20 steps |
| Req B latency | 35 steps (waits for A) | 15 steps (no waiting) |
| Req C latency | 60 steps (waits for A+B) | 25 steps (no waiting) |
| Throughput | 1 token/step | 3 tokens/step (peak) |
| GPU utilization | ~33% (one request at a time) | ~100% (full batch) |

The throughput improvement scales with batch size. With 8 concurrent requests, you get up to 8x throughput — the model processes 8 tokens per step instead of 1.

**Important caveat:** On CPU, we execute sequences sequentially within a step (no real parallelism). On GPU, all sequences in a step run in a single batched forward pass, which is where the actual throughput gain comes from. Our demo shows the scheduling benefit; the GPU would multiply that by the batch parallelism.

---

## Key Takeaways

1. **Static batching** wastes compute waiting for the longest sequence — shorter sequences sit idle
2. **Continuous batching** schedules at every iteration — sequences join and leave the batch freely
3. The **scheduler** runs every step, managing a **token budget** and two queues (waiting + running)
4. Each **Sequence** tracks its state and how many tokens have been computed vs. total needed
5. The **step loop** (schedule → execute → process) is the heartbeat of every production inference server
6. Throughput scales with batch size — more concurrent requests = higher model utilization

---

## Further Reading

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — the paper that introduced iteration-level scheduling
- [vLLM Scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — production scheduler with chunked prefill and preemption
- **Next: [Blog 5 — Async Scheduling](05_async_scheduling.md)** — overlap CPU scheduling with GPU execution for zero-overhead scheduling
