---
title: "Part 5: Async Scheduling"
description: "How async scheduling overlaps CPU output processing with GPU execution, eliminating 5-10ms of dead GPU time per step."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 5
tags: ["async", "scheduling", "GPU optimization"]
---

## What Problem Does This Solve?

Blog 4's continuous batching engine has a step loop: `schedule() → execute() → process_outputs() → repeat`. Each phase runs sequentially. The problem: **the GPU sits idle while the CPU runs the scheduler.**

```
Blog 4 (Synchronous step loop):

CPU: [schedule][prepare]                    [process][schedule][prepare]
GPU:                    [████ execute ████]                            [████ execute ████]
                        ↑                    ↑       ↑                ↑
                        GPU working          GPU idle (waiting for CPU)

Scheduling + preparation takes 1-5ms per step.
At 100 steps/second, that's 100-500ms/s of GPU sitting idle.
Overhead: 10-30% of total time wasted.
```

This overhead grows with scale. At high QPS (queries per second) with large batches, the scheduler does more work: sorting request queues, walking block tables, allocating KV blocks, building input tensors. What starts as 1ms becomes 5ms, and 5ms per step × 200 steps/second = 1 second of idle GPU per second — 50% waste.

The fix: **run the CPU work concurrently with GPU execution.** While the GPU executes batch N, the CPU processes outputs from batch N-1 and schedules batch N+1. When the GPU finishes, the next batch is already prepared — submit it immediately with zero gap.

```
Blog 5 (Async scheduling):

CPU: [sched N][process N-1 + sched N+1][process N + sched N+2]
GPU:          [████ execute N █████████][████ execute N+1 ████]
              ↑                        ↑
              No idle gaps — GPU always busy
```

---

## The Core Idea: Pipeline the CPU and GPU

The key insight is that **scheduling batch N+1 doesn't depend on the GPU results of batch N.** The scheduler only needs to know:
- Which sequences finished (detected during output processing — a CPU operation)
- Which new requests arrived (from the API — a CPU operation)
- How many KV blocks are free (updated during output processing — a CPU operation)

None of these require waiting for the GPU to finish the current forward pass. So we can prepare the next batch while the GPU is still busy.

This is a classic **producer-consumer pipeline**:

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Scheduler   │────→│  Prepared Batch   │────→│  GPU         │
│  (CPU)       │     │  (double buffer)  │     │  (Execute)   │
│              │     │                   │     │              │
│ Runs during  │     │ Buffer A: current │     │ Reads from   │
│ GPU execute  │     │ Buffer B: next    │     │ current buf  │
└──────────────┘     └───────────────────┘     └──────────────┘
```

### How Real Systems Do This

**vLLM V1** runs the engine core in a **separate process** (`EngineCoreProc`). The API server and engine communicate via ZMQ sockets:

```
┌─────────────┐        ZMQ         ┌──────────────────┐
│  API Server │ ←───────────────→  │  EngineCoreProc  │
│  (FastAPI)  │                    │                   │
│  - Accept   │                    │  - Scheduler      │
│    requests │                    │  - GPU execution  │
│  - Stream   │                    │  - Output process │
│    tokens   │                    │                   │
└─────────────┘                    └──────────────────┘
                                   Runs in own process
                                   with its own event loop
```

Inside `EngineCoreProc`, the step loop overlaps CPU work with GPU work using `asyncio`. The GPU forward pass runs asynchronously, and when it completes, the CPU immediately processes outputs and schedules the next batch.

**SGLang** pioneered the "zero-overhead scheduler" approach. Their scheduler runs in a dedicated thread with a lock-free queue for incoming requests. The scheduling overhead is essentially zero because it always completes within the GPU execution time.

---

## How It Works

### The Synchronous Baseline (Blog 4)

In Blog 4's `step()`, everything runs sequentially:

```python
def step():
    batch = scheduler.schedule()         # CPU: ~0.01-5ms
    outputs = execute(batch)             # GPU: ~10-200ms
    process_outputs(outputs)             # CPU: ~0.01-1ms
```

Total step time = schedule + execute + process. The GPU is idle during schedule and process.

### The Async Pattern (Blog 5)

We reorganize the step so that CPU work from the previous step runs concurrently with GPU execution of the current step:

```python
def step():
    # Phase 1 (CPU): Process outputs from PREVIOUS step
    results = process_outputs(pending_outputs)
    
    # Phase 2 (CPU): Schedule CURRENT batch
    batch = scheduler.schedule()
    
    # Phase 3 (GPU): Execute current batch
    # On GPU: this runs on the GPU while CPU is free
    # pending_outputs saved for Phase 1 of NEXT step
    for seq in batch:
        token = execute(seq)
        pending_outputs.append((seq, token))
    
    return results  # results from PREVIOUS step
```

The crucial change: **output processing is deferred by one step.** The model generates tokens in step N, but those tokens are processed (appended to sequences, checked for EOS) at the start of step N+1. This decoupling is what enables the overlap.

```
Step 1: [schedule 1]  [execute 1]          ← no prior outputs to process
Step 2: [process 1 + schedule 2] [execute 2]  ← process outputs from step 1
Step 3: [process 2 + schedule 3] [execute 3]  ← process outputs from step 2
```

On GPU, Phases 1+2 (CPU) and Phase 3 (GPU) run truly concurrently. The scheduling time is hidden behind GPU execution.

### Detailed Waterfall: What Happens Each Step

Here's a detailed timing waterfall showing exactly what runs where across three consecutive steps:

```
                    Step N                Step N+1              Step N+2
                 ─────────────        ─────────────         ─────────────
CPU Thread:     │process(N-1) │      │process(N)   │      │process(N+1) │
                │schedule(N)  │      │schedule(N+1)│      │schedule(N+2)│
                │prepare(N)   │      │prepare(N+1) │      │prepare(N+2) │
                 ─────────────        ─────────────         ─────────────
                ↓ submit       ↓ submit             ↓ submit
GPU:            ╔═════════════╗╔═════════════╗╔═════════════╗
                ║ execute(N)  ║║ execute(N+1)║║ execute(N+2)║
                ║  forward()  ║║  forward()  ║║  forward()  ║
                ║  attention  ║║  attention  ║║  attention  ║
                ║  sampling   ║║  sampling   ║║  sampling   ║
                ╚═════════════╝╚═════════════╝╚═════════════╝
                        ↑                ↑               ↑
                   GPU finishes     GPU finishes     GPU finishes
                   outputs saved    outputs saved    outputs saved
                   for process(N)   for process(N+1) for process(N+2)

Key insight: "process(N)" runs during "execute(N+1)"
             → CPU work is completely hidden behind GPU execution
             → GPU never waits for CPU (as long as CPU finishes before GPU)

Timing (real GPU):
  CPU: schedule + process = 0.1-5ms
  GPU: execute = 10-200ms
  Overlap efficiency: CPU finishes 2-200x before GPU → zero gap
```

### Double Buffering

In production systems, this pattern is enhanced with **double buffering** — two sets of input tensors:

```
Step N:
  GPU reads from Buffer A (executing batch N)
  CPU writes to Buffer B (preparing batch N+1)
  
Step N+1:
  GPU reads from Buffer B (executing batch N+1)
  CPU writes to Buffer A (preparing batch N+2)
  
  → No copies, no synchronization, just swap pointers
```

vLLM's `InputBatch` maintains cached tensors that are updated incrementally — only changed entries are modified, not the entire batch. This reduces preparation time from O(batch_size) to O(num_changes).

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM V1 | Real SGLang |
|----------|-------------|-------------|
| `SyncEngine.step()` | V0 `LLMEngine.step()` (sync) | N/A (SGLang was always async) |
| `AsyncEngine.step()` | `EngineCoreProc.run_busy_loop()` | `Scheduler` in dedicated thread |
| Deferred output processing | `process_model_outputs()` before `schedule()` | `process_batch_result()` + `get_next_batch_to_run()` |
| `_pending_outputs` list | Model output queue | Batch result buffer |
| Threading (demo) | Separate process (`EngineCoreProc`) + ZMQ | Separate thread + shared memory |
| Sequential execution | CUDA streams for async GPU | CUDA streams + FlashInfer |

Key architectural details:

**vLLM V1's `run_busy_loop()`:** The engine core's main loop calls `process_model_outputs()` immediately after the GPU finishes, then calls `schedule()`, then submits the next batch. With CUDA graphs and async launching, the GPU starts the next batch before the CPU finishes scheduling.

**SGLang's zero-overhead scheduler:** SGLang processes the previous batch's results and schedules the next batch in the same "tick." Their tokenizer runs in a separate process, and the scheduler thread receives pre-tokenized requests via a pipe. This pipelining ensures the scheduler is never the bottleneck.

**Why separate processes, not just threads?** Python's GIL (Global Interpreter Lock) prevents true parallelism between threads. vLLM puts the engine in a separate process to get true CPU parallelism between the API server and the engine. SGLang uses a thread but relies on the GIL being released during GPU kernel execution (which PyTorch does).

---

## The Implementation

The complete implementation is in [05_async_scheduling.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/05_async_scheduling.py) (~600 lines). It contains both a `SyncEngine` (Blog 4 baseline) and an `AsyncEngine` (async pattern) for comparison.

### SyncEngine (Baseline)

```python
class SyncEngine:
    def step(self):
        batch = self.scheduler.schedule()      # CPU
        for seq, num_new in batch:
            if seq.is_prefill:
                token = self._execute_prefill(seq)
            else:
                token = self._execute_decode(seq)
            seq.output_token_ids.append(token)  # process immediately
            if finished:
                self.scheduler.finish_sequence(seq)
        return outputs
```

Schedule, execute, process — all in one sequential pass.

### AsyncEngine (Overlapped)

```python
class AsyncEngine:
    def step(self):
        # Phase 1 (CPU): Process outputs from PREVIOUS step
        results = self._process_outputs(self._pending_outputs)
        self._pending_outputs = []
        
        # Phase 2 (CPU): Schedule CURRENT batch
        batch = self.scheduler.schedule()
        
        # Phase 3 (GPU): Execute current batch
        for seq, num_new in batch:
            token = self._execute(seq)
            self._pending_outputs.append((seq, token))
        
        return results  # from previous step!
```

Output processing is decoupled from execution. On GPU, Phases 1+2 would run on CPU while the GPU executes the previous batch.

### Timing Breakdown

The demo measures each phase separately:

```
Avg schedule time: 0.01ms (per step)
Avg execute time:  693.6ms (per step, on CPU)
Avg process time:  0.13ms (per step)

CPU overhead (schedule + process) = 0.14ms
GPU execution = 693.6ms

On GPU: CPU overhead is 0.02% of execution time
        → 100% of scheduling hidden behind GPU work
```

---

## Running the Code

**Demo mode (compares sync vs async):**
```bash
python 05_async_scheduling.py --demo
```

**Server mode:**
```bash
python 05_async_scheduling.py --port 5000

# Streaming request:
curl -N -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "stream": true}'
```

Expected demo output:
```
--- Synchronous Scheduling (Blog 4 style) ---
  [SYNC]
  Steps: 15, Tokens: 46
  Total time: 10884ms
  Throughput: 4.2 tok/s

--- Async Scheduling (Blog 5, overlapped) ---
  [ASYNC]
  Steps: 16, Tokens: 46
  Total time: 10406ms
  Throughput: 4.4 tok/s
  Avg schedule time: 0.01ms (total: 0.2ms)
  Avg execute time: 693.6ms (total: 10403ms)
  Avg process time: 0.13ms (total: 1.9ms)
```

On CPU, the speedup is minimal because both scheduling and execution use the same CPU cores. On GPU, the speedup is much larger because CPU scheduling runs concurrently with GPU execution.

---

## Benchmarks

| Metric | Sync (Blog 4) | Async (Blog 5) | GPU Impact |
|--------|--------------|----------------|------------|
| Schedule overhead | Added to step time | Hidden behind execution | -10-30% step time |
| Process overhead | Added to step time | Hidden behind execution | -1-5% step time |
| GPU utilization | 70-90% | 95-100% | +10-30% throughput |
| Throughput (high QPS) | Limited by CPU overhead | Near GPU peak | 1.1-1.3x speedup |

The impact is most visible at high QPS with large batch sizes:

```
QPS=10, batch_size=4:
  Schedule: 0.5ms, Execute: 50ms → 1% overhead (barely matters)

QPS=1000, batch_size=256:
  Schedule: 5ms, Execute: 20ms → 25% overhead (significant!)
  With async: 0% overhead → 1.25x throughput gain
```

---

## Key Takeaways

1. **Sync scheduling** makes the GPU wait for CPU work (scheduling + output processing) — 10-30% overhead at scale
2. **Async scheduling** overlaps CPU work with GPU execution — the CPU schedules batch N+1 while the GPU runs batch N
3. **Deferred output processing** is the key enabler — tokens from step N are processed at the start of step N+1
4. vLLM V1 uses a **separate process** with ZMQ; SGLang uses a **dedicated scheduler thread** — both achieve zero-overhead scheduling
5. **Double buffering** eliminates memory contention between preparation and execution
6. On CPU the benefit is minimal; on GPU with high QPS it's a 10-30% throughput improvement

---

## Further Reading

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — zero-overhead scheduler
- [vLLM V1 architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) — EngineCoreProc design
- [CUDA Streams and Concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/) — GPU async execution
- **Next: [Blog 6 — Chunked Prefill](06_chunked_prefill.md)** — split long prompts into chunks so decode requests don't starve
