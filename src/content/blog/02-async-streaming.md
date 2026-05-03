---
title: "Part 2: Async Streaming with FastAPI"
description: "Why Flask blocks on long generation, how FastAPI enables true async, and Server-Sent Events for token-by-token streaming."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 2
tags: ["FastAPI", "async", "streaming", "SSE"]
---

## What Problem Does This Solve?

In Blog 1 we built a Flask server that generates text. It works, but it has two critical problems:

1. **The client waits for the entire response.** If generating 100 tokens takes 24 seconds, the client sees nothing for 24 seconds, then gets all 100 tokens at once. ChatGPT and Claude don't work this way — they stream tokens as they're generated, giving instant feedback.

2. **The server is completely blocked during generation.** While one request is generating tokens, the server can't respond to health checks, accept new requests, or do anything else. A load balancer pinging `/health` would think the server is dead.

This blog fixes both problems by replacing Flask with **FastAPI** (an async server) and adding **Server-Sent Events** (SSE) for token streaming.

---

## The Core Idea: Don't Wait, Stream

When you send a message to ChatGPT, you see tokens appear one at a time in the browser. This isn't just a UI trick — the server is actually sending each token as soon as it's generated, using a protocol called **Server-Sent Events (SSE)**.

```
Blog 1 (Flask, blocking):
  Client ──POST──→ Server
  Client            [....generating 100 tokens for 24 seconds....]
  Client ←─────────────────────────────────────────── full response

Blog 2 (FastAPI, streaming):
  Client ──POST──→ Server
  Client ←─ "The"                    (240ms)
  Client ←─ " answer"               (480ms)
  Client ←─ " is"                   (720ms)
  Client ←─ " 4"                    (960ms)
  Client ←─ [DONE]                  (1200ms)
```

The user sees the first token in ~350ms (prefill + first decode step) instead of waiting 24 seconds for everything. Same total generation time, dramatically better user experience.

---

## How It Works

### WSGI vs ASGI: Why Flask Can't Stream

Flask uses **WSGI** (Web Server Gateway Interface), a synchronous protocol from 2003. In WSGI, a request handler is a function that takes a request and returns a response. The function must complete before the response is sent. There's no mechanism to "send part of the response now, more later."

FastAPI uses **ASGI** (Asynchronous Server Gateway Interface), which supports:
- **Async/await**: handlers can pause (await) without blocking the server thread
- **Streaming responses**: send data in chunks as it becomes available
- **Concurrent requests**: multiple requests can be in-flight simultaneously

```
WSGI (Flask):                          ASGI (FastAPI):
┌──────────────────────┐               ┌──────────────────────┐
│ Thread 1:            │               │ Event Loop:          │
│  Req A: generate()   │               │  Req A: generate()   │
│  [blocked 24s]       │               │  Req A: yield token  │
│  Req B: waiting...   │               │  Req B: /health ✓    │ ← handled!
│  Health: waiting...  │               │  Req A: yield token  │
│                      │               │  Req A: yield token  │
│  Req A: done         │               │  Req C: /generate    │ ← accepted!
│  Req B: now starts   │               │  Req A: yield token  │
│  Health: now responds │               │  Req A: done         │
└──────────────────────┘               └──────────────────────┘
```

### Server-Sent Events (SSE)

SSE is a simple protocol where the server pushes data to the client over a single HTTP connection. Each message is a line starting with `data: ` followed by a newline:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"token": "The", "token_id": 450}

data: {"token": " answer", "token_id": 1234}

data: {"token": " is", "token_id": 338}

data: {"token": " 4", "token_id": 29946}

data: [DONE]
```

The client reads these lines as they arrive. In a browser, you'd use `EventSource`; with `curl`, use the `-N` flag (no buffering).

### Streaming Request Lifecycle

Here's the full lifecycle of a streaming request from HTTP arrival to the final token:

```
Client                      FastAPI Server                    Engine
  │                              │                              │
  │── POST /generate ──────────►│                              │
  │   {stream: true}             │                              │
  │                              │── create StreamingResponse ──│
  │                              │                              │
  │◄─ HTTP 200 ─────────────────│                              │
  │   Content-Type:              │                              │
  │   text/event-stream          │                              │
  │                              │── run_in_executor ──────────►│
  │                              │   next(generator)            │── prefill ──┐
  │                              │                              │             │
  │                              │◄─ yield {token: "The"} ─────│◄────────────┘
  │◄─ data: {"token":"The"} ────│                              │
  │                              │                              │
  │   (event loop handles        │── run_in_executor ──────────►│
  │    /health, other reqs)      │   next(generator)            │── decode ───┐
  │                              │                              │             │
  │                              │◄─ yield {token:" answer"} ──│◄────────────┘
  │◄─ data: {"token":" answer"}─│                              │
  │                              │                              │
  │   ... repeat per token ...   │                              │
  │                              │                              │
  │◄─ data: [DONE] ────────────│                              │
  │   connection closed          │                              │
```

### The run_in_executor Pattern

There's a tension: FastAPI is async (non-blocking), but our inference engine is synchronous (blocks the CPU for hundreds of milliseconds per token). If we call `engine.generate()` directly in an async handler, it blocks the event loop and no other requests can be handled.

The solution is `run_in_executor`: run the blocking inference in a **thread pool** while keeping the event loop free:

```
Event Loop (main thread):          Thread Pool:
  accept request A                   
  submit generate() to pool ──→     generate() starts
  handle /health ✓                   [computing token 1...]
  handle /health ✓                   [computing token 2...]
  ...                                [computing token N...]
  pool returns result ←──────────    generate() done
  send response to A
```

For streaming, we wrap each `next(generator)` call in `run_in_executor` so each token yield returns control to the event loop:

```python
async def stream_tokens(prompt, max_tokens, temperature):
    loop = asyncio.get_event_loop()
    gen = engine.generate_stream(prompt, max_tokens, temperature)

    while True:
        try:
            chunk = await loop.run_in_executor(None, next, gen)
            yield f"data: {json.dumps(chunk)}\n\n"
        except StopIteration:
            yield "data: [DONE]\n\n"
            break
```

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Real SGLang |
|----------|-----------|-------------|
| `generate_stream()` (sync generator) | `LLMEngine.generate()` (async) | `generate_request()` |
| `StreamingResponse` + SSE | OpenAI-compatible streaming | SSE streaming endpoint |
| `run_in_executor` (thread pool) | Separate engine process (`EngineCoreProc`) | Scheduler in separate thread |
| Pydantic `GenerateRequest` | OpenAI `ChatCompletionRequest` | OpenAI-compatible models |
| `uvicorn.run()` | Gunicorn + uvicorn workers | Uvicorn with custom router |
| Auto-generated `/docs` | OpenAI API spec | OpenAI-compatible `/v1/chat/completions` |

Key architectural difference: vLLM V1 runs the engine in a **separate process** (not just a thread pool), communicating via ZMQ. This fully isolates the event loop from inference. SGLang similarly runs the scheduler in its own thread. We'll build this architecture in Blog 5 (Async Scheduling).

---

## The Implementation

The complete implementation is in [02_async_streaming.py](../code-blog-simple-vllm/02_async_streaming.py) (~200 lines). Key additions over Blog 1:

### Pydantic Request Validation

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False       # NEW: controls streaming vs batch
```

FastAPI automatically validates the request body against this schema. Invalid requests get a 422 error with details — no manual validation code needed. FastAPI also auto-generates API docs at `/docs`.

### The Streaming Generator

```python
def generate_stream(self, prompt, max_tokens, temperature):
    # ... prefill ...
    yield {"token": token_text, "token_id": next_token, "prefill_time_ms": ...}

    for i in range(max_tokens - 1):
        # ... one decode step ...
        yield {"token": token_text, "token_id": next_token, "step_ms": ...}
```

Instead of collecting all tokens and returning at the end, `generate_stream` **yields** each token as soon as it's generated. Python's `yield` turns this into a generator — it pauses after each yield, waiting for the consumer to request the next token.

### The Async Endpoint

```python
@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    if req.stream:
        return StreamingResponse(
            stream_tokens(req.prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.generate, ...)
        return GenerateResponse(**result)
```

One endpoint, two modes. When `stream=true`, FastAPI sends an SSE stream. When `stream=false`, it runs inference in the thread pool and returns the complete response.

---

## Running the Code

**Demo mode:**
```bash
python 02_async_streaming.py --demo
```

**Server mode:**
```bash
python 02_async_streaming.py --port 5000

# Non-streaming:
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'

# Streaming:
curl -N -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "stream": true}'

# Auto-generated API docs:
open http://localhost:5000/docs
```

---

## Benchmarks

Streaming doesn't change total generation time, but it dramatically changes perceived latency:

| Metric | Non-Streaming | Streaming |
|--------|--------------|-----------|
| Time to first visible token | ~24,000ms (all at once) | ~350ms |
| Total generation time | ~24,000ms | ~24,000ms |
| Server responsiveness | Blocked during generation | Health checks still work |
| Client can cancel early? | No | Yes (close connection) |

The key insight: **streaming doesn't make generation faster — it makes the user experience faster.** The first token appears in milliseconds instead of seconds, and the user can start reading while the rest generates.

---

## Key Takeaways

1. **Flask (WSGI) blocks** on every request — no streaming, no concurrent handling
2. **FastAPI (ASGI)** supports async handlers, streaming responses, and concurrent requests
3. **SSE (Server-Sent Events)** streams tokens one at a time using the `data: {...}\n\n` format
4. **run_in_executor** runs blocking inference in a thread pool so the event loop stays free
5. Streaming doesn't speed up generation — it changes when the user sees the first token

---

## Further Reading

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Server-Sent Events spec](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [uvicorn ASGI server](https://www.uvicorn.org/)
- **Next: [Blog 3 — Paged Attention](03_paged_attention.md)** — manage KV cache memory like an OS manages RAM
