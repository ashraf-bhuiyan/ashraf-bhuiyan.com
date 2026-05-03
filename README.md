# ashraf-bhuiyan.com

Personal website and technical blog — [ashraf-bhuiyan.com](https://ashraf-bhuiyan.com)

Built with [Astro](https://astro.build), deployed via GitHub Actions to GitHub Pages.

---

## Building an LLM Inference Engine from Scratch

A 15-part series teaching how production inference servers like [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) work by building one from scratch. Each post isolates one technique with a runnable Python implementation.

### Series Architecture

```
                          ┌──────────────────────┐
                          │   Blog 1             │
                          │   Naive Inference    │
                          │   ─ foundation ─     │
                          └──────────┬───────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
 ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
 │   INDEPENDENT       │   │   DEPENDENCY CHAIN  │   │   PARALLELISM       │
 │   (toggle on/off)   │   │                     │   │   (pick per HW)     │
 │                     │   │   3 Paged Attention  │   │                     │
 │   2  Async Stream   │   │          │           │   │   9  Tensor ∥       │
 │   7  Prefix Cache   │   │          ▼           │   │  10  Data ∥         │
 │   8  Spec Decode    │   │   4 Cont. Batching   │   │  11  Expert ∥       │
 │  12  CPU Offload    │   │       ┌──┴──┐        │   │                     │
 │  14  Quantization   │   │       ▼     ▼        │   │  DP × TP × EP      │
 │                     │   │    5 Async  6 Chunked│   │   = total GPUs      │
 │                     │   │    Sched    Prefill  │   │                     │
 └─────────┬───────────┘   └─────────┬───────────┘   └─────────┬───────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Blog 13            │
                          │   Disaggregated P/D  │
                          │   (system arch)      │
                          └──────────┬───────────┘
                                     │
                          ╔══════════▼═══════════╗
                          ║   Blog 15            ║
                          ║   Full Architecture  ║
                          ║   (all combined)     ║
                          ╚══════════════════════╝
```

### Posts

| # | Title | Key Concept |
|---|-------|-------------|
| 01 | The Simplest LLM Server | Autoregressive generation, KV cache, prefill vs decode |
| 02 | Async Streaming with FastAPI | Non-blocking I/O, SSE streaming |
| 03 | Paged Attention | Virtual memory for KV cache, block tables |
| 04 | Continuous Batching | Dynamic batch formation, iteration-level scheduling |
| 05 | Async Scheduling | Decoupled scheduler and model executor |
| 06 | Chunked Prefill | Breaking long prefills into chunks |
| 07 | Prefix Caching | Reusing KV cache across requests |
| 08 | Speculative Decoding | Draft model acceleration |
| 09 | Tensor Parallelism | Splitting model across GPUs |
| 10 | Data Parallelism | Replicating model for throughput |
| 11 | Expert Parallelism | MoE model distribution |
| 12 | KV Cache CPU Offloading | Spilling KV cache to host memory |
| 13 | Disaggregated Prefill-Decode | Separating prefill and decode phases |
| 14 | Quantization | Reduced precision inference |
| 15 | The Full Architecture | Combining all techniques |

---

## Development

```bash
npm install          # Install dependencies
npm run dev          # Dev server at localhost:4321
npm run build        # Production build to ./dist/
```

## Deployment

Push to `main` triggers GitHub Actions (`.github/workflows/deploy.yml`):

```
git push → GitHub Actions → astro build → GitHub Pages → ashraf-bhuiyan.com
```

## Adding a New Blog Post

Create `src/content/blog/NN-slug.md`:

```markdown
---
title: "Part N: Title"
description: "One-line description"
pubDate: 2026-MM-DD
series: "Building an LLM Inference Engine from Scratch"
part: N
tags: ["tag1", "tag2"]
---

Content here...
```

Then `git add`, `git commit`, `git push`. The homepage auto-groups posts by `series`.

## Adding a New Series

Use a different `series` value in the frontmatter — the homepage creates a new section automatically.
