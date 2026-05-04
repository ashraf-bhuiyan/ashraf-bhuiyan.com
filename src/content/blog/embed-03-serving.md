---
title: "Serving Embedding Models in vLLM"
description: "Launch with --task embed, use the /v1/embeddings API, configure PoolingParams, and understand the internal pipeline."
pubDate: 2026-05-03
series: "Embeddings, Pooling & Rerankers in vLLM"
part: 3
tags: ["vLLM", "embeddings", "serving", "API"]
---

## What Problem Does This Solve?

You have an embedding model and need to serve it at scale — batch embed millions of documents for indexing, or serve real-time embedding requests for a search API. You could use Sentence-Transformers (simple but slow) or a dedicated embedding server (another service to manage). But if you're already running vLLM for text generation, you can serve embeddings from the same infrastructure.

vLLM handles embedding models as a first-class workload: the same optimizations (paged attention, continuous batching, tensor parallelism) work for embeddings, with the decode loop stripped out.

---

## The Embedding Pipeline vs. the Generation Pipeline

### Generation Pipeline (Standard vLLM)

```
Request → Tokenize → Prefill → [Decode → Sample → Token]×N → Response
                       │               │
                       │               └── repeat until done
                       └── fills the KV cache

Characteristics:
  - Variable number of forward passes (one per output token)
  - KV cache grows over time (decode adds tokens)
  - Output: sequence of tokens
```

### Embedding Pipeline

```
Request → Tokenize → Prefill → Pool → Response
                       │         │
                       │         └── one operation: collapse hidden states
                       └── fills the KV cache (but only used once)

Characteristics:
  - ONE forward pass per request (prefill only, no decode)
  - KV cache is temporary (freed immediately after pooling)
  - Output: one vector per request
```

The embedding pipeline is simpler: no decode loop, no sampling, no token-by-token streaming. Each request is a single forward pass through the transformer followed by pooling. This makes embedding requests more predictable in latency and more efficient in GPU utilization.

---

## Starting vLLM for Embeddings

### Basic Launch

```bash
# Serve a small encoder model
vllm serve BAAI/bge-large-en-v1.5 --task embed

# Serve a large decoder-only embedding model
vllm serve Alibaba-NLP/gte-Qwen2-7B-instruct --task embed

# Serve with tensor parallelism for models that don't fit on one GPU
vllm serve nvidia/NV-Embed-v2 --task embed --tensor-parallel-size 2
```

The `--task embed` flag tells vLLM to:
1. Skip the LM head (no logits, no sampling)
2. Run the `Pooler` after the last transformer layer
3. Expose the `/v1/embeddings` endpoint
4. Disable the `/v1/chat/completions` and `/v1/completions` endpoints

### Key Configuration Options

```bash
vllm serve BAAI/bge-large-en-v1.5 \
  --task embed \
  --max-model-len 512 \          # max input length (save memory)
  --max-num-seqs 256 \           # max concurrent requests in a batch
  --override-pooler-config '{"pooling_type": "MEAN"}'  # override if needed
```

| Option | Purpose | Default |
|--------|---------|---------|
| `--task embed` | Enable embedding mode | Required |
| `--max-model-len` | Maximum input token length | Model's max |
| `--max-num-seqs` | Max requests in a batch | 256 |
| `--override-pooler-config` | Override pooling type | Auto-detected |
| `--tensor-parallel-size` | TP for large models | 1 |

### Auto-Detection

vLLM auto-detects the correct pooling strategy from the model's configuration. For most popular embedding models, you don't need `--override-pooler-config`:

```
BAAI/bge-large-en-v1.5       → CLS pooling (auto-detected)
intfloat/e5-large-v2         → Mean pooling (auto-detected)
Alibaba-NLP/gte-Qwen2-7B    → Last-token pooling (auto-detected)
```

---

## The /v1/embeddings API

### Request Format

vLLM exposes an OpenAI-compatible embeddings endpoint:

```bash
# Single text
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": "How do I reset my password?"
  }'

# Batch of texts (recommended for throughput)
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": [
      "How do I reset my password?",
      "Steps to recover your account credentials",
      "Best pizza restaurants in NYC"
    ],
    "encoding_format": "float"
  }'
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0111, -0.0432, 0.0765, ...]
    },
    {
      "object": "embedding",
      "index": 2,
      "embedding": [-0.0345, 0.0567, -0.0123, ...]
    }
  ],
  "model": "BAAI/bge-large-en-v1.5",
  "usage": {
    "prompt_tokens": 24,
    "total_tokens": 24
  }
}
```

### Encoding Formats

```
"encoding_format": "float"    → JSON array of floats (default, verbose)
"encoding_format": "base64"   → base64-encoded binary (4× smaller response)
```

Use `base64` for production pipelines where you parse embeddings programmatically — the response is significantly smaller over the network.

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Single embedding
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input="How do I reset my password?",
)
embedding = response.data[0].embedding
print(f"Dimension: {len(embedding)}")  # 1024

# Batch embedding (much faster than single requests)
texts = [
    "How do I reset my password?",
    "Steps to recover your account credentials",
    "Best pizza restaurants in NYC",
]
response = client.embeddings.create(
    model="BAAI/bge-large-en-v1.5",
    input=texts,
)
embeddings = [d.embedding for d in response.data]
```

---

## PoolingParams

`PoolingParams` controls post-processing of the embedding:

```python
from vllm import LLM, PoolingParams

llm = LLM(model="BAAI/bge-large-en-v1.5", task="embed")

# Default: no normalization, full dimensions
params = PoolingParams()

# With L2 normalization
params = PoolingParams(normalize=True)

# Truncate dimensions (for Matryoshka models)
params = PoolingParams(dimensions=256)

outputs = llm.embed(["Hello world"], params=params)
```

### Matryoshka Dimension Truncation

For models that support Matryoshka representation learning (Blog B1), you can truncate the embedding to fewer dimensions:

```
Full embedding:  4096 dims × 4 bytes = 16 KB per vector
Truncated:       256 dims × 4 bytes  = 1 KB per vector

Quality tradeoff (GTE-Qwen2-7B, MTEB retrieval):
  4096 dims: 100% quality
  1024 dims: ~98% quality
  256 dims:  ~94% quality
  64 dims:   ~85% quality
```

Truncation happens after pooling — vLLM computes the full embedding and returns only the first N dimensions.

---

## How vLLM Handles Embeddings Internally

### The Internal Pipeline

```
1. Request arrives at /v1/embeddings
   │
2. Tokenizer converts text to token IDs
   │  "How do I reset my password?" → [2347, 567, 432, 8901, 678, 12345, 30]
   │
3. Scheduler adds request to the batch
   │  (same continuous batching as generation — requests join/leave freely)
   │
4. Forward pass (prefill only):
   │  input_ids → Embedding layer → Transformer layers → hidden_states
   │  [batch_size, seq_len, hidden_dim]
   │
5. Pooler:
   │  hidden_states → pooling (CLS/mean/last) → embeddings
   │  [batch_size, hidden_dim]
   │
6. (Optional) Normalize: embedding / ||embedding||₂
   │
7. Return to client
```

### What's Different from Generation

```
Generation:                        Embedding:
  LM head → logits → sample         Pooler → embedding → return
  → next token → decode loop         No decode loop!
  → KV cache grows                   KV cache freed immediately
  → streaming response               Single response (all at once)
```

### The Pooler Class

vLLM's `Pooler` in `vllm/model_executor/pooling_metadata.py`:

```python
class Pooler:
    def __init__(self, pooling_type, normalize, softmax):
        self.pooling_type = pooling_type
        self.normalize = normalize
    
    def forward(self, hidden_states, pooling_metadata):
        if self.pooling_type == PoolingType.CLS:
            # Take the first token's hidden state per sequence
            pooled = hidden_states[pooling_metadata.prompt_token_ids[:, 0]]
        
        elif self.pooling_type == PoolingType.LAST:
            # Take the last real token's hidden state per sequence
            pooled = hidden_states[pooling_metadata.last_token_indices]
        
        elif self.pooling_type == PoolingType.MEAN:
            # Masked mean over real tokens per sequence
            pooled = masked_mean(hidden_states, pooling_metadata.attention_mask)
        
        if self.normalize:
            pooled = F.normalize(pooled, dim=-1)
        
        return pooled
```

---

## Batch Embedding: Throughput and Latency

### Why Batching Matters

```
Single request (1 text, 50 tokens):
  GPU utilization: ~5%
  Throughput: ~500 embeddings/sec
  Latency: ~2ms

Batch of 64 texts (50 tokens each):
  GPU utilization: ~60%
  Throughput: ~15,000 embeddings/sec
  Latency: ~4ms per request

Batch of 256 texts:
  GPU utilization: ~85%
  Throughput: ~30,000 embeddings/sec
  Latency: ~9ms per request
```

Batching increases throughput dramatically because the GPU's compute units are utilized more efficiently. The per-request latency increases slightly, but the throughput-per-GPU improvement is enormous.

### Continuous Batching for Embeddings

vLLM's continuous batching works for embeddings too:

```
Time →

Step 1: [req_1(50 tok), req_2(120 tok), req_3(30 tok)]
  req_1 done → return embedding, free KV
  req_3 done → return embedding, free KV

Step 2: [req_2(120 tok), req_4(80 tok), req_5(60 tok)]
  req_2 done → return embedding, free KV
  req_5 done → return embedding, free KV

Step 3: [req_4(80 tok), req_6(200 tok)]
  ...

Requests with shorter inputs complete faster and make room
for new requests. Same principle as generation batching.
```

### Client-Side Batching

For maximum throughput, send batch requests with multiple texts:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")

async def embed_batch(texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = await client.embeddings.create(
            model="BAAI/bge-large-en-v1.5",
            input=batch,
        )
        embeddings.extend([d.embedding for d in response.data])
    return embeddings

# Embed 10,000 documents
all_embeddings = asyncio.run(embed_batch(documents))
```

---

## Instruction-Prefixed Embedding Models

Some embedding models require a task instruction prefix:

```
E5 models:
  Query:    "query: How do I reset my password?"
  Document: "passage: To reset your password, go to Settings..."

Instructor models:
  Query:    "Represent the question for retrieval: How do I reset my password?"
  Document: "Represent the document for retrieval: To reset your password..."

BGE models:
  Query:    "Represent this sentence for searching relevant passages: ..."
  Document: (no prefix)
```

When serving these models, the prefix must be included in the input text. vLLM doesn't add prefixes automatically — the client is responsible:

```python
# For E5 models
response = client.embeddings.create(
    model="intfloat/e5-large-v2",
    input=[
        "query: How do I reset my password?",
        "passage: To reset your password, go to Settings and click...",
    ],
)
```

Prefix caching (from Inference Blog 7) helps here: the shared instruction prefix is cached across requests, reducing prefill time. More on this in Blog B5.

---

## vLLM vs. Dedicated Embedding Servers

### When to Use vLLM

```
✓ You're already running vLLM for generation
  → Same infrastructure, same monitoring, same deployment pipeline
  → Add --task embed for a second model instance

✓ The embedding model is large (7B+ decoder-only)
  → vLLM's TP, continuous batching, and memory management shine
  → These models need GPU-optimized serving

✓ You need the same optimization stack as generation
  → Paged attention, prefix caching, continuous batching
  → All apply to embedding workloads
```

### When to Use a Dedicated Server (TEI, etc.)

```
✓ The model is small (BERT-class, <1B params)
  → vLLM's scheduler overhead is proportionally larger for small models
  → TEI is lighter-weight and optimized for BERT

✓ You need ONNX Runtime or TensorRT optimization
  → Dedicated servers integrate with these runtimes
  → Can provide 2-3× speedup for small models on specific hardware

✓ You want the simplest possible setup
  → Sentence-Transformers: 3 lines of Python
  → TEI: single Docker container
  → vLLM: more powerful but more configuration
```

### Decision Framework

```
Model size < 1B AND only embedding workload?  → TEI
Model size >= 1B?                             → vLLM
Already running vLLM?                         → vLLM (add another instance)
Need maximum throughput for BERT?             → TEI + ONNX/TensorRT
```

---

## Key Takeaways

1. **`--task embed`** tells vLLM to run in embedding mode — prefill only, no decode loop
2. **`/v1/embeddings`** is OpenAI-compatible — drop-in replacement for OpenAI's embedding API
3. **Continuous batching** works for embeddings — requests join and leave the batch freely
4. **Auto-detection**: vLLM detects the correct pooling strategy from model config
5. **Batch for throughput**: single requests waste GPU; batches of 64-256 achieve 30-60× higher throughput
6. **vLLM for large models**: for 7B+ decoder-only embedding models, vLLM provides the best serving stack

---

## What's Next

Embedding models (bi-encoders) are fast but sacrifice accuracy because query and document are encoded independently. **Blog B4** introduces rerankers (cross-encoders) — models that score query-document pairs jointly for higher accuracy.

---

## Further Reading

- [vLLM Embedding Documentation](https://docs.vllm.ai/en/latest/features/pooling.html)
- [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) — HuggingFace's dedicated embedding server
- [OpenAI Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings) — the API spec vLLM implements
- **Next: [Blog B4 — Reranker & Scoring Models](/blog/embed-04-rerankers/)** — cross-encoders for higher-accuracy ranking
