---
title: "Embedding Models 101: Turning Text into Vectors"
description: "Bi-encoders, contrastive learning, similarity metrics, the model landscape, and decoder-only embedding trends."
pubDate: 2026-05-03
series: "Embeddings, Pooling & Rerankers in vLLM"
part: 1
tags: ["embeddings", "bi-encoder", "contrastive learning", "similarity"]
---

## What Problem Does This Solve?

You have a million documents and a user query: "How do I reset my password?" You need to find the 10 most relevant documents — fast. Keyword search fails when the documents say "credential recovery" instead of "reset password." You need a system that understands *meaning*, not just words.

**Embedding models** solve this. They convert text into fixed-size numerical vectors where semantically similar texts have similar vectors. Instead of matching keywords, you compare vector distances.

```
Embedding model maps text → vector:

  "How do I reset my password?"     → [0.12, -0.34, 0.56, ..., 0.78]  (768 dims)
  "Steps to recover your password"  → [0.11, -0.32, 0.55, ..., 0.76]  (768 dims)
  "Best pizza restaurants in NYC"   → [-0.45, 0.67, -0.12, ..., 0.33] (768 dims)

  cosine_similarity(query, doc1) = 0.97  ← very similar (relevant!)
  cosine_similarity(query, doc2) = 0.12  ← very different (irrelevant)
```

The embedding model captures that "reset password" and "recover password" mean the same thing, even though they share zero keywords. This is the foundation of **semantic search**.

---

## Why Embeddings Matter

Embeddings aren't just for search. They're the backbone of a surprisingly large number of ML systems:

### Semantic Search

Traditional keyword search (BM25, TF-IDF) matches on token overlap. Embedding-based search matches on meaning:

```
Query: "cardiac arrest treatment"

Keyword search finds:
  ✓ "Treatment protocols for cardiac arrest"    (keyword match)
  ✗ "How to manage sudden heart failure"        (no keyword overlap!)

Embedding search finds:
  ✓ "Treatment protocols for cardiac arrest"
  ✓ "How to manage sudden heart failure"         (semantically similar)
  ✓ "Emergency CPR and defibrillation guidelines" (related topic)
```

### Retrieval-Augmented Generation (RAG)

The dominant pattern for grounding LLMs with external knowledge:

```
1. User asks: "What's our refund policy for enterprise customers?"
2. Embed the question → query vector
3. Search your document store for similar vectors → retrieve top-5 docs
4. Feed the retrieved docs + question to an LLM → generate answer

Without RAG: LLM hallucinates a policy
With RAG:    LLM cites the actual policy document
```

Embeddings are the retrieval backbone of every RAG system.

### Other Applications

```
Clustering:       Embed all support tickets → cluster by topic → find trends
Classification:   Embed text → feed vector to a classifier → categorize
Deduplication:    Embed all documents → find near-duplicate pairs → deduplicate
Anomaly detection: Embed logs → flag vectors far from the cluster center
Recommendation:   Embed user query + product descriptions → find nearest products
```

---

## Bi-Encoder Architecture

### How Embedding Models Work

An embedding model is a transformer encoder that processes text and outputs a single vector:

```
Input text: "How do I reset my password?"

Step 1: Tokenize
  ["How", "do", "I", "reset", "my", "password", "?"]

Step 2: Transformer encoder
  Each token → hidden state of size hidden_dim
  [h₀, h₁, h₂, h₃, h₄, h₅, h₆]    (7 vectors, each 768-dim)

Step 3: Pooling (covered in Blog B2)
  Collapse 7 vectors into 1 vector
  → [0.12, -0.34, 0.56, ..., 0.78]   (1 vector, 768-dim)

Step 4: (Optional) Normalization
  L2-normalize the vector to unit length
  → embedding on the unit sphere
```

### The "Bi" in Bi-Encoder

A bi-encoder encodes the query and document **independently** through the same model:

```
           ┌──────────────┐
  Query ──►│  Transformer │──► query_embedding    ─┐
           │  Encoder     │                         ├── cosine_sim → score
  Doc   ──►│  (same model)│──► doc_embedding      ─┘
           └──────────────┘

The query and document NEVER see each other during encoding.
This is the key property that enables scalability.
```

Why "bi"? Because the encoder is used twice — once for the query, once for the document — but these two uses are completely independent. This means:

**Document embeddings can be precomputed and stored.** Encode your million documents once (offline), store the vectors in a database. At query time, only the query needs to be encoded — a single forward pass through the model.

```
Offline (once):
  Document 1 → embed → store [0.12, -0.34, ...]
  Document 2 → embed → store [0.45, 0.11, ...]
  ...
  Document 1,000,000 → embed → store [...]

Online (per query):
  Query → embed → [0.12, -0.33, ...]
  → ANN search against 1M stored vectors
  → Top-10 results in ~5ms
```

This is what makes embedding search scale to millions of documents with millisecond latency.

---

## Training: Contrastive Learning

### The Training Data

Embedding models are trained on (query, positive_document, negative_documents) tuples:

```
Training example:
  query:    "How to make pasta"
  positive: "Boil water, add pasta, cook for 8 minutes, drain"  ← relevant
  negative: "The stock market closed higher today"               ← irrelevant
  negative: "How to repair a bicycle tire"                       ← irrelevant
  negative: "Italian history during the Renaissance"             ← tricky negative!
```

The last negative is a "hard negative" — topically related (Italian) but not actually about making pasta. Hard negatives are critical for training high-quality embeddings.

### The InfoNCE Loss

The standard contrastive loss function:

```
L = -log( exp(sim(q, d⁺) / τ) / Σᵢ exp(sim(q, dᵢ) / τ) )

Where:
  q:     query embedding
  d⁺:    positive document embedding
  dᵢ:    all documents (positive + negatives) in the batch
  sim:   similarity function (usually cosine similarity)
  τ:     temperature parameter (controls sharpness)
```

In plain language: maximize the similarity between the query and the positive document, relative to all negatives. The loss pushes the positive closer in embedding space and pushes negatives further away.

### Multi-Stage Training

Modern embedding models use a multi-stage training pipeline:

```
Stage 1: Weak supervision (large scale)
  Data: title-body pairs from web pages, question-answer pairs
  Scale: 100M+ pairs
  Goal:  learn general text understanding

Stage 2: Fine-tuning (curated)
  Data: manually labeled (query, relevant_doc) pairs
  Scale: 100K-1M pairs
  Goal:  learn task-specific similarity

Stage 3: Hard negative mining
  Data: use the model from Stage 2 to find hard negatives
  Scale: same data, harder negatives
  Goal:  improve discrimination between similar-but-different texts

Each stage produces a better model by training on harder examples.
```

---

## Similarity Metrics

### Cosine Similarity

The most common metric for text embeddings:

```
cosine(a, b) = (a · b) / (||a|| × ||b||)

Range: [-1, 1]
  1.0:  identical direction (most similar)
  0.0:  orthogonal (unrelated)
 -1.0:  opposite direction (most dissimilar)

Properties:
  - Invariant to vector magnitude (only direction matters)
  - Most embedding models are trained to optimize this metric
  - After L2 normalization: cosine(a, b) = a · b (dot product)
```

### Dot Product

```
dot(a, b) = Σ aᵢ × bᵢ

Range: (-∞, +∞)
Properties:
  - Faster to compute (no normalization)
  - Sensitive to vector magnitude
  - For L2-normalized vectors: dot product = cosine similarity
  - Some models (e.g., those using Matryoshka training) work with dot product
```

### Euclidean Distance

```
euclidean(a, b) = √(Σ (aᵢ - bᵢ)²)

Range: [0, +∞)
Properties:
  - Smaller = more similar (opposite convention to similarity)
  - Used in some clustering algorithms (k-means)
  - Less common for text retrieval
```

### Which to Use?

**Use the metric the model was trained with.** Almost always cosine similarity for text embedding models. Check the model card on HuggingFace — it specifies the recommended metric.

```
If you use cosine-trained embeddings with dot product:
  → Rankings might differ (magnitude affects results)
  → But if vectors are L2-normalized, they're equivalent

If you use cosine-trained embeddings with Euclidean:
  → Works but suboptimal (different ranking than intended)
```

---

## The Embedding Model Landscape

### Encoder-Only Models (Traditional)

These are BERT-style models designed specifically for embeddings:

```
Model Family    Params    Dims    Context   Notes
──────────────────────────────────────────────────────────────
BGE (BAAI)      33-326M   768     512-8K    Strong general-purpose
E5 (Microsoft)  33-335M   768     512       Multiple sizes, well-tested
GTE (Alibaba)   33-137M   768     512-8K    Good multilingual
Nomic Embed     137M      768     8,192     Long context, open weights
Jina Embeddings 33-137M   768     8,192     Domain-specific variants
Instructor      335M      768     512       Task-prefixed (add instruction)
```

### Decoder-Only Models (Recent Trend)

The big shift in embedding models: use decoder-only LLMs (Llama, Mistral, Qwen) as embedding models:

```
Model Family       Params    Dims    Context   Notes
────────────────────────────────────────────────────────────────────
GTE-Qwen2          1.5-7B    1536    32K       Qwen2 backbone, strong
E5-Mistral         7B        4096    32K       Mistral backbone
NV-Embed v2        7B        4096    32K       NVIDIA, SOTA on MTEB
SFR-Embedding-2    7B        4096    32K       Salesforce
Linq-Embed-Mistral 7B        4096    32K       Mistral backbone
```

### Why Use a 7B Model for Embeddings?

```
BERT-large (335M):
  + Fast inference (~5ms per query)
  + Runs on CPU
  - 512-token context limit
  - Limited understanding of complex queries

GTE-Qwen2-7B:
  + 32K context (process entire documents)
  + Deeper language understanding
  + Better on complex, nuanced queries
  - Slower inference (~50ms per query on GPU)
  - Requires GPU
```

Decoder-only embedding models are 7-20B parameters — the same size as generative LLMs. They benefit from all the same vLLM optimizations: paged attention, continuous batching, tensor parallelism. This is why serving them with vLLM makes sense.

### MTEB: The Standard Benchmark

The [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard) ranks embedding models across tasks:

```
MTEB leaderboard (simplified):

Rank  Model                    Avg Score   Params
────────────────────────────────────────────────────
1     NV-Embed-v2              72.3        7B
2     SFR-Embedding-2          71.8        7B
3     GTE-Qwen2-7B-instruct    70.2        7B
4     E5-Mistral-7B-instruct   66.6        7B
...
15    BGE-large-en-v1.5        64.2        335M
18    E5-large-v2              62.3        335M

Trend: larger models consistently score higher,
but the gap narrows for specific tasks.
```

---

## Embedding Dimensions and Storage

### How Much Space Do Embeddings Take?

```
Per embedding (FP32):
  768 dims × 4 bytes  = 3,072 bytes ≈ 3 KB
  1536 dims × 4 bytes = 6,144 bytes ≈ 6 KB
  4096 dims × 4 bytes = 16,384 bytes ≈ 16 KB

For 1 million documents:
  768-dim:   3 GB
  1536-dim:  6 GB
  4096-dim: 16 GB

For 100 million documents:
  768-dim:  300 GB
  4096-dim: 1.6 TB
```

### Matryoshka Embeddings

Some models support **Matryoshka representation learning** — you can truncate the embedding to fewer dimensions with minimal quality loss:

```
Full embedding:   4096 dims → 100% quality, 16 KB per vector
Truncated to 1024: first 1024 dims → 98% quality, 4 KB per vector
Truncated to 256:  first 256 dims → 94% quality, 1 KB per vector

The first dimensions carry the most information, like principal components.
This works because models are specifically trained to front-load information.
```

---

## Key Takeaways

1. **Embedding models** convert text to fixed-size vectors where similar texts have similar vectors
2. **Bi-encoders** encode query and document independently — documents can be precomputed and stored, enabling millisecond search at million-doc scale
3. **Contrastive learning** (InfoNCE loss) trains embeddings by pushing positives together and negatives apart
4. **Cosine similarity** is the standard metric — always use what the model was trained with
5. **Decoder-only LLMs** (7B+) are increasingly used as embedding models, achieving state-of-the-art results
6. These large embedding models are exactly the workload vLLM is built for — Blog B3 shows how to serve them

---

## What's Next

An embedding model produces one hidden state per token, but we need one vector per input. **Blog B2** covers pooling strategies — the step that collapses token-level representations into a single embedding.

---

## Further Reading

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — the standard embedding model benchmark
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — the paper that popularized bi-encoder embeddings
- [E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)
- [GTE-Qwen2 Technical Report](https://arxiv.org/abs/2308.03281) — decoder-only embedding model
- **Next: [Blog B2 — Pooling Strategies](/blog/embed-02-pooling/)** — how to go from per-token hidden states to a single embedding
