---
title: "Rerankers and Cross-Encoders in vLLM"
description: "Cross-encoder scoring, --task score, the /v1/score API, retrieve-then-rerank pattern, and reward models."
pubDate: 2026-05-03
series: "Embeddings, Pooling & Rerankers in vLLM"
part: 4
tags: ["reranker", "cross-encoder", "RAG", "scoring"]
---

## What Problem Does This Solve?

Bi-encoder embedding models (Blogs B1-B3) encode query and document independently. This makes them fast — you can precompute document embeddings — but it sacrifices accuracy. The query embedding is computed without ever "seeing" the document, and vice versa.

**Cross-encoders** (rerankers) take a different approach: they encode the query AND document **together**, allowing the model to see interactions between the two texts. This is more accurate but more expensive, because you can't precompute anything — every (query, document) pair requires a forward pass.

```
Bi-encoder (fast, approximate):
  embed("apple stock price")     → q_vec
  embed("Apple releases iPhone") → d_vec
  cosine(q_vec, d_vec) = 0.72    ← high score (both mention "apple")
  
  Problem: model can't distinguish "apple stock" from "Apple iPhone"
  because query and document are encoded separately.

Cross-encoder (slow, accurate):
  score("apple stock price", "Apple releases iPhone") = 0.12
  ← low score! The model sees both texts and understands the mismatch.

  score("apple stock price", "AAPL closes at $185, up 2%") = 0.95
  ← high score! The model connects "apple stock" with "AAPL closes."
```

---

## Bi-Encoder vs. Cross-Encoder

### Architecture Comparison

```
Bi-encoder:
  Query  ──► [Encoder] ──► q_embedding ─┐
                                          ├── cosine(q, d) → score
  Doc    ──► [Encoder] ──► d_embedding ─┘
  
  Two independent forward passes. No interaction between query and doc.

Cross-encoder:
  [Query] [SEP] [Doc] ──► [Encoder] ──► [CLS hidden state] ──► Linear ──► score
  
  ONE forward pass. Query and document tokens attend to each other.
```

### Why Cross-Encoders Are More Accurate

The difference is **cross-attention**. In a cross-encoder, every query token can attend to every document token:

```
Cross-encoder attention for "apple stock price" + "Apple releases iPhone":

  "apple"  attends to "Apple", "releases", "iPhone"
  "stock"  attends to "Apple", "releases", "iPhone"
  "price"  attends to "Apple", "releases", "iPhone"

  The model sees: "stock" and "price" have no match in the document.
  "Apple" in the doc is about the company, but the context is
  consumer electronics, not finance.
  
  Result: low relevance score.
```

A bi-encoder can't do this — the query embedding is computed before the document is even seen. It can only capture that "apple" appears in both texts.

### The Speed-Accuracy Tradeoff

```
                    Speed              Accuracy
                    ─────              ────────
Bi-encoder:         1M docs/sec        Good (NDCG ~0.45)
                    (precomputed docs)

Cross-encoder:      ~100 pairs/sec     Great (NDCG ~0.55)
                    (must score each pair)

Cross-encoder can't search 1M docs — it would take hours.
Bi-encoder can't match cross-encoder quality.
```

---

## The Retrieve-Then-Rerank Pattern

The standard solution: use both.

```
Stage 1: RETRIEVAL (bi-encoder, fast)
  Query: "apple stock price"
  → Embed query
  → ANN search in 1M document embeddings
  → Return top-100 candidates (5ms)

Stage 2: RERANKING (cross-encoder, accurate)
  Score each of the 100 candidates with the cross-encoder:
    score("apple stock price", doc_1) = 0.92
    score("apple stock price", doc_2) = 0.15
    score("apple stock price", doc_3) = 0.87
    ...
  → Re-sort by cross-encoder score
  → Return top-10 (500ms)

Result: near-cross-encoder quality at near-bi-encoder speed.
```

Why this works:
- Stage 1 filters 1M docs to 100 candidates — fast, approximate
- Stage 2 accurately ranks the 100 candidates — slow, but only 100 pairs
- The bi-encoder's job is recall (don't miss relevant docs); the cross-encoder's job is precision (rank them correctly)

### Pipeline Diagram

```
User Query
    │
    ▼
┌─────────────────┐     ┌───────────────────┐
│  Bi-encoder     │     │  Vector Database   │
│  (vLLM embed)   │────►│  (FAISS, Milvus)  │
│                 │     │  1M doc embeddings │
└─────────────────┘     └────────┬──────────┘
                                 │ top-100
                                 ▼
                        ┌─────────────────┐
                        │  Cross-encoder  │
                        │  (vLLM score)   │
                        │  Score 100 pairs│
                        └────────┬────────┘
                                 │ top-10
                                 ▼
                        ┌─────────────────┐
                        │  LLM Generator  │
                        │  (vLLM generate)│
                        │  Answer + docs  │
                        └─────────────────┘
```

All three components can run on vLLM — embeddings (`--task embed`), reranking (`--task score`), and generation (default).

---

## Cross-Encoder Architecture

### Input Construction

The cross-encoder concatenates query and document with a separator:

```
For BERT-based cross-encoders:
  Input: [CLS] query tokens [SEP] document tokens [SEP]
  
  Example:
  [CLS] apple stock price [SEP] AAPL closes at $185, up 2% today [SEP]

For decoder-only cross-encoders:
  Input: query tokens [SEP] document tokens
  (pooling on last token, similar to decoder-only embeddings)
```

### Scoring Head

After the transformer processes the concatenated input:

```
[CLS] hidden state → Linear layer → scalar score
  (hidden_dim)      (hidden_dim → 1)  (1 value)

The linear layer is a learned classification head.
It maps the [CLS] representation to a relevance score.
```

### Training

```
Training data: (query, document, relevance_label) tuples
  - Binary: relevant (1) or not relevant (0)
  - Graded: relevance on a scale (0, 1, 2, 3)

Loss function:
  - Binary cross-entropy for binary labels
  - Margin-based ranking loss for graded labels
  - Contrastive loss (similar to bi-encoder training)

The model learns to output high scores for relevant pairs
and low scores for irrelevant pairs.
```

### Score Interpretation

```
Cross-encoder scores are logits, NOT probabilities:
  score = 4.2   ← high relevance
  score = -1.3  ← low relevance
  score = 0.1   ← borderline

Scores are relative, not absolute:
  - Compare scores within a single query's candidates
  - Don't compare scores across different queries
  - "4.2 for query A" doesn't mean the same as "4.2 for query B"

To convert to probabilities: sigmoid(score)
  sigmoid(4.2)  = 0.985 (98.5% relevant)
  sigmoid(-1.3) = 0.214 (21.4% relevant)
```

---

## Serving Rerankers in vLLM

### Starting the Server

```bash
# BERT-based reranker
vllm serve BAAI/bge-reranker-v2-m3 --task score

# Large cross-encoder
vllm serve BAAI/bge-reranker-v2-gemma --task score
```

### The /v1/score API

```bash
# Score a single pair
curl http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": "What is the capital of France?",
    "text_2": "Paris is the capital and most populous city of France."
  }'

# Score multiple pairs (batch)
curl http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": "What is the capital of France?",
    "text_2": [
      "Paris is the capital and most populous city of France.",
      "France is a country in Western Europe.",
      "The Eiffel Tower is in Paris.",
      "Berlin is the capital of Germany."
    ]
  }'
```

### Response Format

```json
{
  "id": "score-abc123",
  "object": "list",
  "data": [
    {"index": 0, "object": "score", "score": 0.956},
    {"index": 1, "object": "score", "score": 0.312},
    {"index": 2, "object": "score", "score": 0.728},
    {"index": 3, "object": "score", "score": 0.021}
  ],
  "model": "BAAI/bge-reranker-v2-m3",
  "usage": {"prompt_tokens": 89, "total_tokens": 89}
}
```

### Python Client

```python
import requests

def rerank(query, documents, model="BAAI/bge-reranker-v2-m3"):
    response = requests.post(
        "http://localhost:8000/v1/score",
        json={
            "model": model,
            "text_1": query,
            "text_2": documents,
        }
    )
    scores = response.json()["data"]
    
    # Sort by score descending
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1]["score"],
        reverse=True,
    )
    return ranked

# Rerank candidates
query = "How do I reset my password?"
candidates = [
    "To change your email, go to Settings > Email.",
    "To reset your password, click 'Forgot Password' on the login page.",
    "Contact support for account recovery assistance.",
    "Our password policy requires 12+ characters.",
]

ranked = rerank(query, candidates)
for doc, score_data in ranked:
    print(f"  {score_data['score']:.3f}  {doc}")

# Output:
#   0.952  To reset your password, click 'Forgot Password' on the login page.
#   0.734  Contact support for account recovery assistance.
#   0.289  Our password policy requires 12+ characters.
#   0.043  To change your email, go to Settings > Email.
```

---

## Beyond Reranking: Other Scoring Models

vLLM's `--task score` isn't limited to rerankers. Any model that takes one or two inputs and produces a scalar score works:

### Reward Models

Used in RLHF to score (prompt, response) pairs for alignment quality:

```bash
vllm serve OpenAssistant/reward-model-deberta-v3-large --task score
```

```json
{
  "text_1": "Explain quantum computing simply.",
  "text_2": "Quantum computing uses qubits that can be 0 and 1 simultaneously, enabling parallel computation on certain problems."
}
// Response: {"score": 0.85}  ← good response
```

### Classification Models

Single-input models that classify text:

```bash
vllm serve cardiffnlp/twitter-roberta-base-sentiment --task classify
```

Classification uses `--task classify` instead of `--task score`, returning class probabilities instead of a single score.

### Sentence Similarity

Score the similarity between two sentences (different from reranking — symmetric, not query-document):

```json
{
  "text_1": "The cat sat on the mat",
  "text_2": "A feline was resting on the rug"
}
// Response: {"score": 0.89}  ← high similarity
```

---

## Building a Full RAG Pipeline with vLLM

Three vLLM instances working together:

```python
import numpy as np
from openai import OpenAI

# Instance 1: Embedding model (port 8001)
embed_client = OpenAI(base_url="http://localhost:8001/v1", api_key="unused")

# Instance 2: Reranker model (port 8002)  
score_client = OpenAI(base_url="http://localhost:8002/v1", api_key="unused")

# Instance 3: Generative LLM (port 8003)
gen_client = OpenAI(base_url="http://localhost:8003/v1", api_key="unused")

def rag_pipeline(query, document_store, top_k_retrieve=100, top_k_rerank=5):
    # Stage 1: Embed the query
    q_response = embed_client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input=query,
    )
    query_embedding = np.array(q_response.data[0].embedding)
    
    # Stage 2: Retrieve top-K candidates from vector store
    candidates = document_store.search(query_embedding, top_k=top_k_retrieve)
    
    # Stage 3: Rerank candidates
    import requests
    rerank_response = requests.post(
        "http://localhost:8002/v1/score",
        json={
            "model": "BAAI/bge-reranker-v2-m3",
            "text_1": query,
            "text_2": [c["text"] for c in candidates],
        }
    )
    scores = rerank_response.json()["data"]
    
    # Sort by reranker score and take top-5
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1]["score"], reverse=True)
    top_docs = [c["text"] for c, s in ranked[:top_k_rerank]]
    
    # Stage 4: Generate answer with retrieved context
    context = "\n\n".join(top_docs)
    response = gen_client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": f"Answer using this context:\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content
```

---

## Popular Reranker Models

```
Model                          Params   Type       Context   Notes
──────────────────────────────────────────────────────────────────────
bge-reranker-v2-m3             568M     BERT       512       Multilingual
bge-reranker-v2-gemma          2B       Decoder    8K        Higher quality
bge-reranker-v2-minicpm        2.4B     Decoder    8K        Chinese focus
jina-reranker-v2               137M     BERT       8K        Long context
mxbai-rerank-large-v1          335M     BERT       512       English
ms-marco-MiniLM-L-12-v2       33M      BERT       512       Lightweight
```

### Choosing a Reranker

```
Latency-critical (real-time search):
  → ms-marco-MiniLM-L-12-v2 (33M, very fast)
  → jina-reranker-v2 (137M, good balance)

Quality-critical (RAG, legal, medical):
  → bge-reranker-v2-gemma (2B, highest quality)
  → bge-reranker-v2-m3 (568M, multilingual)

Long documents (context > 512 tokens):
  → jina-reranker-v2 (8K context)
  → bge-reranker-v2-gemma (8K context)
```

---

## Key Takeaways

1. **Cross-encoders score query-document pairs jointly** — every query token attends to every document token, enabling fine-grained relevance judgments
2. **The retrieve-then-rerank pattern** combines bi-encoder speed (retrieve top-100) with cross-encoder accuracy (rerank to top-10)
3. **vLLM serves rerankers with `--task score`** — the `/v1/score` endpoint accepts (text_1, text_2) pairs and returns scores
4. **Scores are relative logits**, not probabilities — compare within a query, not across queries
5. **Beyond reranking**: the same scoring API works for reward models, sentence similarity, and other pairwise tasks
6. **Full RAG pipeline**: vLLM can serve all three components — embeddings, reranking, and generation

---

## What's Next

You can serve embeddings and rerankers. **Blog B5** covers optimization: maximizing throughput for embedding workloads, prefix caching for instruction-prefixed models, and benchmarking methodology.

---

## Further Reading

- [Cross-Encoders (Sentence-Transformers docs)](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) — popular reranker model
- [MS MARCO Leaderboard](https://microsoft.github.io/msmarco/) — reranking benchmark
- **Next: [Blog B5 — High-Throughput Embedding & Reranking](/blog/embed-05-optimization/)** — optimizing for production workloads
