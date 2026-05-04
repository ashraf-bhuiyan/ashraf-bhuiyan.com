---
title: "Pooling Strategies: From Token Embeddings to Sentence Vectors"
description: "CLS, mean, last-token, and weighted mean pooling. Normalization, wrong-pooling pitfalls, and how to choose."
pubDate: 2026-05-03
series: "Embeddings, Pooling & Rerankers in vLLM"
part: 2
tags: ["pooling", "CLS", "mean pooling", "embeddings"]
---

## What Problem Does This Solve?

A transformer processes text and produces one hidden state per token. But an embedding model needs **one vector per input**, not one per token. If your input is "How do I reset my password?" (7 tokens), the transformer gives you 7 vectors. You need 1.

```
Transformer output for "How do I reset my password?":

  "How"      → h₀ = [0.12, -0.34, 0.56, ..., 0.78]     ┐
  "do"       → h₁ = [0.23, -0.11, 0.43, ..., 0.65]     │
  "I"        → h₂ = [0.08, -0.45, 0.67, ..., 0.82]     │  7 vectors
  "reset"    → h₃ = [0.34, -0.28, 0.51, ..., 0.71]     ├  (each 768-dim)
  "my"       → h₄ = [0.19, -0.39, 0.48, ..., 0.69]     │
  "password" → h₅ = [0.31, -0.22, 0.55, ..., 0.74]     │
  "?"        → h₆ = [0.15, -0.41, 0.59, ..., 0.77]     ┘

  Need: ONE vector [?, ?, ?, ..., ?] that represents the whole input
```

**Pooling** is the operation that collapses these per-token representations into a single vector. The choice of pooling strategy is tightly coupled to how the model was trained — using the wrong strategy silently produces bad embeddings with no error message.

---

## CLS Token Pooling

### How It Works

Add a special `[CLS]` (classification) token at the beginning of the input. Use its hidden state as the embedding:

```
Input with CLS token:
  [CLS] "How" "do" "I" "reset" "my" "password" "?"

Transformer output:
  h_CLS = [0.42, -0.18, 0.63, ..., 0.85]    ← this is the embedding
  h₁    = [0.12, -0.34, 0.56, ..., 0.78]    ← ignored
  h₂    = [0.23, -0.11, 0.43, ..., 0.65]    ← ignored
  ...

Embedding = h_CLS
```

### Why It Works

The `[CLS]` token attends to all other tokens through the self-attention mechanism. By the final layer, its hidden state has aggregated information from the entire input:

```
Attention pattern of [CLS] token:
  [CLS] attends to → "How" "do" "I" "reset" "my" "password" "?"
  
  After 12+ layers of attention, [CLS] has "seen" everything.
  Its hidden state is a summary of the full input.
```

### When It's Used

```
Models using CLS pooling:
  ✓ BERT (original)
  ✓ RoBERTa
  ✓ Most encoder-only models trained in the BERT tradition
  ✓ Some cross-encoder rerankers (Blog B4)
```

### Limitations

```
1. All representational burden is on ONE token
   → For long inputs, one vector may not capture everything

2. BERT's [CLS] token wasn't designed for sentence similarity
   → BERT's [CLS] without fine-tuning is actually poor for embeddings
   → Sentence-BERT paper showed this explicitly

3. Position 0 bias
   → The [CLS] token is always at position 0
   → Positional encoding gives it a fixed "perspective"
```

---

## Mean Pooling

### How It Works

Average the hidden states of all tokens (excluding padding):

```
Input: "How do I reset my password?"   (7 tokens)
Padding: [PAD] [PAD] [PAD]             (3 padding tokens)

Hidden states:
  h₀ = [0.12, -0.34, 0.56, ...]    "How"        ← include
  h₁ = [0.23, -0.11, 0.43, ...]    "do"         ← include
  h₂ = [0.08, -0.45, 0.67, ...]    "I"          ← include
  h₃ = [0.34, -0.28, 0.51, ...]    "reset"      ← include
  h₄ = [0.19, -0.39, 0.48, ...]    "my"         ← include
  h₅ = [0.31, -0.22, 0.55, ...]    "password"   ← include
  h₆ = [0.15, -0.41, 0.59, ...]    "?"          ← include
  h₇ = [0.00,  0.00, 0.00, ...]    [PAD]        ← EXCLUDE
  h₈ = [0.00,  0.00, 0.00, ...]    [PAD]        ← EXCLUDE
  h₉ = [0.00,  0.00, 0.00, ...]    [PAD]        ← EXCLUDE

Embedding = mean(h₀, h₁, h₂, h₃, h₄, h₅, h₆)   ← only real tokens
          = [(0.12+0.23+...+0.15)/7, (-0.34-0.11+...-0.41)/7, ...]
```

### Implementation

```python
def mean_pooling(hidden_states, attention_mask):
    # attention_mask: 1 for real tokens, 0 for padding
    # hidden_states: [batch, seq_len, hidden_dim]
    
    # Expand mask to match hidden_states shape
    mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    
    # Sum hidden states of real tokens
    sum_hidden = (hidden_states * mask).sum(dim=1)  # [batch, hidden_dim]
    
    # Count real tokens per sequence
    count = mask.sum(dim=1)  # [batch, 1]
    
    # Average
    return sum_hidden / count  # [batch, hidden_dim]
```

The `attention_mask` is critical — without it, padding tokens dilute the average and produce worse embeddings.

### Why It Works

```
Mean pooling:
  Every token contributes equally
  → Information is distributed, not bottlenecked
  → More robust for long inputs than CLS pooling
  → The "average opinion" of all tokens about what the text means
```

### When It's Used

```
Models using mean pooling:
  ✓ Sentence-BERT / Sentence-Transformers
  ✓ E5 family (E5-small, E5-base, E5-large)
  ✓ BGE family (BGE-small, BGE-base, BGE-large)
  ✓ GTE (encoder variants)
  ✓ Most modern encoder-only embedding models
  
Mean pooling is the most popular pooling strategy for embeddings.
```

---

## Last-Token Pooling

### How It Works

Use the hidden state of the **last token** (or the EOS/end-of-sequence token) as the embedding:

```
Input: "How do I reset my password?"

Tokenized: ["How", "do", "I", "reset", "my", "password", "?", "<EOS>"]

Hidden states:
  h₀ = [...]    "How"
  h₁ = [...]    "do"
  h₂ = [...]    "I"
  h₃ = [...]    "reset"
  h₄ = [...]    "my"
  h₅ = [...]    "password"
  h₆ = [...]    "?"
  h₇ = [...]    "<EOS>"    ← this is the embedding

Embedding = h₇ (the last token's hidden state)
```

### Why It Works for Decoder-Only Models

In decoder-only models (GPT, Llama, Mistral, Qwen), attention is **causal** — each token can only attend to previous tokens:

```
Causal attention pattern:
  Token 0 ("How")      attends to: [How]
  Token 1 ("do")       attends to: [How, do]
  Token 2 ("I")        attends to: [How, do, I]
  Token 3 ("reset")    attends to: [How, do, I, reset]
  ...
  Token 7 ("<EOS>")    attends to: [How, do, I, reset, my, password, ?, <EOS>]
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    The LAST token has seen EVERYTHING
```

The last token is the only token that has attended to the entire input. It's the natural "summary" position in a causal model — analogous to `[CLS]` in an encoder model, but at the end instead of the beginning.

### When It's Used

```
Models using last-token pooling:
  ✓ GTE-Qwen2 (1.5B, 7B)
  ✓ E5-Mistral-7B
  ✓ NV-Embed-v2
  ✓ SFR-Embedding-2
  ✓ Most decoder-only LLMs used as embedding models
  
If the model is decoder-only → almost certainly last-token pooling.
```

### Important Nuance: "Last Token" vs. "Last Position"

```
Input:   "How do I reset my password?"
Tokens:  ["How", "do", "I", "reset", "my", "password", "?"]
                                                          ↑
                                                    last REAL token

If the sequence is padded to length 10:
  ["How", "do", "I", "reset", "my", "password", "?", [PAD], [PAD], [PAD]]
                                                  ↑                    ↑
                                            last REAL token      last POSITION

WRONG: use h₉ (the last position)  ← this is padding!
RIGHT: use h₆ (the last real token) ← this is "?"
```

vLLM handles this correctly by tracking the actual sequence length, not the padded length.

---

## Weighted Mean Pooling

### How It Works

Like mean pooling, but tokens at different positions get different weights:

```
Linear weighting (later tokens get higher weight):
  weights = [1, 2, 3, 4, 5, 6, 7] / sum([1,2,3,4,5,6,7])
          = [0.036, 0.071, 0.107, 0.143, 0.179, 0.214, 0.250]

  embedding = Σ (wᵢ × hᵢ)
```

The intuition: in many texts, the most important information is near the end (the conclusion, the key statement). Linear weighting gives more importance to later tokens.

### Attention-Weighted Pooling

A more sophisticated variant: learn a small attention head that scores each token's importance:

```
Attention weights:
  score("How") = 0.05       ← low (generic word)
  score("reset") = 0.35     ← high (key action)
  score("password") = 0.40  ← high (key noun)
  score("?") = 0.02         ← low (punctuation)
  ...

  embedding = Σ (attention_score(hᵢ) × hᵢ)
```

### When It's Used

```
Weighted pooling is rare in practice:
  - Most models use mean or last-token
  - The quality improvement over mean pooling is marginal
  - Adds complexity without clear benefit for most tasks
  
Notable exception:
  - Some reward models use attention-weighted pooling
  - Custom models trained for specific domains
```

---

## How to Know Which Pooling to Use

### Rule 1: Check the Model Card

Every HuggingFace model card specifies the pooling strategy:

```
Model card for BAAI/bge-large-en-v1.5:
  "Pooling: CLS"

Model card for intfloat/e5-large-v2:
  "Pooling: Mean"

Model card for Alibaba-NLP/gte-Qwen2-7B-instruct:
  "Pooling: Last Token"
```

### Rule 2: Check the Config

Look for `pooling_mode` in the model's configuration:

```json
// sentence_transformers config (1_Pooling/config.json)
{
  "word_embedding_dimension": 768,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": false
}
```

### Rule 3: Follow the Architecture Convention

```
BERT-based model:
  → Probably CLS or mean pooling
  → Check model card to confirm

Decoder-only LLM (Llama, Mistral, Qwen):
  → Almost certainly last-token pooling
  → Confirmed by checking the model class

Sentence-Transformers model:
  → Pooling config is in the model directory
  → 1_Pooling/config.json tells you exactly
```

### What Happens If You Get It Wrong?

```
Model trained with mean pooling, you use CLS pooling:

  Correct (mean):  cosine_sim("cat", "kitten") = 0.92    ✓
  Wrong (CLS):     cosine_sim("cat", "kitten") = 0.67    ✗

  The rankings are degraded — similar texts score lower,
  dissimilar texts score randomly. There's no error message.
  The model just silently produces bad embeddings.

Model trained with last-token, you use mean pooling:

  Correct (last):  cosine_sim("reset password", "change password") = 0.95  ✓
  Wrong (mean):    cosine_sim("reset password", "change password") = 0.71  ✗

  Mean pooling dilutes the last-token's summary with tokens
  that weren't trained to contribute to the embedding.
```

This is one of the most common mistakes in embedding pipelines — and one of the hardest to debug because there's no error, just silently worse results.

---

## Normalization

### L2 Normalization

After pooling, many models apply L2 normalization to project the embedding onto the unit sphere:

```
raw_embedding = pool(hidden_states)
normalized = raw_embedding / ||raw_embedding||₂

Where ||v||₂ = √(v₁² + v₂² + ... + vₙ²)

Result: ||normalized||₂ = 1.0 (unit vector)
```

### Why Normalize?

```
Without normalization:
  cosine(a, b) = (a · b) / (||a|| × ||b||)
  → Need to compute norms every time
  → Dot product ≠ cosine similarity

With normalization:
  ||a|| = ||b|| = 1.0
  cosine(a, b) = a · b   (just a dot product!)
  → Faster similarity computation
  → Dot product = cosine similarity
  → Compatible with all ANN libraries (FAISS, etc.)
```

### Which Models Normalize?

```
Models that normalize by default:
  ✓ E5 family (all variants)
  ✓ GTE-Qwen2
  ✓ NV-Embed
  
Models that DON'T normalize:
  ✗ Some BGE variants
  ✗ Some older BERT models
  
Check the model card. When in doubt, normalize yourself
— it never hurts (cosine similarity is invariant to scaling).
```

vLLM's `PoolingParams` supports normalization, so you can enable it at serving time regardless of the model's default.

---

## Pooling in vLLM

vLLM's `Pooler` class implements all these strategies:

```
vLLM Pooler:
  PoolingType.CLS  → return hidden_states[0]     (first token)
  PoolingType.LAST → return hidden_states[-1]     (last real token)
  PoolingType.MEAN → return mean(hidden_states)   (masked average)
  PoolingType.ALL  → return all hidden_states     (no pooling, for special models)

vLLM auto-detects the pooling type from the model's configuration.
You can override with --override-pooler-config if needed.
```

More details on vLLM's embedding serving in Blog B3.

---

## Summary: Pooling Strategy Quick Reference

```
Strategy       How                    Used By              Best For
──────────────────────────────────────────────────────────────────────
CLS            First token ([CLS])    BERT, RoBERTa        Encoder models
Mean           Average all tokens     E5, BGE, GTE         Most encoder models
Last-token     Last real token        GTE-Qwen2, NV-Embed  Decoder-only LLMs
Weighted Mean  Weighted average       Rare (custom)        Specialized models
```

---

## Key Takeaways

1. **Pooling collapses per-token hidden states into one embedding vector** — the bridge between the transformer and the embedding
2. **CLS pooling**: use the `[CLS]` token's hidden state. Works for BERT-style models.
3. **Mean pooling**: average all token hidden states. Most popular for modern encoder models.
4. **Last-token pooling**: use the last token's hidden state. Natural for causal (decoder-only) models.
5. **Using the wrong pooling silently degrades quality** — no error, just worse similarity scores. Always check the model card.
6. **L2 normalization** makes dot product = cosine similarity, simplifying downstream computation.

---

## What's Next

You understand embeddings (Blog B1) and pooling (this blog). **Blog B3** shows how to serve embedding models in vLLM — the `--task embed` flag, the `/v1/embeddings` API, and how vLLM handles the non-generative pipeline internally.

---

## Further Reading

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — introduced mean pooling for BERT embeddings
- [How to Use BERT's [CLS] Token](https://arxiv.org/abs/2101.00438) — analysis of CLS vs. mean pooling
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — truncatable embeddings
- **Next: [Blog B3 — Serving Embeddings in vLLM](/blog/embed-03-serving/)** — `--task embed` and the `/v1/embeddings` API
