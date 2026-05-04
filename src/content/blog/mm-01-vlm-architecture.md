---
title: "VLM Architecture Primer: How Vision Meets Language"
description: "Vision encoders (ViT, SigLIP), projection strategies, token interleaving, and the LLaVA/Qwen2-VL/InternVL model families."
pubDate: 2026-05-03
series: "Vision, Language & Audio Models in vLLM"
part: 1
tags: ["VLM", "vision encoder", "ViT", "multi-modal"]
---

## What Problem Does This Solve?

An LLM processes sequences of token embeddings. It has no concept of pixels, colors, or visual structure. But humans routinely ask "What's in this image?" or "Read the text in this document." Vision-language models (VLMs) bridge this gap by converting images into token-like embeddings that the LLM can process alongside text.

```
Text-only LLM:
  Input:  "Describe a cat"  → [token₁, token₂, token₃]
  Output: "A cat is a small domesticated..."

Vision-language model:
  Input:  "Describe this image:" + [🖼️ cat photo]
          → [token₁, token₂, token₃, img₁, img₂, ..., img₅₇₆]
  Output: "The image shows an orange tabby cat sitting on..."
```

The challenge is converting pixels into a sequence of embeddings that the LLM treats identically to text tokens. This blog explains the three architectural components that make this possible.

---

## The Three Components

Every VLM follows the same high-level pattern:

```
┌────────────┐     ┌─────────────┐     ┌──────────────┐
│   Vision   │     │  Projection │     │     LLM      │
│   Encoder  │────►│   Layer     │────►│   Backbone   │
│            │     │             │     │              │
│ Image →    │     │ vision_dim  │     │ text tokens  │
│ patch      │     │     →       │     │     +        │
│ features   │     │ llm_dim     │     │ visual tokens│
└────────────┘     └─────────────┘     └──────────────┘
    SigLIP             MLP               Llama/Qwen
    ViT                Linear            Mistral
    DINOv2             Cross-attn        Gemma
```

---

## Component 1: Vision Encoder

### What It Does

The vision encoder converts a raw image (a grid of pixels) into a sequence of feature vectors. The dominant architecture is the **Vision Transformer (ViT)**.

### How ViT Works

```
Step 1: Split the image into patches

  224×224 image with 14×14 pixel patches:
  
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │p0│p1│p2│p3│p4│p5│p6│p7│p8│p9│..│..│..│..│..│p₂₅₅│
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴────┘
  
  224/14 = 16 patches per row
  16 × 16 = 256 patches total

Step 2: Linear projection

  Each 14×14×3 patch (588 values) → linear layer → patch embedding (vision_dim)
  
  256 patches → 256 patch embeddings, each of size vision_dim (e.g., 1152)

Step 3: Add position embeddings

  Each patch gets a learned position embedding
  → Patch 0 always corresponds to top-left, Patch 255 to bottom-right

Step 4: Transformer layers

  Run all 256 patch embeddings through N transformer layers
  (typically 24-48 layers, same architecture as text transformers)
  
  Output: 256 refined patch features, each of size vision_dim
```

### Common Vision Encoders

```
Encoder       Params   Output dim   Training         Used by
────────────────────────────────────────────────────────────────
SigLIP-400M   400M     1152         Sigmoid loss     LLaVA, InternVL, PaliGemma
CLIP ViT-L    304M     1024         Contrastive      Early LLaVA, some Flamingo
DINOv2-L      304M     1024         Self-supervised  Some research VLMs
InternViT-6B  6B       3200         Custom           InternVL2 (very large encoder)
```

### Resolution and Token Count

The relationship between image resolution, patch size, and token count:

```
Resolution    Patch size    Patches (tokens)    KV cache per image
                                                (7B model, FP16)
──────────────────────────────────────────────────────────────────
224 × 224     14 × 14       16 × 16 = 256       134 MB
336 × 336     14 × 14       24 × 24 = 576       302 MB
448 × 448     14 × 14       32 × 32 = 1024      537 MB
672 × 672     14 × 14       48 × 48 = 2304      1,208 MB

Higher resolution → more patches → more visual tokens
  → better visual detail → more memory per request
```

This is the fundamental tradeoff: higher resolution gives better visual understanding but consumes proportionally more KV cache memory, reducing the number of concurrent requests.

---

## Component 2: Projection Layer

### The Dimensionality Problem

The vision encoder outputs features of size `vision_dim`. The LLM expects embeddings of size `llm_dim`. These are usually different:

```
SigLIP encoder:  vision_dim = 1152
Llama-3.1-8B:    llm_dim = 4096

1152 ≠ 4096 → need a projection layer
```

### Projection Strategies

**Strategy 1: MLP Projection (LLaVA-style)**

The simplest and most common approach — a 2-layer MLP:

```
patch_feature (1152) → Linear(1152, 4096) → GELU → Linear(4096, 4096) → visual_token (4096)

Applied independently to each patch:
  256 patch features → 256 visual tokens
  
  Token count: unchanged (256 in → 256 out)
  Pro: simple, effective, each visual token maps to one image patch
  Con: token count equals patch count (can be high for large images)
```

Used by: LLaVA-1.5, LLaVA-1.6, LLaVA-OneVision

**Strategy 2: Cross-Attention (Flamingo-style)**

Learnable query tokens attend to the vision features:

```
Query tokens: Q₁, Q₂, ..., Q₆₄  (64 learned queries, each llm_dim)
Vision features: P₁, P₂, ..., P₂₅₆  (256 patch features)

Cross-attention:
  Q₁ attends to [P₁...P₂₅₆] → refined Q₁
  Q₂ attends to [P₁...P₂₅₆] → refined Q₂
  ...
  Q₆₄ attends to [P₁...P₂₅₆] → refined Q₆₄

Output: 64 visual tokens (regardless of image resolution!)

Token count: fixed at 64 (or another constant)
Pro: fixed token count, memory-predictable
Con: may lose fine-grained spatial details (256 patches → 64 tokens)
```

Used by: Flamingo, some Qwen-VL variants

**Strategy 3: Perceiver Resampler**

Like cross-attention but with additional self-attention among the query tokens:

```
1. Cross-attention: queries attend to vision features
2. Self-attention: queries attend to each other
3. Repeat for N layers

Output: fixed number of visual tokens with richer representations
```

Used by: Some research models, early Qwen-VL

**Strategy 4: Token Merging (Qwen2-VL)**

Process at native resolution, then merge adjacent tokens to reduce count:

```
Native resolution → ViT → patch features → merge 2×2 neighbors

Before merging: 48 × 48 = 2,304 features
After merging:  24 × 24 = 576 visual tokens (4× reduction)

Pro: native resolution (no forced resize), reduced token count
Con: variable token count depending on image size
```

---

## Component 3: Token Interleaving

### Combining Visual and Text Tokens

After projection, visual tokens are interleaved with text tokens into a single sequence:

```
Text prompt: "Describe this image:\n"
Image: [cat photo] → 576 visual tokens

Combined sequence:
  [BOS] "Describe" "this" "image" ":" "\n" [IMG₁] [IMG₂] ... [IMG₅₇₆] [EOS]
   ↑     ↑         ↑      ↑       ↑   ↑    ↑      ↑           ↑        ↑
   text   text      text   text   text text  visual visual      visual   text

The LLM processes this as one sequence.
Visual tokens participate in self-attention just like text tokens.
```

### Placeholder Tokens

In the tokenized text, special placeholder tokens mark where images should go:

```
Before preprocessing:
  User message: "What's in this image? <image>"
  
Tokenized:
  ["What", "'s", "in", "this", "image", "?", "<image>"]
                                                 ↑
                                           placeholder (1 token)

After preprocessing:
  ["What", "'s", "in", "this", "image", "?", IMG₁, IMG₂, ..., IMG₅₇₆]
                                               ↑                    ↑
                                          576 visual tokens replace
                                          the 1 placeholder token
```

The number of placeholder tokens must match the number of visual tokens exactly. The chat template handles this mapping.

### How the LLM Sees It

After interleaving, the LLM processes the combined sequence with standard causal attention:

```
Attention for text token "orange":
  "orange" attends to → [BOS, "Describe", ..., IMG₁, IMG₂, ..., IMG₅₇₆]
  
  The model can "look at" the image while generating text.
  It sees visual features from different parts of the image
  through the patch tokens.

Attention for visual token IMG₁₀₀:
  IMG₁₀₀ attends to → [BOS, "Describe", ..., IMG₁, ..., IMG₉₉]
  
  Visual tokens attend to preceding text AND other visual tokens.
  This lets the model build a unified understanding of text + image.
```

---

## Architecture Patterns

### LLaVA: Project and Concatenate

```
Image → SigLIP (384×384, 27×27 patches) → MLP → 729 visual tokens
                                                    │
Text → Tokenizer → text tokens ────────────────────┤
                                                    ▼
                                        Concat → Llama/Vicuna → output

Characteristics:
  - Simple architecture, very effective
  - Token count = patch count (576-729 typically)
  - MLP projection is lightweight (~10M params)
  - Most popular VLM architecture pattern
  
Models: LLaVA-1.5, LLaVA-1.6, LLaVA-OneVision
```

### Qwen2-VL: Dynamic Resolution

```
Image (native res) → ViT (native resolution, 2D RoPE)
                           │
                     Token merging (2×2 → 1)
                           │
                     Variable visual tokens
                           │
Text → Tokenizer → text tokens ────┤
                                    ▼
                          Concat → Qwen2 → output

Characteristics:
  - No forced resize — process at native resolution
  - 2D Rotary Position Embeddings preserve spatial structure
  - Token merging reduces count by 4×
  - Token count varies per image: (H/14 × W/14) / 4
  - Excellent for variable-resolution inputs
  
Models: Qwen2-VL-2B, Qwen2-VL-7B, Qwen2-VL-72B
```

### InternVL2: Dynamic Tiling

```
Image → Split into 448×448 tiles + thumbnail
          │
          ├─ Tile 1 → InternViT → 256 tokens
          ├─ Tile 2 → InternViT → 256 tokens
          ├─ ...
          └─ Thumbnail → InternViT → 256 tokens
                                        │
                             Concat all tile tokens
                                        │
Text → Tokenizer → text tokens ────────┤
                                        ▼
                              Concat → InternLM2/Llama → output

Characteristics:
  - High-res images split into multiple 448×448 tiles
  - Each tile processed independently through the encoder
  - Token count = (num_tiles + 1) × 256
  - Handles very high-resolution images (add more tiles)
  - Large encoder: InternViT-6B (6 billion params)
  
Models: InternVL2-1B, InternVL2-8B, InternVL2-26B, InternVL2-76B
```

### PaliGemma: Native Multi-Modal

```
Image → SigLIP → Linear projection → visual tokens (prefix)
                                          │
Text → Tokenizer → text tokens ──────────┤ (appended after)
                                          ▼
                                   Gemma → output

Characteristics:
  - Image tokens ALWAYS come first (prefix position)
  - Trained from scratch as multi-modal (not post-hoc adapter)
  - Fixed resolution (224×224 or 448×448)
  - Simpler than LLaVA — no special chat template handling
  
Models: PaliGemma, PaliGemma2
```

---

## The Token Count Problem

### Memory Impact

Visual tokens consume KV cache memory just like text tokens:

```
Per-token KV cache (Llama-3.1-8B, FP16):
  32 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes
  = 32 × 8 × 128 × 2 × 2 = 131,072 bytes ≈ 128 KB per token

Per-image KV cache:
  576 tokens × 128 KB = 72 MB    (LLaVA, 336×336)
  1024 tokens × 128 KB = 128 MB  (448×448)
  2304 tokens × 128 KB = 288 MB  (672×672)

For comparison:
  A 500-word text prompt ≈ 700 tokens × 128 KB = 87.5 MB
  One 336×336 image uses as much KV cache as ~560 words of text
```

### Concurrent Request Impact

```
Available KV cache memory: 40 GB (after model weights)

Text-only requests (500 tokens avg):
  40 GB / 62.5 MB per request = 640 concurrent requests

VLM requests (500 text + 576 image = 1076 tokens):
  40 GB / 134 MB per request = 298 concurrent requests
  → 2.1× fewer concurrent requests!

VLM requests with 3 images (500 text + 1728 image = 2228 tokens):
  40 GB / 278 MB per request = 143 concurrent requests
  → 4.5× fewer concurrent requests!
```

This is why token reduction strategies (cross-attention, token merging) matter for production VLM serving — fewer visual tokens per image means more concurrent requests.

---

## Key Takeaways

1. **VLMs have three components**: vision encoder (image → patches), projection layer (vision_dim → llm_dim), and LLM backbone
2. **Vision encoders (ViT/SigLIP)** split images into patches and process them through transformer layers, producing one feature per patch
3. **Projection strategies** differ in whether they preserve or reduce token count: MLP (preserve), cross-attention (reduce to fixed count), token merging (reduce by 4×)
4. **Visual tokens are interleaved** with text tokens — the LLM processes them identically via self-attention
5. **The token count problem**: each image = hundreds of tokens of KV cache, directly reducing concurrent request capacity
6. **Different architectures** (LLaVA, Qwen2-VL, InternVL) make different tradeoffs on resolution, token count, and flexibility

---

## What's Next

You understand how VLMs work architecturally. **Blog C2** shows how to serve them in vLLM — sending images, formatting requests, and handling the preprocessing pipeline.

---

## Further Reading

- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — the LLaVA paper
- [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191) — dynamic resolution VLM
- [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238)
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) — the most popular VLM encoder
- **Next: [Blog C2 — Serving VLMs in vLLM](/blog/mm-02-serving-vlms/)** — your first image + text request
