---
title: "Multi-Modal Internals: How vLLM Routes Modalities"
description: "MultiModalRegistry, plugins, processor pipeline, placeholder token replacement, and KV cache token estimation."
pubDate: 2026-05-03
series: "Vision, Language & Audio Models in vLLM"
part: 5
tags: ["multi-modal", "internals", "registry", "processor"]
---

## What Problem Does This Solve?

vLLM supports images, video, and audio — three fundamentally different modalities with different preprocessing requirements, different encoders, and different token counts. Yet they all flow through the same engine: the same scheduler, the same KV cache manager, the same continuous batching system.

This blog explains the internal architecture that makes this work: how vLLM discovers a model's multi-modal capabilities, preprocesses different modalities, replaces placeholder tokens with multi-modal embeddings, and handles the KV cache implications.

---

## The MultiModalRegistry

### Purpose

The `MultiModalRegistry` is the central coordination point. It answers: "What multi-modal capabilities does this model have?"

```
When a model loads, it registers its capabilities:

  Qwen2-VL:     image ✓, video ✓, audio ✗
  Qwen2-Audio:  image ✗, video ✗, audio ✓
  LLaVA-1.5:    image ✓, video ✗, audio ✗
  
The registry stores:
  - Supported modalities (image, video, audio)
  - Maximum inputs per modality
  - Preprocessing class for each modality
  - How to map modality data to model inputs
```

### How It Works

```
Request arrives: "What's in this image?" + [image_url]
         │
         ▼
MultiModalRegistry.has_modality("image")?
├── Yes → route to image processing pipeline
│         → fetch image, preprocess, encode
│
└── No  → return error "Model doesn't support images"

This check happens BEFORE any expensive computation.
```

### Why a Registry?

Without a registry, vLLM would need per-model if/else logic:

```python
# BAD: hardcoded per-model logic
if model_name == "Qwen2-VL":
    supports_image = True
    supports_video = True
elif model_name == "LLaVA":
    supports_image = True
    supports_video = False
# ... 50 more models
```

With a registry, each model self-declares its capabilities:

```python
# GOOD: model registers itself
class Qwen2VLForConditionalGeneration(nn.Module):
    def get_multimodal_config(self):
        return MultiModalConfig(
            modalities=["image", "video"],
            max_image_per_prompt=5,
            max_video_per_prompt=1,
        )
```

New models register themselves — no changes to the engine code.

---

## MultiModalPlugin System

### Per-Modality Plugins

Each modality has a plugin that handles its specific processing:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ ImagePlugin  │     │ VideoPlugin  │     │ AudioPlugin  │
│              │     │              │     │              │
│ Parse:       │     │ Parse:       │     │ Parse:       │
│  URL/base64  │     │  URL/file    │     │  WAV/MP3     │
│  → PIL Image │     │  → frames   │     │  → waveform  │
│              │     │              │     │              │
│ Validate:    │     │ Validate:    │     │ Validate:    │
│  format,     │     │  format,     │     │  format,     │
│  size limits │     │  frame count │     │  duration    │
│              │     │              │     │              │
│ Preprocess:  │     │ Preprocess:  │     │ Preprocess:  │
│  resize,     │     │  sample      │     │  resample,   │
│  normalize   │     │  frames,     │     │  mel spec    │
│              │     │  resize each │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Plugin Architecture

Each plugin implements a standard interface:

```python
class MultiModalPlugin:
    def parse_input(self, data):
        """Convert API format to internal representation"""
        # image_url → PIL Image
        # audio base64 → numpy waveform
        
    def validate(self, data, limits):
        """Check size, format, count limits"""
        # Image too large? Audio too long? Too many inputs?
        
    def get_max_per_prompt(self, model_config):
        """How many of this modality per request?"""
        # e.g., max 5 images, max 1 video
```

### Extensibility

The plugin system means new modalities can be added without modifying the core engine:

```
Want to add 3D point cloud support?
  1. Create PointCloudPlugin
  2. Implement parse/validate/preprocess
  3. Register with MultiModalRegistry
  4. The engine handles scheduling, batching, KV cache — unchanged
```

---

## MultiModalProcessor: The Pipeline

### End-to-End Processing

The `MultiModalProcessor` coordinates the full pipeline for a multi-modal request:

```
Raw API Request
      │
      ▼
┌─────────────────────────────────────────────────┐
│              MultiModalProcessor                │
│                                                 │
│  1. PARSE ─────────────────────────────────────│
│     Extract multi-modal data from API request   │
│     → image URLs, audio base64, video files     │
│                                                 │
│  2. VALIDATE ──────────────────────────────────│
│     Check against limits:                       │
│     - Image count <= limit_mm_per_prompt.image  │
│     - Audio duration <= max_audio_duration      │
│     - Total tokens won't exceed max_model_len   │
│                                                 │
│  3. PREPROCESS ────────────────────────────────│
│     Model-specific transformations:             │
│     Image: resize, normalize, pixel_values      │
│     Audio: resample, mel spectrogram            │
│     Video: sample frames, resize each           │
│                                                 │
│  4. TOKEN COUNT ESTIMATION ────────────────────│
│     Estimate visual/audio tokens from input     │
│     dimensions BEFORE running the encoder       │
│     → Used by scheduler for memory planning     │
│                                                 │
│  5. BUILD MODEL INPUTS ────────────────────────│
│     Combine: token_ids + multi-modal data       │
│     → Ready for the model forward pass          │
└─────────────────────────────────────────────────┘
```

### Model-Specific Processors

Each VLM implements its own processor because preprocessing is model-specific:

```
LLaVA processor:
  Image → resize to 336×336 → normalize → pixel_values

Qwen2-VL processor:
  Image → keep native resolution → normalize → pixel_values
        → calculate dynamic token count

InternVL2 processor:
  Image → split into 448×448 tiles → normalize each tile
        → tile_count × 256 tokens per tile
```

vLLM loads the correct processor based on the model class.

---

## Placeholder Token Replacement

### The Core Mechanism

During the forward pass, placeholder tokens in the text sequence are replaced with actual multi-modal embeddings:

```python
# Step 1: Text embedding (standard)
text_embeddings = self.embed_tokens(input_ids)
# Shape: [batch_size, seq_len, hidden_dim]
# Placeholder tokens have placeholder embeddings (meaningless)

# Step 2: Multi-modal encoding
image_features = self.vision_encoder(pixel_values)  # [num_images, num_patches, vision_dim]
image_embeds = self.projection(image_features)        # [num_images, num_tokens, llm_dim]

# Step 3: Replace placeholders with actual embeddings
for i, positions in enumerate(image_token_positions):
    text_embeddings[batch_idx, positions] = image_embeds[i]
# Now text_embeddings contains real visual information at image positions

# Step 4: Standard LLM forward pass
output = self.language_model(inputs_embeds=text_embeddings)
```

### Position Matching

The number of placeholder tokens must exactly match the number of multi-modal embeddings:

```
Tokenized input:
  ["What", "is", "this", "<img>", "<img>", ..., "<img>", "?"]
                           ↑                          ↑
                      position 3                position 578
                      (576 placeholders)

Visual embeddings: [576, 4096]  ← exactly 576 embeddings

Replace:
  text_embeddings[3:579] = visual_embeddings[0:576]

If the counts don't match → crash.
The chat template ensures they match.
```

---

## KV Cache Implications

### Visual Tokens in the KV Cache

After prefill, visual tokens have KV cache entries just like text tokens:

```
Prefill produces KV cache for all tokens:
  [BOS, "What", "is", IMG₁, IMG₂, ..., IMG₅₇₆, "?"]
   ↑     ↑       ↑     ↑     ↑          ↑        ↑
   text  text   text  visual visual    visual   text
   KV    KV     KV    KV     KV        KV       KV

All tokens have KV cache entries. The scheduler allocates
blocks for ALL tokens (text + visual), not just text.
```

### Token Count Estimation

The scheduler needs to know how many tokens a request will use *before* running the model. For text-only requests, this is just `len(token_ids)`. For multi-modal requests, visual/audio tokens must be estimated:

```
Estimation:
  text_tokens = len(tokenize(prompt))
  
  For LLaVA (fixed resolution):
    image_tokens = 576 per image  (always the same)
    total_tokens = text_tokens + num_images × 576
  
  For Qwen2-VL (dynamic resolution):
    image_tokens = (H/14 × W/14) / 4  per image  (depends on resolution!)
    total_tokens = text_tokens + Σ image_tokens_i
  
  For video:
    video_tokens = num_frames × tokens_per_frame
    total_tokens = text_tokens + video_tokens

Estimation happens BEFORE encoding:
  → From image dimensions, calculate expected token count
  → Scheduler uses this for memory planning
  → Over-estimation wastes capacity; under-estimation can cause OOM
```

### Why Estimation Matters

```
Without estimation:
  1. Schedule request (assume it fits)
  2. Run vision encoder → produce 2,304 tokens (high-res image)
  3. Allocate KV cache → OOM! Not enough memory!
  4. Request fails after wasting GPU time on the encoder

With estimation:
  1. Estimate: this image will produce ~2,304 tokens
  2. Check: 2,304 × 128 KB = 295 MB → do we have space?
  3. No → keep in queue until space is available
  4. Yes → schedule, encode, allocate KV cache (guaranteed to fit)
```

---

## Multi-Modal Data Flow Through the Engine

### Complete Request Lifecycle

```
1. API Server receives request with image URL
   │
2. MultiModalProcessor.parse()
   │  Extract image URL from the message content
   │  Validate against limit_mm_per_prompt
   │
3. MultiModalProcessor.preprocess()
   │  Fetch image → decode → resize → normalize
   │  Estimate token count from image dimensions
   │  (runs on CPU, in parallel with other requests' GPU work)
   │
4. Tokenizer
   │  Convert text to token IDs
   │  Insert placeholder tokens for images
   │  token_ids = [BOS, ..., IMG_PAD, IMG_PAD, ..., IMG_PAD, ..., EOS]
   │
5. Scheduler
   │  Estimate total tokens = text + visual
   │  Check memory budget (KV cache availability)
   │  Admit to batch when space is available
   │
6. ModelRunner.execute_model()
   │
   ├── 6a. Embed text tokens
   │       text_embeds = embed_tokens(token_ids)
   │
   ├── 6b. Run vision encoder (GPU)
   │       image_features = vision_encoder(pixel_values)
   │
   ├── 6c. Project vision features
   │       image_embeds = projection(image_features)
   │
   ├── 6d. Replace placeholders
   │       text_embeds[placeholder_positions] = image_embeds
   │
   └── 6e. LLM forward pass
           hidden_states = transformer(text_embeds)
   │
7. Sample next token
   │
8. Return token to client
   │
9. Decode loop (steps 6e-8, no more vision encoding)
   │  Visual tokens are already in the KV cache
   │  Only the new text token needs processing
```

### Key Insight: Vision Encoding Only During Prefill

```
Prefill (step 6):
  - Vision encoder runs once
  - All visual tokens get KV cache entries
  - Most expensive step (compute-heavy)

Decode (step 9, repeated):
  - No vision encoding (already done)
  - Visual KV is already cached
  - Only process 1 new text token per step
  - Same speed as text-only decode!
```

This means the vision encoder overhead is a one-time cost per request. Once prefill is done, decode runs at the same speed as text-only inference.

---

## Model-Specific Customization Points

### How VLMs Integrate with vLLM

Each VLM model class implements specific methods:

```python
class SomeVLM(nn.Module):
    
    def get_multimodal_embeddings(self, pixel_values, ...):
        """Encode and project multi-modal inputs"""
        features = self.vision_encoder(pixel_values)
        embeddings = self.projection(features)
        return embeddings
    
    def get_input_embeddings(self, input_ids, multimodal_embeddings):
        """Merge multi-modal embeddings with text embeddings"""
        text_embeds = self.embed_tokens(input_ids)
        text_embeds[placeholder_mask] = multimodal_embeddings
        return text_embeds
    
    def get_multimodal_config(self):
        """Declare supported modalities and limits"""
        return MultiModalConfig(modalities=["image"], ...)
```

### Adding a New VLM to vLLM

```
1. Implement the model class with:
   - get_multimodal_embeddings()
   - get_input_embeddings()
   - get_multimodal_config()
   
2. Create a processor (preprocessing pipeline)

3. Register in the model registry

4. The engine handles everything else:
   - Scheduling
   - Continuous batching
   - KV cache management
   - Prefix caching
   - Chunked prefill
```

---

## Key Takeaways

1. **MultiModalRegistry** is a self-declaration system — each model registers what modalities it supports
2. **MultiModalPlugin** handles per-modality parsing, validation, and preprocessing
3. **MultiModalProcessor** coordinates the end-to-end pipeline from API request to model inputs
4. **Placeholder replacement** swaps special tokens with actual multi-modal embeddings during the forward pass
5. **Token count estimation** happens before encoding — the scheduler needs to know memory requirements upfront
6. **Vision/audio encoding only happens during prefill** — decode runs at text-only speed because visual KV is already cached

---

## What's Next

You understand how vLLM handles multi-modal inputs internally. **Blog C6** covers optimization: prefix caching for shared images, chunked prefill with large visual inputs, tensor parallelism for vision encoders, and throughput tuning.

---

## Further Reading

- [vLLM Multi-Modal Source Code](https://github.com/vllm-project/vllm/tree/main/vllm/multimodal) — the implementation
- [vLLM Adding a New Model Guide](https://docs.vllm.ai/en/latest/contributing/model/index.html) — how to add new models
- **Next: [Blog C6 — Optimizing Multi-Modal Serving](/blog/mm-06-optimization/)** — production-grade multi-modal tuning
