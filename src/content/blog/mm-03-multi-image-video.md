---
title: "Multi-Image and Video Input"
description: "Token count explosion with multiple images, video frame sampling, dynamic vs fixed resolution, and memory management."
pubDate: 2026-05-03
series: "Vision, Language & Audio Models in vLLM"
part: 3
tags: ["multi-image", "video", "tokens", "memory"]
---

## What Problem Does This Solve?

Single-image VLMs answer "What's in this image?" But real applications need more:

- **Multi-image comparison**: "What's different between these two screenshots?"
- **Document processing**: "Analyze pages 1-5 of this PDF" (each page = an image)
- **Video understanding**: "Describe what happens in this clip"
- **Interleaved conversations**: "Here's image A. Now here's image B. How do they compare?"

Each additional image multiplies the visual token count — and the memory cost. This blog covers how to handle multi-image and video input in vLLM, and how to manage the resulting memory pressure.

---

## Multi-Image Input

### Sending Multiple Images

The OpenAI vision API format supports multiple `image_url` entries:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare these two images. What are the differences?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/before.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/after.jpg"}}
      ]
    }],
    "max_tokens": 300
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the key differences between these product photos?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/product_v1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/product_v2.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/product_v3.jpg"}},
        ],
    }],
    max_tokens=500,
)
```

### How Multiple Images Are Processed

Each image is independently preprocessed and encoded:

```
Image 1 → fetch → resize → normalize → ViT encoder → project → 576 tokens
Image 2 → fetch → resize → normalize → ViT encoder → project → 576 tokens
Image 3 → fetch → resize → normalize → ViT encoder → project → 576 tokens

Combined sequence:
  [text tokens] [IMG1₁..IMG1₅₇₆] [IMG2₁..IMG2₅₇₆] [IMG3₁..IMG3₅₇₆] [more text]
  
  Total visual tokens: 3 × 576 = 1,728
```

The vision encoder runs once per image. Images don't interact during encoding — each is processed independently. Interaction only happens in the LLM, where all visual tokens participate in the same self-attention computation.

---

## The Token Count Explosion

### Memory Math

```
Llama-3.1-8B, FP16:
  KV cache per token ≈ 128 KB

Token count by image count:
  1 image  (576 tokens):   74 MB
  3 images (1,728 tokens): 221 MB
  5 images (2,880 tokens): 369 MB
  10 images (5,760 tokens): 737 MB

Add text tokens (200 avg):
  1 image:  (576+200) × 128 KB = 99 MB
  5 images: (2,880+200) × 128 KB = 394 MB
  10 images: (5,760+200) × 128 KB = 762 MB
```

### Concurrent Request Impact

```
Available KV cache: 40 GB

Text-only (200 tokens avg):
  40 GB / 25 MB = 1,600 concurrent requests

1 image per request:
  40 GB / 99 MB = 404 concurrent requests

5 images per request:
  40 GB / 394 MB = 101 concurrent requests

10 images per request:
  40 GB / 762 MB = 52 concurrent requests
```

More images per request → drastically fewer concurrent requests. This is why `--limit-mm-per-prompt` exists.

### Controlling Image Count

```bash
# Limit to 5 images per request
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --limit-mm-per-prompt image=5

# Limit to 3 images and 1 video
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --limit-mm-per-prompt image=3,video=1
```

Requests exceeding the limit receive an error before processing — preventing a single request from consuming all available memory.

---

## High-Resolution Images

### Resolution and Token Count

For dynamic-resolution models, higher-resolution images produce more tokens:

```
Qwen2-VL (native resolution with 2× token merging):
  Resolution      Patches    After merge    KV cache
  ───────────────────────────────────────────────────
  224 × 224       16 × 16    8 × 8 = 64     8 MB
  448 × 448       32 × 32    16 × 16 = 256  33 MB
  896 × 896       64 × 64    32 × 32 = 1024 131 MB
  1344 × 1344     96 × 96    48 × 48 = 2304 295 MB

InternVL2 (dynamic tiling, 448×448 tiles):
  Image size      Tiles    Tokens              KV cache
  ───────────────────────────────────────────────────────
  448 × 448       1+1      (1+1) × 256 = 512   66 MB
  896 × 448       2+1      (2+1) × 256 = 768   98 MB
  896 × 896       4+1      (4+1) × 256 = 1280  164 MB
  1344 × 896      6+1      (6+1) × 256 = 1792  229 MB
```

### Controlling Resolution

```bash
# Qwen2-VL: limit maximum pixels
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 602112}'

# max_pixels = 602112 → max ~24×24 patches after merge → 576 tokens
# max_pixels = 1003520 → max ~31×31 patches after merge → ~961 tokens
```

Lower `max_pixels` = fewer tokens per image = more concurrent requests, but lower visual detail. This is the primary quality-throughput tradeoff for VLM serving.

---

## Video Input

### How Video Works in VLMs

VLMs don't process raw video streams. Instead, video is sampled into frames, and each frame is treated as an image:

```
Video (30 seconds, 30 FPS) = 900 raw frames

Sampling strategy: 1 frame per second → 30 frames
Each frame → vision encoder → 576 tokens per frame

Total: 30 × 576 = 17,280 visual tokens
KV cache: 17,280 × 128 KB = 2.2 GB (for ONE video!)
```

Video is the most memory-intensive modality by far.

### Sending Video Requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe what happens in this video."},
        {"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}}
      ]
    }],
    "max_tokens": 500
  }'
```

### Frame Sampling Strategies

```
Uniform sampling (most common):
  Extract N frames evenly spaced throughout the video
  30-second video, N=16: one frame every ~2 seconds
  
FPS-based:
  Extract frames at a fixed rate
  30-second video, 0.5 FPS: 15 frames
  
Keyframe-based:
  Only extract frames with significant visual changes
  (scene cuts, motion bursts)
  → Fewer frames for static scenes, more for action

The sampling strategy depends on the model.
Qwen2-VL uses uniform sampling with configurable frame count.
```

### Controlling Video Frame Count

```bash
# Limit video frames via mm-processor-kwargs
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --mm-processor-kwargs '{"max_pixels": 602112, "fps": 1.0}' \
  --limit-mm-per-prompt video=1
```

```
Frame count vs. memory:
  8 frames:   8 × 576 = 4,608 tokens → 590 MB
  16 frames: 16 × 576 = 9,216 tokens → 1.2 GB
  32 frames: 32 × 576 = 18,432 tokens → 2.4 GB
  64 frames: 64 × 576 = 36,864 tokens → 4.7 GB
```

### Models with Video Support

```
Model                  Video?   Max frames   Notes
──────────────────────────────────────────────────────
Qwen2-VL              Yes      Configurable  Native video support
Qwen2.5-VL            Yes      Configurable  Improved video
LLaVA-OneVision        Yes      Configurable  Multi-image + video
InternVL2              Yes      Configurable  Dynamic tiling per frame
Phi-3.5-Vision         No       —             Image only
PaliGemma              No       —             Image only
Llama 3.2 Vision       No       —             Image only
```

---

## Interleaved Image-Text Conversations

### Multi-Turn with Images

Models like Qwen2-VL and LLaVA-OneVision support images across conversation turns:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                {"type": "text", "text": "What animal is this?"},
            ],
        },
        {
            "role": "assistant",
            "content": "This is an orange tabby cat sitting on a windowsill.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}},
                {"type": "text", "text": "And what about this one?"},
            ],
        },
    ],
    max_tokens=200,
)
```

### Memory Accumulation in Multi-Turn

Each turn's images add to the cumulative token count:

```
Turn 1: "What's this?" + [image_1]
  Tokens: 5 text + 576 image = 581 + assistant response (~50 tokens)

Turn 2: "And this?" + [image_2]
  Tokens: previous (631) + 3 text + 576 image = 1,210 + response (~50 tokens)

Turn 3: "Compare them"
  Tokens: previous (1,260) + 2 text = 1,262 + response (~100 tokens)

Total context: 1,362 tokens
  → All previous images are still in the KV cache
  → Deep conversations with many images can exhaust context length
```

---

## Dynamic vs. Fixed Resolution in Practice

### Fixed Resolution (Predictable)

```
LLaVA-1.5: always 336×336, always 576 tokens

Pros:
  ✓ Predictable memory per request
  ✓ Scheduler knows exact token count before processing
  ✓ Uniform batch characteristics → stable latency

Cons:
  ✗ High-res images lose detail (downsampled to 336×336)
  ✗ Small images are upsampled (wasted tokens)
  ✗ Bad for OCR/document tasks (need high resolution)
```

### Dynamic Resolution (Flexible)

```
Qwen2-VL: native resolution, variable token count

Pros:
  ✓ Best visual quality — process at actual resolution
  ✓ Small images use few tokens, large images use many
  ✓ Great for mixed workloads

Cons:
  ✗ Unpredictable memory per request
  ✗ A single high-res image can produce 2,304+ tokens unexpectedly
  ✗ Harder for the scheduler to plan capacity
```

### Mitigation for Dynamic Resolution

```bash
# Cap the worst case with max_pixels
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --mm-processor-kwargs '{"max_pixels": 602112}' \
  --max-model-len 8192

# This ensures:
# - No single image exceeds ~576 tokens
# - max-model-len can handle the worst case
# - Memory planning is bounded
```

---

## Memory Management for Multi-Modal Requests

### How the Scheduler Handles Visual Tokens

```
1. Request arrives with image(s)
2. vLLM estimates visual token count from image dimensions
   (before running the vision encoder)
3. Scheduler checks: total tokens (text + visual) fit in budget?
   ├── Yes → admit to batch
   └── No → wait in queue
4. Forward pass: vision encoding + LLM prefill
5. KV cache allocated for all tokens (text + visual)
6. During decode: no new visual tokens (they're all in the KV cache)
```

### Chunked Prefill for Multi-Image Requests

From Inference Blog 6: chunked prefill splits long prefills into chunks to prevent latency spikes.

This is critical for multi-image requests:

```
Without chunked prefill:
  Request with 5 images (2,880 tokens): prefill takes ~800ms
  → All decode requests blocked for 800ms
  → P99 ITL spikes from 30ms to 800ms

With chunked prefill:
  2,880 tokens split into 512-token chunks
  → 6 chunks, interleaved with decode steps
  → P99 ITL stays under 50ms
```

The vision encoder runs before chunked prefill — it produces all visual embeddings at once. Then the LLM's processing of those embeddings is chunked:

```
Step 1: Vision encoder (all images, ~50ms) — not chunked
Step 2: LLM prefill chunk 1 (512 tokens) + decode for other requests
Step 3: LLM prefill chunk 2 (512 tokens) + decode for other requests
...
Step 7: LLM prefill complete → request enters decode phase
```

---

## Use Cases

### Document Analysis (Multi-Page)

```python
# Process a multi-page document (each page as an image)
pages = [f"data:image/png;base64,{encode_page(i)}" for i in range(5)]

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            *[{"type": "image_url", "image_url": {"url": p}} for p in pages],
            {"type": "text", "text": "Summarize the key points from these 5 pages."},
        ],
    }],
    max_tokens=1000,
)
```

### Before/After Comparison

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image 1 is before renovation, Image 2 is after. What changed?"},
            {"type": "image_url", "image_url": {"url": "before.jpg"}},
            {"type": "image_url", "image_url": {"url": "after.jpg"}},
        ],
    }],
    max_tokens=300,
)
```

### Video Summarization

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "file:///data/meeting.mp4"}},
            {"type": "text", "text": "Summarize what happens in this meeting recording."},
        ],
    }],
    max_tokens=1000,
)
```

---

## Key Takeaways

1. **Multi-image requests multiply visual tokens** — 5 images = 2,880 tokens = 369 MB of KV cache per request
2. **Video is extremely expensive** — 32 frames = 18,432 tokens = 2.4 GB per request
3. **`--limit-mm-per-prompt`** prevents single requests from consuming all GPU memory
4. **Dynamic resolution** gives better quality but unpredictable memory — cap with `max_pixels`
5. **Chunked prefill is essential** for multi-image/video to avoid blocking decode requests
6. **Multi-turn conversations accumulate** — all previous images stay in the KV cache

---

## What's Next

Beyond images and video, vLLM also supports audio input. **Blog C4** covers audio and speech models — how audio becomes tokens and how to serve models like Qwen2-Audio and Ultravox.

---

## Further Reading

- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) — multi-image and video VLM
- [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191) — native video support
- **Next: [Blog C4 — Audio & Speech Models](/blog/mm-04-audio-models/)** — the audio modality in vLLM
