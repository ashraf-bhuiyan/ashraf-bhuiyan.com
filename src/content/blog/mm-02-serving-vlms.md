---
title: "Serving VLMs in vLLM: Your First Image + Text Request"
description: "Launch a VLM, send image requests via the OpenAI vision API, understand chat templates and the preprocessing pipeline."
pubDate: 2026-05-03
series: "Vision, Language & Audio Models in vLLM"
part: 2
tags: ["VLM", "serving", "OpenAI API", "image"]
---

## What Problem Does This Solve?

You understand how VLMs convert images into tokens (Blog C1). Now you want to actually serve one — send an image and a question, get a text answer. vLLM handles all the complexity: fetching images, preprocessing, vision encoding, token interleaving, and generation. You just send a request.

---

## Starting vLLM with a VLM

### Basic Launch

Most VLMs work out of the box — vLLM auto-detects multi-modal capabilities from the model's architecture:

```bash
# Qwen2-VL (dynamic resolution, strong general-purpose)
vllm serve Qwen/Qwen2-VL-7B-Instruct

# LLaVA-OneVision (latest LLaVA, multi-image support)
vllm serve llava-hf/llava-onevision-qwen2-7b-ov-hf

# InternVL2 (dynamic tiling, high resolution)
vllm serve OpenGVLab/InternVL2-8B

# Llama 3.2 Vision
vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct
```

No `--task` flag needed (unlike embeddings which require `--task embed`). VLMs are generative models — they output text just like any other LLM.

### Key Configuration Options

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --max-model-len 32768 \
  --limit-mm-per-prompt image=5,video=1 \
  --mm-processor-kwargs '{"max_pixels": 1003520}'
```

| Option | Purpose | Default |
|--------|---------|---------|
| `--max-model-len` | Max total tokens (text + visual) | Model's max |
| `--limit-mm-per-prompt` | Max images/videos/audios per request | Model-specific |
| `--mm-processor-kwargs` | Model-specific preprocessing options | Model defaults |
| `--tensor-parallel-size` | TP for large VLMs | 1 |

### Why `--max-model-len` Matters More for VLMs

Each image contributes hundreds of visual tokens:

```
Text-only request:  "Describe a cat" → ~5 tokens
VLM request:        "Describe this:" + image → 5 + 576 = 581 tokens

With 3 images: 5 + 3×576 = 1,733 tokens
With high-res: 5 + 2,304 = 2,309 tokens

If max-model-len is too low, requests with images may be rejected.
Set it high enough for your expected image count + resolution.
```

---

## Sending Image Requests

### OpenAI Vision API Format

vLLM implements the OpenAI vision API, so you can use the same format:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 200
  }'
```

### Image Input Methods

**Method 1: URL** — vLLM fetches the image

```json
{
  "type": "image_url",
  "image_url": {"url": "https://example.com/photo.jpg"}
}
```

**Method 2: Base64** — embed the image data inline

```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
}
```

**Method 3: Local file** — reference a file on the server

```json
{
  "type": "image_url",
  "image_url": {"url": "file:///path/to/image.jpg"}
}
```

### Supported Image Formats

```
JPEG (.jpg, .jpeg)  — most common, good for photos
PNG (.png)          — lossless, supports transparency
WebP (.webp)        — modern format, good compression
GIF (.gif)          — first frame extracted
BMP (.bmp)          — uncompressed
TIFF (.tiff)        — high quality, large files
```

### Python Client

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Method 1: URL
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail."},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
    max_tokens=300,
)
print(response.choices[0].message.content)

# Method 2: Base64 (for local files)
with open("photo.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What text is visible in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ],
    }],
    max_tokens=500,
)
```

---

## Chat Templates and Image Placeholders

### How It Works

Each VLM has its own chat template that specifies where images go. When you send an OpenAI-format request, vLLM:

1. Converts the structured message to the model's native format
2. Inserts image placeholder tokens at the right positions
3. The preprocessing pipeline replaces placeholders with visual embeddings

```
Your request:
  {"type": "text", "text": "What's in this image?"}
  {"type": "image_url", "image_url": {"url": "..."}}

vLLM converts to (Qwen2-VL format):
  "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What's in this image?<|im_end|>\n<|im_start|>assistant\n"

Tokenized (simplified):
  [..., VISION_START, IMAGE_PAD×576, VISION_END, "What", "'s", ...]
                      ↑
                These 576 IMAGE_PAD tokens get replaced
                with actual visual embeddings during forward pass
```

You don't need to know the model's native format — vLLM handles the conversion. But understanding it helps when debugging.

### Model-Specific Formats

```
LLaVA:
  "<image>\nWhat's in this image?"
  → <image> expands to 576 visual tokens

Qwen2-VL:
  "<|vision_start|><|image_pad|><|vision_end|>What's in this image?"
  → <|image_pad|> repeats for each visual token

InternVL2:
  "<img><IMG_CONTEXT></img>\nWhat's in this image?"
  → <IMG_CONTEXT> expands to tile_count × 256 tokens

Llama 3.2 Vision:
  "<|image|>What's in this image?"
  → <|image|> expands to the visual token count
```

---

## The Preprocessing Pipeline

### Step-by-Step

When vLLM receives an image request:

```
1. FETCH (CPU)
   │  Download image from URL or decode base64
   │  → Raw bytes (JPEG/PNG/WebP)
   │
2. DECODE (CPU)
   │  Decode image format to pixel array
   │  → PIL Image (H × W × 3)
   │
3. RESIZE (CPU)
   │  Resize to model's expected resolution
   │  LLaVA: always 336×336
   │  Qwen2-VL: keep native, within min/max bounds
   │  InternVL: tile into 448×448 tiles
   │  → Resized image(s)
   │
4. NORMALIZE (CPU)
   │  Apply model-specific normalization
   │  Subtract mean, divide by std (per RGB channel)
   │  → Normalized tensor [C, H, W]
   │
5. ENCODE (GPU)
   │  Run through vision encoder (ViT/SigLIP)
   │  → Patch features [num_patches, vision_dim]
   │
6. PROJECT (GPU)
   │  Projection layer (MLP/cross-attention)
   │  → Visual tokens [num_tokens, llm_dim]
   │
7. INTERLEAVE (GPU)
   │  Replace placeholder tokens with visual embeddings
   │  → Combined text+visual sequence
   │
8. GENERATE (GPU)
   │  Standard autoregressive generation
   │  → Output text
```

Steps 1-4 happen on CPU (preprocessing). Steps 5-8 happen on GPU. The CPU preprocessing runs in parallel with GPU work from other requests, so it rarely becomes a bottleneck.

---

## Practical Examples

### Image Captioning

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/sunset.jpg"}},
            {"type": "text", "text": "Describe this image in one paragraph."},
        ],
    }],
    max_tokens=200,
)
```

### Visual Question Answering (VQA)

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
            {"type": "text", "text": "What is the highest value shown in this bar chart?"},
        ],
    }],
    max_tokens=100,
)
```

### Document Understanding / OCR

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Extract all text from this receipt image and list the items with their prices."},
        ],
    }],
    max_tokens=500,
)
```

### Diagram Analysis

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/architecture.png"}},
            {"type": "text", "text": "Explain the system architecture shown in this diagram."},
        ],
    }],
    max_tokens=400,
)
```

---

## Common Pitfalls

### 1. Context Length Exceeded

```
Error: "Input too long: 35,000 tokens exceed max_model_len=32,768"

Cause: high-resolution image produced too many visual tokens
Fix:   increase --max-model-len or reduce image resolution:
       --mm-processor-kwargs '{"max_pixels": 602112}'
```

### 2. Slow First Request

```
Observation: first request takes 10+ seconds, subsequent requests are fast

Cause: vision encoder warmup (CUDA kernels compiled on first use)
Fix:   this is normal — the first request warms up the GPU.
       Send a throwaway request before serving traffic.
```

### 3. Out of Memory

```
Error: "CUDA out of memory"

Cause: images inflate token count → KV cache overflow
Fix:   reduce --max-model-len, limit images with --limit-mm-per-prompt,
       or reduce image resolution with --mm-processor-kwargs
```

### 4. Wrong Image Format

```
Error: "Cannot decode image"

Cause: corrupted file, unsupported format, or truncated download
Fix:   verify the image opens with PIL:
       from PIL import Image; Image.open("file.jpg").verify()
```

---

## Supported VLM Models

A selection of models supported by vLLM:

```
Model                       Params   Images   Video   Resolution
─────────────────────────────────────────────────────────────────
LLaVA-1.5                  7/13B    Single   No      336×336
LLaVA-1.6 (LLaVA-NeXT)     7/13B    Single   No      Dynamic tiles
LLaVA-OneVision             7/72B    Multi    Yes     Dynamic
Qwen2-VL                    2/7/72B  Multi    Yes     Native resolution
Qwen2.5-VL                  3/7/72B  Multi    Yes     Native resolution
InternVL2                   1-76B    Multi    Yes     Dynamic tiles
PaliGemma / PaliGemma2      3/28B    Single   No      Fixed (224/448)
Phi-3-Vision                4.2B     Multi    No      Dynamic tiles
Phi-3.5-Vision              4.2B     Multi    No      Dynamic tiles
Pixtral (Mistral)           12B      Multi    No      Native resolution
Molmo                       7/72B    Multi    No      Fixed
Llama 3.2 Vision            11/90B   Single   No      Fixed
```

Check the vLLM documentation for the latest supported model list — new VLMs are added frequently.

---

## Key Takeaways

1. **VLMs work out of the box** in vLLM — just `vllm serve model-name`, no special flags needed
2. **Use the OpenAI vision API format** — structured messages with `image_url` entries
3. **Three image input methods**: URL (vLLM fetches), base64 (inline), or local file
4. **Chat templates handle placeholder insertion** — you don't need to know the model's native format
5. **Watch the token count**: images can use 576-2,304 tokens of KV cache. Set `--max-model-len` and `--limit-mm-per-prompt` accordingly.
6. **Preprocessing runs on CPU** in parallel with GPU work — rarely a bottleneck

---

## What's Next

Single-image requests are the starting point. **Blog C3** covers multi-image and video input — sending multiple images per request, processing video as frames, and managing the memory explosion.

---

## Further Reading

- [vLLM Multi-Modal Documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html)
- [OpenAI Vision API Reference](https://platform.openai.com/docs/guides/vision) — the API spec vLLM implements
- **Next: [Blog C3 — Multi-Image & Video Input](/blog/mm-03-multi-image-video/)** — handling multiple images and video frames
