---
title: "Audio and Speech Models in vLLM"
description: "Whisper encoder, mel spectrograms, Qwen2-Audio, Ultravox, and strategies for handling long audio."
pubDate: 2026-05-03
series: "Vision, Language & Audio Models in vLLM"
part: 4
tags: ["audio", "Whisper", "speech", "mel spectrogram"]
---

## What Problem Does This Solve?

Vision-language models understand images. But what about audio? Meetings need transcription. Voice assistants need to understand spoken commands. Podcast analysis requires processing hours of audio. Audio-language models extend the same pattern: audio encoder → projection → LLM.

Unlike traditional speech-to-text systems (which transcribe audio to text, then feed text to an LLM), audio-language models process audio *directly* — no intermediate transcription step. This enables richer understanding: tone of voice, speaker emotion, background sounds, and even music.

---

## Audio-Language Model Architecture

### The Same Pattern as VLMs

```
VLM:   Image → Vision Encoder → Projection → LLM → Text output
Audio: Audio → Audio Encoder  → Projection → LLM → Text output

The architecture is identical — only the encoder changes.
```

### Full Pipeline

```
┌───────────────┐     ┌────────────┐     ┌─────────────┐     ┌──────────┐
│  Raw Audio    │     │   Audio    │     │ Projection  │     │   LLM    │
│  Waveform     │────►│  Encoder   │────►│   Layer     │────►│ Backbone │
│               │     │            │     │             │     │          │
│  16kHz mono   │     │ Mel spec → │     │ audio_dim → │     │ text +   │
│  1D signal    │     │ features   │     │ llm_dim     │     │ audio    │
└───────────────┘     └────────────┘     └─────────────┘     └──────────┘
                       Whisper             MLP/Linear          Llama/Qwen
```

---

## Audio Preprocessing: From Waveform to Features

### Raw Audio

Audio is a 1D signal — amplitude values sampled at a fixed rate:

```
Raw waveform (16kHz):
  Sample rate: 16,000 samples per second
  30 seconds of audio: 16,000 × 30 = 480,000 samples
  1 minute: 960,000 samples
  1 hour: 57,600,000 samples

Each sample is a float (amplitude at that instant in time).
The raw waveform is too low-level for a transformer to process.
```

### Mel Spectrogram

The standard preprocessing step converts the waveform to a 2D time-frequency representation:

```
Step 1: Short-Time Fourier Transform (STFT)
  Split audio into overlapping windows (25ms each, 10ms hop)
  Apply FFT to each window → frequency spectrum per window
  
  30 seconds → 3,000 windows × 201 frequency bins

Step 2: Mel scale mapping
  Map the linear frequency bins to mel scale
  (logarithmic scale matching human hearing perception)
  
  201 frequency bins → 80 mel bins (standard for Whisper)

Step 3: Log compression
  Apply log to the mel spectrogram
  → Compresses the dynamic range (makes quiet sounds visible)

Result: 2D feature map [time_steps, mel_bins]
  30 seconds → [3,000, 80]
  
  This is the "image" equivalent for audio.
  The audio encoder processes this 2D feature map.
```

```
Mel spectrogram visualization:

  Frequency ▲
  (mel)     │ ░░░░████░░░░░░░░░░█████░░░░  ← high frequencies
            │ ░░████████░░░░░░████████░░░░
            │ ██████████████████████████░░  ← speech formants
            │ ████████████████████████████  ← low frequencies
            └────────────────────────────► Time
              0s         15s           30s
              
  Brighter = more energy at that frequency at that time
```

---

## Audio Encoder Architectures

### Whisper Encoder

The most widely used audio encoder, from OpenAI:

```
Architecture:
  Input: mel spectrogram [3000, 80] (30-second chunk)
  Conv layers: 2 convolutions (stride 2 → downsample by 4×)
  Positional encoding: sinusoidal
  Transformer layers: 12-32 layers (depending on model size)
  Output: [1500, encoder_dim] (one feature per 20ms of audio)

Sizes:
  Whisper-tiny:    39M params, encoder_dim = 384
  Whisper-base:    74M params, encoder_dim = 512
  Whisper-small:   244M params, encoder_dim = 768
  Whisper-medium:  769M params, encoder_dim = 1024
  Whisper-large:   1.55B params, encoder_dim = 1280

Key property:
  30 seconds of audio → 1,500 encoder features
  Each feature represents ~20ms of audio
```

Originally designed for ASR (Automatic Speech Recognition), Whisper's encoder is now reused as a feature extractor in audio-language models.

### Qwen2-Audio Encoder

Modified Whisper encoder fine-tuned for general audio understanding:

```
Changes from standard Whisper:
  - Additional training on diverse audio (not just speech)
  - Handles music, environmental sounds, multiple speakers
  - Variable-length audio support (not just 30-second chunks)
  - Output: [time_steps, 1280] features
```

---

## Audio Tokens: Count and Memory

### Token Count Calculation

```
Audio → Mel spectrogram → Audio encoder → Projection → Audio tokens

30 seconds of audio:
  Mel spectrogram: [3000, 80]
  After encoder (Whisper-large): [1500, 1280]
  After downsampling (2×): [750, 1280]
  After projection: [750, llm_dim]
  
  = 750 audio tokens for 30 seconds

1 minute = 1,500 tokens
5 minutes = 7,500 tokens
1 hour = 90,000 tokens (!!!)
```

### Memory Comparison Across Modalities

```
Modality         Input                    Tokens    KV cache (8B model)
──────────────────────────────────────────────────────────────────────
Text             500 words                ~700      90 MB
Image            336×336 photo            576       74 MB
Audio            30 seconds speech        750       96 MB
Video (16 frames) 16 seconds clip         9,216     1.2 GB
Audio            5 minutes recording      7,500     960 MB

Audio is:
  - More expensive than a single image
  - Much cheaper than video (no visual redundancy across frames)
  - Scales linearly with duration (unlike images which are fixed)
```

---

## Serving Audio Models in vLLM

### Supported Models

```
Model            Params   Audio?   Image?   Notes
──────────────────────────────────────────────────────
Qwen2-Audio      7B       Yes      No       General audio understanding
Ultravox         7B       Yes      No       Real-time speech + text
```

### Launching

```bash
# Qwen2-Audio
vllm serve Qwen/Qwen2-Audio-7B-Instruct

# Ultravox (Whisper + Llama)
vllm serve fixie-ai/ultravox-v0_4
```

### Sending Audio Requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-Audio-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-encoded-wav>",
            "format": "wav"
          }
        }
      ]
    }],
    "max_tokens": 500
  }'
```

### Audio Input Methods

```
Base64 inline:
  {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}

URL:
  {"type": "audio_url", "audio_url": {"url": "https://example.com/speech.wav"}}

Local file:
  {"type": "audio_url", "audio_url": {"url": "file:///path/to/audio.wav"}}

Supported formats: WAV, MP3, FLAC, OGG
  (decoded to raw waveform internally, resampled to 16kHz)
```

### Python Client

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Encode audio file
with open("recording.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is being said in this audio? Also describe any background sounds."},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
        ],
    }],
    max_tokens=500,
)
print(response.choices[0].message.content)
```

---

## The Audio Preprocessing Pipeline

```
1. DECODE (CPU)
   │  WAV/MP3/FLAC → raw waveform (1D array of floats)
   │  → [num_samples] (e.g., 480,000 for 30s at 16kHz)
   │
2. RESAMPLE (CPU)
   │  Resample to model's expected rate (usually 16kHz)
   │  44.1kHz MP3 → 16kHz waveform
   │
3. MEL SPECTROGRAM (CPU)
   │  STFT → Mel scale → Log compression
   │  → [time_steps, mel_bins] (e.g., [3000, 80])
   │
4. CHUNK (CPU, if needed)
   │  Split into 30-second chunks (Whisper's fixed input size)
   │  → Multiple mel spectrograms for long audio
   │
5. ENCODE (GPU)
   │  Audio encoder (Whisper/custom)
   │  → [time_steps', encoder_dim]
   │
6. DOWNSAMPLE (GPU, optional)
   │  Merge adjacent frames to reduce token count
   │  → [time_steps'/2, encoder_dim]
   │
7. PROJECT (GPU)
   │  Linear/MLP projection
   │  → [num_audio_tokens, llm_dim]
   │
8. INTERLEAVE (GPU)
   │  Replace audio placeholders with audio embeddings
   │  → Combined text + audio sequence
   │
9. GENERATE (GPU)
      Standard autoregressive generation
```

---

## Ultravox: Real-Time Speech Understanding

### Why Ultravox Is Different

Traditional approach (two-stage):
```
Speech → Whisper (ASR) → Text → LLM → Response
                    ↑              ↑
              ~2 seconds      ~1 second
              Total: ~3 seconds
              
Problem: information loss in ASR (tone, emphasis, hesitation)
```

Ultravox (end-to-end):
```
Speech → Whisper encoder → Projection → Llama/Mistral → Response
                                              ↑
                                    ~1.5 seconds total
                                    
Benefit: no ASR bottleneck, preserves audio nuances
```

### Capabilities

```
- Direct speech understanding (no intermediate text)
- Mixed speech + text input ("Here's a recording [audio]. What does the speaker mean?")
- Lower latency than two-stage approaches
- Preserves prosody, emphasis, and speaking style in understanding
```

---

## Use Cases

### Transcription

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": "Transcribe this audio accurately."},
        ],
    }],
    max_tokens=1000,
)
```

### Audio Analysis

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": "Describe the sounds in this recording. Is there music, speech, or environmental sounds?"},
        ],
    }],
    max_tokens=300,
)
```

### Meeting Summarization

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": meeting_audio_b64, "format": "wav"}},
            {"type": "text", "text": "Summarize the key discussion points and action items from this meeting."},
        ],
    }],
    max_tokens=1000,
)
```

---

## Handling Long Audio

### The Duration Problem

```
Audio duration vs. token count:
  30 seconds:   750 tokens    → manageable
  5 minutes:   7,500 tokens   → significant memory
  30 minutes: 45,000 tokens   → exceeds most context limits
  1 hour:     90,000 tokens   → doesn't fit in any model
```

### Strategies for Long Audio

**Strategy 1: Chunking with overlap**
```
Split 30-minute audio into 5-minute chunks with 30-second overlap.
Process each chunk independently, then combine results.

  Chunk 1: 0:00 - 5:00  → process → summary_1
  Chunk 2: 4:30 - 9:30  → process → summary_2
  Chunk 3: 9:00 - 14:00 → process → summary_3
  ...
  
  Final: combine summaries into one
```

**Strategy 2: Selective processing**
```
For meetings: extract speech segments only (skip silence)
  30 minutes of meeting → 18 minutes of speech → 27,000 tokens
  Still large, but 40% reduction

For podcasts: extract the most "interesting" segments
  Voice activity detection → timestamp segments
  Process only the densest speech segments
```

**Strategy 3: Aggressive downsampling**
```
Increase the downsampling factor in the audio encoder.
Standard: 2× downsample (750 tokens per 30s)
Aggressive: 4× downsample (375 tokens per 30s)

Trade: fewer tokens = less detail but longer audio fits
```

---

## Key Takeaways

1. **Audio models follow the VLM pattern**: audio encoder (Whisper) → projection → LLM
2. **Audio preprocessing**: waveform → mel spectrogram → encoder features → LLM tokens
3. **Token count scales linearly with duration**: ~750 tokens per 30 seconds of audio
4. **Audio is more efficient than video** but more expensive than images for equivalent information
5. **Long audio is challenging**: chunk into segments or use selective processing for recordings longer than a few minutes
6. **Ultravox enables end-to-end speech understanding** without an intermediate ASR step

---

## What's Next

You've seen images (Blogs C1-C3) and audio (this blog). **Blog C5** dives into vLLM's multi-modal internals — how the `MultiModalRegistry`, `InputMapper`, and `MultiModalProcessor` coordinate all these modalities through a unified system.

---

## Further Reading

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — OpenAI's speech model
- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
- [Ultravox](https://github.com/fixie-ai/ultravox) — end-to-end speech-language model
- **Next: [Blog C5 — Multi-Modal Internals](/blog/mm-05-internals/)** — how vLLM routes and processes different modalities
