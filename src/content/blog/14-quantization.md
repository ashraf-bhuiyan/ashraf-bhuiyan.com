---
title: "Part 14: Quantization"
description: "Number formats from FP32 to INT4, symmetric quantization, weight-only dequantize-on-the-fly, and nibble packing for 2-4x memory reduction."
pubDate: 2026-05-02
series: "Building an LLM Inference Engine from Scratch"
part: 14
tags: ["quantization", "INT8", "INT4", "GPTQ", "AWQ"]
---

## What Problem Does This Solve?

A 70B parameter model in FP16 consumes 140GB of memory -- just for the weights. An H100 has 80GB of HBM. You need two GPUs at minimum, plus a tensor-parallel setup, plus the communication overhead between them. And that is before the KV cache eats another chunk of memory during inference.

Now imagine the same 70B model in INT4: 70 billion parameters at 0.5 bytes each equals 35GB. That fits on a single H100 with 45GB left over for KV cache. One GPU instead of two. Half the hardware cost. Half the power consumption.

```
FP16 (16-bit):
  70B params x 2 bytes = 140 GB
  Need: 2x H100 (80GB each) + tensor parallelism
  Cost: ~$60,000/year cloud rental

INT4 (4-bit):
  70B params x 0.5 bytes = 35 GB
  Need: 1x H100 (80GB)
  Cost: ~$30,000/year cloud rental

Same model. Half the GPUs.
```

Quantization trades numerical precision for memory savings. The question is how much quality you lose -- and the answer, with the right techniques, is surprisingly little.

---

## The Core Idea

Quantization replaces high-precision floating-point weights with lower-precision integers (or lower-precision floats). Instead of storing each weight as a 16-bit or 32-bit float, you store it as an 8-bit or 4-bit integer plus a small scaling factor that maps the integer back to an approximate float.

The math is simple. Given a weight tensor W:

```
Quantize:
  scale = max(|W|) / max_int_value
  W_int = round(W / scale)

Dequantize:
  W_approx = W_int * scale
```

For INT8 (range -128 to 127), `max_int_value` is 127. For INT4 (range -8 to 7), it is 7. The scale factor captures the magnitude of the original weights, and the integer captures the relative value within that range.

```
                     Quantize                    Dequantize
  FP32 Weight  ──────────────────►  INT8 + Scale ──────────────────►  FP32 (approx)
  [ 0.0723 ]      scale = 0.0033    [ 22 ]          22 * 0.0033       [ 0.0726 ]
  [-0.1541 ]      W / scale         [-47 ]         -47 * 0.0033       [-0.1551 ]
  [ 0.2890 ]      then round        [ 88 ]          88 * 0.0033       [ 0.2904 ]
  [ 0.4156 ]                        [127 ]         127 * 0.0033       [ 0.4191 ]
```

The error is small because INT8 has 256 levels -- enough to represent most weight distributions with minimal distortion. INT4 has only 16 levels, so the approximation is coarser, but the memory savings are dramatic.

---

## How It Works

### Number Formats: The Bit-Level View

Every number format in deep learning is a tradeoff between range, precision, and storage size:

```
FP32 (32 bits): [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
                 1    8 exp              23 mantissa
                 Range: +/- 3.4e38      Precision: ~7 decimal digits

FP16 (16 bits): [S][EEEEE][MMMMMMMMMM]
                 1   5 exp    10 mantissa
                 Range: +/- 65504      Precision: ~3.3 decimal digits

BF16 (16 bits): [S][EEEEEEEE][MMMMMMM]
                 1    8 exp     7 mantissa
                 Range: +/- 3.4e38     Precision: ~2.4 decimal digits

FP8 E4M3 (8 bits): [S][EEEE][MMM]
                     1  4 exp  3 mantissa
                     Range: +/- 448    Precision: ~1.7 decimal digits

FP8 E5M2 (8 bits): [S][EEEEE][MM]
                     1   5 exp  2 mantissa
                     Range: +/- 57344  Precision: ~1.2 decimal digits

INT8 (8 bits): [SSSSSSSS]   (two's complement)
               Range: -128 to 127     256 uniformly spaced levels

INT4 (4 bits): [SSSS]       (two's complement)
               Range: -8 to 7        16 uniformly spaced levels
```

Key distinctions:

- **BF16 vs FP16**: BF16 trades mantissa precision for FP32-matching exponent range. Better for training where large gradients matter. FP16 is more precise but overflows at 65504.
- **FP8 E4M3 vs E5M2**: E4M3 is used for weights and activations (better precision). E5M2 is used for gradients (wider range needed). On H100 GPUs, both have hardware tensor core support.
- **FP8 vs INT8**: FP8 preserves floating-point dynamic range -- values near zero get more precision, large values get less. INT8 spaces values uniformly. For weight distributions that cluster near zero (most neural networks), FP8 can be more accurate at the same bit width.

```
Memory for a 70B-parameter model:
  ┌────────┬──────┬───────────────┬──────────────────┐
  | Format | Bits | Memory (70B)  | Values per byte  |
  +--------+------+---------------+------------------+
  | FP32   |  32  |  280 GB       | 0.25             |
  | FP16   |  16  |  140 GB       | 0.5              |
  | BF16   |  16  |  140 GB       | 0.5              |
  | FP8    |   8  |   70 GB       | 1                |
  | INT8   |   8  |   70 GB       | 1                |
  | INT4   |   4  |   35 GB       | 2                |
  └────────┴──────┴───────────────┴──────────────────┘
```

### Symmetric Quantization

Our implementation uses **symmetric quantization** -- the zero point is always zero, and the scale is derived from the absolute maximum of the weights:

```python
def quantize_per_tensor_int8(weight):
    w_max = weight.abs().max()
    scale = w_max / 127.0
    w_int8 = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return w_int8, scale
```

Asymmetric quantization adds a zero-point offset (`W_int = round((W - zero_point) / scale)`) to handle distributions that are not centered at zero. Symmetric is simpler and works well for most weight distributions, which tend to be roughly symmetric around zero.

### Per-Tensor vs Per-Channel Quantization

Per-tensor quantization uses a single scale for the entire weight matrix. This is fast but imprecise -- one outlier row with large values forces a coarse scale for all rows, wasting precision in rows with small values.

Per-channel quantization (per output channel, i.e., per row) gives each row its own scale:

```
Per-tensor (1 scale for entire matrix):
  ┌─────────────────────────────┐
  │ Row 0: [0.01, 0.02, 0.03]  │  All rows share
  │ Row 1: [0.10, 0.20, 0.30]  │  scale = 5.0/127
  │ Row 2: [1.00, 2.00, 5.00]  │  = 0.0394
  └─────────────────────────────┘
  Row 0 values map to: [0, 1, 1] -- terrible resolution!
  Row 2 values map to: [25, 51, 127] -- fine resolution

Per-channel (1 scale per row):
  ┌─────────────────────────────┐
  │ Row 0: scale = 0.03/127    │  Each row gets its own scale
  │ Row 1: scale = 0.30/127    │  tailored to its range
  │ Row 2: scale = 5.00/127    │
  └─────────────────────────────┘
  Row 0 values map to: [42, 85, 127] -- much better!
  Row 2 values map to: [25, 51, 127] -- same as before
```

The demo shows this concretely: per-channel INT8 produces **11.6x lower mean absolute error** than per-tensor on a real model layer. The overhead is minimal -- one extra float32 scale per output channel, which is negligible compared to the weight matrix itself.

### Weight-Only Quantization: Store INT, Compute in FP

Our approach is **weight-only quantization** with dequantize-on-the-fly. The weights are stored in low precision (INT8 or INT4) to save memory, but during the forward pass they are dequantized back to floating point before the matrix multiply:

```
Forward pass with weight-only quantization:

  Input (FP16)         INT8 Weights + Scale
       |                      |
       |                  Dequantize
       |                  (INT8 * scale → FP16)
       |                      |
       └──────── matmul ──────┘
                   |
              Output (FP16)
```

This is the simplest form of quantization. It saves memory but does not save compute -- the matrix multiply still happens in FP16. The dequantization overhead is small because it is a simple element-wise multiply.

More advanced approaches quantize both weights and activations, then use integer or FP8 tensor cores for the matrix multiply itself. This saves both memory and compute, but requires hardware support (INT8 tensor cores on A100, FP8 tensor cores on H100).

Here's how the three quantization approaches differ:

```
Weight-only quantization (what we implement):
  ┌──────────┐     ┌──────────────┐     ┌───────────┐
  │ INT8     │────►│ Dequantize   │────►│ FP16      │
  │ Weights  │     │ (INT8×scale  │     │ Weights   │
  │ (stored) │     │  → FP16)     │     │ (on-the-  │
  └──────────┘     └──────────────┘     │  fly)     │
                                         └─────┬─────┘
  ┌──────────┐                                 │
  │ FP16     │─────────────── matmul ──────────┤
  │ Input    │                                 │
  └──────────┘                                 ▼
                                         ┌───────────┐
  Saves: memory                          │ FP16      │
  Compute: same FLOPs                    │ Output    │
                                         └───────────┘

Weight + Activation quantization (FP8 on H100):
  ┌──────────┐                         ┌───────────┐
  │ FP8      │────── FP8 matmul ──────►│ FP16/FP32 │
  │ Weights  │     (tensor core)       │ Output    │
  └──────────┘         ▲               └───────────┘
                       │
  ┌──────────┐         │
  │ FP16     │── quantize to FP8 ──┘
  │ Input    │   (dynamic, per-tensor)
  └──────────┘

  Saves: memory AND compute (2x tensor core throughput)
```

### INT4 Packing: Two Values in One Byte

INT4 values range from -8 to 7 -- four bits. But hardware is byte-addressable, so you cannot store a single 4-bit value in memory. The solution is to pack two INT4 values into one uint8 byte:

```
Packing two INT4 values into one byte:

  Value A = 5  (binary: 0101)
  Value B = -3 (binary: 1101, as unsigned nibble)

  Packed byte: [0101 | 1101] = 0x5D
                 A       B
               high    low
              nibble  nibble

Unpacking:
  A = (packed >> 4) & 0x0F  = 0101 = 5
  B = packed & 0x0F         = 1101 → sign-extend → -3
```

Sign extension is needed because the 4-bit value is stored unsigned in the nibble. If the value exceeds 7 (i.e., bit 3 is set), it represents a negative number and must be adjusted by subtracting 16.

### Production Methods: GPTQ, AWQ, SmoothQuant, FP8

Our naive round-to-nearest approach works, but production systems use calibration-based methods that achieve dramatically lower error at the same bit width.

**GPTQ (Generative Pre-Trained Transformer Quantization):** Quantizes weights column by column, using the inverse Hessian matrix to compensate for quantization error. When you quantize column j, the error is distributed across the not-yet-quantized columns j+1, j+2, ... so that the overall output of the layer changes as little as possible. Think of it as "budgeting" error -- columns that contribute more to the output (high Hessian diagonal) are quantized more carefully.

**AWQ (Activation-Aware Weight Quantization):** Observes that a small fraction of weights matter far more than others -- specifically, the weights connected to channels with large activation magnitudes. AWQ scales up these "salient" channels before quantization (giving them more integer levels) and scales down the others. The scaling factors are determined by running calibration data through the model and measuring activation magnitudes.

**SmoothQuant:** Addresses the problem of outlier activations. Some activation channels have values 100x larger than others, making activation quantization difficult. SmoothQuant migrates this difficulty from activations to weights by dividing activations by a per-channel smoothing factor and multiplying weights by the same factor. Since weights have a more uniform distribution, they absorb the extra range more gracefully.

```
SmoothQuant transformation:
  Y = X * W
  Y = (X / s) * (s * W)    where s = per-channel smoothing factor
       ^^^^      ^^^^
   Easier to     Slightly harder to
   quantize      quantize (but weights
   (smaller      are more uniform than
    outliers)    activations)
```

**FP8 (E4M3/E5M2):** Instead of integer quantization, FP8 keeps the floating-point format but reduces to 8 bits. E4M3 (4 exponent, 3 mantissa bits) is used for weights and activations; E5M2 (5 exponent, 2 mantissa bits) is used for gradients. H100 GPUs have native FP8 tensor cores that achieve 2x the throughput of FP16 tensor cores. FP8 preserves dynamic range naturally -- values near zero get more relative precision -- making it less sensitive to outliers than INT8.

---

## How vLLM/SGLang Implements This

| Our Code | Real vLLM | Notes |
|----------|-----------|-------|
| `quantize_per_channel_int8()` | `torch.int8` quantization via Marlin/Machete kernels | Fused dequant + matmul in custom CUDA |
| `quantize_per_channel_int4()` | GPTQ/AWQ 4-bit via `marlin`, `exllama`, `exllamav2` kernels | Hessian/activation-aware, not naive round |
| `QuantizedLinearINT8` | `Int8LinearMethod` / `CompressedTensorsLinearMethod` | Selects optimal kernel per GPU arch |
| `QuantizedLinearINT4` | `GPTQLinearMethod` / `AWQLinearMethod` | Loads pre-quantized checkpoints |
| `dequantize_int8()` (separate step) | Fused dequant in GEMM kernel | No separate dequant pass -- fused for speed |
| Per-channel scale | Per-group scale (group_size=128) | Finer granularity than per-channel |
| No FP8 support | `Fp8LinearMethod` with E4M3 | H100 tensor core acceleration |
| `model.eval()` + manual replace | `quantization` config in `LLMEngine` | Auto-selects method from checkpoint metadata |

Key details in production:

**Fused dequantize-GEMM kernels:** Our implementation dequantizes weights to FP16, then calls `F.linear()` -- two separate operations. Production systems like Marlin fuse the dequantization into the matrix multiply kernel itself. The INT4 values are unpacked and converted to FP16 on-the-fly inside the CUDA kernel, directly into the register file, avoiding a full-size intermediate FP16 tensor in memory.

**Per-group quantization:** Rather than per-tensor (one scale) or per-channel (one scale per row), production systems use per-group with a typical group size of 128. Each contiguous group of 128 elements along the input dimension shares a scale. This provides finer granularity than per-channel while keeping scale overhead small (one FP16 scale per 128 weights = 0.8% overhead).

**Checkpoint-based quantization:** GPTQ and AWQ quantization is done offline as a preprocessing step. The quantized weights, scales, and zero points are saved to a checkpoint. vLLM detects the quantization format from the checkpoint metadata and loads the appropriate kernel. There is no online quantization during model loading.

**FP8 on H100:** vLLM supports FP8 E4M3 weight-activation quantization using the H100's native FP8 tensor cores. This provides 2x higher throughput than FP16 with minimal quality loss, and no custom quantization preprocessing is needed -- dynamic FP8 quantization can be applied at runtime.

---

## The Implementation

The complete implementation is in [14_quantization.py](https://github.com/ashraf-bhuiyan/code-blog-simple-vllm/blob/main/14_quantization.py) (~690 lines).

### Quantization Primitives

INT8 per-channel quantization is the workhorse:

```python
def quantize_per_channel_int8(weight: torch.Tensor):
    w_max = weight.abs().amax(dim=1)       # max per row
    scales = w_max / 127.0                 # one scale per output channel
    scales = scales.clamp(min=1e-10)       # avoid division by zero
    w_scaled = weight / scales.unsqueeze(1)
    w_int8 = torch.clamp(torch.round(w_scaled), -128, 127).to(torch.int8)
    return w_int8, scales
```

Each row gets its own scale, computed as `max(|row|) / 127`. The `unsqueeze(1)` broadcasts the per-row scale across all columns during division.

INT4 adds the packing step:

```python
def quantize_per_channel_int4(weight: torch.Tensor):
    out_features, in_features = weight.shape
    w_max = weight.abs().amax(dim=1)
    scales = w_max / 7.0                   # INT4 range is -8 to 7
    scales = scales.clamp(min=1e-10)
    w_scaled = weight / scales.unsqueeze(1)
    w_int4 = torch.clamp(torch.round(w_scaled), -8, 7).to(torch.int8)

    # Pack two INT4 values into one INT8 byte
    even = (w_int4[:, 0::2] & 0x0F).to(torch.uint8)
    odd = (w_int4[:, 1::2] & 0x0F).to(torch.uint8)
    packed = (even << 4) | odd

    return packed, scales, weight.shape
```

The packing extracts even-indexed columns into the high nibble and odd-indexed columns into the low nibble of each byte. This halves the storage.

### Quantized Linear Layers

The quantized linear layer stores weights as integers and dequantizes during the forward pass:

```python
class QuantizedLinearINT8(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly: INT8 * scale → float
        w_float = dequantize_int8(self.weight_int8, self.scales).to(x.dtype)
        out = F.linear(x, w_float)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out
```

The `from_float()` class method converts a standard `nn.Linear` to quantized form:

```python
@classmethod
def from_float(cls, linear: nn.Linear):
    q = cls(linear.in_features, linear.out_features,
            bias=linear.bias is not None)
    w_int8, scales = quantize_per_channel_int8(linear.weight.data)
    q.weight_int8.copy_(w_int8)
    q.scales.copy_(scales)
    return q
```

### Model-Wide Quantization

The `quantize_model()` function walks the model's module tree and replaces every `nn.Linear` with its quantized equivalent:

```python
def quantize_model(model, bits=8):
    QuantizedClass = QuantizedLinearINT8 if bits == 8 else QuantizedLinearINT4
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                quantized = QuantizedClass.from_float(child)
                setattr(module, child_name, quantized)
    return model
```

This replaces layers in-place. After quantization, the original FP32 weights are freed, and the model holds only INT8/INT4 weights plus their scales.

### Inference Engine

The `QuantizedEngine` wraps model loading, quantization, and generation into one class:

```python
class QuantizedEngine:
    def __init__(self, model_name, bits=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.fp32_memory = model_memory_mb(self.model)

        if bits is not None:
            quantize_model(self.model, bits=bits)

        self.quantized_memory = model_memory_mb(self.model)
```

Passing `bits=None` keeps FP32, `bits=8` gives INT8, `bits=4` gives INT4. The engine tracks both original and quantized memory usage for comparison.

---

## Running the Code

**Demo mode** (quantizes a real model, compares precision levels):

```bash
python 14_quantization.py --demo
```

**Server mode** (serve with a specific precision):

```bash
# FP32 (no quantization)
python 14_quantization.py --port 5000

# INT8 quantization
python 14_quantization.py --port 5000 --bits 8

# INT4 quantization
python 14_quantization.py --port 5000 --bits 4

# Send a request
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 20}'

# Check memory usage
curl http://localhost:5000/health
```

**Custom model:**

```bash
python 14_quantization.py --demo --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Expected demo output:

```
--- Part 1: How Quantization Works ---

  Original FP32 weights (8 values):
    ['0.0723', '-0.1541', '0.2890', '-0.0312', '0.4156', '-0.3678', '0.1234', '-0.0891']
    Memory: 32 bytes (4 bytes each)

  INT8 quantized (scale=0.003272):
    [22, -47, 88, -10, 127, -112, 38, -27]
    Max error: 0.001091

  INT4 quantized (scale=0.059371):
    Packed: [133, 246, 114, 161] (4 bytes)
    Max error: 0.035914

--- Part 4: Memory Comparison ---

  ┌──────────┬────────────┬────────────┬───────────┐
  | Precision| Memory     | % of FP32  | KL Div    |
  +----------+------------+------------+-----------+
  | FP32     | 4196.4 MB  |    100%    |   0.0     |
  | INT8     | 1238.5 MB  |     30%    |  0.000509 |
  | INT4     |  745.2 MB  |     18%    |  0.351619 |
  └──────────┴────────────┴────────────┴───────────┘
```

Part 5 shows generation quality side by side:

```
--- Part 5: Generation Quality ---

  Prompt: "The capital of France is"
    FP32: "Paris, which is also the largest city in the"
    INT8: "Paris, which is also the largest city in the"
    INT4: "Paris.\nThe French Republic is a unitary semi-"

  Prompt: "Machine learning is"
    FP32: "a subset of artificial intelligence that uses"
    INT8: "a subset of artificial intelligence that uses"
    INT4: "a type of artificial intelligence that allows"
```

INT8 produces output identical to FP32 -- the quantization error is below the noise floor of greedy decoding. INT4 diverges slightly: same factual content but different phrasing, because the coarser quantization shifts probability mass enough to alter token selection.

---

## Benchmarks

Results from quantizing TinyLlama-1.1B-Chat-v1.0 (1.1 billion parameters):

| Metric | FP32 | INT8 | INT4 |
|--------|------|------|------|
| Model memory | 4196.4 MB | 1238.5 MB (30%) | 745.2 MB (18%) |
| Memory savings | -- | 70% | 82% |
| KL divergence from FP32 | 0.0 | 0.000509 | 0.351619 |
| Top-1 prediction match | -- | Identical | Often matches |
| Generated text quality | Reference | Identical to FP32 | Minor phrasing differences |

Per-tensor vs per-channel comparison (on a single linear layer):

| Quantization Granularity | Mean Absolute Error | Relative Quality |
|--------------------------|--------------------:|:----------------:|
| Per-tensor INT8 | 0.00021173 | 1x |
| Per-channel INT8 | 0.00001824 | 11.6x better |

The 11.6x improvement from per-channel quantization comes at almost zero cost -- just one extra float32 value per output channel. For a layer with 2048 output channels, that is 8KB of scales versus millions of bytes saved from INT8 compression.

**Scaling to 70B models (projected):**

| Config | Memory | GPUs Required (80GB H100) |
|--------|--------|--------------------------|
| FP16 | 140 GB | 2 |
| INT8 | 70 GB | 1 |
| INT4 (GPTQ/AWQ) | 35 GB | 1 (with room for KV cache) |
| FP8 (H100 native) | 70 GB | 1 (with 2x FLOPS vs FP16) |

---

## Key Takeaways

1. **Quantization trades precision for memory.** INT8 reduces memory by 50% vs FP16 (75% vs FP32) with negligible quality loss. INT4 reduces memory by 75% vs FP16 with small but measurable degradation.

2. **Per-channel quantization is dramatically better than per-tensor.** Giving each output channel its own scale factor captures the different magnitude distributions across neurons, yielding 11.6x lower error in our demo.

3. **Weight-only quantization stores INT but computes in FP.** The weights are dequantized on-the-fly during the forward pass. This saves memory but not compute FLOPs. Full weight+activation quantization with INT8/FP8 tensor cores saves both.

4. **INT4 packing stores two values per byte.** Since hardware is byte-addressable, two 4-bit values are packed into one uint8 using bit shifting. This is how GPTQ and AWQ checkpoints are stored.

5. **Production methods (GPTQ, AWQ, SmoothQuant) use calibration data** to minimize quantization error. GPTQ compensates error via the Hessian; AWQ protects weights connected to large activations; SmoothQuant smooths outlier activations. All achieve far better quality than naive rounding at the same bit width.

6. **FP8 is the hardware-native path on modern GPUs.** H100 tensor cores natively support FP8 E4M3, providing 2x throughput over FP16 with floating-point dynamic range instead of uniform integer spacing.

---

## Further Reading

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) -- Hessian-based weight quantization
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) -- protecting salient weights
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438) -- migrating quantization difficulty from activations to weights
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) -- the E4M3/E5M2 format specification
- [vLLM Quantization Documentation](https://docs.vllm.ai/en/latest/features/quantization/) -- supported formats and configuration
- [Marlin: Mixed-Precision GEMM Kernels](https://github.com/IST-DASLab/marlin) -- fused INT4 dequantize-matmul kernels used by vLLM
- **Next: [Blog 15 -- Full Architecture](15_full_architecture.md)** -- putting all the pieces together into a complete inference engine
