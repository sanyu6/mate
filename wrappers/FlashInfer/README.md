# FlashInfer Compatibility Wrapper (flashinfer)

`flashinfer` preserves a focused set of FlashInfer Python APIs while running
the operations on MUSA through MATE.

## Overview

The wrapper currently covers the following integration areas:

| Area | Public surface |
| --- | --- |
| GEMM | BF16/FP16 BMM, FP8 BMM, groupwise FP8 GEMM, and grouped FP8 GEMM |
| Norm | RMSNorm, LayerNorm, fused add RMSNorm, quantized outputs, and diffusion-model fused norms |
| Quantization | FlashInfer MXFP4/MXFP8 and bit packing APIs with native SF layout |
| MLA RoPE | Fused RoPE and FP8 quantization for 512-value NoPE and 64-value RoPE components |
| Sparse MLA decode | FP8 sparse decode with optional reusable scheduler metadata |
| GLM NoPE sparse MLA | Native `mla.SparseMLASm120Wrapper` with compact 528-byte or padded 656-byte KV rows |

The GEMM APIs align with official FlashInfer v0.6.17. This is a compatibility
wrapper rather than a complete FlashInfer distribution; only the APIs listed
below are provided.

## Package and import

- Distribution name: `flashinfer-python`
- Import path: `flashinfer`
- Runtime backend: MATE operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`. Use
`python -m pip show flashinfer-python` to distinguish this wrapper from the
native package.

```python
import flashinfer
import flashinfer.decode
import flashinfer.gated_act_mxfp8
import flashinfer.gemm
import flashinfer.norm
import flashinfer.quantization
import flashinfer.rope
import flashinfer.mla
```

## Requirements

- TorchMUSA is installed and the MUSA runtime environment is configured.
- A matching MATE version is installed.
- Operator inputs are placed on a MUSA device.

## Build and installation

Run the following commands from `wrappers/FlashInfer`. Install the local MATE
checkout first, then install the wrapper in editable mode:

```bash
python -m pip install --no-build-isolation --no-deps -e ../.. -v
python -m pip install --no-build-isolation --no-deps -e .
```

Build and install a local wheel with:

```bash
python -m build --wheel
python -m pip install --no-deps dist/flashinfer_python-*.whl
```

## GEMM APIs

```python
from flashinfer.gemm import (
    batch_deepgemm_fp8_nt_groupwise,
    bmm_bf16,
    bmm_fp8,
    gemm_fp8_nt_groupwise,
    group_deepgemm_fp8_nt_groupwise,
)
```

`bmm_bf16` and `bmm_fp8` are also exported from the `flashinfer` package root.

| API | Supported behavior | Backend |
| --- | --- | --- |
| `bmm_bf16` | FP16 or BF16 inputs; BF16, FP16, or FP32 output | `auto`, `mudnn` |
| `bmm_fp8` | FP8 inputs with scalar or rank-3 scales; BF16 or FP16 output | `auto`, `mudnn` |
| `gemm_fp8_nt_groupwise` | Groupwise FP8 NT GEMM; optional FP8 output scaling | `auto`, `mudnn`, or `mubin` depending on output mode |
| `group_deepgemm_fp8_nt_groupwise` | Contiguous grouped FP8 NT GEMM with BF16 output | MATE grouped GEMM |
| `batch_deepgemm_fp8_nt_groupwise` | Masked batched FP8 NT GEMM with BF16 output | MATE grouped GEMM |

The BMM signatures follow FlashInfer's argument order:

```python
bmm_bf16(A, B, out=None, out_dtype=torch.bfloat16, backend="auto")
bmm_fp8(A, B, A_scale, B_scale, dtype, out=None, backend="auto")
```

`A` has shape `(batch, m, k)` and `B` has shape `(batch, k, n)`. The wrapper
passes a transpose view of `B` to MATE without making a layout copy. Scalar
FP8 scales select per-tensor scaling; rank-3 scales are a MATE extension for
channelwise scaling. When `out` is supplied, the wrapper writes into and
returns that tensor.

For `gemm_fp8_nt_groupwise`, `auto` selects `mudnn` for BF16 or FP16 output.
Supplying `output_scale` selects FP8 E4M3 output and makes `auto` select
`mubin`.

## Norm APIs

```python
from flashinfer.norm import (
    fused_add_rmsnorm,
    fused_add_rmsnorm_fp8_block_quant,
    fused_add_rmsnorm_quant,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    layernorm,
    layernorm_quant,
    rmsnorm,
    rmsnorm_quant,
)
```

The wrapper also exports the fused DiT LayerNorm, fused QK RMSNorm/RoPE, and
fused RMSNorm/SiLU APIs implemented by MATE. All supported Norm functions are
available both from `flashinfer.norm` and the `flashinfer` package root.

Quantized RMSNorm and LayerNorm functions follow FlashInfer's output-first
signatures and write into the supplied FP8 tensor. Python float scales are
converted to device-local FP32 scalar tensors. Fused add RMSNorm updates the
residual in place; its non-quantized variant also replaces the input with the
normalized output. The `enable_pdl` argument is accepted for compatibility and
has no effect on MUSA.

`fused_add_rmsnorm_fp8_block_quant` produces FP8 E4M3 activations together
with dynamic FP32 scales for groups of 128 consecutive hidden elements. MATE
uses a contiguous row-major scale tensor with shape
`[rows, hidden_size // 128]`; this intentionally differs from FlashInfer's
swizzled DeepGEMM scale layout.

## Quantization APIs

```python
from flashinfer.quantization import mxfp4_quantize, mxfp8_quantize

q4, sf4 = mxfp4_quantize(x)  # MP32: packed E2M1 + 1x32 E8M0
q8, sf8 = mxfp8_quantize(x)  # MP32: E4M3 + 1x32 E8M0
```

The following basic quantization functions are available from `flashinfer.quantization` and
the package root. They call `mate.quantization` public APIs. Inputs may have
non-contiguous outer dimensions; only the K stride must be one. K must be a
positive multiple of 32, as required by the upstream MX quantization APIs.

**Quantizers return MN-major scale factors.** For input shape
`[..., K]`, MX scale tensors have shape `[..., K // 32]` and uint8
E8M0 data. SF strides are `(1, M)` for 2D input or `(M * (K // 32), 1, M)`
for 3D input, ready for MP32 MXFP GEMM. The wrapper does not swizzle, pad,
or flatten scales. FlashInfer's
`sfLayout`, `sf_swizzle_layout`, `is_sf_swizzled_layout`, and
`is_sf_8x4_layout` arguments are accepted and ignored. Consumers must use the
native layout even when their call passes an NVIDIA layout selector.

| Entry | Behavior |
| --- | --- |
| `mxfp4_quantize` | Packed uint8 FP4 payload and native 1x32 E8M0 scales |
| `fp4_quantize` | FlashInfer signature; requires `sf_vec_size=32`, `sf_use_ue8m0=True`, and no global scale |
| `mxfp4_dequantize` | Reads native scales on MUSA and returns FP32 on CPU, matching upstream's return device |
| `mxfp8_quantize` | E4M3 payload and native 1x32 E8M0 scales; supports `alignment=32` and preserves logical K without payload padding |
| `packbits`, `segment_packbits` | FlashInfer-compatible bit packing through MATE |

The official fused SwiGLU MXFP8 rowwise entry is also available:

```python
from flashinfer.gated_act_mxfp8 import silu_and_mul_mxfp8_quantize

q, col_q, sf, col_sf = silu_and_mul_mxfp8_quantize(gated_input)
```

`gated_input` is BF16 with shape `[M, 2*K]`, with gate before up; M and K
must be positive multiples of 128. Rows may be non-contiguous, with unit
stride along K. The wrapper supports `rowwise=True, colwise=False` and calls
MATE's existing fused kernel. `q` is E4M3 `[M, K]`; `sf` is E8M0 with dtype
`torch.float8_e8m0fnu`, shape `[M, K // 32]`, and MN-major strides `(1, M)`
for direct consumption by MP32 MXFP GEMM. The disabled
colwise outputs are empty tensors with the corresponding dtypes on the input
device. Colwise and backward quantization are not provided.

The wrapper exports only FlashInfer API names and signatures. For fused
FP4, clamp SwiGLU/SiTU, FP8 1x128 FP32 scales, or other native extensions,
call `mate.quantization` directly. The backend/PDL compatibility arguments
do not change MATE's implementation selection. NVFP4 1x16 and 4over6 recipes
are not provided; the generic `fp4_quantize` retains upstream's signature
defaults and rejects that unsupported default recipe. Use `mxfp4_quantize`
for MXFP4 without extra options. CPU dequantization helpers are not exported.

The compatibility signatures were checked against the
[official FlashInfer quantization module](https://github.com/flashinfer-ai/flashinfer/tree/c94c1a9e4bf151e56080168ac880ebf0651af317/flashinfer/quantization).

## FP8 MLA APIs

```python
from flashinfer.decode import (
    get_batch_decode_metadata_mla,
    trtllm_batch_decode_with_kv_cache_mla,
)
from flashinfer.rope import mla_rope_quantize_fp8
```

### RoPE and FP8 quantization

`mla_rope_quantize_fp8` accepts BF16 query/key components and an FP32 or BF16
cosine/sine cache. It produces FP8 E4M3 outputs for a 512-value NoPE component
and a 64-value RoPE component. Optional output buffers may be strided in their
leading dimensions, so the outputs can be written directly into slices of a
merged `[..., 576]` tensor:

```python
q_nope_out = q_fp8[..., :512]
q_rope_out = q_fp8[..., 512:]
```

The quantization scales are multipliers applied before the FP8 cast:

```text
Q_fp8  = cast(Q  * quant_scale_q)
KV_fp8 = cast(KV * quant_scale_kv)
```

### Sparse MLA decode

`trtllm_batch_decode_with_kv_cache_mla` supports:

- A contiguous FP8 E4M3 query with shape `[batch, q_len, heads, 576]`.
- A contiguous FP8 E4M3 KV cache with shape `[pages, page_size, 576]` or
  `[pages, 1, page_size, 576]`.
- Contiguous int32 physical-token indices with shape
  `[batch, q_len, sparse_mla_top_k]`.
- A contiguous int32 `seq_lens` tensor with shape `[batch]`.
- Optional preallocated BF16 output and FP32 LSE buffers.

The decode path requires `kv_lora_rank=512`, `qk_rope_head_dim=64`, and a
positive `sparse_mla_top_k` that is divisible by 64. Unsupported shapes,
dtypes, layouts, devices, and tensor-valued decode scales fail explicitly.

`workspace_buffer` and FlashInfer implementation-selection arguments are kept
for call compatibility. MATE owns the runtime storage and always dispatches
its FP8 sparse MLA kernel, independent of the supplied decode backend name.

When RoPE quantization uses non-unit multipliers, fold their reciprocal scales
into the decode scales:

```python
q_descale = 1.0 / quant_scale_q
kv_descale = 1.0 / quant_scale_kv
bmm1_scale = attention_scale * q_descale * kv_descale
bmm2_scale = kv_descale
```

### Metadata reuse

Decode prepares scheduler metadata on every call unless a metadata object is
provided. Reuse metadata while `seq_lens`, query shape, and sparse top-k remain
unchanged:

```python
metadata = flashinfer.decode.get_batch_decode_metadata_mla(
    query=query,
    seq_lens=seq_lens,
    sparse_mla_top_k=topk,
)
out, lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    ...,
    workspace_buffer=None,
    return_lse=True,
    metadata=metadata,
)
```

Prepare a new metadata object after changing `seq_lens`, query shape, or top-k.
Do not share one metadata object between overlapping decode calls.

## Example

[`examples/fp8_mla_decode.py`](examples/fp8_mla_decode.py) demonstrates the
full RoPE-to-FP8-to-sparse-decode path with non-unit quantization scales,
merged 576-value buffers, and reusable metadata:

```bash
python examples/fp8_mla_decode.py
```

For the native MATE API and its complete parameter documentation, see
[`mate.sparse_mla_interface`](../../docs/source/api/sparse_mla.rst).

### GLM-5.3-Flash NoPE

`flashinfer.mla.SparseMLASm120Wrapper` accepts BF16 absorbed Q512 and byte caches
with 512 E4M3 latent values followed by four arbitrary FP32 group-128 scales.
Compact rows use 528 bytes; legacy rows may retain 128 trailing padding bytes.
`supported_sparse_mla_sm120_configs()["glm53_nope"].bytes_per_token` reports 528
so compatible framework allocators can reserve the compact size.

```python
from flashinfer.mla import SparseMLASm120Wrapper
from mate.sparse_mla.glm_nope import write_glm_nope_cache
import torch

# latent: [new_tokens, 512] BF16; slots: int64 physical token slots.
cache = torch.empty((num_pages, page_size, 1, 528), dtype=torch.uint8, device="musa")
write_glm_nope_cache(latent, cache, slots)
runner = SparseMLASm120Wrapper(
    max_num_tokens=max_tokens, max_num_heads=heads,
    kv_scale_format="arbitrary_fp32", device="musa",
)
# q: [tokens, heads, 512]; indices: [tokens, candidates] int32.
output = torch.empty_like(q)
runner.run(q, cache.squeeze(2), indices, output, model_softmax_scale,
           topk_length=candidate_prefix_end)
```

Pass the model's softmax scale after weight absorption (GLM commonly uses
0.0625); it must not be inferred from the absorbed dimension. Candidate prefix
ends include holes: the three KPool tail candidates at columns 2048–2050 remain
valid even when earlier columns contain `-1`. Nonaligned candidate arrays are
padded internally; empty candidate rows yield zero output and positive-infinite
LSE. `run` writes the supplied output and returns LSE only with `return_lse=True`.
Caller scratch arguments are accepted; MATE manages its own scheduled workspace.

This entry explicitly supports the GLM inline-scale format. The distinct
V4.1 528-byte page-footer format uses the FlashMLA wrapper. Whole-model GLM
serving also requires the framework's other model operators.
