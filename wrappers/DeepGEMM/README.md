# DeepGEMM Compatibility Wrapper (deep-gemm)

`deep-gemm` is a compatibility wrapper package that preserves the
`deep_gemm` import path while running on MUSA through MATE GEMM operators.

## Overview

This wrapper is designed for projects that already target DeepGEMM-style
Python APIs. It helps run existing integrations on MUSA through MATE with
minimal code changes.

The wrapper API aligns with the official DeepGEMM repository v2.1.1.

## Package and import

- Package name: `deep-gemm`
- Import path: `deep_gemm`
- Runtime backend: MATE GEMM and logits operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.6+musa`. Use `python -m pip show deep-gemm` to distinguish this
wrapper from the native package. The Python API compatibility baseline is
official DeepGEMM v2.1.1.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.

## Build

Build a wheel from the `wrappers/DeepGEMM` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under:

```text
dist/
```

## Installation

For delivered packages, install from the external MUSA wheel source:

```bash
python -m pip install deep-gemm \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

This installs the matching `mate` dependency automatically.

For local wrapper development, install from source:

```bash
python -m pip install --no-build-isolation --no-deps -e ../.. -v
python -m pip install --no-build-isolation --no-deps -e .
```

Install a built local wheel:

```bash
python -m pip install --no-deps dist/deep_gemm-*.whl
```

If you previously installed the legacy `mate-deep-gemm` package, uninstall it
before installing `deep-gemm` so the environment does not keep stale wrapper
metadata.

```bash
python -m pip uninstall -y mate-deep-gemm
python -m pip install --no-deps dist/deep_gemm-*.whl
```

## Import

Import the package directly:

```python
import deep_gemm
```

Import individual APIs:

```python
from deep_gemm import (
    bf16_gemm_nn,
    bf16_gemm_nt,
    bf16_gemm_tn,
    bf16_gemm_tt,
    cublaslt_gemm_nn,
    cublaslt_gemm_nt,
    cublaslt_gemm_tn,
    cublaslt_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nt_masked,
    m_grouped_fp8_gemm_nn_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    k_grouped_bf16_gemm_tn_contiguous,
    fp8_fp4_gemm_nt,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    k_grouped_fp8_gemm_tn_contiguous,
    fp8_gemm_nn,
    fp8_gemm_nt,
    fp8_gemm_tn,
    fp8_gemm_tt,
    fp8_einsum,
    fp8_gemm_nt_skip_head_mid,
    tf32_hc_prenorm_gemm,
    get_paged_mqa_logits_metadata,
    fp8_paged_mqa_logits,
    fp8_mqa_logits,
)
```

## Public APIs

Dense BF16 GEMM:

- `bf16_gemm_nt`
- `bf16_gemm_nn`
- `bf16_gemm_tn`
- `bf16_gemm_tt`

cuBLASLt-compatible GEMM:

- `cublaslt_gemm_nt`
- `cublaslt_gemm_nn`
- `cublaslt_gemm_tn`
- `cublaslt_gemm_tt`

Grouped GEMM:

- `m_grouped_bf16_gemm_nt_contiguous`
- `m_grouped_bf16_gemm_nt_masked`
- `m_grouped_fp8_fp4_gemm_nt_contiguous`
- `m_grouped_fp8_fp4_gemm_nt_masked`
- `m_grouped_fp8_gemm_nn_contiguous`
- `m_grouped_fp8_gemm_nt_contiguous`
- `m_grouped_fp8_gemm_nt_masked`
- `k_grouped_fp8_gemm_nt_contiguous`
- `k_grouped_fp8_gemm_tn_contiguous`
- `k_grouped_bf16_gemm_tn_contiguous`
- Legacy aliases: `fp8_m_grouped_gemm_nt_masked`, `bf16_m_grouped_gemm_nt_masked`

Dense FP8 GEMM:

- `fp8_fp4_gemm_nt`
- `fp8_gemm_nt`
- `fp8_gemm_nn`
- `fp8_gemm_tn`
- `fp8_gemm_tt`
- `fp8_einsum`
- `fp8_gemm_nt_skip_head_mid`

Einsum:

- `einsum`

HyperConnection prenorm GEMM:

- `tf32_hc_prenorm_gemm`

MQA logits APIs:

- `get_paged_mqa_logits_metadata`
- `fp8_paged_mqa_logits`
- `fp8_mqa_logits`

Utility helpers re-exported from `deep_gemm.utils`:

- `bench`, `bench_kineto`, `calc_diff`
- `get_num_sms`, `set_num_sms`
- `get_tc_util`, `set_tc_util`
- `get_mk_alignment_for_contiguous_layout`
- `get_col_major_tma_aligned_tensor`
- `get_mn_major_tma_aligned_tensor`
- `transform_sf_into_required_layout`

Testing helpers are implemented in `mate.testing.deep_gemm` and are also
available from the upstream-style path:

- `deep_gemm.testing.bench`
- `deep_gemm.testing.bench_kineto`
- `deep_gemm.testing.calc_diff`

## Contiguous Grouped GEMM Alignment

`get_mk_alignment_for_contiguous_layout()` returns the M-axis padding alignment
used by DeepGEMM-compatible contiguous grouped GEMM wrappers. It defaults to
`128` on MP31 and MP32. It can be overridden with:

```bash
export MATE_DEEPGEMM_MK_ALIGNMENT=256
```

Only `128` and `256` are supported. Use the returned value when padding each
expert segment and building `grouped_layout` for
`m_grouped_{fp8,bf16}_gemm_nt_contiguous`. Set the environment variable before
starting Python; the helper caches the configured value.

For FP8-by-FP8 and FP4-by-FP4, grouped APIs resolve an omitted `recipe` from the architecture and, for
MP32 M-grouped GEMM, B's SF dtype:

| Architecture / B SF dtype | M-group NT contiguous / masked | K-group TN contiguous |
| --- | --- | --- |
| MP31 | `(1, 128, 128)` | `(1, 1, 128)` |
| MP32 / FP32 | `(1, 128, 128)` | Unsupported |
| MP32 / INT32 | `(1, 1, 128)` | Unsupported |
| MP32 / UINT8 | `(1, 1, 32)` | `(1, 1, 32)` |

On MP32, `m_grouped_fp8_fp4_gemm_nt_{contiguous,masked}` and their FP8 aliases
**automatically convert source SF** for FP8-by-FP8 and FP4-by-FP4.
The following scale-conversion rules apply to these two paths. They accept FP32 power-of-two
scales, INT32 words packing four successive K UE8M0 codes (least significant
byte first), and UINT8 UE8M0 codes. K-major and MN-major positive-stride inputs,
including padding and storage offsets, are supported. Conversion uses TileLang
(`mate[tilelang]`) only when needed; native UINT8 MN-major scales are reused
without loading TileLang or launching a conversion kernel.

Specify the source quantization with `recipe=(gran_m, gran_n, gran_k)` or both
`recipe_a=(gran_m, gran_k_a)` and `recipe_b=(gran_n, gran_k_b)`. Each side supports
`(1,32)`, `(1,128)` and `(128,128)`, including different K granularities for A/B.
With `R=ceil(M_or_N/gran_mn)` and `S=ceil(K/gran_k)`, FP32/UINT8 SF has logical
shape `[R,S]` or `[G,R,S]`; INT32 SF replaces `S` with `ceil(S/4)`. Both memory
majors keep this axis order. Contiguous A SF is 2D; masked A and B SF are 3D.
K counts logical values, including for packed FP4 operands.

```python
# FP32 sfa: [G, M, ceil(K/128)]; sfb: [G, ceil(N/128), ceil(K/128)].
# The wrapper converts SF and dispatches the native GEMM with (1,1,32).
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    (a, sfa), (b, sfb), d, masked_m, expected_m, recipe=(1, 128, 128),
)
```

`disable_ue8m0_cast=None` (the wrapper default) or `False` permits FP32
conversion. `True` rejects FP32 but still permits INT32/UINT8 layout conversion.
FP32 conversion extracts exponent bits without rounding or requantizing the
operands, so scales must already represent UE8M0 powers of two. Valid scale
codes must exclude `0xFF`; unused bytes in the last INT32 word are ignored.
Conversion runs on the current stream on each call, including graph replay;
converted tensors are not cached across calls.

For reusable scales, explicit preprocessing remains available through
`transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None,
is_sfa=False, disable_ue8m0_cast=False)`. It accepts a per-operand recipe or a
three-value recipe with `is_sfa` selecting A/B. Pass its output to GEMM with
the default UINT8 recipe or `(1,1,32)`. The native helper is
`mate.gemm.transform_mxfp_sf`. On MP31, the wrapper helper keeps the existing
behavior of returning contiguous scales with unchanged dtype and granularity.
Hardware-swizzled SF, K-grouped independently packed scales and PSUM gap layouts
are unsupported by these conversions. Dense and K-grouped GEMM do not perform
this automatic conversion.

Native MATE MX scales must be **MN-major**, with stride 1 in M/N and no padding.
With `Q = ceil(K/32)`, ragged A scales have logical shape `[M, Q]` and
strides `[1, M]`; masked A scales have shape `[G, Mmax, Q]` and strides
`[Mmax * Q, 1, Mmax]`. B scales have shape `[G, N, Q]` and strides
`[N * Q, 1, N]`. K-major scales and the old `[Q, M]` / `[G, Q, N]`
axis orders are rejected. Existing dense scales in those old layouts can
expose the new shape and strides without copying, including when `Q == 1`:

```python
# sfa: dense [Q, M]; sfb: dense [G, Q, N]; both UINT8 UE8M0.
sfa = sfa.as_strided((M, Q), (1, M))
sfb = sfb.as_strided((G, N, Q), (N * Q, 1, N))
```

K-group scales are contiguous `[sum(ceil(k_i/32)), M or N]`, with each `k_i`
a positive multiple of 32. Native MP32 MXFP APIs and the K-grouped wrapper
require UINT8 scales for 32-element blocks; they reject 128-granularity recipes,
FP32 scales and packed INT32 scales.

The grouped wrapper uses the DeepGEMM 559d79f parameter names and order, while
retaining MATE-only `alignment_m`, `backend`, and overlap parameters at the end
of the signatures. PSUM and zero-padding layouts remain unsupported. The
wrapper keeps runnable MATE defaults for unsupported capabilities.

On MP32, FP8-by-FP8 and FP4-by-FP4 grouped wrappers call MATE's `*_moe_gemm_mxfp` entry points.
M-grouped GEMM supports all four E4M3/E5M2 A/B combinations and packed
E2M1-by-E2M1, with BF16 output. K-grouped GEMM supports all four FP8
combinations with BF16 or FP32 output. Both native paths use `(1, 1, 32)`.
FP4 values are packed two E2M1 values per int8 byte, with the first in the low
nibble and the second in the high nibble. After wrapper conversion, FP4 and FP8
both use **uint8 UE8M0 scales** in the MN-major layout described above. Each
scale byte covers 32 logical values (16 packed data bytes for FP4). Direct
MATE `*_8bit` APIs forward to MXFP on MP32 for compatibility. FP6 and FP4-by-FP8 combinations need additional kernel support.

On MP31, FP8-by-FP8 uses the existing FP8 path. FP8 E4M3 A plus packed E2M1
FP4 B dispatches to MATE's native W4A8 backend. Native MATE scale tuples
work with the wrapper defaults;
DeepGEMM FP32 logical scales are adapted for `recipe_a=(1, 128)` and
`recipe_b=(1, 32)` when `K=128`. Larger K is rejected because the current MATE
W4A8 kernel has one A scale per row. The dense `fp8_fp4_gemm_nt` entry point
still rejects packed FP4 operands.


The MP32 FP8-by-FP4 path calls MATE native MXFP APIs with the MUTLASS RS backend.
It requires E4M3 A, packed E2M1 B (int8), and BF16 D:

- A/B/D and int32 metadata must be contiguous; all tensors must share a device.
  N must be a multiple of 4 and K a multiple of 8. Scale groups use
  `Q=ceil(K/32)`; the last group may contain fewer than 32 values.
- Recipes default to A=(1,32), B=(1,32); A=(1,128) is also supported.
  Scales accept FP32 powers of two (zero for padding), uint8 UE8M0,
  or packed int32 (four UE8M0 codes per word). Values are caller-guaranteed;
  UE8M0 code 0xFF is invalid.
- For contiguous A `[M,K]`, M is the total padded row count; A/ScaleA are
  flattened over M. Pad each expert segment to alignment_m (128 or 256).
  Each tile starts with its group ID in [0,G) and contains only that ID or -1
  padding, or is entirely padding.
- For masked A `[G,M,K]`, M is the allocated row capacity per expert and must
  be a multiple of 4. Valid token counts `masked_m[g]` need no alignment.
  Require 0 <= masked_m[g] <= M and expected_m > 0.
  Metadata values are caller-guaranteed.
- Supports auto/mutlass backends; no PSUM or overlap.
  Contiguous padding output is unspecified; masked invalid rows remain untouched
  for nonzero K. Zero G/M/N returns without launching; zero K clears the entire output.

Legacy per-row ScaleA and residual/epilogue ScaleB tuples keep the existing
mixed-dtype route. Its DeepGEMM FP32 scale adapter remains limited to K=128;
that limit does not apply to the native MP32 block-scale route.
The dense FP8/FP4 entry point still rejects packed FP4 operands.
For K-grouped GEMM, `c=None` clears D, an independent C is copied to D, and
`c is d` keeps in-place accumulation.

## Quick Start

Minimal import example:

```python
import deep_gemm
```

An example script is provided at:

```text
examples/run_deep_gemm.py
```

## Examples

Run the bundled example:

```bash
python examples/run_deep_gemm.py
```

## Notes

- This wrapper preserves the DeepGEMM-style Python surface, but execution is provided by MATE on MUSA
- `get_paged_mqa_logits_metadata(..., block_kv, ...)` currently requires `block_kv == 64`
- `fp8_gemm_nt_skip_head_mid` writes the GEMM result into a full-width D tensor
  according to `head_splits=(left, mid, right)` and leaves the middle portion of each head untouched
- The example script currently demonstrates the FP8 grouped GEMM path
