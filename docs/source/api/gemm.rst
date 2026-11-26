.. _apigemm:

GEMM
=========

For framework integrations that already target DeepGEMM style Python APIs,
prefer the ``deep-gemm`` wrapper first. Use the MATE APIs below when wrapper
coverage is not enough.

MoE GEMM
-------------------------

.. currentmodule:: mate.gemm

The direct MoE GEMM entrypoints below cover 16-bit, 8-bit, and mixed-dtype
W4A8 paths. On MP32, use the ``*_mxfp`` entry points for grouped MXFP8
and MXFP4. MXFP8 supports E4M3/E5M2 independently for A and B. The entry
points select the format from the input dtypes and use 32-element scale groups. All MX scales must be MN-major, contiguous in M/N
without padding. For ``Q=ceil(K/32)``, ragged A scales have shape ``[M,Q]``
and strides ``[1,M]``; masked A scales have shape ``[G,M,Q]`` and strides
``[M*Q,1,M]``. B scales have shape ``[G,N,Q]`` and strides ``[N*Q,1,N]``.
K-major scale inputs are rejected. The ``*_8bit`` MP32 entry points forward
to MXFP for compatibility. ``GemmMixedDType.S4FP8`` uses
``b_quant_recipe=(1, 128)``. For ``masked_moe_gemm_mixed_dtype``, compatible
S4FP8 decode workloads can use ``backend="mutlass"``. ``backend="auto"``
selects MUTLASS for supported non-overlap inputs, generally when the dispatch
M is at most 32. Grouped-A ``a_quant_recipe=(1, 128)`` requires MUTLASS.
``GemmMixedDType.FP4FP8`` uses E2M1 weights, E8M0 residual scales, an FP32
epilogue scale, ``a_quant_recipe=(1, -1)``, and
``b_quant_recipe=(1, 32)`` with the MUBIN backend.

.. autofunction:: ragged_m_moe_gemm_8bit
.. autofunction:: ragged_m_moe_gemm_16bit
.. autofunction:: ragged_k_moe_gemm_8bit
.. autofunction:: ragged_k_moe_gemm_16bit
.. autofunction:: masked_moe_gemm_8bit
.. autofunction:: transform_mxfp_sf
.. autofunction:: ragged_m_moe_gemm_mxfp
.. autofunction:: masked_moe_gemm_mxfp
.. autofunction:: ragged_k_moe_gemm_mxfp
.. autofunction:: masked_moe_gemm_16bit
.. autofunction:: ragged_moe_gemm_mixed_dtype
.. autofunction:: masked_moe_gemm_mixed_dtype

MXFP Scale Conversion
---------------------

On MP32, the DeepGEMM wrapper's
``m_grouped_fp8_fp4_gemm_nt_contiguous`` and
``m_grouped_fp8_fp4_gemm_nt_masked`` APIs (including their FP8 aliases)
automatically convert source SF before calling the native GEMM. FP32,
packed INT32 and UINT8 inputs may be K-major or MN-major positive-stride
views, including padding and storage offsets. Conversion requires the
``mate[tilelang]`` extra only when SF is not already native.

Specify the input ``recipe=(gran_m,gran_n,gran_k)`` or both per-operand
``recipe_a`` and ``recipe_b``. Each side accepts ``(1,32)``, ``(1,128)`` or
``(128,128)``; the per-operand form permits different K granularities.
When omitted, B's SF dtype selects ``(1,128,128)`` for FP32,
``(1,1,128)`` for INT32, or ``(1,1,32)`` for UINT8.
With ``R=ceil(mn/gran_mn)`` and ``S=ceil(K/gran_k)``, FP32/UINT8 SF has
shape ``[R,S]`` or ``[G,R,S]``; INT32 SF replaces ``S`` with ``ceil(S/4)``.
INT32 packs four successive K UE8M0 codes, least significant byte first.
The logical axis order is the same for both majors. Contiguous A SF is 2D;
masked A and B SF are 3D. For packed FP4, K is the unpacked logical size.

``disable_ue8m0_cast=None`` or ``False`` permits FP32 conversion; ``True``
rejects FP32 while permitting INT32/UINT8 layout conversion. FP32 scales
must already represent UE8M0 powers of two: conversion extracts the exponent
without rounding or requantizing the operands. Converted SF is UINT8
MN-major with qtile 32, and the wrapper always passes ``(1,1,32)`` to the
native GEMM. Conversion uses the current stream and observes updated source
SF during graph replay. Converted tensors are not cached across calls.
Already-native UINT8 inputs are reused without conversion or TileLang.

For reusable scales, explicitly call
``deep_gemm.transform_sf_into_required_layout`` or
``mate.gemm.transform_mxfp_sf`` before GEMM. The native helper accepts the
per-operand recipes above and broadcasts coarser scales to 32-element
blocks. Consume its output using the default UINT8 recipe or ``(1,1,32)``.
Native MATE MXFP APIs still require UINT8 MN-major SF with the shape and
strides specified above. Dense and K-grouped wrapper APIs do not perform
this automatic conversion. Hardware-swizzled SF, K-grouped independently
packed scales and PSUM gap layouts are unsupported by these conversions.

MP32 block-scaled FP8 x FP4 grouped GEMM
---------------------------------------

Use ``mate.gemm.ragged_m_moe_gemm_mxfp`` or
``mate.gemm.masked_moe_gemm_mxfp`` with E4M3 A, packed E2M1 B (int8),
native K32 UINT8 MN-major scales, and BF16 output. This combination uses
``backend="mutlass"`` (also selected by ``auto``) on MP32.

A/B/output and int32 metadata must be contiguous on the same device.
N must be a multiple of 4 and K a multiple of 8. Scale groups use
``Q=ceil(K/32)``; the last group may contain fewer than 32 values.

For contiguous A shaped ``[M,K]``, M is the total padded row count. Pad each
expert segment to ``alignment_m`` (128 by default, or 256); each tile must
contain one expert ID or -1 padding.

For masked A shaped ``[G,M,K]``, M is the allocated row capacity per expert
and must be a multiple of 4. Valid token counts ``masked_tokens_info[g]``
need no alignment. Callers guarantee ``0 <= masked_tokens_info[g] <= M``
and positive ``expect_tokens``.
Contiguous padding output is unspecified; masked invalid rows are untouched
for nonzero K. Zero G/M/N returns without launching; zero K clears the entire
output.
PSUM, overlap, and adding C are unsupported.

DeepGEMM-compatible ``m_grouped_fp8_fp4_gemm_nt_*`` names are provided by
the wrapper. For this FP8-by-FP4 path, source recipes default to A=(1,32),
B=(1,32); A=(1,128) is also supported. The wrapper converts FP32,
packed INT32, or UINT8 scales using the existing MXFP scale utilities.


Dense GEMM
-------------------------

.. currentmodule:: mate.gemm

.. autofunction:: bmm

DeepGemm Lighting Indexer
-------------------------

.. currentmodule:: mate.deep_gemm
.. autofunction:: fp8_einsum
.. autofunction:: tf32_hc_prenorm_gemm
.. autofunction:: fp8_mqa_logits
.. autofunction:: fp8_gemm_nt_skip_head_mid
.. autofunction:: get_paged_mqa_logits_metadata
.. autofunction:: paged_mqa_logits
.. autofunction:: fp8_paged_mqa_logits
.. autofunction:: get_mqa_logits_metadata
.. autofunction:: mqa_logits
