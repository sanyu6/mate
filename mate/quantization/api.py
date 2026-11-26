"""Public MUSA quantization and bit-packing APIs.

Import these functions from ``mate.quantization``. Floating-point quantization
uses row-major payloads and MN-major scales, with one scale for a partial
final block. Only the last input dimension must be contiguous; batch and row
strides are preserved when reading inputs. Backend selection is an
implementation detail.
"""

from typing import Literal

import torch

from mate.api_logging import mate_api

from .tilelang.silu_and_mul_quantize import _silu_and_mul_quantize_impl


__all__ = [
    "fp4_quantize",
    "fp4_dequantize",
    "fp8_quantize",
    "fp8_dequantize",
    "silu_and_mul_quantize",
    "packbits",
    "segment_packbits",
]


@mate_api
def fp4_quantize(
    x: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    sf_vec_size: int = 32,
    sf_type: Literal["e8m0", "e4m3"] | None = None,
    is_global_scale_inversed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 values to packed E2M1 FP4 with block scales.

    Each row is divided into blocks of 32 values along K. The raw scale is
    ``amax(abs(block)) / 6``, optionally multiplied by ``global_scale`` (or
    divided by it in inverse mode), then encoded in ``sf_type``. Quantization
    divides values by the decoded scale before FP4 conversion. E8M0 rounds
    scales up to a power of two within ``[2**-127, 2**127]``; E4M3 uses
    round-to-nearest conversion. E8M0 code zero represents ``2**-127``.

    Args:
        x: Finite FP16/BF16 tensor on a MUSA device, shaped ``[M, K]`` or
            ``[B, M, K]``. All dimensions must be nonempty and K must be even.
            ``x.stride(-1)`` must be 1; batch/row strides may vary. K need
            not be a multiple of 16 or 32.
        global_scale: Optional finite FP32 tensor of shape ``[1]`` on the
            same device. Adjusts the raw block scale before encoding; it
            does not require an additional factor during dequantization.
            Omitting it is equivalent to a factor of one.
        sf_vec_size: Values sharing one scale. Only 32 is supported.
        sf_type: Scale encoding, ``"e8m0"`` or ``"e4m3"``. None selects
            ``"e8m0"``. Scales are returned as raw uint8 bytes.
        is_global_scale_inversed: Divide by ``global_scale`` instead of
            multiplying. The supplied global scale must then be positive.
            Has no effect when ``global_scale`` is None.

    Returns:
        ``(q, sf)`` on the input device, both preserving rank. ``q`` is
        contiguous uint8 with shape ``x.shape[:-1] + (K // 2,)``: the earlier
        FP4 value occupies the low nibble of each byte. ``sf`` is uint8 with
        shape ``x.shape[:-1] + (ceil_div(K, 32),)``. The final scale covers
        only valid values in a partial block. With ``Q=ceil_div(K, 32)``,
        MN-major sf strides are ``(1, M)`` for 2D input or ``(M * Q, 1, M)``
        for 3D input. E8M0 scales can be consumed directly by MP32 MXFP GEMM.
        Scales have no swizzle or padding.

    Raises:
        ValueError: An input shape, stride, dtype, device, scale parameter,
            or block size is unsupported.

    Examples:
        >>> from mate.quantization import fp4_quantize, fp4_dequantize
        >>> x = torch.randn(16, 48, device="musa", dtype=torch.bfloat16)
        >>> q, sf = fp4_quantize(x)
        >>> restored = fp4_dequantize(q, sf, output_dtype=x.dtype)
        >>> restored.shape
        torch.Size([16, 48])
    """
    from .tilelang.fp4_quantize import _fp4_quantize_impl

    return _fp4_quantize_impl(
        x, global_scale, sf_vec_size, sf_type, is_global_scale_inversed
    )


@mate_api
def fp4_dequantize(
    q: torch.Tensor,
    sf: torch.Tensor,
    sf_type: Literal["e8m0", "e4m3"] = "e8m0",
    sf_vec_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode packed E2M1 FP4 and multiply by its per-block scale.

    This consumes the payload and scales returned by :func:`fp4_quantize` or
    FP4 :func:`silu_and_mul_quantize`. The low nibble is decoded before the
    high nibble. No additional global-scale factor is applied.

    Args:
        q: MUSA tensor with dtype uint8 or ``torch.float4_e2m1fn_x2``, shaped
            ``[M, K // 2]`` or ``[B, M, K // 2]``. Each element stores two
            FP4 values; all dimensions must be nonempty. The packed-byte
            dimension must have unit stride; batch/row strides may vary.
        sf: Raw uint8 scale bytes on the same device, shaped
            ``q.shape[:-1] + (ceil_div(K, 32),)``. Row-major and MN-major
            layouts are supported; all scale strides may differ from q's.
        sf_type: Encoding used by the quantizer, ``"e8m0"`` (default) or
            ``"e4m3"``. E8M0 byte zero decodes to ``2**-127``.
        sf_vec_size: Logical FP4 values sharing one scale. Only 32 is supported.
        output_dtype: Output dtype: FP32 (default), FP16, or BF16.

    Returns:
        A contiguous tensor on q's device, with dtype ``output_dtype`` and
        shape ``q.shape[:-1] + (2 * q.shape[-1],)``. Each value is the decoded
        FP4 value multiplied by the decoded scale of its 32-value block.

    Raises:
        ValueError: The shape, stride, dtype, device, encoding, or block size
            of the supplied tensors is unsupported or inconsistent.

    Examples:
        >>> from mate.quantization import fp4_quantize, fp4_dequantize
        >>> x = torch.randn(4, 64, device="musa", dtype=torch.float16)
        >>> q, sf = fp4_quantize(x, sf_type="e4m3")
        >>> restored = fp4_dequantize(q, sf, sf_type="e4m3")
        >>> restored.dtype
        torch.float32
    """
    from .tilelang.fp4_dequantize import _fp4_dequantize_impl

    return _fp4_dequantize_impl(q, sf, sf_type, sf_vec_size, output_dtype)


@mate_api
def fp8_quantize(
    x: torch.Tensor,
    sf_vec_size: int = 32,
    sf_type: Literal["e8m0", "e4m3"] | None = None,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 values to FP8 with one scale per 32 K values.

    For each block, the raw scale is ``amax(abs(block)) / fp8_max``, where
    ``fp8_max`` is 448 for E4M3 or 57344 for E5M2. E8M0 rounds this scale up
    to a power of two within ``[2**-127, 2**127]``. Positive E4M3 scales are
    clamped to ``[2**-9, 448]`` and rounded to nearest. Quantization divides
    by the decoded scale and saturates payloads to the finite FP8 range.
    Zero blocks have E4M3 scale zero or E8M0 code zero (``2**-127``).

    Args:
        x: Finite FP16/BF16 MUSA tensor shaped ``[M, K]`` or ``[B, M, K]``.
            All dimensions must be nonempty, K must be even, and the last
            dimension must have unit stride. Batch/row strides may vary;
            K need not be a multiple of 16 or 32.
        sf_vec_size: Values sharing one scale. Only 32 is supported by this
            API; fused activation quantization also supports 128.
        sf_type: Scale encoding, ``"e8m0"`` or ``"e4m3"``. None selects
            ``"e8m0"``. Both encodings are returned as raw uint8 bytes.
        output_dtype: ``torch.float8_e4m3fn`` (default) or ``torch.float8_e5m2``.

    Returns:
        ``(q, sf)`` on x's device, both preserving rank. q is contiguous with
        x's shape and dtype ``output_dtype``. sf is uint8 with shape
        ``x.shape[:-1] + (ceil_div(K, 32),)``. The final block may be partial;
        with ``Q=ceil_div(K, 32)``, MN-major sf strides are ``(1, M)`` for 2D
        input or ``(M * Q, 1, M)`` for 3D input. E8M0 scales can be consumed
        directly by MP32 MXFP GEMM. There is no padding or scale swizzle.

    Raises:
        ValueError: An input shape, stride, dtype, device, scale encoding,
            output dtype, or block size is unsupported.

    Examples:
        >>> from mate.quantization import fp8_quantize, fp8_dequantize
        >>> x = torch.randn(8, 48, device="musa", dtype=torch.bfloat16)
        >>> q, sf = fp8_quantize(x)
        >>> restored = fp8_dequantize(q, sf)
        >>> restored.shape
        torch.Size([8, 48])
    """
    from .tilelang.fp8_quantize import _fp8_quantize_impl

    return _fp8_quantize_impl(x, sf_vec_size, sf_type, output_dtype)


@mate_api
def fp8_dequantize(
    q: torch.Tensor,
    sf: torch.Tensor,
    sf_type: Literal["e8m0", "e4m3"] = "e8m0",
    sf_vec_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Multiply FP8 values by decoded block scales.

    This consumes the 1x32 uint8-scale format from :func:`fp8_quantize` or
    :func:`silu_and_mul_quantize`. The fused API's 1x128 FP32 scales are not
    accepted here; those can be expanded along K and multiplied in FP32.

    Args:
        q: MUSA tensor of dtype ``torch.float8_e4m3fn`` or
            ``torch.float8_e5m2``, shaped ``[M, K]`` or ``[B, M, K]``. All
            dimensions must be nonempty, K must be even, and the last
            dimension must have unit stride. Batch/row strides may vary.
        sf: Raw uint8 scale bytes on q's device, shaped
            ``q.shape[:-1] + (ceil_div(K, 32),)``. Row-major and MN-major
            layouts are supported; all scale strides may differ from q's.
        sf_type: Encoding used by the quantizer: ``"e8m0"`` (default) or
            ``"e4m3"``. E8M0 byte zero decodes to ``2**-127``.
        sf_vec_size: FP8 values sharing one scale. Only 32 is supported.
        output_dtype: Output dtype: FP32 (default), FP16, or BF16.

    Returns:
        A contiguous tensor with q's shape and device and dtype
        ``output_dtype``. Each output is ``q.float() * decoded_block_scale``,
        converted to the requested dtype after multiplication.

    Raises:
        ValueError: A tensor shape, stride, dtype, device, scale encoding,
            or block size is unsupported or inconsistent.

    Examples:
        >>> from mate.quantization import fp8_quantize, fp8_dequantize
        >>> x = torch.randn(4, 64, device="musa", dtype=torch.float16)
        >>> q, sf = fp8_quantize(x, output_dtype=torch.float8_e5m2)
        >>> restored = fp8_dequantize(q, sf, output_dtype=x.dtype)
        >>> restored.dtype
        torch.float16
    """
    from .tilelang.fp8_dequantize import _fp8_dequantize_impl

    return _fp8_dequantize_impl(q, sf, sf_type, sf_vec_size, output_dtype)


@mate_api
def silu_and_mul_quantize(
    input: torch.Tensor,
    output_dtype: torch.dtype = torch.float4_e2m1fn_x2,
    *,
    global_scale: torch.Tensor | None = None,
    sf_vec_size: int = 32,
    sf_type: Literal["e8m0", "e4m3", "fp32"] | None = None,
    is_global_scale_inversed: bool = False,
    activation: Literal["swiglu", "situ"] = "swiglu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply SwiGLU, clamp SwiGLU, or SiTU with FP4/FP8 or BF16 output.

    Split input into gate and up halves along the last dimension. Ordinary
    SwiGLU computes ``gate * sigmoid(gate) * up``. With ``swiglu_limit=L``,
    first replace gate by ``min(gate, L)`` and up by ``clamp(up, -L, L)``;
    this is the clamp SwiGLU used by DeepSeek V4/V4.1 (L=10).

    SiTU computes ``beta * tanh(gate / beta) * sigmoid(gate) * softcap(up)``,
    where ``softcap(up) = linear_beta * tanh(up / linear_beta)``. Setting
    ``situ_linear_beta=None`` uses up directly. Kimi K3 uses beta=4 and
    linear_beta=25. Activations are evaluated in FP32 and quantized directly,
    without an intermediate FP16/BF16 rounding or materialized activation.
    With BF16 output, the activation is rounded directly to BF16 without
    quantization or scale computation.

    Args:
        input: Finite FP16/BF16 MUSA tensor shaped ``[M, 2*K]`` or
            ``[B, M, 2*K]``, with gate before up. All dimensions must be
            nonempty and output K must be even for both FP4 and FP8.
            BF16 output supports any positive K.
            The last dimension must have unit stride; batch/row strides
            may vary. Partial final scale blocks are supported.
        output_dtype: ``torch.float4_e2m1fn_x2`` (default),
            ``torch.float8_e4m3fn``, ``torch.float8_e5m2``, or
            ``torch.bfloat16``. FP4 output
            is returned as packed uint8; FP8/BF16 retain the selected dtype.
            For BF16, scale options must retain their defaults.
        global_scale: FP4 only. Optional finite FP32 tensor of shape ``[1]``
            on input's device. Multiplies the raw ``amax / 6`` scale before
            encoding, or divides it in inverse mode. Returned scales
            already include this factor; dequantization needs no extra factor.
        sf_vec_size: Values sharing one scale: 32 for FP4; 32 or 128 for FP8.
            The 1x128 FP8 recipe uses FP32 scales, matching RMSNorm block
            quantization: ``scale = max(amax, 1e-4) / fp8_max``.
        sf_type: ``"e8m0"`` or ``"e4m3"`` for 1x32; ``"fp32"`` for
            1x128 FP8. None selects ``"e8m0"`` for 32 and ``"fp32"`` for
            128. The 1x32 scale encoding follows :func:`fp4_quantize` or
            :func:`fp8_quantize`, respectively.
        is_global_scale_inversed: Divide by ``global_scale`` instead of
            multiplying. A supplied global scale must then be positive.
            Has no effect when no global scale is supplied.
        activation: ``"swiglu"`` (default) or ``"situ"``.
        swiglu_limit: Optional finite positive clamp bound for SwiGLU only.
            None disables clamping; use 10.0 for DeepSeek V4/V4.1.
        situ_beta: Finite positive gate softcap for SiTU, default 4.0.
            Ignored for SwiGLU.
        situ_linear_beta: Finite positive up softcap for SiTU, default 25.0.
            None disables the up softcap. Ignored for SwiGLU.

    Returns:
        ``(q, sf)`` on input's device. q preserves rank and is contiguous.
        For BF16, q has shape ``input.shape[:-1] + (K,)`` and sf is None.
        For quantized output, sf also preserves rank.
        FP4 q is uint8 with shape ``input.shape[:-1] + (K // 2,)`` and the
        earlier value in the low nibble. FP8 q has shape
        ``input.shape[:-1] + (K,)`` and dtype ``output_dtype``. sf has shape
        ``input.shape[:-1] + (ceil_div(K, sf_vec_size),)``: raw uint8 scale
        bytes for 1x32, FP32 values for 1x128. sf uses MN-major storage:
        with ``Q=ceil_div(K, sf_vec_size)``, its strides are ``(1, M)`` for
        2D input or ``(M * Q, 1, M)`` for 3D input. E8M0 1x32 scales can
        be consumed directly by MP32 MXFP GEMM without a layout transform.
        There is no scale padding or swizzle.

    Raises:
        ValueError: Input layout/dtype/device, activation parameters, output
            format, or the combination of scale options is unsupported.

    Examples:
        >>> from mate.quantization import silu_and_mul_quantize
        >>> x = torch.randn(16, 512, device="musa", dtype=torch.bfloat16)
        >>> q, sf = silu_and_mul_quantize(x)  # SwiGLU to MXFP4
        >>> q, sf = silu_and_mul_quantize(
        ...     x, output_dtype=torch.float8_e4m3fn, swiglu_limit=10.0
        ... )  # clamp SwiGLU to MXFP8
        >>> q, sf = silu_and_mul_quantize(
        ...     x, output_dtype=torch.float8_e4m3fn,
        ...     activation="situ", sf_vec_size=128
        ... )  # SiTU to FP8 with FP32 block scales
        >>> restored = q.float() * sf.repeat_interleave(128, dim=-1)[..., :256]
        >>> y, sf = silu_and_mul_quantize(
        ...     x, output_dtype=torch.bfloat16, activation="situ"
        ... )  # BF16 SiTU output; sf is None
    """
    return _silu_and_mul_quantize_impl(
        input,
        output_dtype,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        sf_type=sf_type,
        is_global_scale_inversed=is_global_scale_inversed,
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


@mate_api
def packbits(
    x: torch.Tensor, bitorder: Literal["big", "little"] = "big"
) -> torch.Tensor:
    """Pack truth values from a 1D tensor into bytes.

    Each input element is converted to bool: zero becomes 0, nonzero becomes
    1. Consecutive groups of eight bits form one byte. Missing elements in
    the final group are zero, and an empty input returns an empty output.

    Args:
        x: Contiguous 1D tensor on a MUSA device, of bool or a numeric dtype
            convertible to bool. Numeric inputs are converted before packing.
        bitorder: ``"big"`` (default) places the first element of each group
            in bit 7; ``"little"`` places it in bit 0.

    Returns:
        A contiguous uint8 MUSA tensor of shape ``(ceil_div(x.numel(), 8),)``.
        Tail padding occupies the unused low bits for big order or high bits
        for little order.

    Raises:
        ValueError: x is not 1D/on MUSA, or bitorder is unsupported.
        RuntimeError: A MUSA device is unavailable.

    Examples:
        >>> from mate.quantization import packbits
        >>> x = torch.tensor([True, False, True], device="musa")
        >>> packbits(x).cpu().tolist()
        [160]
        >>> packbits(x, bitorder="little").cpu().tolist()
        [5]
    """
    from .tilelang.packbits import _packbits_impl

    return _packbits_impl(x, bitorder)


@mate_api
def segment_packbits(
    x: torch.Tensor,
    indptr: torch.Tensor,
    bitorder: Literal["big", "little"] = "big",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length segments of truth values independently.

    Segment i is ``x[indptr[i]:indptr[i + 1]]``. Each segment starts at a
    fresh output byte and pads its final byte with zero bits. Empty segments
    are allowed and consume no bytes. Nonzero numeric inputs are treated as
    True, as in :func:`packbits`.

    Args:
        x: Contiguous 1D bool or numeric tensor on a MUSA device.
        indptr: Contiguous 1D int32/int64 tensor on x's device, of shape
            ``[num_segments + 1]``. Offsets must start at zero, be
            nondecreasing, and end at ``x.numel()``. Offsets and packed
            output offsets must fit signed int32; int64 offsets are
            converted to int32 internally.
        bitorder: ``"big"`` (default) puts the first bit of each group in
            bit 7; ``"little"`` puts it in bit 0.

    Returns:
        ``(packed, packed_indptr)`` on x's device. packed is contiguous uint8
        with ``sum(ceil_div(segment_length, 8))`` elements. packed_indptr is
        int32 with indptr's shape and contains offsets measured in output
        bytes, beginning with zero. All-empty segments return an empty packed
        tensor and zero offsets.

    Raises:
        ValueError: A tensor rank, device, offset dtype/order/final value,
            or bitorder is unsupported.
        RuntimeError: A MUSA device is unavailable.

    Note:
        Offset validation and output sizing read values on the host and
        synchronize with the device; this API is not graph-capture safe.

    Examples:
        >>> from mate.quantization import segment_packbits
        >>> x = torch.tensor([1, 0, 1, 1, 0], device="musa")
        >>> indptr = torch.tensor([0, 3, 5], device="musa", dtype=torch.int32)
        >>> packed, offsets = segment_packbits(x, indptr)
        >>> packed.cpu().tolist(), offsets.cpu().tolist()
        ([160, 128], [0, 1, 2])
    """
    from .tilelang.segment_packbits import _segment_packbits_impl

    return _segment_packbits_impl(x, indptr, bitorder)
