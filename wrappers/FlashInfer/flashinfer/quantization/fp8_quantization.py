"""FlashInfer MXFP8 entry point backed by MATE's native scale layout."""

import torch

from mate.quantization import fp8_quantize as _mate_fp8_quantize

from ..tllm_enums import SfLayout


def mxfp8_quantize(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: bool | None = None,
    backend: str = "cuda",
    sf_swizzle_layout: SfLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP8 E4M3 with native MUSA MN-major E8M0 scales.

    Args:
        input: FP16/BF16 MUSA tensor shaped ``[M, K]`` or ``[B, M, K]``.
            K must be a positive multiple of 32 with unit stride. Outer
            strides may vary.
        is_sf_swizzled_layout: Compatibility argument, ignored. Scales
            always use the native MN-major layout.
        alignment: Must be 32. The payload preserves the logical K dimension.
        enable_pdl: Compatibility argument, ignored on MUSA.
        backend: Compatibility argument; MATE selects the implementation.
        sf_swizzle_layout: Compatibility argument, ignored. No scale
            swizzle or layout conversion is performed.

    Returns:
        ``(q, sf)`` on input's device. q is contiguous with input's shape
        and dtype ``torch.float8_e4m3fn``; sf is uint8 with shape
        ``input.shape[:-1] + (K // 32,)`` and MN-major strides ``(1, M)`` for
        2D input or ``(M * (K // 32), 1, M)`` for 3D input. Scales encode the ceiling
        power of two of ``amax / 448``; code zero is ``2**-127``. The scales
        retain input rank and have no padding or flattening.

    Raises:
        NotImplementedError: A payload alignment other than 32 is requested.
        ValueError: Input dtype, device, shape, or stride is unsupported.
    """
    del is_sf_swizzled_layout, enable_pdl, backend, sf_swizzle_layout
    if alignment != 32:
        raise NotImplementedError("MUSA MXFP8 supports alignment=32")
    if input.shape[-1] % 32:
        raise ValueError("FlashInfer MXFP8 requires K to be a multiple of 32")
    return _mate_fp8_quantize(input)
