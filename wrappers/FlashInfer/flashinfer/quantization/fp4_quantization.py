"""FlashInfer MXFP4 entry points backed by MATE's native scale layout."""

import torch

from mate.quantization import fp4_dequantize as _mate_fp4_dequantize
from mate.quantization import fp4_quantize as _mate_fp4_quantize

from ..tllm_enums import SfLayout


def fp4_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    is_global_scale_inversed: bool = False,
    enable_pdl: bool | None = None,
    backend: str = "cuda",
    nvfp4_4over6=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16/BF16 input using the supported MXFP4 recipe on MUSA.

    Args:
        input: MUSA tensor of shape ``[M, K]`` or ``[B, M, K]``. K must
            be a positive multiple of 32 with unit stride; outer strides
            may vary.
        global_scale: Must be None for this MXFP4 compatibility entry.
        sf_vec_size: Must be 32. The upstream NVFP4 default of 16 is retained
            in the signature but is unsupported; use :func:`mxfp4_quantize`
            for the native default recipe.
        sf_use_ue8m0: Must be True, selecting E8M0 block scales.
        is_sf_swizzled_layout: Accepted for call compatibility and ignored.
        is_sf_8x4_layout: Accepted for call compatibility and ignored.
        is_global_scale_inversed: Has no effect without a global scale.
        enable_pdl: Accepted for call compatibility and ignored on MUSA.
        backend: Upstream backend name, accepted and ignored. MATE selects
            the MUSA implementation.
        nvfp4_4over6: Must be None; NVFP4 recipes are not implemented.

    Returns:
        Packed uint8 payload of shape ``input.shape[:-1] + (K // 2,)``
        and uint8 E8M0 scales of shape
        ``input.shape[:-1] + (ceil_div(K, 32),)``. The earlier value occupies
        the low nibble. With ``Q=ceil_div(K, 32)``, MN-major scale strides are
        ``(1, M)`` for 2D input or ``(M * Q, 1, M)`` for 3D input. Scales have
        no swizzle, flattening, or padding, regardless of the layout arguments.

    Raises:
        NotImplementedError: The requested recipe is not MXFP4 without a
            global scale. Native MATE scale options remain available from
            ``mate.quantization.fp4_quantize``.
        ValueError: Input dtype, device, shape, or stride is unsupported.
    """
    del is_sf_swizzled_layout, is_sf_8x4_layout, is_global_scale_inversed
    del enable_pdl, backend
    if sf_vec_size != 32 or not sf_use_ue8m0:
        raise NotImplementedError(
            "MUSA supports MXFP4 here: use sf_vec_size=32, sf_use_ue8m0=True "
            "or mxfp4_quantize(input)"
        )
    if global_scale is not None or nvfp4_4over6 is not None:
        raise NotImplementedError(
            "The MXFP4 wrapper does not support global_scale or NVFP4 recipes"
        )
    return mxfp4_quantize(input)


def mxfp4_quantize(
    a: torch.Tensor,
    backend: str = "cuda",
    enable_pdl: bool | None = None,
    sfLayout: SfLayout = SfLayout.layout_128x4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to packed MXFP4 with native MUSA MN-major E8M0 scales.

    Args:
        a: FP16/BF16 MUSA tensor shaped ``[M, K]`` or ``[B, M, K]``.
            K must be a positive multiple of 32, and only its stride must
            be 1. Outer strides may vary.
        backend: Accepted for compatibility; MATE selects the implementation.
        enable_pdl: Accepted for compatibility; ignored on MUSA.
        sfLayout: Accepted for compatibility; all values return the native
            MN-major SF layout, without a layout conversion.

    Returns:
        uint8 ``(q, sf)`` on a's device. q is contiguous with shape
        ``a.shape[:-1] + (K // 2,)``, with low nibble first; sf has shape
        ``a.shape[:-1] + (K // 32,)`` and MN-major strides ``(1, M)`` for 2D
        input or ``(M * (K // 32), 1, M)`` for 3D input. Scales encode the ceiling
        power of two of each block's ``amax / 6``. Code zero is ``2**-127``.
    """
    del backend, enable_pdl, sfLayout
    if a.shape[-1] % 32:
        raise ValueError("FlashInfer MXFP4 requires K to be a multiple of 32")
    return _mate_fp4_quantize(a)


def mxfp4_dequantize(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    sfLayout: SfLayout = SfLayout.layout_128x4,
) -> torch.Tensor:
    """Decode native MXFP4 and return FP32 on the CPU.

    Args:
        a_fp4: Packed uint8 or float4_e2m1fn_x2 tensor shaped ``[M, K/2]``
            or ``[B, M, K/2]`` with unit last stride.
        a_sf: uint8 E8M0 scales on the same device, shaped
            ``a_fp4.shape[:-1] + (ceil_div(K, 32),)``. Both native MN-major
            and row-major scales are supported.
        sfLayout: Compatibility argument, ignored. Scales are read using
            their actual strides, without an NVIDIA layout conversion.

    Returns:
        Contiguous CPU FP32 tensor shaped ``a_fp4.shape[:-1] + (K,)``,
        matching the upstream entry's return device. Decoding uses MATE on
        MUSA, followed by a device-to-host copy. To keep the result on MUSA,
        call ``mate.quantization.fp4_dequantize`` directly.
    """
    del sfLayout
    return _mate_fp4_dequantize(a_fp4, a_sf).cpu()
