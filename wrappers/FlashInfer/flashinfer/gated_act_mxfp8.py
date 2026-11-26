"""FlashInfer fused SwiGLU MXFP8 quantization backed by MATE."""

import torch

from mate.quantization import silu_and_mul_quantize as _mate_silu_and_mul_quantize

__all__ = ["silu_and_mul_mxfp8_quantize"]


def silu_and_mul_mxfp8_quantize(
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply SwiGLU and quantize each row to MXFP8 with native E8M0 scales.

    Split ``gated_input`` into gate and up halves and compute
    ``gate * sigmoid(gate) * up`` in FP32. Quantize the result directly to
    E4M3, sharing one scale per 32 consecutive K elements. Positive scales
    round up to a power of two; scale code zero represents ``2**-127``.

    Args:
        gated_input: Finite BF16 MUSA tensor shaped ``[M, 2*K]``, with gate
            before up. M and K must be positive multiples of 128, matching
            the upstream shape contract. Only the last dimension must have
            unit stride; rows may be non-contiguous.
        rowwise: Must be True. Enables quantization along the K dimension.
        colwise: Must be False. Quantization along M is not implemented.

    Returns:
        ``(row_output, col_output, row_scales, col_scales)`` on input's device.
        ``row_output`` is contiguous E4M3 with shape ``[M, K]`` and dtype
        ``torch.float8_e4m3fn``. ``row_scales`` has shape ``[M, K // 32]``
        and dtype ``torch.float8_e8m0fnu`` in MN-major layout, with strides
        ``(1, M)``. Its dtype view does not copy or rearrange scales.
        ``col_output`` and ``col_scales`` have shape ``[0]`` and the same
        dtypes as the corresponding row tensors.

    Raises:
        NotImplementedError: colwise quantization is requested.
        TypeError: Input dtype is not BF16.
        ValueError: Both orientations are disabled, or input shape, device,
            or stride is unsupported.
    """
    if not rowwise and not colwise:
        raise ValueError("at least one of rowwise or colwise must be enabled")
    if colwise:
        raise NotImplementedError("MUSA gated MXFP8 supports rowwise quantization only")
    if gated_input.dtype != torch.bfloat16:
        raise TypeError("gated_input must have bfloat16 dtype")
    if gated_input.ndim != 2:
        raise ValueError("gated_input must have shape [M, 2K]")
    rows, doubled_k = gated_input.shape
    if rows <= 0 or doubled_k <= 0 or rows % 128 or doubled_k % 256:
        raise ValueError("M and K must be positive multiples of 128")

    row_output, row_scales = _mate_silu_and_mul_quantize(
        gated_input, output_dtype=torch.float8_e4m3fn
    )
    return (
        row_output,
        row_output.new_empty(0),
        row_scales.view(torch.float8_e8m0fnu),
        row_scales.new_empty(0).view(torch.float8_e8m0fnu),
    )
