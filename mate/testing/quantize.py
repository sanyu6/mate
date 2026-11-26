import torch


_MXFP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)


__all__ = [
    "dequantize_mxfp4",
    "dequantize_mxfp8",
    "quantize_mxfp4",
    "quantize_mxfp8",
]


def _quantize_ue8m0_scale(
    x_grouped: torch.Tensor, max_value: float
) -> tuple[torch.Tensor, torch.Tensor]:
    block_scale = x_grouped.abs().amax(dim=-1) / max_value
    # Round positive FP32 scales up to UE8M0 without log2 rounding error.
    scale_bits = block_scale.contiguous().view(torch.int32)
    exponent = (scale_bits >> 23) & 0xFF
    mantissa = scale_bits & 0x7FFFFF
    increment_exponent = (mantissa != 0) & ((exponent != 0) | (mantissa > 0x400000))
    exponent = (exponent + increment_exponent.to(exponent.dtype)).clamp(max=254)
    scale_ue8m0 = torch.where(block_scale > 0, exponent, 0).to(torch.uint8)

    inverse_scale = torch.ldexp(
        torch.ones_like(block_scale), 127 - scale_ue8m0.to(torch.int32)
    )
    return scale_ue8m0, inverse_scale


def _dequantize_ue8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    # Code zero is the subnormal value 2^-127, not a zero scale.
    bits = scale.to(torch.int32) << 23
    return torch.where(scale == 0, 0x00400000, bits).view(torch.float32)


def quantize_mxfp8(
    x: torch.Tensor, output_dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP8 quantization of complete 32-value blocks to E4M3/E5M2."""
    if output_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("MXFP8 output_dtype must be float8_e4m3fn or float8_e5m2")
    dim = x.size(-1)
    if dim <= 0 or dim % 32 != 0:
        raise ValueError(
            "MXFP8 quantization requires a positive final dimension divisible by 32"
        )

    groups = dim // 32
    x_grouped = x.float().reshape(*x.shape[:-1], groups, 32)
    max_value = torch.finfo(output_dtype).max
    scale_ue8m0, inverse_scale = _quantize_ue8m0_scale(x_grouped, max_value)
    x_fp8 = (x_grouped * inverse_scale.unsqueeze(-1)).clamp(-max_value, max_value)
    x_fp8 = x_fp8.reshape_as(x).to(output_dtype)
    return x_fp8, scale_ue8m0.contiguous()


def dequantize_mxfp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode row-major MXFP8, including a partial final 32-value block."""
    if (
        x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or scale.dtype != torch.uint8
    ):
        raise TypeError(
            "MXFP8 values must be float8_e4m3fn/float8_e5m2 and scales must be uint8"
        )
    expected_scale_shape = (*x.shape[:-1], (x.size(-1) + 31) // 32)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"MXFP8 scales must have shape {expected_scale_shape}")

    logical_scale = _dequantize_ue8m0_scale(scale)
    return x.float() * logical_scale.repeat_interleave(32, dim=-1)[..., : x.size(-1)]


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _MXFP4_DTYPE is None:
        raise RuntimeError(
            "MXFP4 quantization requires torch.float4_e2m1fn_x2, "
            "which is unavailable in this PyTorch build"
        )
    dim = x.size(-1)
    if dim <= 0 or dim % 32 != 0:
        raise ValueError(
            "MXFP4 quantization requires a positive final dimension divisible by 32"
        )

    groups = dim // 32
    x_grouped = x.float().reshape(*x.shape[:-1], groups, 32)
    scale_ue8m0, inverse_scale = _quantize_ue8m0_scale(x_grouped, 6.0)
    x_scaled = x_grouped * inverse_scale.unsqueeze(-1)

    # E2M1 magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}; midpoint ties round to even.
    magnitude = x_scaled.abs()
    codes = torch.zeros_like(x_scaled, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        codes += (magnitude > boundary).to(torch.uint8)
    for upper_even_tie in (0.75, 1.75, 3.5):
        codes += (magnitude == upper_even_tie).to(torch.uint8)
    codes |= (torch.signbit(x_scaled) & ~torch.isnan(x_scaled)).to(torch.uint8) << 3

    codes = codes.reshape(*x.shape[:-1], dim // 2, 2)
    packed = (codes[..., 0] & 0x0F) | ((codes[..., 1] & 0x0F) << 4)

    return packed.view(_MXFP4_DTYPE), scale_ue8m0.contiguous()


def dequantize_mxfp4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode row-major packed MXFP4, including a partial final scale block."""
    if _MXFP4_DTYPE is None:
        raise RuntimeError(
            "MXFP4 dequantization requires torch.float4_e2m1fn_x2, "
            "which is unavailable in this PyTorch build"
        )
    if x.dtype != _MXFP4_DTYPE or scale.dtype != torch.uint8:
        raise TypeError(
            "MXFP4 values and scales must have float4_e2m1fn_x2 and uint8 dtypes"
        )
    cols = x.size(-1) * 2
    expected_scale_shape = (*x.shape[:-1], (cols + 31) // 32)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"MXFP4 scales must have shape {expected_scale_shape}")

    packed = x.view(torch.uint8)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)

    e2m1_values = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=x.device,
    )
    values = e2m1_values[(codes & 0x07).to(torch.int64)]
    values = torch.where((codes & 0x08) != 0, -values, values)

    logical_scale = _dequantize_ue8m0_scale(scale)
    return values * logical_scale.repeat_interleave(32, dim=-1)[..., :cols]
