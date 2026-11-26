"""Functional coverage for the public quantization APIs.

The functions in this module intentionally have no pytest fixtures or hidden
state, so they can also be called directly by external validation scripts.
"""

import numpy as np
import pytest
import torch

from mate.quantization import (
    fp4_quantize,
    fp4_dequantize,
    fp8_quantize,
    fp8_dequantize,
    silu_and_mul_quantize,
    packbits,
    segment_packbits,
)
from mate.testing import (
    dequantize_mxfp4,
    dequantize_mxfp8,
    quantize_mxfp4,
    quantize_mxfp8,
)

# FlashInfer's test_situ_f32_constant_and_runtime_beta compares the FP32 gate
# against its mathematical reference with these tolerances (not MoE tolerances).
# https://github.com/flashinfer-ai/flashinfer/blob/5b550737e6d8b698c0fb7a5026cce816a3f461e5/tests/moe/test_cute_dsl_bf16_gather_grouped_gemm.py#L348
_SITU_RTOL = 1e-5
_SITU_ATOL = 2e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1, 2), "contiguous"),
        ((6, 4096), "contiguous"),
        ((3, 66), "offset"),
        ((2, 7, 1056), "aligned_pitch"),
        ((2, 7, 4096), "rows"),
        ((2, 7, 512), "broadcast_batch"),
        ((2, 7, 512), "batch_transpose"),
        ((3, 1024), "broadcast"),
        ((3, 1024), "pitch"),
        ((1032, 4096), "contiguous"),
        ((2, 1032, 4160), "aligned_pitch"),
    ],
)
def test_situ_bf16_output(dtype, shape, layout):
    device = _musa_device()
    torch.manual_seed(27)
    x = _tensor_view(torch.randn(shape, dtype=dtype, device=device) * 8, layout)
    before = x.cpu()
    for beta, linear_beta in ((4.0, 25.0), (4.0, None), (4.0, 3.0), (2.0, None)):
        y, sf = silu_and_mul_quantize(
            x,
            output_dtype=torch.bfloat16,
            activation="situ",
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        reference, error = _reference_situ_with_error(
            *x.chunk(2, dim=-1), beta, linear_beta
        )
        assert sf is None
        assert y.dtype == torch.bfloat16 and y.device == x.device
        assert y.shape == (*shape[:-1], shape[-1] // 2) and y.is_contiguous()
        # FP32 activation tolerance plus one correctly rounded BF16 store.
        error += reference.abs() * (2**-8) + 2**-133
        assert ((y.cpu().float() - reference).abs() <= error).all()
    torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("aligned", [False, True])
def test_gated_bf16_extremes_and_clamp(aligned):
    device = _musa_device()
    gate = torch.tensor(
        [-1e4, -100, -25, -4, -1e-4, -1e-7, 0, 1e-7, 1e-4, 4, 25, 100, 1e4],
        dtype=torch.bfloat16,
        device=device,
    )
    if aligned:
        gate = torch.cat((gate, torch.zeros(3, device=device, dtype=gate.dtype)))
    x = torch.cat(
        (
            gate[:, None].expand(-1, gate.numel()),
            gate[None, :].expand(gate.numel(), -1),
        ),
        dim=1,
    )
    for options in ({"activation": "situ"}, {}, {"swiglu_limit": 10.0}):
        y, sf = silu_and_mul_quantize(x, output_dtype=torch.bfloat16, **options)
        reference = _reference_swiglu(x, **options)
        assert sf is None
        if options.get("activation") == "situ":
            reference, error = _reference_situ_with_error(*x.chunk(2, dim=-1))
        else:
            error = 2e-5 + reference.abs() * 1e-5
        error += reference.abs() * (2**-8) + 2**-133
        assert torch.isfinite(y).all()
        assert ((y.cpu().float() - reference).abs() <= error).all()


def test_gated_bf16_reject_scale_options():
    x = torch.ones((1, 32), dtype=torch.bfloat16, device=_musa_device())
    for options in (
        {"sf_type": "e8m0"},
        {"sf_vec_size": 128},
        {"is_global_scale_inversed": True},
        {"global_scale": torch.ones(1, device=x.device)},
    ):
        with pytest.raises(ValueError, match="BF16 output"):
            silu_and_mul_quantize(x, output_dtype=torch.bfloat16, **options)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_situ_bf16_finite_encodings(dtype):
    device = _musa_device()
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values[~torch.isfinite(values)] = 0
    # Every finite gate and up encoding, with opposite signs to exercise both
    # saturated softcaps and sigmoid underflow without overflowing the reference.
    x = torch.cat((values, values.flip(0))).reshape(1, -1).to(device)
    y, sf = silu_and_mul_quantize(x, output_dtype=torch.bfloat16, activation="situ")
    reference, error = _reference_situ_with_error(*x.chunk(2, dim=-1))
    error += reference.abs() * (2**-8) + 2**-133
    assert sf is None and torch.isfinite(y).all()
    assert ((y.cpu().float() - reference).abs() <= error).all()


def _musa_device() -> torch.device:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is required for quantization functional tests")
    return torch.device("musa")


def _reference_fp4_dequantize(q, sf, sf_type):
    q = q.view(torch.float4_e2m1fn_x2)
    if sf_type == "e8m0":
        return dequantize_mxfp4(q, sf)
    # Reuse the shared E2M1 decoder with unit MX scales, then apply E4M3 scales.
    decoded = dequantize_mxfp4(q, torch.full_like(sf, 127))
    scale = sf.view(torch.float8_e4m3fn).float()
    return decoded * scale.repeat_interleave(32, dim=-1)[..., : decoded.shape[-1]]


def _group_values(x, sf_vec_size=32):
    values = x.detach().cpu().float()
    values = torch.nn.functional.pad(values, (0, -x.shape[-1] % sf_vec_size))
    return values.reshape(*x.shape[:-1], -1, sf_vec_size)


def _reference_scale(raw, sf_type, *, fp8=False):
    if sf_type == "e8m0":
        mantissa, exponent = torch.frexp(raw.double().clamp(max=2.0**127))
        exponent = (exponent - (mantissa == 0.5).to(exponent.dtype)).clamp(-127, 127)
        exponent = torch.where(raw > 0, exponent, -127)
        return (exponent + 127).to(torch.uint8), torch.ldexp(
            torch.ones_like(raw), exponent
        )
    if fp8:
        raw = torch.where(raw > 0, raw.clamp(2.0**-9, 448.0), 0.0)
    # MUSA scale conversion saturates to finite E4M3, including FP4 scales.
    encoded = raw.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return encoded.view(torch.uint8), encoded.float()


def _reference_fp4_quantize(
    x, sf_type, global_scale=1.0, inverse=False, encoded_scale=None
):
    values = _group_values(x)
    if sf_type == "e8m0" and global_scale == 1.0 and encoded_scale is None:
        q, sf = quantize_mxfp4(values.flatten(-2))
        return q.view(torch.uint8)[..., : x.shape[-1] // 2], sf
    # Preserve FP32 reciprocal-multiply rounding before applying global_scale.
    # Without global_scale, finite FP16/BF16 maxima encode as with division.
    raw = values.abs().amax(dim=-1) * (1.0 / 6.0)
    raw = raw / global_scale if inverse else raw * global_scale
    sf, scale = _reference_scale(raw, sf_type)
    if encoded_scale is not None:
        sf = encoded_scale
        scale = (
            sf.view(torch.float8_e4m3fn).float()
            if sf_type == "e4m3"
            else torch.pow(2.0, sf.float() - 127)
        )
    normalized = values * torch.where(scale > 0, scale.reciprocal(), 0.0).unsqueeze(-1)
    magnitude = normalized.abs()
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code = torch.bucketize(magnitude, boundaries)
    # E2M1 uses round-to-nearest-even at each representable midpoint.
    tie = magnitude == boundaries[code.clamp(max=6)]
    code = code + (tie & (code % 2 == 1))
    code = (code | (torch.signbit(normalized).long() << 3)).to(torch.uint8)
    code = code.flatten(-2)[..., : x.shape[-1]]
    return code[..., 0::2] | (code[..., 1::2] << 4), sf


def _reference_fp8_quantize(
    x, output_dtype, sf_type, sf_vec_size=32, encoded_scale=None
):
    values = _group_values(x, sf_vec_size)
    if sf_type == "e8m0" and sf_vec_size == 32 and encoded_scale is None:
        q, sf = quantize_mxfp8(values.flatten(-2), output_dtype=output_dtype)
        return q[..., : x.shape[-1]], sf
    amax = values.abs().amax(dim=-1)
    q_max = torch.finfo(output_dtype).max
    if sf_type == "fp32":
        amax = amax.clamp_min(1e-4)
        sf = amax * (1.0 / q_max)
        inv_scale = q_max / amax
    else:
        raw = amax * (1.0 / q_max)
        sf, scale = _reference_scale(raw, sf_type, fp8=True)
        inv_scale = torch.where(scale > 0, scale.reciprocal(), 0.0)
    if encoded_scale is not None:
        sf = encoded_scale
        scale = (
            sf
            if sf_type == "fp32"
            else sf.view(torch.float8_e4m3fn).float()
            if sf_type == "e4m3"
            else torch.pow(2.0, sf.float() - 127)
        )
        inv_scale = torch.where(scale > 0, scale.reciprocal(), 0.0)
    q = (values * inv_scale.unsqueeze(-1)).clamp(-q_max, q_max).to(output_dtype)
    return q.flatten(-2)[..., : x.shape[-1]], sf


def _reference_fp8_dequantize(q, sf, sf_type, output_dtype):
    q, sf = q.detach().cpu(), sf.detach().cpu()
    if sf_type == "e8m0":
        return dequantize_mxfp8(q, sf).to(output_dtype)
    scale = sf.view(torch.float8_e4m3fn).float()
    return (q.float() * scale.repeat_interleave(32, dim=-1)[..., : q.shape[-1]]).to(
        output_dtype
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_mxfp_reference_minimum_scale(q_dtype):
    # Code zero must retain small nonzero values, including signed zeros.
    # Run on CPU so device denormal handling cannot mask a reference error.
    fp4 = q_dtype == torch.float4_e2m1fn_x2
    q_max = 6.0 if fp4 else torch.finfo(q_dtype).max
    normalized = torch.tensor([[0.0, -0.0, 0.5, -0.5, 1.0, -1.0, q_max, -q_max] * 4])
    values = normalized * (2.0**-127)
    q, sf = (
        quantize_mxfp4(values) if fp4 else quantize_mxfp8(values, output_dtype=q_dtype)
    )
    assert q.dtype == q_dtype
    torch.testing.assert_close(sf, torch.zeros((1, 1), dtype=torch.uint8))
    restored = (dequantize_mxfp4 if fp4 else dequantize_mxfp8)(q, sf)
    torch.testing.assert_close(restored, values, rtol=0, atol=0)
    torch.testing.assert_close(torch.signbit(restored), torch.signbit(values))
    # Independently check the payload against known exactly representable values.
    if fp4:
        expected = torch.tensor([[0x80, 0x91, 0xA2, 0xF7] * 4], dtype=torch.uint8)
    else:
        expected = normalized.to(q_dtype).view(torch.uint8)
    torch.testing.assert_close(q.view(torch.uint8), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "q_dtype", [torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_mxfp_reference_partial_dequantize(q_dtype):
    fp4 = q_dtype == torch.float4_e2m1fn_x2
    dequantize = dequantize_mxfp4 if fp4 else dequantize_mxfp8
    values = torch.linspace(-6, 6, 2 * 3 * 64).reshape(2, 3, 64)
    q, sf = (
        quantize_mxfp4(values) if fp4 else quantize_mxfp8(values, output_dtype=q_dtype)
    )
    complete = dequantize(q, sf)
    for cols in (2, 16, 34, 48):
        # Payload and scales deliberately have different noncontiguous strides.
        payload = q.view(torch.uint8)[..., : cols // 2 if fp4 else cols].view(q_dtype)
        scales = sf[..., : (cols + 31) // 32]
        torch.testing.assert_close(
            dequantize(payload, scales), complete[..., :cols], rtol=0, atol=0
        )


def test_packbits_functional() -> None:
    device = _musa_device()
    for size in (0, 1, 7, 8, 9, 4097):
        x = (torch.arange(size, device=device) % 3 != 0).bool()
        for bitorder in ("big", "little"):
            expected = torch.from_numpy(
                np.packbits(x.cpu().numpy(), bitorder=bitorder)
            ).to(device)
            torch.testing.assert_close(packbits(x, bitorder), expected)


def test_segment_packbits_functional() -> None:
    device = _musa_device()
    lengths = torch.tensor([0, 1, 7, 8, 9, 17], dtype=torch.int32, device=device)
    indptr = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), lengths.cumsum(0))
    )
    x = (torch.arange(int(indptr[-1].item()), device=device) % 3 != 0).bool()
    expected_lengths = (lengths + 7) // 8
    expected_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            expected_lengths.cumsum(0),
        )
    ).to(torch.int32)
    for bitorder in ("big", "little"):
        actual, actual_indptr = segment_packbits(x, indptr, bitorder)
        expected = torch.cat(
            [
                torch.from_numpy(
                    np.packbits(
                        x[indptr[i] : indptr[i + 1]].cpu().numpy(),
                        bitorder=bitorder,
                    )
                )
                for i in range(lengths.numel())
            ]
        ).to(device)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_indptr, expected_indptr)


def _tensor_view(x, layout):
    """Construct views with independent logical values and physical layouts."""
    if layout == "contiguous":
        return x
    if layout == "broadcast":
        return x[..., :1, :].expand_as(x)
    if layout == "broadcast_batch":
        return x[:1].expand_as(x)
    if layout == "batch_transpose":
        return x.transpose(0, 1).contiguous().transpose(0, 1)
    if layout in ("offset", "pitch", "aligned_offset", "aligned_pitch"):
        padding, offset = {
            "offset": (16, 1),
            "pitch": (1, 0),
            "aligned_offset": (16, 8),
            "aligned_pitch": (8, 0),
        }[layout]
        backing = torch.empty(
            (*x.shape[:-1], x.shape[-1] + padding), dtype=x.dtype, device=x.device
        )
        view = backing[..., offset : offset + x.shape[-1]]
    elif layout == "rows":
        backing = torch.empty(
            (*x.shape[:-2], x.shape[-2] * 2, x.shape[-1]),
            dtype=x.dtype,
            device=x.device,
        )
        view = backing[..., ::2, :]
    else:
        # Odd row padding and a one-element offset invalidate vector alignment.
        backing = torch.empty(
            (*x.shape[:-1], x.shape[-1] + 3), dtype=x.dtype, device=x.device
        )
        view = backing[..., 1 : x.shape[-1] + 1]
    view.copy_(x)
    return view


def _reference_swiglu(
    x, activation="swiglu", swiglu_limit=None, situ_beta=4.0, situ_linear_beta=25.0
):
    gate, up = x.detach().cpu().float().chunk(2, dim=-1)
    if activation == "situ":
        gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
        if situ_linear_beta is not None:
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
        return gate * up
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    return (gate * (1.0 / (torch.exp2(-gate * 1.4426950408889634) + 1.0))) * up


def _assert_mn_major_scales(sf):
    rows, groups = sf.shape[-2:]
    assert sf.stride() == ((rows * groups, 1, rows) if sf.ndim == 3 else (1, rows))


def _reference_situ_with_error(gate, up, beta=4.0, linear_beta=25.0):
    """Extend the standalone gate tolerance to MATE's two-branch SiTU product."""
    gate, up = gate.detach().cpu().double(), up.detach().cpu().double()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    gate_error = _SITU_ATOL + _SITU_RTOL * gate.abs()
    up_error = torch.zeros_like(up)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
        # Apply the same FP32 activation budget to the softcapped up branch.
        up_error = _SITU_ATOL + _SITU_RTOL * up.abs()
    error = gate_error * up.abs() + up_error * gate.abs() + gate_error * up_error
    return (gate * up).float(), error.float()


def _assert_activation_quantization(
    q, sf, reference, error, output_dtype, sf_type, size
):
    """Check scale and payload rounding of an activation within its error budget."""
    is_fp4 = output_dtype == torch.float4_e2m1fn_x2
    q_max = 6.0 if is_fp4 else torch.finfo(output_dtype).max
    amax_low = _group_values((reference.abs() - error).clamp_min(0), size).amax(-1)
    amax_high = _group_values(reference.abs() + error, size).amax(-1)
    if sf_type == "fp32":
        sf_low = amax_low.clamp_min(1e-4) * (1.0 / q_max)
        sf_high = amax_high.clamp_min(1e-4) * (1.0 / q_max)
    else:
        sf_low, _ = _reference_scale(amax_low * (1.0 / q_max), sf_type, fp8=not is_fp4)
        sf_high, _ = _reference_scale(
            amax_high * (1.0 / q_max), sf_type, fp8=not is_fp4
        )
    sf = sf.detach().cpu()
    assert ((sf >= sf_low) & (sf <= sf_high)).all(), "Scale exceeds activation budget"

    def quantized_values(values):
        if is_fp4:
            packed, _ = _reference_fp4_quantize(values, sf_type, encoded_scale=sf)
            return dequantize_mxfp4(
                packed.view(torch.float4_e2m1fn_x2), torch.full_like(sf, 127)
            )
        quantized, _ = _reference_fp8_quantize(
            values, output_dtype, sf_type, size, encoded_scale=sf
        )
        return quantized.float()

    # Use each returned scale to interpret its payload: an allowed E8M0 boundary
    # crossing changes both the exponent and the codes. Numerical intervals also
    # handle signed zero, where integer code differences are not meaningful.
    low, high = quantized_values(reference - error), quantized_values(reference + error)
    actual = (
        dequantize_mxfp4(
            q.detach().cpu().view(torch.float4_e2m1fn_x2), torch.full_like(sf, 127)
        )
        if is_fp4
        else q.detach().cpu().float()
    )
    assert torch.isfinite(actual).all()
    assert ((actual >= low) & (actual <= high)).all(), (
        "Payload exceeds activation budget"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sf_type", ["e8m0", "e4m3"])
@pytest.mark.parametrize(
    "q_dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("fused", [False, True])
def test_quantization_strides_and_tails(dtype, sf_type, q_dtype, fused):
    device = _musa_device()
    is_fp4 = q_dtype == torch.uint8
    quant = (
        (silu_and_mul_quantize if fused else fp4_quantize)
        if is_fp4
        else (silu_and_mul_quantize if fused else fp8_quantize)
    )
    kwargs = {} if is_fp4 else {"output_dtype": q_dtype}
    cases = [
        ((1, 2), "contiguous"),
        ((7, 16), "contiguous"),
        ((7, 48), "contiguous"),
        ((7, 1008), "contiguous"),
        ((2, 3, 48), "batch_transpose"),
        ((7, 48), "offset"),
        ((7, 48), "pitch"),
        ((7, 30), "offset"),
        ((2, 3, 34), "batch_transpose"),
        ((31, 62), "padded"),
        ((32, 64), "rows"),
        ((33, 66), "broadcast"),
        ((2, 3, 8194), "batch_transpose"),
        ((7, 32), "pitch"),
        ((2, 32, 4096), "aligned_offset"),
        ((2, 256, 1040), "aligned_pitch"),
        ((2, 32, 4096), "broadcast_batch"),
    ]
    torch.manual_seed(17)
    for shape, layout in cases:
        input_shape = (*shape[:-1], shape[-1] * (2 if fused else 1))
        x = _tensor_view(torch.randn(input_shape, dtype=dtype, device=device), layout)
        q, sf = quant(x, sf_type=sf_type, **kwargs)
        reference_input = _reference_swiglu(x) if fused else x
        expected_q, expected_sf = (
            _reference_fp4_quantize(reference_input, sf_type)
            if is_fp4
            else _reference_fp8_quantize(reference_input, q_dtype, sf_type)
        )
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8).cpu(), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        assert q.shape == (*shape[:-1], shape[-1] // 2 if is_fp4 else shape[-1])
        assert sf.shape == (*shape[:-1], (shape[-1] + 31) // 32)
        assert q.dtype == q_dtype and sf.dtype == torch.uint8
        assert q.is_contiguous()
        _assert_mn_major_scales(sf)
        # Independent q and sf strides catch accidental reuse of input strides.
        q_view = _tensor_view(q.view(torch.uint8), "padded").view(q.dtype)
        sf_view = sf
        if is_fp4:
            actual = fp4_dequantize(q_view, sf_view, sf_type)
            expected = _reference_fp4_dequantize(q.cpu(), sf.cpu(), sf_type)
        else:
            actual = fp8_dequantize(q_view, sf_view, sf_type)
            expected = _reference_fp8_dequantize(q, sf, sf_type, torch.float32)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_quantization_zero_and_tail_maximum(fused):
    device = _musa_device()
    for k in (2, 16, 30, 32, 34, 48, 66, 1008):
        x = torch.zeros(
            (3, k * (2 if fused else 1)), dtype=torch.bfloat16, device=device
        )
        if fused:
            x[:, :k] = 1.0
        x[1, -1] = 6.0
        x[2, -1] = -6.0
        values = _reference_swiglu(x) if fused else x
        for sf_type in ("e8m0", "e4m3"):
            for q_dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
                if q_dtype == torch.uint8:
                    fn = silu_and_mul_quantize if fused else fp4_quantize
                    q, sf = fn(x, sf_type=sf_type)
                    expected_q, expected_sf = _reference_fp4_quantize(values, sf_type)
                else:
                    fn = silu_and_mul_quantize if fused else fp8_quantize
                    q, sf = fn(x, sf_type=sf_type, output_dtype=q_dtype)
                    expected_q, expected_sf = _reference_fp8_quantize(
                        values, q_dtype, sf_type
                    )
                torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
                torch.testing.assert_close(
                    q.view(torch.uint8).cpu(),
                    expected_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("sf_type", ["e8m0", "e4m3"])
@pytest.mark.parametrize("inverse", [False, True])
def test_fp4_global_scale_with_tails(fused, sf_type, inverse):
    device = _musa_device()
    scale = torch.tensor([2.0], dtype=torch.float32, device=device)
    fn = silu_and_mul_quantize if fused else fp4_quantize
    for k, layout in ((64, "rows"), (34, "padded"), (48, "rows"), (1008, "rows")):
        x = _tensor_view(
            torch.randn(
                (3, k * (2 if fused else 1)), dtype=torch.float16, device=device
            ),
            layout,
        )
        q, sf = fn(
            x, global_scale=scale, sf_type=sf_type, is_global_scale_inversed=inverse
        )
        values = _reference_swiglu(x) if fused else x
        expected_q, expected_sf = _reference_fp4_quantize(values, sf_type, 2.0, inverse)
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("sf_type", ["e8m0", "e4m3"])
@pytest.mark.parametrize(
    "q_dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("fused", [False, True])
def test_quantization_wide_rows_and_tails(sf_type, q_dtype, fused):
    """Check public quantization results across wide rows and partial groups."""
    device = _musa_device()
    fp4 = q_dtype == torch.uint8
    fn = (
        (silu_and_mul_quantize if fused else fp4_quantize)
        if fp4
        else (silu_and_mul_quantize if fused else fp8_quantize)
    )
    kwargs = {} if fp4 else {"output_dtype": q_dtype}
    for rows, cols in (
        (32, 4096),
        (32, 4112),
        (2048, 2080),
        (8192, 2048),
    ):
        x = torch.randn(
            (rows, cols * (2 if fused else 1)), dtype=torch.bfloat16, device=device
        )
        q, sf = fn(x, sf_type=sf_type, **kwargs)
        _assert_mn_major_scales(sf)
        # Compare all outputs for aligned and unaligned storage of the same input.
        expected_q, expected_sf = fn(
            _tensor_view(x, "padded"), sf_type=sf_type, **kwargs
        )
        torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        # Also check sampled rows against the independent CPU reference.
        selected = torch.tensor([0, rows // 2, rows - 1], device=device)
        values = _reference_swiglu(x[selected]) if fused else x[selected]
        reference_q, reference_sf = (
            _reference_fp4_quantize(values, sf_type)
            if fp4
            else _reference_fp8_quantize(values, q_dtype, sf_type)
        )
        torch.testing.assert_close(sf[selected].cpu(), reference_sf, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8)[selected].cpu(),
            reference_q.view(torch.uint8),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "q_dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dequantization_layouts(q_dtype, output_dtype):
    device = _musa_device()
    for cols in (2, 16, 34, 48, 64, 2080):
        x = torch.randn((2, 3, cols), dtype=torch.bfloat16, device=device)
        fp4 = q_dtype == torch.uint8
        for sf_type in ("e8m0", "e4m3"):
            q, sf = (
                fp4_quantize(x, sf_type=sf_type)
                if fp4
                else fp8_quantize(x, sf_type=sf_type, output_dtype=q_dtype)
            )
            for q_layout, sf_layout in (
                ("contiguous", "rows"),
                ("rows", "padded"),
                ("padded", "batch_transpose"),
                ("broadcast", "broadcast"),
                ("batch_transpose", "rows"),
                ("offset", "padded"),
                ("pitch", "batch_transpose"),
            ):
                q_view = _tensor_view(q.view(torch.uint8), q_layout)
                sf_view = _tensor_view(sf, sf_layout)
                if fp4:
                    # Both public FP4 storage dtypes must preserve byte strides.
                    actual = fp4_dequantize(
                        q_view.view(torch.float4_e2m1fn_x2),
                        sf_view,
                        sf_type,
                        output_dtype=output_dtype,
                    )
                    expected = _reference_fp4_dequantize(
                        q_view.cpu(), sf_view.cpu(), sf_type
                    ).to(output_dtype)
                else:
                    q_view = q_view.view(q_dtype)
                    actual = fp8_dequantize(
                        q_view, sf_view, sf_type, output_dtype=output_dtype
                    )
                    expected = _reference_fp8_dequantize(
                        q_view, sf_view, sf_type, output_dtype
                    )
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sf_type", ["e8m0", "e4m3"])
@pytest.mark.parametrize("output_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_saturation(dtype, sf_type, output_dtype):
    device = _musa_device()
    # Every finite input encoding exercises subnormals, signed zeros, scale
    # boundaries and overflow of E4M3 scales without relying on random samples.
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values = values[torch.isfinite(values)]
    values = torch.nn.functional.pad(values, (0, -values.numel() % (16 * 2048)))
    base = values.reshape(1, -1, 2048)
    for cols in (2048, 2064, 4096, 4112):
        cpu_input = torch.nn.functional.pad(base, (0, cols - 2048))
        # Include a nonzero final half-group with both signs.
        if cols % 32:
            cpu_input[..., -16:] = torch.tensor(
                [torch.finfo(dtype).max, -torch.finfo(dtype).max], dtype=dtype
            ).repeat(8)
        storage = torch.empty(
            (*cpu_input.shape[:-1], cols + 8), dtype=dtype, device=device
        )
        x = storage[..., :cols]
        x.copy_(cpu_input)
        q, sf = fp8_quantize(x, sf_type=sf_type, output_dtype=output_dtype)
        expected_q, expected_sf = _reference_fp8_quantize(
            cpu_input, output_dtype, sf_type
        )
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8).cpu(), expected_q.view(torch.uint8), rtol=0, atol=0
        )

        q, sf = fp8_quantize(
            _tensor_view(x, "padded"), sf_type=sf_type, output_dtype=output_dtype
        )
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8).cpu(), expected_q.view(torch.uint8), rtol=0, atol=0
        )

    # Finite BF16 inputs can produce infinite fused activations. Conversion
    # must still saturate payloads to finite FP8 values in both output formats.
    gate = torch.full((2, 2064), 32.0, dtype=dtype, device=device)
    up = torch.full_like(gate, torch.finfo(dtype).max)
    up[1].neg_()
    x = torch.cat((gate, up), dim=-1)
    q, sf = silu_and_mul_quantize(x, sf_type=sf_type, output_dtype=output_dtype)
    expected_q, expected_sf = _reference_fp8_quantize(
        _reference_swiglu(x), output_dtype, sf_type
    )
    torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
    torch.testing.assert_close(
        q.view(torch.uint8).cpu(), expected_q.view(torch.uint8), rtol=0, atol=0
    )


def test_fp4_e8m0_scale_rounding_boundaries():
    device = _musa_device()
    # A maximum of six gives raw FP4 scale one. A finite global_scale then
    # exposes both FP32 neighbors of every E8M0 power through the public API.
    powers = torch.pow(2.0, torch.arange(-127, 128, dtype=torch.float64)).float()
    factors = torch.cat(
        (
            torch.nextafter(powers, torch.zeros_like(powers)),
            powers,
            torch.nextafter(powers, torch.full_like(powers, float("inf"))),
            torch.tensor([0.0, -0.0, 2.0**-149, -1.0, torch.finfo(torch.float32).max]),
        )
    )
    source = torch.tensor(
        [[6.0, -6.0, 3.0, -3.0, 0.0, -0.0, 1.0, -1.0] * 8], dtype=torch.bfloat16
    )
    x = source.to(device)
    for factor in factors.tolist():
        global_scale = torch.tensor([factor], dtype=torch.float32, device=device)
        q, sf = fp4_quantize(x, global_scale=global_scale, sf_type="e8m0")
        expected_q, expected_sf = _reference_fp4_quantize(source, "e8m0", factor)
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp4_e8m0_scale_extremes(dtype):
    device = _musa_device()
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values = values[torch.isfinite(values)]
    values = values.repeat((256 * 512 + values.numel() - 1) // values.numel())
    cpu_input = values[: 256 * 512].reshape(1, 256, 512)
    x = cpu_input.to(device)
    for factor in (1.0, 2.0**100, 2.0**-100, 0.0, -1.0):
        global_scale = torch.tensor([factor], dtype=torch.float32, device=device)
        q, sf = fp4_quantize(x, global_scale=global_scale, sf_type="e8m0")
        expected_q, expected_sf = _reference_fp4_quantize(
            cpu_input, "e8m0", global_scale=factor
        )
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_fp4_wide_rows_global_scale(dtype):
    device = _musa_device()
    rows, cols = 8192, 16384
    x = torch.randn((rows, 2 * cols + 8), dtype=dtype, device=device)[:, : 2 * cols]
    x[0] = 0
    selected = [0, rows // 2, rows - 1]
    values = _reference_swiglu(x[selected])
    for global_scale in (None, 1.5, 2.0**100, 2.0**-100):
        for inverse in (False, True):
            if global_scale is None and inverse:
                continue
            gs = (
                None
                if global_scale is None
                else torch.tensor([global_scale], dtype=torch.float32, device=device)
            )
            q, sf = silu_and_mul_quantize(
                x,
                output_dtype=torch.float4_e2m1fn_x2,
                sf_type="e8m0",
                global_scale=gs,
                is_global_scale_inversed=inverse,
            )
            expected_q, expected_sf = _reference_fp4_quantize(
                values, "e8m0", 1.0 if global_scale is None else global_scale, inverse
            )
            torch.testing.assert_close(q[selected].cpu(), expected_q, rtol=0, atol=0)
            torch.testing.assert_close(sf[selected].cpu(), expected_sf, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sf_type", ["e8m0", "e4m3"])
def test_fp4_finite_encodings(dtype, sf_type):
    device = _musa_device()
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values = values[torch.isfinite(values)]
    # Every finite magnitude becomes a group maximum. Mixed signs, zeros and
    # smaller values also exercise payload rounding with the encoded scale.
    cpu_input = values.repeat_interleave(32).reshape(1, -1, 512)
    cpu_input[..., 1::4].neg_()
    cpu_input[..., 2::4] *= 0.5
    cpu_input[..., 3::4] = 0
    for cols in (512, 528):
        source = torch.nn.functional.pad(cpu_input, (0, cols - 512))
        x = torch.empty((*source.shape[:-1], cols + 8), dtype=dtype, device=device)[
            ..., :cols
        ]
        x.copy_(source)
        q, sf = fp4_quantize(x, sf_type=sf_type)
        expected_q, expected_sf = _reference_fp4_quantize(source, sf_type)
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
        torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_silu_fp8_block128(dtype, output_dtype):
    device = _musa_device()
    torch.manual_seed(121)
    for shape, layout in (
        ((7, 2), "contiguous"),
        ((7, 30), "offset"),
        ((7, 128), "contiguous"),
        ((7, 144), "pitch"),
        ((2, 3, 258), "batch_transpose"),
        ((7, 528), "padded"),
        ((7, 4096), "contiguous"),
        ((2048, 4112), "rows"),
        ((8192, 144), "contiguous"),
    ):
        x = _tensor_view(
            torch.randn((*shape[:-1], 2 * shape[-1]), device=device, dtype=dtype),
            layout,
        )
        # Include zero/tiny groups and a tail whose maximum is at its last value.
        x[..., 0, :] = 0
        x[..., 1, :] = 1e-5
        x[..., 2, shape[-1] - 1] = 1
        x[..., 2, -1] = 16
        q, sf = silu_and_mul_quantize(x, output_dtype=output_dtype, sf_vec_size=128)
        assert q.shape == shape and q.dtype == output_dtype
        assert sf.shape == (*shape[:-1], (shape[-1] + 127) // 128)
        assert sf.dtype == torch.float32
        _assert_mn_major_scales(sf)
        expected_q, expected_sf = _reference_fp8_quantize(
            _reference_swiglu(x), output_dtype, "fp32", 128
        )
        torch.testing.assert_close(sf.cpu(), expected_sf, rtol=3e-6, atol=0)
        # FP32 SiLU/scale arithmetic can straddle an FP8 rounding midpoint.
        actual = q.view(torch.uint8).cpu().to(torch.int16)
        expected = expected_q.view(torch.uint8).to(torch.int16)
        assert (actual - expected).abs().max() <= 1
        torch.testing.assert_close(
            q[..., 0, :].float(), torch.zeros_like(q[..., 0, :].float())
        )
        torch.testing.assert_close(
            sf[..., 0, :],
            torch.full_like(sf[..., 0, :], 1e-4 / torch.finfo(output_dtype).max),
            rtol=1e-6,
            atol=0,
        )


@pytest.mark.parametrize("activation", ["clamped_swiglu", "situ"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "output_dtype", [torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_model_activation_quantization(activation, dtype, output_dtype):
    device = _musa_device()
    options = (
        {"swiglu_limit": 10.0}
        if activation == "clamped_swiglu"
        else {"activation": "situ"}
    )
    is_fp4 = output_dtype == torch.float4_e2m1fn_x2
    torch.manual_seed(122)
    recipes = [(32, "e8m0"), (32, "e4m3")]
    if not is_fp4:
        recipes.append((128, "fp32"))
    for rows, cols, layout in (
        (7, 30, "offset"),
        (7, 144, "pitch"),
        (2048, 2048, "contiguous"),
    ):
        x = _tensor_view(
            torch.randn((rows, cols * 2), dtype=dtype, device=device) * 16, layout
        )
        # Cover the asymmetric negative gate, both up clamp boundaries, tiny
        # values, and large finite values in every format's activation path.
        special = torch.tensor(
            [-100, -25, -10, -4, -1e-4, 0, 1e-4, 4, 10, 25, 100],
            device=device,
            dtype=dtype,
        )
        x[0, : special.numel()] = special
        x[0, cols : cols + special.numel()] = special.flip(0)
        x[1] = 0
        reference = _reference_swiglu(x, **options)
        if activation == "situ":
            reference, error = _reference_situ_with_error(*x.chunk(2, dim=-1))
        for size, sf_type in recipes:
            q, sf = silu_and_mul_quantize(
                x,
                output_dtype=output_dtype,
                sf_vec_size=size,
                sf_type=sf_type,
                **options,
            )
            _assert_mn_major_scales(sf)
            expected_q, expected_sf = (
                _reference_fp4_quantize(reference, sf_type)
                if is_fp4
                else _reference_fp8_quantize(reference, output_dtype, sf_type, size)
            )
            if activation == "situ":
                _assert_activation_quantization(
                    q, sf, reference, error, output_dtype, sf_type, size
                )
                # Zero-input behavior is an API invariant, independent of the
                # approximation tolerance used for nonzero activations.
                torch.testing.assert_close(
                    q[1].view(torch.uint8).cpu(),
                    expected_q[1].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(sf[1].cpu(), expected_sf[1], rtol=0, atol=0)
                continue
            if sf_type == "fp32":
                torch.testing.assert_close(sf.cpu(), expected_sf, rtol=3e-6, atol=0)
            else:
                # Activation approximation can move amax across an
                # encoded-scale midpoint. Bound that change in raw FP32 space.
                q_max = 6.0 if is_fp4 else torch.finfo(output_dtype).max
                raw = _group_values(reference, size).abs().amax(-1) * (1.0 / q_max)
                low, _ = _reference_scale(raw * (1.0 - 3e-6), sf_type, fp8=not is_fp4)
                high, _ = _reference_scale(raw * (1.0 + 3e-6), sf_type, fp8=not is_fp4)
                assert ((sf.cpu() >= low) & (sf.cpu() <= high)).all()
            # Check payload rounding against the actual encoded scale; a scale
            # crossing an E8M0 boundary doubles the scale and halves payloads.
            expected_q, _ = (
                _reference_fp4_quantize(reference, sf_type, encoded_scale=sf.cpu())
                if is_fp4
                else _reference_fp8_quantize(
                    reference, output_dtype, sf_type, size, encoded_scale=sf.cpu()
                )
            )
            if is_fp4:
                # Compare nibbles separately; packed-byte differences have no
                # meaningful distance when either neighbor crosses a midpoint.
                actual = q.cpu()
                for shift in (0, 4):
                    diff = ((actual >> shift) & 15).to(torch.int16) - (
                        (expected_q >> shift) & 15
                    ).to(torch.int16)
                    assert diff.abs().max() <= 1
            else:
                diff = q.view(torch.uint8).cpu().to(torch.int16) - expected_q.view(
                    torch.uint8
                ).to(torch.int16)
                assert diff.abs().max() <= 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("recipe", ["mxfp4", "mxfp8", "fp8_128"])
@pytest.mark.parametrize(
    "options",
    [
        {"swiglu_limit": 10.0},
        {"swiglu_limit": 3.0},
        {"activation": "situ"},
        {"activation": "situ", "situ_beta": 2.0, "situ_linear_beta": None},
        {"activation": "situ", "situ_beta": 25.0},
    ],
)
def test_model_activation_finite_encodings(dtype, recipe, options):
    device = _musa_device()
    # Exhaust every finite encoding of the public input dtype. Dense samples
    # and tiny signed values add coverage near activation/quantization boundaries.
    finite = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    finite = finite[torch.isfinite(finite)]
    tiny = torch.logspace(-40, -1, 512).to(dtype)
    gate = torch.cat((finite, torch.linspace(-100, 100, 100001).to(dtype), tiny, -tiny))
    gate = torch.nn.functional.pad(gate, (0, -gate.numel() % 128)).reshape(-1, 128)
    up = gate.flatten().roll(1031).reshape_as(gate).clamp(-100, 100)
    source = torch.cat((gate, up), dim=-1)
    output_dtype = torch.float4_e2m1fn_x2 if recipe == "mxfp4" else torch.float8_e4m3fn
    size = 128 if recipe == "fp8_128" else 32
    sf_type = "fp32" if size == 128 else "e8m0"
    if options.get("activation") == "situ":
        reference, error = _reference_situ_with_error(
            gate,
            up,
            options.get("situ_beta", 4.0),
            options.get("situ_linear_beta", 25.0),
        )
    else:
        limit = options["swiglu_limit"]
        reference = (
            torch.nn.functional.silu(gate.double().clamp(max=limit))
            * up.double().clamp(-limit, limit)
        ).float()
        error = reference.abs() * 5e-6 + 1e-35
    for layout in ("contiguous", "padded"):
        x = _tensor_view(source.to(device), layout)
        q, sf = silu_and_mul_quantize(
            x, output_dtype=output_dtype, sf_type=sf_type, sf_vec_size=size, **options
        )
        _assert_activation_quantization(
            q, sf, reference, error, output_dtype, sf_type, size
        )


@pytest.mark.parametrize("activation", ["clamped_swiglu", "situ"])
@pytest.mark.parametrize("recipe", ["mxfp4", "mxfp8", "fp8_128"])
@pytest.mark.parametrize("cols", [144, 2064, 8208])
def test_model_activation_batched_strides(activation, recipe, cols):
    device = _musa_device()
    options = {"activation": "situ"} if activation == "situ" else {"swiglu_limit": 10.0}
    options.update(
        output_dtype=(
            torch.float4_e2m1fn_x2 if recipe == "mxfp4" else torch.float8_e4m3fn
        ),
        sf_vec_size=128 if recipe == "fp8_128" else 32,
    )
    # K16 tail, padded rows, sliced batches and an aligned storage offset.
    # Wide rows cover two groups per thread followed by a partial K16 group.
    backing = torch.randn((6, 2062, 2 * cols + 32), device=device, dtype=torch.bfloat16)
    strided = backing[::2, ::2, 8 : 8 + 2 * cols]
    for x in (strided, strided[:1].expand(3, -1, -1)):
        q, sf = silu_and_mul_quantize(x, **options)
        if activation == "situ":
            selected = [0, x.shape[1] // 2, x.shape[1] - 1]
            reference, error = _reference_situ_with_error(
                *x[:, selected].chunk(2, dim=-1)
            )
            _assert_activation_quantization(
                q.view(torch.uint8)[:, selected].view(q.dtype),
                sf[:, selected],
                reference,
                error,
                options["output_dtype"],
                "fp32" if recipe == "fp8_128" else "e8m0",
                options["sf_vec_size"],
            )
        expected = [silu_and_mul_quantize(batch.contiguous(), **options) for batch in x]
        torch.testing.assert_close(
            q.view(torch.uint8).cpu(),
            torch.stack([pair[0].view(torch.uint8).cpu() for pair in expected]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sf.cpu(), torch.stack([pair[1].cpu() for pair in expected]), rtol=0, atol=0
        )


def test_situ_fp8_block128_wide_rows():
    device = _musa_device()
    rows, cols = 8192, 16384
    storage = torch.randn((rows, 2 * cols + 8), device=device, dtype=torch.bfloat16)
    storage[0] = 0
    x = storage[:, : 2 * cols].unsqueeze(0).expand(2, -1, -1)
    q, sf = silu_and_mul_quantize(
        x, activation="situ", output_dtype=torch.float8_e4m3fn, sf_vec_size=128
    )
    selected = [0, rows // 2, rows - 1]
    reference, error = _reference_situ_with_error(*x[:, selected].chunk(2, dim=-1))
    _assert_activation_quantization(
        q.view(torch.uint8)[:, selected].view(q.dtype),
        sf[:, selected],
        reference,
        error,
        torch.float8_e4m3fn,
        "fp32",
        128,
    )
    torch.testing.assert_close(
        q[0].view(torch.uint8), q[1].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(sf[0], sf[1], rtol=0, atol=0)
