"""Numerical coverage of the public FlashInfer quantization wrapper on MUSA."""

import pytest
import torch
import torch_musa  # noqa: F401

import flashinfer
from flashinfer import quantization
from flashinfer.gated_act_mxfp8 import silu_and_mul_mxfp8_quantize
from flashinfer.tllm_enums import SfLayout
from mate.testing import (
    dequantize_mxfp4,
    quantize_mxfp4,
    quantize_mxfp8,
    supported_musa_compute_capability,
)


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("api", ["fp4_quantize", "mxfp4_quantize", "mxfp8_quantize"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 96), (2, 3, 160)])
def test_mxfp_native_scales(api, dtype, shape):
    torch.manual_seed(57)
    # Pitched rows and sliced batches exercise input layouts without a copy.
    storage_shape = (*shape[:-1], shape[-1] + 32)
    if len(shape) == 3:
        storage_shape = (2 * shape[0], *storage_shape[1:])
    x = torch.randn(storage_shape, dtype=dtype, device="musa")[..., : shape[-1]]
    if len(shape) == 3:
        x = x[::2]
    x[..., 0, :] = 0
    x[..., -1, -1] = 32
    values = x.cpu().float()
    fp4 = api != "mxfp8_quantize"
    expected_q, expected_sf = (quantize_mxfp4 if fp4 else quantize_mxfp8)(values)

    for layout in SfLayout:
        if api == "fp4_quantize":
            options = dict(
                sf_vec_size=32,
                sf_use_ue8m0=True,
                is_sf_swizzled_layout=layout != SfLayout.layout_linear,
                is_sf_8x4_layout=layout == SfLayout.layout_8x4,
            )
        elif fp4:
            options = dict(sfLayout=layout)
        else:
            options = dict(sf_swizzle_layout=layout)
        q, sf = getattr(flashinfer, api)(x, **options)
        assert q.shape == expected_q.shape and q.is_contiguous()
        assert sf.shape == expected_sf.shape
        rows, groups = sf.shape[-2:]
        assert sf.stride() == ((rows * groups, 1, rows) if sf.ndim == 3 else (1, rows))
        torch.testing.assert_close(
            q.view(torch.uint8).cpu(), expected_q.view(torch.uint8)
        )
        torch.testing.assert_close(sf.cpu(), expected_sf)
        if fp4:
            restored = quantization.mxfp4_dequantize(q, sf, sfLayout=layout)
            expected = dequantize_mxfp4(expected_q, expected_sf)
            assert restored.device.type == "cpu"
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("shape", [(128, 128), (256, 2048)])
@pytest.mark.parametrize("pitched", [False, True])
def test_silu_and_mul_mxfp8_rowwise(shape, pitched):
    torch.manual_seed(57)
    rows, cols = shape
    storage = torch.randn(
        (rows, 2 * cols + (32 if pitched else 0)),
        dtype=torch.bfloat16,
        device="musa",
    )
    x = storage[:, : 2 * cols]
    x[0, :cols] = 0
    x[-1, cols - 1] = 16
    x[-1, -1] = -32
    gate, up = x.cpu().float().chunk(2, dim=-1)
    expected_q, expected_sf = quantize_mxfp8(torch.nn.functional.silu(gate) * up)

    q, col_q, sf, col_sf = silu_and_mul_mxfp8_quantize(x)
    assert q.shape == shape and q.dtype == torch.float8_e4m3fn
    assert sf.shape == (rows, cols // 32) and sf.dtype == torch.float8_e8m0fnu
    assert col_q.shape == (0,) and col_q.dtype == q.dtype
    assert col_sf.shape == (0,) and col_sf.dtype == sf.dtype
    assert all(t.device == x.device for t in (q, col_q, sf, col_sf))
    assert all(t.is_contiguous() for t in (q, col_q, col_sf))
    assert sf.stride() == (1, rows)
    torch.testing.assert_close(sf.view(torch.uint8).cpu(), expected_sf, rtol=0, atol=0)
    torch.testing.assert_close(
        q.view(torch.uint8).cpu(), expected_q.view(torch.uint8), rtol=0, atol=0
    )
