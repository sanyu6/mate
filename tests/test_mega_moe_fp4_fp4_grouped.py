"""Native FP4 SS host/JIT/device integration, including independent references."""

import pytest
import torch

pytest.importorskip("torch_musa")

from mate.mega_moe_fp4_fp4 import (
    dequantize_fp4_reference,
    fp4_fp4_grouped_gemm,
    pack_fp4,
    prepare_fp4_operand,
)

pytestmark = pytest.mark.skipif(not torch.musa.is_available(), reason="requires MUSA")


def _operand(groups, rows, k, seed, scale_low):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (groups, rows, k), generator=gen, dtype=torch.uint8)
    scales = torch.randint(
        scale_low,
        scale_low + 3,
        (groups, rows, k // 32),
        generator=gen,
        dtype=torch.uint8,
    )
    return pack_fp4(codes), scales


def _device_operand(packed, scales):
    return tuple(t.to("musa") for t in prepare_fp4_operand(packed, scales))


@pytest.fixture(autouse=True)
def _mp32():
    torch.musa.set_device(0)
    props = torch.musa.get_device_properties(0)
    if (props.major, props.minor) != (3, 2):
        pytest.skip("requires MP32")


@pytest.mark.parametrize(
    "shape", [(1, 128, 256, 256), (3, 129, 264, 512), (2, 256, 512, 768)]
)
def test_grouped_full_reference(shape):
    groups, m, n, k = shape
    a, sa = _operand(groups, m, k, 81, 125)
    b, sb = _operand(groups, n, k, 91, 120)
    ref = torch.bmm(
        dequantize_fp4_reference(a, sa), dequantize_fp4_reference(b, sb).transpose(1, 2)
    ).bfloat16()
    out = torch.empty((groups, m, n), device="musa", dtype=torch.bfloat16)
    result = fp4_fp4_grouped_gemm(
        out, _device_operand(a, sa), _device_operand(b, sb), num_mps=1
    )
    assert result is out
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


def test_bf16_output_exceeds_fp16_range():
    # A real BF16 epilogue must not round through FP16 (which would overflow).
    a = torch.full((1, 64, 128), 0x77, dtype=torch.uint8, device="musa")
    b = torch.full((1, 256, 128), 0x77, dtype=torch.uint8, device="musa")
    sa = torch.full((1, 8, 64), 131, dtype=torch.uint8, device="musa")
    sb = torch.full((1, 8, 256), 131, dtype=torch.uint8, device="musa")
    out = torch.empty((1, 64, 256), device="musa", dtype=torch.bfloat16)
    fp4_fp4_grouped_gemm(out, (a, sa), (b, sb), num_mps=1)
    assert bool((out.cpu() == 2359296).all())


def test_nondefault_stream_and_replay():
    a, sa = _operand(2, 128, 256, 82, 125)
    b, sb = _operand(2, 256, 256, 92, 120)
    ref = torch.bmm(
        dequantize_fp4_reference(a, sa), dequantize_fp4_reference(b, sb).transpose(1, 2)
    ).bfloat16()
    stream = torch.musa.Stream()
    with torch.musa.stream(stream):
        ad, bd = _device_operand(a, sa), _device_operand(b, sb)
        out = torch.empty((2, 128, 256), dtype=torch.bfloat16, device="musa")
        for _ in range(3):
            fp4_fp4_grouped_gemm(out, ad, bd, num_mps=1)
    stream.synchronize()
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


def test_host_and_python_validation():
    a, sa = _device_operand(*_operand(1, 128, 256, 83, 125))
    b, sb = _device_operand(*_operand(1, 256, 256, 93, 120))
    out = torch.empty((1, 128, 256), device="musa", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="num_mps"):
        fp4_fp4_grouped_gemm(out, (a, sa), (b, sb), num_mps=0)
    with pytest.raises(ValueError, match="scales"):
        fp4_fp4_grouped_gemm(out, (a, sa[:, :-1].contiguous()), (b, sb))
    from mate.jit.mega_moe_fp4_fp4 import get_fp4_grouped_module

    name, module = get_fp4_grouped_module()
    with pytest.raises(Exception, match="unexpected shape"):
        module.get_function(name)(out, a, sa[:, :-1].contiguous(), b, sb, 1)


@pytest.mark.parametrize(
    "tokens,k", [(4096, 4096), (8192, 4096), (4096, 2048), (8192, 2048)]
)
def test_flash_32_groups_sampled_reference(tokens, k):
    groups, m, n = 32, tokens * 6 // 32, 4096
    a, sa = _operand(groups, m, k, 84, 125)
    b, sb = _operand(groups, n, k, 94, 120)
    out = torch.empty((groups, m, n), device="musa", dtype=torch.bfloat16)
    mps = min(44, torch.musa.get_device_properties(0).multi_processor_count)
    fp4_fp4_grouped_gemm(
        out, _device_operand(a, sa), _device_operand(b, sb), num_mps=mps
    )
    assert bool(torch.isfinite(out).all())
    cols = torch.tensor([0, 31, n // 2, n - 1])
    rows = torch.tensor([0, m // 2, m - 1])
    for expert in (0, groups // 2, groups - 1):
        ref_a = dequantize_fp4_reference(a[expert, rows], sa[expert, rows])
        ref_b = dequantize_fp4_reference(b[expert, cols], sb[expert, cols])
        ref = (ref_a @ ref_b.T).bfloat16()
        got = out[expert, rows.to("musa")[:, None], cols.to("musa")[None, :]].cpu()
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
