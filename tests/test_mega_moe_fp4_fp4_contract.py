"""Independent numeric checks for E2M1 packing and the FP4 CPU golden path."""

import pytest
import torch

from mate.mega_moe_fp4_fp4 import (
    dequantize_fp4_reference,
    pack_fp4,
    prepare_fp4_operand,
    quantize_fp4_reference,
    unpack_fp4,
)


def test_all_nibble_pairs():
    codes = torch.cartesian_prod(torch.arange(16), torch.arange(16)).to(torch.uint8)
    packed = pack_fp4(codes)
    assert torch.equal(packed[:, 0], codes[:, 0] + 16 * codes[:, 1])
    assert torch.equal(unpack_fp4(packed), codes)


def test_known_encoding_and_signed_zero():
    x = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        * 2
    )
    packed, scales = quantize_fp4_reference(x)
    assert scales.tolist() == [127]
    assert unpack_fp4(packed).tolist() == list(range(16)) * 2
    decoded = dequantize_fp4_reference(packed, scales)
    assert torch.equal(decoded, x)
    assert torch.equal(torch.signbit(decoded), torch.signbit(x))


def test_nearest_even_midpoints():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4)
    p, s = quantize_fp4_reference(x)
    assert unpack_fp4(p).tolist() == [0, 2, 2, 4, 4, 6, 6, 7] * 4
    assert s.tolist() == [127]


def test_scale_boundary_and_zero_blocks():
    x = torch.zeros(3, 32)
    x[1, 0] = 6
    x[2, 0] = torch.nextafter(torch.tensor(6.0), torch.tensor(float("inf")))
    p, s = quantize_fp4_reference(x)
    assert s.flatten().tolist() == [127, 127, 128]
    assert torch.count_nonzero(p[0]) == 0


def test_ue8m0_extremes():
    codes = torch.zeros(2, 32, dtype=torch.uint8)
    codes[:, 0] = 2  # 1.0
    got = dequantize_fp4_reference(
        pack_fp4(codes), torch.tensor([[0], [254]], dtype=torch.uint8)
    )
    assert got[:, 0].double().tolist() == [2.0**-127, 2.0**127]


def test_native_scale_layout():
    p = torch.zeros(2, 3, 32, dtype=torch.uint8)
    s = torch.arange(12, dtype=torch.uint8).reshape(2, 3, 2)
    ap, sf = prepare_fp4_operand(p, s)
    assert ap.shape == (2, 3, 32) and sf.shape == (2, 2, 3)
    assert sf.is_contiguous() and torch.equal(sf[1, :, 2], s[1, 2])


def test_prepared_scales_reject_reserved_byte():
    with pytest.raises(ValueError, match="reserved"):
        prepare_fp4_operand(
            torch.zeros(1, 1, 16, dtype=torch.uint8),
            torch.full((1, 1, 1), 255, dtype=torch.uint8),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        quantize_fp4_reference(torch.full((32,), value))


def test_invalid_contracts():
    with pytest.raises(ValueError):
        pack_fp4(torch.tensor([16, 0], dtype=torch.uint8))
    with pytest.raises(ValueError):
        pack_fp4(torch.zeros(3, dtype=torch.uint8))
    with pytest.raises(TypeError):
        pack_fp4(torch.zeros(2))
    with pytest.raises(ValueError):
        quantize_fp4_reference(torch.ones(31))
    with pytest.raises(ValueError, match="reserved"):
        dequantize_fp4_reference(
            torch.zeros(16, dtype=torch.uint8), torch.tensor([255], dtype=torch.uint8)
        )


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_moe_weight_prepare_matches_online_swizzle(noncontiguous):
    from mate.mega_moe_fp4_fp4 import transform_weights_for_fp4_fp4_mega_moe

    g, n, k = 2, 64, 256
    packed = torch.arange(g * n * (k // 2)).to(torch.uint8).reshape(g, n, k // 2)
    scales = (
        (torch.arange(g * n * (k // 32)) % 255).to(torch.uint8).reshape(g, n, k // 32)
    )
    if noncontiguous:
        packed = packed.transpose(1, 2).contiguous().transpose(1, 2)
        scales = scales.transpose(1, 2).contiguous().transpose(1, 2)
    original = (packed.clone(), scales.clone())
    w1, w2 = transform_weights_for_fp4_fp4_mega_moe((packed, scales), (packed, scales))
    # Independent hardware-formula oracle, rather than the implementation's reshape.
    indices = []
    for i in range(n):
        inner = i % 32
        indices.append(i // 32 * 32 + inner * 8 // 32 + inner * 8 % 32)
    expected = scales.transpose(1, 2).index_select(-1, torch.tensor(indices))
    assert torch.equal(w2[0], packed)
    assert torch.equal(w2[1], expected)
    assert all(t.is_contiguous() for t in (*w1, *w2))
    assert torch.equal(w1[1], scales.transpose(1, 2))
    assert torch.equal(packed, original[0]) and torch.equal(scales, original[1])
    assert torch.equal(prepare_fp4_operand(packed, scales)[1], scales.transpose(1, 2))
    if not noncontiguous:
        assert w2[0].data_ptr() == packed.data_ptr()


def test_moe_weight_prepare_rejects_bad_scales_and_rows():
    from mate.mega_moe_fp4_fp4 import transform_weights_for_fp4_fp4_mega_moe

    w = (
        torch.zeros(1, 64, 128, dtype=torch.uint8),
        torch.zeros(1, 64, 8, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="aligned"):
        transform_weights_for_fp4_fp4_mega_moe(w, (w[0][:, :33], w[1][:, :33]))
    w[1][0, 0, 0] = 255
    with pytest.raises(ValueError, match="reserved"):
        transform_weights_for_fp4_fp4_mega_moe(w, w)


def test_grouped_gemm_rejects_moe_sfb_layout():
    from mate.mega_moe_fp4_fp4 import (
        _validate_fp4_grouped_gemm,
        transform_weights_for_fp4_fp4_mega_moe,
    )

    w = (
        torch.zeros(1, 64, 128, dtype=torch.uint8),
        torch.zeros(1, 64, 8, dtype=torch.uint8),
    )
    a, b = transform_weights_for_fp4_fp4_mega_moe(w, w)
    # The layout check must run before output/device validation or compilation.
    with pytest.raises(ValueError, match="ordinary group GEMM"):
        _validate_fp4_grouped_gemm(None, a, b)


def test_fused_stage1_scale_layout_and_cache_invalidation():
    from types import SimpleNamespace
    from mate._fp4_moe_buffer import FP4FP4SymmBuffer

    buffer = object.__new__(FP4FP4SymmBuffer)
    buffer.config = SimpleNamespace(experts=16, ranks=8, intermediate=64, hidden=64)
    packed = torch.arange(2 * 128 * 32).to(torch.uint8).reshape(2, 128, 32)
    scales = (torch.arange(2 * 2 * 128) % 251).to(torch.uint8).reshape(2, 2, 128)
    original = (packed.clone(), scales.clone())
    rows, scale_rows = [], []
    # Hardware oracle: gate32/up32 interleave, then the documented online
    # swizzle formula target=(inner*8)//32+(inner*8)%32 for SFB only.
    for group in range(2):
        for half in range(2):
            for inner in range(32):
                rows.append(half * 64 + group * 32 + inner)
                target = inner * 8 // 32 + inner * 8 % 32
                scale_rows.append(half * 64 + group * 32 + target)

    def expected():
        return (
            packed.index_select(1, torch.tensor(rows)),
            scales.index_select(2, torch.tensor(scale_rows)),
        )

    weights = (packed, scales)
    first = buffer.fused_stage1_weights(weights)
    assert all(torch.equal(a, b) for a, b in zip(first, expected()))
    assert all(torch.equal(a, b) for a, b in zip(weights, original))
    assert buffer.fused_stage1_weights(weights) is first
    packed[0, 0, 0].bitwise_xor_(1)
    scales[1, 0, 37].bitwise_xor_(1)
    refreshed = buffer.fused_stage1_weights(weights)
    assert refreshed is not first
    assert all(torch.equal(a, b) for a, b in zip(refreshed, expected()))
    assert all(t.is_contiguous() for t in refreshed)
