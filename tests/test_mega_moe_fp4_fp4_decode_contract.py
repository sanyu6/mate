"""CPU-only contracts for the experimental unified M32 decode entry."""

from types import SimpleNamespace

import pytest
import torch

from mate._fp4_moe_buffer import FP4FP4SymmBuffer, FP4SymmetricLayout
from mate.jit.mega_moe_fp4_fp4 import FP4PrefillConfig
from mate.jit.mega_moe_fp4_fp4_decode import gen_fp4_decode_spec
from mate.mega_moe_fp4_fp4 import (
    _decode_settings,
    _decode_weight,
    fp4_fp4_decode_stage1,
    fp4_fp4_mega_moe_decode,
    transform_weights_for_fp4_fp4_mega_moe,
)


def buffer_stub():
    s = object.__new__(FP4FP4SymmBuffer)
    s.group = SimpleNamespace()
    s.config = FP4PrefillConfig(8, 256, 256, 256, 32, 2)
    s.common_mps = 44
    s.decode_enabled = True
    s.buffer = torch.empty(0, dtype=torch.uint8)
    return s


@pytest.mark.parametrize("tokens", [1, 31, 32, 33, 127, 128])
def test_active_tokens_do_not_require_m32_alignment(tokens):
    assert _decode_settings(buffer_stub(), tokens, None, None) == (44, float("inf"))


@pytest.mark.parametrize("tokens", [0, -1, 129, True, 32.0])
def test_decode_token_limit(tokens):
    with pytest.raises(ValueError, match="1..128"):
        _decode_settings(buffer_stub(), tokens, 44, 10.0)


@pytest.mark.parametrize("mps", [0, 1, 3, 45, 46, True, 44.0])
def test_decode_requires_legal_resident_grid(mps):
    with pytest.raises(ValueError, match="even resident"):
        _decode_settings(buffer_stub(), 1, mps, 10.0)


def test_unified_buffer_requires_explicit_decode_reservation():
    s = buffer_stub()
    s.decode_enabled = False
    with pytest.raises(ValueError, match="enable_decode"):
        _decode_settings(s, 1, 44, 10.0)
    s.group = None
    with pytest.raises(ValueError, match="live"):
        _decode_settings(s, 1, 44, 10.0)


@pytest.mark.parametrize("clamp", [-1, float("nan")])
def test_bad_activation_clamp(clamp):
    with pytest.raises(ValueError, match="activation_clamp"):
        _decode_settings(buffer_stub(), 1, 44, clamp)


def test_decode_accepts_the_existing_prepared_weights():
    s = buffer_stub()
    c = s.config
    groups = c.experts // c.ranks
    w1 = (
        torch.zeros(groups, 2 * c.intermediate, c.hidden // 2, dtype=torch.uint8),
        torch.full(
            (groups, 2 * c.intermediate, c.hidden // 32), 127, dtype=torch.uint8
        ),
    )
    w2 = (
        torch.zeros(groups, c.hidden, c.intermediate // 2, dtype=torch.uint8),
        torch.full((groups, c.hidden, c.intermediate // 32), 127, dtype=torch.uint8),
    )
    first, second = transform_weights_for_fp4_fp4_mega_moe(w1, w2)
    _decode_weight(s, first, 1)
    _decode_weight(s, second, 2)
    with pytest.raises(ValueError, match="transform_weights"):
        _decode_weight(s, tuple(second), 2)
    # Input prefix still describes padded prefill capacity, not active decode M.
    prefix = FP4SymmetricLayout(c.tokens, c.hidden, c.topk)
    assert prefix.offsets["x_sf"] >= prefix.offsets["x"] + c.tokens * c.hidden // 2


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_decode_spec_generation_without_build_or_launch(stage):
    config = buffer_stub().config
    name, spec = gen_fp4_decode_spec(config, stage, num_mps=44 if stage else 0)
    assert "decode_" in name and "_m32_" in name
    source = next(iter(spec.generated_sources.values()))
    assert "DecodeLayout<8, 256, 256, 256, 32, 2>" in source
    if stage:
        assert f"mp32_decode_stage{stage}.hpp" in source
        assert source.count("<<<") == 1
        assert "prefill_stage" not in source
    else:
        assert "<<<" not in source


@pytest.mark.parametrize("stage,mps", [(3, 44), (1, 1), (2, 45), (0, 44)])
def test_invalid_specs_rejected_before_build(stage, mps):
    with pytest.raises(ValueError):
        gen_fp4_decode_spec(buffer_stub().config, stage, num_mps=mps)


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_decode_situ_jit_isolated_and_default_abi_preserved(linear_beta):
    c = buffer_stub().config
    default_name, default = gen_fp4_decode_spec(c, 1, num_mps=44)
    name, situ = gen_fp4_decode_spec(
        c, 1, num_mps=44, activation="situ", use_situ_linear=linear_beta is not None
    )
    assert name != default_name and "actsitu" not in default_name
    assert f"actsitu_lin{int(linear_beta is not None)}" in name
    for spec, is_situ in [(default, False), (situ, True)]:
        source = next(iter(spec.generated_sources.values()))
        signature = source[source.index("void ") :].split("{", 1)[0]
        assert ("float beta" in signature) == is_situ
        assert ("situ_stage1_kernel" in source) == is_situ
        # Both occupancy-selected versions must use the same activation.
        assert source.count(
            "true, true>" if linear_beta is not None else "true, false>"
        ) == (2 if is_situ else 0)


@pytest.mark.parametrize(
    "stage,activation,linear",
    [(0, "situ", False), (2, "situ", False), (1, "swiglu", True)],
)
def test_invalid_decode_activation_specialization(stage, activation, linear):
    with pytest.raises(ValueError):
        gen_fp4_decode_spec(
            buffer_stub().config,
            stage,
            num_mps=44 if stage else 0,
            activation=activation,
            use_situ_linear=linear,
        )


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("linear", [False, True])
def test_decode_bad_situ_options_fail_before_ipc(beta, linear):
    options = dict(
        activation="situ",
        activation_clamp=None,
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )
    options["activation_linear_beta" if linear else "activation_beta"] = beta
    with pytest.raises(ValueError, match="finite and positive"):
        fp4_fp4_decode_stage1(None, None, 1, **options)
    with pytest.raises(ValueError, match="finite and positive"):
        fp4_fp4_mega_moe_decode(None, None, None, None, **options)


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(activation="situ", activation_clamp=None), "required"),
        (dict(activation="situ", activation_beta=4.0), "only valid for SwiGLU"),
        (dict(activation="swiglu", activation_beta=4.0), "only valid for SiTU"),
        (dict(activation="unknown"), "activation must be"),
    ],
)
def test_decode_invalid_activation_combinations(options, match):
    with pytest.raises(ValueError, match=match):
        fp4_fp4_decode_stage1(None, None, 1, **options)
    with pytest.raises(ValueError, match=match):
        fp4_fp4_mega_moe_decode(None, None, None, None, **options)


@pytest.mark.parametrize("feature_tile", [64, 128, 256])
def test_swap_decode_weight_layout_and_mutation_cache(feature_tile):
    """Both native operands have independent row/scale axes after swapping."""
    s = buffer_stub()
    g, h, i = 4, 512, 256
    generator = torch.Generator().manual_seed(918)
    logical1 = (
        torch.randint(256, (g, 2 * i, h // 2), dtype=torch.uint8, generator=generator),
        torch.randint(255, (g, 2 * i, h // 32), dtype=torch.uint8, generator=generator),
    )
    logical2 = (
        torch.randint(256, (g, h, i // 2), dtype=torch.uint8, generator=generator),
        torch.randint(255, (g, h, i // 32), dtype=torch.uint8, generator=generator),
    )
    first, second = transform_weights_for_fp4_fp4_mega_moe(logical1, logical2)
    for stage, prepared, logical in [(1, first, logical1), (2, second, logical2)]:
        packed, scales = s.swap_decode_weights(prepared, stage, feature_tile)
        assert s.swap_decode_weights(prepared, stage, feature_tile)[0] is packed
        n, k = packed.shape[1], packed.shape[2] * 2
        restored = (
            packed.view(g, n // feature_tile, k // 256, feature_tile, 128)
            .permute(0, 1, 3, 2, 4)
            .reshape(g, n, k // 2)
        )
        restored_scales = scales.transpose(1, 2).contiguous()
        if stage == 1:
            restored = (
                restored.view(g, i // 32, 2, 32, h // 2)
                .permute(0, 2, 1, 3, 4)
                .reshape_as(logical[0])
            )
            restored_scales = (
                restored_scales.view(g, i // 32, 2, 32, h // 32)
                .permute(0, 2, 1, 3, 4)
                .reshape_as(logical[1])
            )
        if stage == 2:
            restored = (
                restored.view(g, n // 128, 4, 32, k // 2)
                .permute(0, 1, 3, 2, 4)
                .reshape_as(logical[0])
            )
            restored_scales = (
                restored_scales.view(g, n // 128, 4, 32, k // 32)
                .permute(0, 1, 3, 2, 4)
                .reshape_as(logical[1])
            )
        torch.testing.assert_close(restored, logical[0], rtol=0, atol=0)
        torch.testing.assert_close(restored_scales, logical[1], rtol=0, atol=0)
        before = packed.clone()
        prepared[0].bitwise_xor_(1)
        updated = s.swap_decode_weights(prepared, stage, feature_tile)[0]
        assert updated is not packed
        torch.testing.assert_close(updated, before ^ 1, rtol=0, atol=0)


def test_swap_decode_jit_isolated_from_default_and_situ():
    c = buffer_stub().config
    for stage in (1, 2):
        name, spec = gen_fp4_decode_spec(c, stage, num_mps=44, swap_ab=True)
        default, _ = gen_fp4_decode_spec(c, stage, num_mps=44)
        assert name != default and "swapab_f256_p3" in name
        assert ", true, 256>" in next(iter(spec.generated_sources.values()))
    with pytest.raises(ValueError, match="SwiGLU"):
        gen_fp4_decode_spec(c, 1, num_mps=44, activation="situ", swap_ab=True)


def test_kpacked_decode_preserves_online_weight_and_scale_contract():
    s = buffer_stub()
    s.config = FP4PrefillConfig(8, 256, 512, 512, 32, 2)
    rng = torch.Generator().manual_seed(937)
    operands = []
    for n, k in [(1024, 512), (512, 512)]:
        operands.append(
            (
                torch.randint(256, (4, n, k // 2), dtype=torch.uint8, generator=rng),
                torch.randint(255, (4, n, k // 32), dtype=torch.uint8, generator=rng),
            )
        )
    first, second = transform_weights_for_fp4_fp4_mega_moe(*operands)
    for stage, weights in [(1, first), (2, second)]:
        expected = s.fused_stage1_weights(weights) if stage == 1 else weights
        packed, scales = s.packed_decode_weights(weights, stage)
        g, n, half_k = packed.shape
        restored = (
            packed.view(g, n // 256, half_k // 128, 256, 128)
            .permute(0, 1, 3, 2, 4)
            .reshape_as(packed)
        )
        torch.testing.assert_close(restored, expected[0], rtol=0, atol=0)
        assert scales is expected[1]
        assert s.packed_decode_weights(weights, stage)[0] is packed
        weights[0].bitwise_xor_(1)
        changed = s.packed_decode_weights(weights, stage)[0]
        torch.testing.assert_close(changed, packed ^ 1, rtol=0, atol=0)


def test_kpacked_decode_jit_is_exclusive_and_isolated():
    c = buffer_stub().config
    for stage in (1, 2):
        name, spec = gen_fp4_decode_spec(c, stage, num_mps=44, pack_b=True)
        assert "packb_f256_p3" in name
        assert ", false, 256, true>" in next(iter(spec.generated_sources.values()))
        with pytest.raises(ValueError, match="exclusive"):
            gen_fp4_decode_spec(c, stage, num_mps=44, pack_b=True, swap_ab=True)


@pytest.mark.parametrize("backend", ["swap_decode_weights", "packed_decode_weights"])
def test_decode_prepacking_cache_survives_alternating_layers(backend):
    s = buffer_stub()
    prepare = getattr(s, backend)
    pair = (
        torch.zeros(4, 512, 128, dtype=torch.uint8),
        torch.full((4, 8, 512), 127, dtype=torch.uint8),
    )
    other = tuple(t.clone() for t in pair)
    first = prepare(pair, 1)
    prepare(other, 1)
    assert prepare(pair, 1)[0] is first[0]
    pair[0].bitwise_xor_(1)
    assert prepare(pair, 1)[0] is not first[0]


@pytest.mark.parametrize("stage", [1, 2])
def test_w4a4_numerical_and_double_buffered_specs(stage):
    c = buffer_stub().config
    name, spec = gen_fp4_decode_spec(
        c,
        stage,
        num_mps=44,
        sglang_numerics=True,
        line_padded_scales=stage == 1,
        pack_b=True,
        double_buffered=True,
    )
    assert "_sgnum_" in name and "_p2c3_bound" in name
    source = next(iter(spec.generated_sources.values()))
    assert "using Stage" in source and "44, 3" in source
    assert "Tier2" in source
    if stage == 1:
        assert "Stage::kScalePoolTokens" in source


@pytest.mark.parametrize(
    "options",
    [
        dict(sglang_numerics=True, swap_ab=True),
        dict(line_padded_scales=True, swap_ab=True),
        dict(double_buffered=True, activation="situ"),
        dict(sglang_numerics=1),
    ],
)
def test_w4a4_invalid_specializations_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        gen_fp4_decode_spec(buffer_stub().config, 1, num_mps=44, **options)
