"""SiTU option validation, JIT isolation and independent activation oracles."""

import math
import re
from types import SimpleNamespace

import pytest
import torch

import mate.jit.mega_moe_fp4_fp4 as jit

from mate.mega_moe_fp4_fp4 import (
    _fp4_prefill_activation_options,
    fp4_fp4_mega_moe,
    fp4_fp4_stage1,
)
from mega_moe_fp4_fp4_reference import activation_reference


def test_situ_softcap_and_original_gate_sigmoid():
    first = torch.cat(
        (torch.full((1, 32), 4.0), torch.full((1, 32), 32.0)), dim=-1
    ).bfloat16()
    expected_gate = 4.0 * math.tanh(1.0) / (1.0 + math.exp(-4.0))
    for linear_beta in (None, 25.0):
        up = 32.0 if linear_beta is None else 25.0 * math.tanh(32.0 / 25.0)
        expected = torch.full((1, 32), expected_gate * up).bfloat16().float()
        got = activation_reference(
            first,
            None,
            activation="situ",
            activation_beta=4.0,
            activation_linear_beta=linear_beta,
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_situ_zero_and_bounded_output():
    gate = torch.tensor([0.0, 100.0, -100.0, 4.0]).expand(32, 4).T.bfloat16()
    up = torch.full_like(gate, 100.0)
    got = activation_reference(
        torch.cat((gate, up), -1),
        None,
        activation="situ",
        activation_beta=4,
        activation_linear_beta=25,
    )
    assert torch.count_nonzero(got[0]) == 0
    assert torch.isfinite(got).all() and got.abs().max() <= 100
    assert got[1, 0] == 100


def test_valid_activation_options():
    assert _fp4_prefill_activation_options("SWIGLU", 10.0, None, None) == (
        "swiglu",
        1.0,
        1.0,
    )
    assert _fp4_prefill_activation_options("SiTU", None, 4.0, None) == (
        "situ",
        4.0,
        1.0,
    )
    assert _fp4_prefill_activation_options("situ", None, 4.0, 25.0) == (
        "situ",
        4.0,
        25.0,
    )


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("linear", [False, True])
def test_nonfinite_or_nonpositive_situ_options_fail_before_launch(beta, linear):
    options = dict(
        activation="situ",
        activation_clamp=None,
        activation_beta=4.0,
        activation_linear_beta=25.0,
    )
    options["activation_linear_beta" if linear else "activation_beta"] = beta
    with pytest.raises(ValueError, match="finite and positive"):
        fp4_fp4_stage1(None, None, 1, **options)
    with pytest.raises(ValueError, match="finite and positive"):
        fp4_fp4_mega_moe(None, None, None, None, **options)


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(activation="situ", activation_clamp=None), "required"),
        (dict(activation="situ", activation_beta=4.0), "only valid for SwiGLU"),
        (dict(activation="swiglu", activation_beta=4.0), "only valid for SiTU"),
        (dict(activation="swiglu", activation_linear_beta=25.0), "only valid for SiTU"),
        (dict(activation="unknown"), "activation must be"),
    ],
)
def test_invalid_option_combinations(options, match):
    with pytest.raises(ValueError, match=match):
        fp4_fp4_stage1(None, None, 1, **options)


def test_jit_activations_have_distinct_names_and_preserve_swiglu_ffi(monkeypatch):
    captures = {}

    def fake_spec(name, sources, **kwargs):
        captures[name] = next(iter(kwargs["generated_sources"].values()))
        return SimpleNamespace(build_and_load=lambda: object())

    monkeypatch.setattr(jit, "gen_jit_spec", fake_spec)
    jit._build_fp4_prefill_module.cache_clear()
    try:
        config = jit.FP4PrefillConfig(8, 256, 256, 256, 32, 2)
        swiglu, _ = jit.get_fp4_prefill_stage_module(config, 1, num_mps=44)
        plain_situ, _ = jit.get_fp4_prefill_stage_module(
            config, 1, num_mps=44, activation="situ"
        )
        linear_situ, _ = jit.get_fp4_prefill_stage_module(
            config, 1, num_mps=44, activation="situ", use_situ_linear=True
        )
        assert len({swiglu, plain_situ, linear_situ}) == 3
        assert (
            "actsitu" not in swiglu
            and "actsitu_lin0" in plain_situ
            and "actsitu_lin1" in linear_situ
        )
        for name, situ in [(swiglu, False), (plain_situ, True), (linear_situ, True)]:
            source = captures[name]
            signature = source[source.index("void " + name) :].split("{", 1)[0]
            assert bool(re.search(r"float\s+beta", signature)) == situ
            assert ("fused_situ_stage1_kernel" in source) == situ
        with pytest.raises(ValueError, match="only valid for Stage1"):
            jit.get_fp4_prefill_stage_module(config, 2, num_mps=44, activation="situ")
        with pytest.raises(ValueError, match="requires SiTU"):
            jit.get_fp4_prefill_stage_module(
                config, 1, num_mps=44, use_situ_linear=True
            )
    finally:
        jit._build_fp4_prefill_module.cache_clear()
