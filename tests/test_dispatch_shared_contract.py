from types import SimpleNamespace
import pytest
import torch
import mate
from mate.arch import MP31_ARCH, MP32_ARCH
from mate.ops import Op
from mate.ops.moe import Fp4DispatchBuffer, Fp4DispatchSharedGemm
from mate.ops.moe.dispatch_shared._input import _DispatchSharedInput
from mate.ops.moe.dispatch_shared.impl import DispatchSharedImpl
from mate.jit.dispatch_shared import gen_dispatch_shared_spec
from mate.jit.mega_moe_fp4_fp4 import FP4PrefillConfig


def inputs():
    b = object.__new__(Fp4DispatchBuffer)
    b.closed = False
    b.max_tokens = 128
    b._sym = SimpleNamespace(
        config=SimpleNamespace(hidden=256), common_mps=36, buffer=torch.empty(0)
    )
    x = torch.zeros((24, 256), dtype=torch.float8_e4m3fn)
    w = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)
    xs = torch.full((8, 24), 127, dtype=torch.uint8).t()
    ws = torch.full((8, 128), 127, dtype=torch.uint8).t()
    return _DispatchSharedInput(
        b, x, xs, w, ws, torch.empty(24, 128, dtype=torch.bfloat16)
    )


def test_export_and_selection():
    assert mate.ops.moe.Fp4DispatchSharedGemm is Fp4DispatchSharedGemm
    assert issubclass(Fp4DispatchSharedGemm, Op)
    assert Fp4DispatchSharedGemm.resolve_backend(MP32_ARCH).fused
    assert not Fp4DispatchSharedGemm.resolve_backend(
        MP32_ARCH, override_backend="mp32_serial"
    ).fused
    with pytest.raises(ValueError, match="MP32"):
        Fp4DispatchSharedGemm.resolve_backend(MP31_ARCH)


def test_valid_inputs_retain_the_same_buffers():
    v = inputs()
    original = v.output
    DispatchSharedImpl().sanitize(v)
    assert v.output is original


@pytest.mark.parametrize("ctas", [0, 37, True, 1.5])
def test_invalid_ctas_rejected_before_build(ctas):
    v = inputs()
    v.num_ctas = ctas
    with pytest.raises(ValueError, match="num_ctas"):
        DispatchSharedImpl().sanitize(v)


def test_scale_layout_is_explicit():
    v = inputs()
    v.x_scale = v.x_scale.contiguous()
    with pytest.raises(ValueError, match="MN-major"):
        DispatchSharedImpl().sanitize(v)


def test_shared_precision_and_output_shape():
    v = inputs()
    v.x = v.x.bfloat16()
    with pytest.raises(ValueError, match="E4M3"):
        DispatchSharedImpl().sanitize(v)
    v = inputs()
    v.output = v.output.float()
    with pytest.raises(ValueError, match="BF16"):
        DispatchSharedImpl().sanitize(v)
    v = inputs()
    v.feature_tile = 0
    with pytest.raises(ValueError, match="tile"):
        DispatchSharedImpl().sanitize(v)


@pytest.mark.parametrize("fused", [False, True])
def test_native_specialization_has_one_launch(fused):
    c = FP4PrefillConfig(4, 256, 5120, 256, 384, 6)
    name, spec = gen_dispatch_shared_spec(c, fused=fused)
    source = next(iter(spec.generated_sources.values()))
    assert source.count("<<<") == 1
    assert "dispatch_shared_kernel" in source
    assert "std::map" in source
    assert "shared_output" in source
    assert "4608,24,true," + str(fused).lower() in source.replace(" ", "")


def test_jit_rejects_invalid_configs_before_compiling():
    c = FP4PrefillConfig(4, 256, 5120, 256, 384, 6)
    for options in (
        {"num_ctas": 0},
        {"feature_tile": 0},
        {"shared_features": 15},
        {"stages": 1},
    ):
        with pytest.raises(ValueError):
            gen_dispatch_shared_spec(c, **options)
    with pytest.raises(TypeError):
        gen_dispatch_shared_spec(c, fused=1)
