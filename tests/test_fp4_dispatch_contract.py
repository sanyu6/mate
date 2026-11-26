from pathlib import Path

import pytest
import torch
import mate
from mate.arch import MP31_ARCH, MP32_ARCH
from mate.ops import Impl, Op
from mate.ops.moe import Fp4Dispatch, Fp4DispatchBuffer, Fp4DispatchResult
from mate.ops.moe.fp4_dispatch._input import _Fp4DispatchInput
from mate.ops.moe.fp4_dispatch.impls.mp32 import Mp32DispatchImpl
from mate.jit.fp4_dispatch_only import gen_fp4_dispatch_spec
from mate.jit.mega_moe_fp4_fp4 import FP4PrefillConfig
from fp4_dispatch_reference import dispatch_reference


def test_export_and_selection():
    assert mate.ops.moe.Fp4Dispatch is Fp4Dispatch
    assert issubclass(Fp4Dispatch, Op)
    assert isinstance(Fp4Dispatch.resolve_backend(MP32_ARCH), Impl)
    with pytest.raises(ValueError, match="mp32"):
        Fp4Dispatch.resolve_backend(MP31_ARCH)
    with pytest.raises(ValueError, match="mp32"):
        Fp4Dispatch.resolve_backend(MP32_ARCH, override_backend="unknown")


@pytest.mark.parametrize("tokens", [0, 129, -1, True, 1.5])
def test_active_tokens_rejected_without_device_work(tokens):
    b = object.__new__(Fp4DispatchBuffer)
    b.closed = False
    b.max_tokens = 128
    with pytest.raises(ValueError, match="tokens"):
        Mp32DispatchImpl().sanitize(_Fp4DispatchInput(b, tokens))


def test_closed_buffer_and_missing_build_inputs():
    b = object.__new__(Fp4DispatchBuffer)
    b.closed = True
    with pytest.raises(ValueError, match="live"):
        Fp4Dispatch(b, 24)
    with pytest.raises(TypeError, match="_Fp4DispatchInput"):
        Fp4Dispatch.build()


@pytest.mark.parametrize("padded", [False, True])
def test_ffi_contract_has_one_dispatch_launch_and_no_gemm(padded):
    c = FP4PrefillConfig(4, 256, 5120, 256, 384, 6)
    name, spec = gen_fp4_dispatch_spec(c, 36, padded)
    source = next(iter(spec.generated_sources.values()))
    assert source.count("<<<") == 1
    assert "dispatch_only_kernel" in source
    assert "musaOccupancyMaxActiveBlocksPerMultiprocessor" in source
    native = (
        Path(__file__).resolve().parents[1]
        / "include/mate/mega_moe/decode/fp4_fp4/dispatch_only.hpp"
    ).read_text()
    assert "mainloop.mma" not in native and "Pool::load" not in native
    assert "get_l1_arrival_count_ptr" not in native
    assert native.count("mtlink_barrier<") == 2
    assert (
        "get_expert_send_count_ptr(e), 0ull" in native
        or "get_expert_send_count_ptr(e),0ull" in native
    )
    assert (
        "get_expert_recv_count_sum_ptr(e), 0ull" in native
        or "get_expert_recv_count_sum_ptr(e),0ull" in native
    )
    assert "_sf" + str(int(padded)) + "_" in name


def test_output_snapshot_does_not_alias_reusable_views():
    tensors = [torch.ones(1) for _ in range(6)]
    r = Fp4DispatchResult(*tensors, True)
    snap = r.clone()
    for t in r.tensors:
        t.zero_()
    assert all(t.item() == 1 for t in snap.tensors)
    assert snap.line_padded_scales


def test_reference_duplicated_payload_routes_remain_separate():
    peer = dict(
        payload=[[17, 18]], scales=[[127]], ids=[[0, 1]], weights=[[0.25, 0.75]]
    )
    rows, offsets, _ = dispatch_reference([peer], 0, 2)
    assert [len(v) for v in rows] == [1, 1]
    assert offsets == [0, 32, 64]
    assert rows[0][0][1] == rows[1][0][1]
    assert rows[0][0][0][2] != rows[1][0][0][2]


def test_reference_skew_disabled_and_invalid():
    peer = dict(
        payload=[[1], [2]],
        scales=[[127], [128]],
        ids=[[0, -1], [0, 99]],
        weights=[[1.0, 0.0], [0.5, 0.5]],
    )
    rows, offsets, invalid = dispatch_reference([peer, peer], 0, 4)
    assert [len(v) for v in rows] == [4, 0]
    assert offsets == [0, 32, 32] and invalid


def test_family_import_has_no_device_query():
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("MATE_MUSA_ARCH_LIST", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from types import SimpleNamespace
def fail(*args,**kwargs):
    raise AssertionError('device query during import')
torch.musa=SimpleNamespace(is_available=fail,get_device_properties=fail,device_count=fail)
import mate
assert mate.ops.moe.Fp4Dispatch.__name__=='Fp4Dispatch'
""",
        ],
        env=env,
        check=True,
        timeout=30,
    )


def test_prequantized_input_preserves_prefix_and_checks_layout():
    from types import SimpleNamespace

    b = object.__new__(Fp4DispatchBuffer)
    b.closed = False
    b.max_tokens = 4
    b._sym = SimpleNamespace(
        config=SimpleNamespace(hidden=256, topk=2),
        buffer=torch.empty(0),
        x=torch.full((4, 128), 238, dtype=torch.uint8),
        x_sf=torch.zeros(4, 8, dtype=torch.uint8),
        topk_idx=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 2),
    )
    x = torch.arange(256).to(torch.uint8).reshape(2, 128)
    sf = torch.arange(16).to(torch.uint8).reshape(2, 8)
    ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, -0.5], [0.75, 1.0]])
    b.prepare_fp4(x, sf, ids, weights)
    assert torch.equal(b._sym.x[:2], x)
    assert torch.all(b._sym.x[2:] == 238)
    assert torch.equal(b._sym.x_sf[:2], sf)
    assert torch.equal(b._sym.topk_idx[:2], ids.long())
    assert torch.equal(b._sym.topk_weights[:2], weights)
    strided = torch.empty(8, 2, dtype=torch.uint8).t()
    with pytest.raises(ValueError, match="stride"):
        b.prepare_fp4(x, strided, ids, weights)
    with pytest.raises(ValueError, match="dtype"):
        b.prepare_fp4(x, sf, ids, weights.double())


def test_pull_is_default_and_legacy_is_explicit():
    assert Fp4Dispatch.resolve_backend(MP32_ARCH).pull
    assert Fp4Dispatch.resolve_backend(MP32_ARCH, override_backend="mp32_pull").pull
    assert not Fp4Dispatch.resolve_backend(MP32_ARCH, override_backend="mp32").pull


@pytest.mark.parametrize("ctas", [0, 37, True, 1.5])
def test_pull_grid_validation_before_build(ctas):
    c = FP4PrefillConfig(4, 256, 5120, 256, 384, 6)
    with pytest.raises(ValueError, match="num_ctas"):
        gen_fp4_dispatch_spec(c, 36, True, pull=True, num_ctas=ctas)


def test_pull_spec_no_compute_launch_and_default_grid():
    c = FP4PrefillConfig(4, 256, 5120, 256, 384, 6)
    name, spec = gen_fp4_dispatch_spec(c, 36, True, pull=True, num_ctas=24)
    source = next(iter(spec.generated_sources.values()))
    assert source.count("<<<") == 1 and "dispatch_pull_kernel" in source
    assert "PullDispatchOnly<L,24,true,32>" in source
    assert "static thread_local int checked_device" in source
    assert "_pull_final_c24_flags32" in name
