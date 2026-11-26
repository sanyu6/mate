"""Exercise the public-baseline to MP32 wrapper ABI without launching a GPU."""

import inspect

import pytest
import torch

from mate.arch import MP32_ARCH
from mate.sparse_mla.tilelang import sparse_mla_decode as dispatch
from mate.sparse_mla.tilelang.mp32 import (
    sparse_mla_v4_mp32_decode_fwd_scheduled as mp32,
)


@pytest.mark.parametrize("cache_bytes", [584, 528])
@pytest.mark.parametrize("fused", [False, True])
def test_mp32_dispatch_preserves_each_wrapper_signature(
    monkeypatch, cache_bytes, fused
):
    monkeypatch.setattr(dispatch, "get_musa_arch", lambda _: MP32_ARCH)
    selected = []

    def stub(name, function):
        signature = inspect.signature(function)

        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            selected.append((name, bound.arguments))
            return "out", "lse"

        return call

    monkeypatch.setattr(
        dispatch, "_decode_model1_mp32", stub("mp32", dispatch._decode_model1_mp32)
    )
    monkeypatch.setattr(
        dispatch, "_decode_model1", stub("public", dispatch._decode_model1)
    )
    result = dispatch.sparse_mla_decode_fwd(
        q=torch.zeros((1, 1, 8, 512), dtype=torch.bfloat16),
        k_cache=torch.zeros((1, 64, 1, cache_bytes), dtype=torch.uint8),
        indices=torch.zeros((1, 1, 64), dtype=torch.int32),
        head_dim_v=512,
        softmax_scale=512**-0.5,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        topk_length=None,
        extra_topk_length=None,
        tile_scheduler_metadata=torch.zeros((8, 8), dtype=torch.int32),
        num_splits=torch.zeros((2,), dtype=torch.int32),
        metadata_getter=lambda **kwargs: pytest.fail("metadata already provided"),
        _fused_output=object() if fused else None,
    )
    assert result == ("out", "lse")
    assert selected[0][0] == ("public" if fused else "mp32")


def test_mp32_runtime_allocation_matches_public_split_api(monkeypatch):
    monkeypatch.setattr(
        mp32,
        "jit_for_tensor_addressing",
        lambda *args: lambda *args, **kwargs: lambda *args: None,
    )
    # One page, eight heads, eight scheduler parts: every allocation stays on CPU.
    out, lse = mp32.sparse_mla_decode_fwd_scheduled_interface_v4(
        torch.zeros((1, 1, 8, 512), dtype=torch.bfloat16),
        torch.zeros((1, 64 * 584), dtype=torch.float8_e4m3fn),
        torch.zeros((1, 64 * 584 // 2), dtype=torch.bfloat16),
        torch.zeros((1, 64 * 584), dtype=torch.uint8),
        torch.zeros((1, 1, 1, 64), dtype=torch.int32),
        torch.zeros((8, 8), dtype=torch.int32),
        torch.zeros((2,), dtype=torch.int32),
    )
    assert out.shape == (1, 1, 8, 512)
    assert lse.shape == (1, 8, 1)
