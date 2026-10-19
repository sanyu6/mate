"""Exercise the SGLang fixed-query adapter with real kernels and graph replay."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch_musa  # noqa: F401

import mate
from mate.jit.mla_ops import get_mla_ops_module
from mate.testing import supported_musa_compute_capability

spec = importlib.util.spec_from_file_location(
    "joyai_mla_adapter", Path(__file__).parents[1] / "benchmarks/joyai_mla/adapter.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch,length", [(1, 129), (1, 10000), (5, 129), (5, 10000)])
def test_fixed_decode_adapter(dtype, batch, length):
    torch.manual_seed(42)
    heads, page = 16, 64
    pages = (length + page - 1) // page
    q = torch.randn(batch, heads, 576, device="musa", dtype=dtype)
    kv = torch.randn(batch * pages, page, 1, 576, device="musa", dtype=dtype)
    lengths = torch.full((batch,), length, device="musa", dtype=torch.int32)
    workspace = torch.empty(16 * 1024 * 1024, device="musa", dtype=torch.uint8)
    kwargs = dict(
        q=q[..., 512:],
        qv=q[..., :512],
        k_cache=kv[..., 512:],
        v_cache=kv[..., :512],
        cache_seqlens=lengths,
        page_table=torch.arange(batch * pages, device="musa", dtype=torch.int32).view(
            batch, pages
        ),
        cu_seqlens_q=torch.arange(batch + 1, device="musa", dtype=torch.int32),
        max_seqlen_q=1,
        softmax_scale=192**-0.5,
        causal=True,
        scheduler_metadata=(workspace, False),
    )
    assert adapter.supports_fixed_decode(kwargs)
    baseline = mate.flash_attn_with_kvcache(**kwargs)
    kwargs["scheduler_metadata"] = (workspace, True)
    module = get_mla_ops_module()
    out, lse = adapter.fixed_decode(module, kwargs)
    torch.testing.assert_close(out, baseline, atol=1.5e-2, rtol=1e-2)
    latent = kv[..., :512].cpu().float().view(batch, pages * page, 512)[:, :length]
    rope = kv[..., 512:].cpu().float().view(batch, pages * page, 64)[:, :length]
    logits = (
        q[..., :512].cpu().float() @ latent.transpose(1, 2)
        + q[..., 512:].cpu().float() @ rope.transpose(1, 2)
    ) * 192**-0.5
    ref = logits.softmax(-1) @ latent
    torch.testing.assert_close(out.cpu().float(), ref, atol=1.5e-2, rtol=1e-2)
    torch.testing.assert_close(
        lse[:, 0].cpu(), logits.logsumexp(-1), atol=1.5e-2, rtol=1e-2
    )
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = adapter.fixed_decode(module, kwargs)[0]
    for _ in range(3):
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(captured, out, atol=0, rtol=0)
    assert not adapter.supports_fixed_decode(dict(kwargs, max_seqlen_q=2))
    assert not adapter.supports_fixed_decode(dict(kwargs, return_softmax_lse=True))
