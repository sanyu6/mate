from itertools import accumulate
from pathlib import Path

import pytest
import torch
import torch_musa  # noqa: F401

from mate.testing import supported_musa_compute_capability


@pytest.fixture(scope="module")
def combine():
    from mate.jit import env as jit_env
    from mate.jit.core import gen_jit_spec
    from mate.jit.mla_ops import CUDA_FLAGS, CXX_FLAGS, INCLUDE_PATHS

    module = gen_jit_spec(
        "test_mla_combine",
        sources=[
            Path(__file__).parent / "csrc" / "mla_combine_test.mu",
            jit_env.MATE_CSRC_DIR / "attention_combine.mu",
        ],
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    ).build_and_load()
    return module.get_function("test_mla_combine")


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "heads,counts",
    [
        (16, [53]),
        (16, [12, 13, 12, 12, 13]),
        (32, [120]),
        (32, [1, 3, 17, 31, 45]),
        (16, [1, 1, 1, 1, 1]),
        (16, [7] * 6),  # Batch size does not limit the parallel path.
        (64, [53]),  # Query/head row fallback.
    ],
)
def test_mla_combine(combine, dtype, masked, heads, counts):
    torch.manual_seed(42)
    batch = len(counts)
    total_splits = sum(counts)
    o_accum = torch.randn(total_splits, heads, 512, device="musa")
    lse_accum = torch.randn(total_splits, heads, device="musa")
    if masked:
        # Undefined partial O must not contaminate zero-mass splits.
        lse_accum[::3] = -torch.inf
        o_accum[::3] = torch.nan
        lse_accum[:, 0] = -torch.inf
        o_accum[:, 0] = torch.nan

    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)
    splits = torch.tensor(prefix, dtype=torch.int32, device="musa")
    out = torch.full((batch, heads, 512), torch.nan, dtype=dtype, device="musa")
    lse = torch.full((batch, heads), torch.nan, device="musa")
    # Keep the reference independent of device reduction edge-case behavior.
    o_cpu, lse_cpu = o_accum.cpu(), lse_accum.cpu()
    expected, expected_lse = [], []
    for b, count in enumerate(counts):
        begin, end = prefix[b : b + 2]
        partial_lse = lse_cpu[begin:end]
        normalizer = torch.logsumexp(partial_lse, dim=0)
        normalizer = torch.where(torch.isneginf(normalizer), torch.inf, normalizer)
        weight = (partial_lse - normalizer).exp()
        partial_o = torch.where(weight[..., None] == 0, 0, o_cpu[begin:end])
        reference = (partial_o * weight[..., None]).sum(dim=0)
        expected.append(reference)
        expected_lse.append(normalizer)
        if count == 1:
            # The main kernel bypasses partials and writes these rows directly.
            out[b] = reference.to(dtype)
            lse[b] = normalizer

    num_mp_parts = max(total_splits - batch, max(counts), 1)
    combine(o_accum, lse_accum, splits, out, lse, num_mp_parts)
    torch.testing.assert_close(
        out.cpu(), torch.stack(expected).to(dtype), atol=1e-3, rtol=1e-2
    )
    torch.testing.assert_close(
        lse.cpu(), torch.stack(expected_lse), atol=2e-6, rtol=2e-6
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "lengths,heads,kv_length",
    [([1], 16, 1), ([1, 2, 1], 8, 97), ([1] * 6, 16, 10000), ([1, 2, 1], 32, 129)],
)
def test_mla_combine_varlen(dtype, causal, lengths, heads, kv_length):
    import mate

    torch.manual_seed(42)
    batch, page = len(lengths), 64
    pages = (kv_length + page - 1) // page
    total_q = sum(lengths)
    q = torch.randn(total_q, heads, 576, dtype=dtype, device="musa")
    kv = torch.randn(batch * pages, page, 1, 576, dtype=dtype, device="musa")
    cu_q = torch.tensor(
        [0, *accumulate(lengths)],
        dtype=torch.int32,
        device="musa",
    )
    lens = torch.full((batch,), kv_length, dtype=torch.int32, device="musa")
    table = torch.arange(batch * pages, dtype=torch.int32, device="musa").view(
        batch, pages
    )
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="musa")
    scale = 576**-0.5

    # Explicit tuple metadata selects MLA rather than the default FMHA backend.
    def run(initialized):
        return mate.flash_attn_with_kvcache(
            q=q[..., 512:],
            k_cache=kv[..., 512:],
            v_cache=kv[..., :512],
            qv=q[..., :512],
            cache_seqlens=lens,
            page_table=table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=max(lengths),
            softmax_scale=scale,
            causal=causal,
            scheduler_metadata=(workspace, initialized),
            return_softmax_lse=True,
        )[:2]

    out, lse = run(False)
    q_cpu = q.cpu().float()
    kv_cpu = kv.cpu().float().view(batch, pages * page, 576)[:, :kv_length]
    expected, expected_lse = [], []
    start = 0
    for b, length in enumerate(lengths):
        logits = (
            torch.einsum("qhd,kd->hqk", q_cpu[start : start + length], kv_cpu[b])
            * scale
        )
        if causal:
            mask = torch.arange(kv_length)[None, :] > (
                kv_length - length + torch.arange(length)[:, None]
            )
            logits.masked_fill_(mask[None, :, :], -torch.inf)
        expected.append(
            torch.einsum("hqk,kd->qhd", logits.softmax(-1), kv_cpu[b, :, :512])
        )
        expected_lse.append(logits.logsumexp(-1))
        start += length
    torch.testing.assert_close(
        out.cpu(), torch.cat(expected).to(dtype), atol=1.5e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        lse.cpu(), torch.cat(expected_lse, dim=1), atol=1.5e-2, rtol=1e-2
    )

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = run(True)
    for _ in range(3):
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(captured[0].cpu(), out.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(captured[1].cpu(), lse.cpu(), atol=0, rtol=0)
