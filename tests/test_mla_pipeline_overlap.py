import pytest
import torch
import torch_musa  # noqa: F401

import mate
import mate.mha_interface as mi
from mate.testing import supported_musa_compute_capability


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch,length,heads,one_split",
    [
        (1, 129, 16, True),
        (1, 10000, 16, False),
        (5, 10000, 16, False),
        (1, 6000, 32, False),
    ],
)
def test_mla_pipeline_overlap(dtype, causal, batch, length, heads, one_split):
    torch.manual_seed(42)
    pages = (length + 63) // 64
    ql = torch.randn(batch, 1, heads, 512, dtype=dtype, device="musa")
    qr = torch.randn(batch, 1, heads, 64, dtype=dtype, device="musa")
    kv = torch.randn(batch * pages, 64, 1, 512, dtype=dtype, device="musa")
    kr = torch.randn(batch * pages, 64, 1, 64, dtype=dtype, device="musa")
    lens = torch.full((batch,), length, dtype=torch.int32, device="musa")
    table = torch.arange(batch * pages, dtype=torch.int32, device="musa").view(
        batch, pages
    )
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="musa")
    scale = 576**-0.5

    def run(initialized=True):
        return mate.flash_attn_with_kvcache(
            qr,
            kr,
            kv,
            qv=ql,
            cache_seqlens=lens,
            page_table=table,
            max_seqlen_q=1,
            softmax_scale=scale,
            causal=causal,
            scheduler_metadata=(workspace, initialized),
            return_softmax_lse=True,
        )[:2]

    run(False)
    if one_split:
        assert batch == 1
        metadata, splits = mi._prepare_mla_scheduler_metadata_from_workspace(
            workspace, True, ql, kv, lens, None, 1
        )
        metadata.zero_()
        metadata[:, 0] = 1
        metadata[0, :5] = torch.tensor(
            [0, 0, 0, pages, 0], dtype=torch.int32, device="musa"
        )
        splits.copy_(torch.tensor([0, 1], dtype=torch.int32, device="musa"))

    latent = kv.cpu().float().view(batch, pages * 64, 512)[:, :length]
    rope = kr.cpu().float().view(batch, pages * 64, 64)[:, :length]
    logits = (
        ql.cpu().float()[:, 0] @ latent.transpose(1, 2)
        + qr.cpu().float()[:, 0] @ rope.transpose(1, 2)
    ) * scale
    ref = (logits.softmax(-1) @ latent).to(dtype).unsqueeze(1)
    ref_lse = logits.logsumexp(-1).unsqueeze(1)
    first = None
    for _ in range(5):
        out, lse = run()
        actual, actual_lse = out.cpu(), lse.cpu()
        torch.testing.assert_close(actual, ref, atol=1.5e-2, rtol=1e-2)
        torch.testing.assert_close(actual_lse, ref_lse, atol=1.5e-2, rtol=1e-2)
        if first is None:
            first = actual
        else:
            torch.testing.assert_close(actual, first, atol=0, rtol=0)

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = [run() for _ in range(3)]
    for _ in range(3):
        graph.replay()
        torch.musa.synchronize()
        for out, lse in captured:
            torch.testing.assert_close(out.cpu(), ref, atol=1.5e-2, rtol=1e-2)
            torch.testing.assert_close(lse.cpu(), ref_lse, atol=1.5e-2, rtol=1e-2)
