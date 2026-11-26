"""Exact independent W4A4 decode contract and queued graph input updates."""

import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from mate.mega_moe_fp4_fp4 import (
    FP4FP4SymmBuffer,
    fp4_fp4_mega_moe_decode,
    transform_weights_for_fp4_fp4_mega_moe,
    quantize_fp4_reference,
)
from mate.jit.mega_moe_fp4_fp4_decode import (
    get_fp4_decode_module,
    get_fp4_decode_input_module,
)
from mega_moe_fp4_fp4_reference import (
    rank_weights,
    routes_for,
    w4a4_partials_reference,
    w4a4_combine_reference,
)
from test_mega_moe_fp4_fp4_pipeline import collective_check


try:
    import torch_musa
except ImportError:
    torch_musa = None


def test_owner_partial_rounding_is_part_of_the_contract():
    values = torch.tensor([[[1.0], [1.0 / 256], [-1.0]]], dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1, 4]])
    result = w4a4_combine_reference(values, ids, 4, 2)
    assert result.item() == 0.0
    assert values.float().sum(1).item() == 1.0 / 256


@pytest.mark.parametrize("ranks", [1, 4])
def test_w4a4_decode_graph_and_updates(ranks):
    if torch_musa is None:
        pytest.skip("torch_musa unavailable")
    if not torch.musa.is_available() or torch.musa.device_count() < ranks:
        pytest.skip("requires MP32 GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            str(Path(__file__).resolve()),
        ],
        check=True,
        timeout=900,
    )


def main():
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    dist.init_process_group("gloo")
    b, h, i, k = 24, 256, 256, 2
    s = FP4FP4SymmBuffer(dist.group.WORLD, world * 4, 32, k, h, i, enable_decode=True)
    w1, w2 = transform_weights_for_fp4_fp4_mega_moe(
        tuple(
            t.to("musa") for t in rank_weights(rank, 4, 2 * i, h, 1, logical_only=True)
        ),
        tuple(t.to("musa") for t in rank_weights(rank, 4, h, i, 2, logical_only=True)),
    )
    options = dict(
        sglang_numerics=True, line_padded_scales=True, pack_b=True, double_buffered=True
    )
    s.allocate_workspace()
    s.packed_decode_weights(w1, 1)
    s.packed_decode_weights(w2, 2)
    for stage in (1, 2):
        get_fp4_decode_module(
            s.config,
            stage,
            num_mps=s.common_mps,
            sglang_numerics=True,
            line_padded_scales=stage == 1,
            pack_b=True,
            double_buffered=True,
        )
    get_fp4_decode_input_module()
    rng = torch.Generator().manual_seed(924 + rank)
    fixtures = []
    for phase, pattern in enumerate(("balanced", "skew", "disabled", "all_disabled")):
        hidden = (torch.randn(b, h, generator=rng) * (0.5 + phase)).bfloat16()
        if pattern == "all_disabled":
            hidden.zero_()
        ids = routes_for(rank, b, k, world * 4, pattern).int()
        weights = torch.softmax(torch.randn(b, k, generator=rng), -1)
        q, sf = quantize_fp4_reference(hidden)
        partials = w4a4_partials_reference(q, sf, ids.long(), weights, i)
        expected = w4a4_combine_reference(partials, ids.long(), 4, world)
        fixtures.append((tuple(t.to("musa") for t in (hidden, ids, weights)), expected))
    x, ids, weights = (torch.empty_like(v) for v in fixtures[0][0])
    out = torch.empty(b, h, dtype=torch.bfloat16, device="musa")

    def run():
        s.prepare_decode_input(x, ids, weights)
        return fp4_fp4_mega_moe_decode(out, w1, w2, s, **options)

    graph = None
    try:
        for values, expected in fixtures:
            hidden, route_ids, route_weights = values
            q, sf = quantize_fp4_reference(hidden.cpu())
            zero_groups = (
                hidden.cpu().float().reshape(b, h // 32, 32).abs().amax(-1) == 0
            )
            sf[zero_groups] = 0  # packed W4A4's zero-group UE8M0 representation
            for index_type in (torch.int32, torch.int64):
                s.prepare_decode_input(hidden, route_ids.to(index_type), route_weights)
                torch.musa.synchronize()

                def verify_input():
                    torch.testing.assert_close(s.x[:b].cpu(), q, rtol=0, atol=0)
                    torch.testing.assert_close(s.x_sf[:b].cpu(), sf, rtol=0, atol=0)
                    torch.testing.assert_close(
                        s.topk_idx[:b].cpu(), route_ids.cpu().long(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        s.topk_weights[:b].cpu(), route_weights.cpu(), rtol=0, atol=0
                    )

                collective_check(verify_input)
            for destination, source in zip((x, ids, weights), values):
                destination.copy_(source)
            run()
            torch.musa.synchronize()
            collective_check(
                lambda: torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            )
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            run()
        queued = []
        for step in range(32):
            values, expected = fixtures[step % len(fixtures)]
            for destination, source in zip((x, ids, weights), values):
                destination.copy_(source)
            out.fill_(float("nan"))
            graph.replay()
            queued.append((out.clone(), expected))
        torch.musa.synchronize()

        def verify():
            for got, expected in queued:
                torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)

        collective_check(verify)
        if rank == 0:
            print("W4A4_CPU_GRAPH_EXACT_PASS", world, flush=True)
    finally:
        if graph is not None:
            graph.reset()
        s.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
