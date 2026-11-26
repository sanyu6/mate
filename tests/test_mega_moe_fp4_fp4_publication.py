"""Regression for remote Stage2 partial publication across changing routes."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from mate.mega_moe_fp4_fp4 import (
    FP4FP4SymmBuffer,
    fp4_fp4_mega_moe,
    transform_weights_for_fp4_fp4_mega_moe,
)
from mate.jit.mega_moe_fp4_fp4 import get_fp4_prefill_stage_module
from mega_moe_fp4_fp4_reference import (
    rank_inputs,
    rank_weights,
    reference_partials,
    routes_for,
)


def test_flash_peer_partial_publication():
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available() or torch.musa.device_count() < 4:
        pytest.skip("requires four MP32 devices")
    visible = os.environ.get("MUSA_VISIBLE_DEVICES")
    device_mask = ",".join(visible.split(",")[:4]) if visible else "0,1,2,3"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(Path(__file__).resolve()),
            "--iterations",
            "12",
        ],
        env=dict(os.environ, MUSA_VISIBLE_DEVICES=device_mask, OMP_NUM_THREADS="4"),
        check=True,
        timeout=600,
    )


def main():
    import torch_musa  # noqa: F401

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world != 4:
        parser.error("this regression requires EP4")
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    # Control collectives must not enqueue work on a resident prefill GPU grid.
    dist.init_process_group("gloo")
    tokens, hidden, intermediate, experts, topk = 4096, 4096, 2048, 256, 6
    s = FP4FP4SymmBuffer(dist.group.WORLD, experts, tokens, topk, hidden, intermediate)
    s.allocate_workspace()
    try:
        x, sf = rank_inputs(rank, tokens, hidden)
        s.x.copy_(x)
        s.x_sf.copy_(sf)
        s.topk_weights.fill_(1.0 / topk)
        w1, w2 = transform_weights_for_fp4_fp4_mega_moe(
            tuple(
                v.to("musa")
                for v in rank_weights(
                    rank,
                    experts // world,
                    2 * intermediate,
                    hidden,
                    1,
                    logical_only=True,
                )
            ),
            tuple(
                v.to("musa")
                for v in rank_weights(
                    rank, experts // world, hidden, intermediate, 2, logical_only=True
                )
            ),
        )
        get_fp4_prefill_stage_module(s.config, 1, num_mps=s.common_mps)
        get_fp4_prefill_stage_module(s.config, 2, num_mps=s.common_mps)
        get_fp4_prefill_stage_module(s.config, 3)
        s.fused_stage1_weights(w1)
        balanced = routes_for(rank, tokens, topk, experts, "balanced")
        skew = routes_for(rank, tokens, topk, experts, "skew")
        skew[::3, 0] = -1
        active_tokens = 129
        selected = torch.tensor([0, 64, active_tokens - 1])
        partials = reference_partials(
            x[selected], sf[selected], skew[selected], intermediate
        )
        expected = torch.zeros((len(selected), hidden))
        # Round each FMA once to FP32, then round the final output to BF16.
        route_weight = float(torch.tensor(1.0 / topk))
        for slot in range(topk):
            expected = (
                expected.double() + partials[:, slot].double() * route_weight
            ).float()
        expected = expected.bfloat16()
        out = torch.empty((tokens, hidden), device="musa", dtype=torch.bfloat16)
        selected_gpu = selected.to("musa")
        torch.musa.synchronize()
        dist.barrier()
        torch.musa.synchronize()
        for iteration in range(args.iterations):
            # All ranks first populate their own experts; skew then leaves only
            # rank 0 writing peer partials while the other GPUs finish early.
            # Do not insert a host barrier or D2H read between these launches:
            # it can hide a dirty cached store that is flushed at kernel exit.
            s.topk_idx.copy_(balanced)
            fp4_fp4_mega_moe(out, w1, w2, s)
            s.topk_idx.copy_(skew)
            fp4_fp4_mega_moe(out[:active_tokens], w1, w2, s)
            error = None
            try:
                torch.testing.assert_close(
                    out[selected_gpu].cpu(), expected, rtol=0, atol=0
                )
                assert s.status.cpu().item() == 0
            except Exception as exc:
                error = repr(exc)
            errors = [None] * world
            dist.all_gather_object(errors, error)
            assert not any(errors), (iteration, errors)
            if rank == 0:
                print("PEER_PARTIAL_PUBLICATION_PASS", iteration, flush=True)
    finally:
        s.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
