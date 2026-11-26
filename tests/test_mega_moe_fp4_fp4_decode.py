"""Device bring-up tests for M32 decode; not run during static development."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mate.jit.mega_moe_fp4_fp4 import get_fp4_prefill_stage_module
from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_module
from mate.mega_moe_fp4_fp4 import (
    FP4FP4SymmBuffer,
    fp4_fp4_mega_moe,
    fp4_fp4_mega_moe_decode,
    transform_weights_for_fp4_fp4_mega_moe,
)
from mega_moe_fp4_fp4_reference import (
    rank_inputs,
    rank_weights,
    reference_partials,
    routes_for,
)
from test_mega_moe_fp4_fp4_pipeline import collective_check


@pytest.mark.parametrize("ranks", [1, 4, 8])
@pytest.mark.parametrize(
    "activation,linear_beta", [("swiglu", None), ("situ", None), ("situ", 25.0)]
)
def test_decode_replay_and_unified_prefill_switch(ranks, activation, linear_beta):
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available() or torch.musa.device_count() < ranks:
        pytest.skip("requires MP32 devices")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            str(Path(__file__).resolve()),
            "--activation",
            activation,
            "--activation-linear-beta",
            "none" if linear_beta is None else str(linear_beta),
        ],
        check=True,
        timeout=900,
    )


def main():
    import torch_musa  # noqa: F401
    import torch.distributed as dist

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--activation-beta", type=float, default=4.0)
    parser.add_argument(
        "--activation-linear-beta",
        type=lambda value: None if value.lower() == "none" else float(value),
        default=25.0,
    )
    args = parser.parse_args()
    options, reference_options, jit_options = {}, {}, {}
    if args.activation == "situ":
        options = dict(
            activation="situ",
            activation_clamp=None,
            activation_beta=args.activation_beta,
            activation_linear_beta=args.activation_linear_beta,
        )
        reference_options = dict(
            activation="situ",
            clamp=None,
            activation_beta=args.activation_beta,
            activation_linear_beta=args.activation_linear_beta,
        )
        jit_options = dict(
            activation="situ", use_situ_linear=args.activation_linear_beta is not None
        )
    rank, world, local = (
        int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    dist.init_process_group("mccl", device_id=torch.device("musa", local))
    s = FP4FP4SymmBuffer(
        dist.group.WORLD, world * 4, 256, 2, 256, 256, enable_decode=True
    )
    logical1 = rank_weights(rank, 4, 512, 256, 1, logical_only=True)
    logical2 = rank_weights(rank, 4, 256, 256, 2, logical_only=True)
    w1, w2 = transform_weights_for_fp4_fp4_mega_moe(
        tuple(t.to("musa") for t in logical1),
        tuple(t.to("musa") for t in logical2),
    )
    s.allocate_workspace()
    s.fused_stage1_weights(w1)
    for stage in (1, 2):
        get_fp4_decode_module(
            s.config, stage, num_mps=s.common_mps, **(jit_options if stage == 1 else {})
        )
    for stage in (1, 2, 3):
        get_fp4_prefill_stage_module(
            s.config,
            stage,
            **({"num_mps": s.common_mps} if stage < 3 else {}),
            **(jit_options if stage == 1 else {}),
        )
    x, sf = rank_inputs(rank, 256, 256)
    cases = [
        (1, "balanced", "decode"),
        (31, "skew", "decode"),
        (32, "disabled", "decode"),
        (256, "balanced", "prefill"),
        (33, "skew", "decode"),
        (127, "all_disabled", "decode"),
        (128, "balanced", "decode"),
    ]
    outputs = [
        torch.empty(tokens, 256, dtype=torch.bfloat16, device="musa")
        for tokens, _, _ in cases
    ]
    inputs = []
    device_inputs = []
    for phase, (tokens, pattern, _) in enumerate(cases):
        changed = x[:tokens].clone()
        changed[:, 0] ^= phase & 1
        ids = routes_for(rank, tokens, 2, world * 4, pattern)
        inputs.append((changed, sf[:tokens], ids))
        device_inputs.append(tuple(t.to("musa") for t in inputs[-1]))
    snapshots = []
    try:
        # Queue the entire replay, including decode -> prefill -> decode,
        # without host/device synchronization between pipeline launches.
        for (tokens, pattern, mode), out, (dx, ds, di) in zip(
            cases, outputs, device_inputs
        ):
            s.x[:tokens].copy_(dx)
            s.x_sf[:tokens].copy_(ds)
            s.topk_idx[:tokens].copy_(di)
            s.topk_weights[:tokens].fill_(0.5)
            function = (
                fp4_fp4_mega_moe if mode == "prefill" else fp4_fp4_mega_moe_decode
            )
            function(out, w1, w2, s, **options)
            snapshots.append(s.status.clone())
        torch.musa.synchronize()

        def verify():
            for case, out, status, (a, scales, ids) in zip(
                cases, outputs, snapshots, inputs
            ):
                expected = (
                    reference_partials(a, scales, ids, 256, **reference_options)
                    .float()
                    .sum(1)
                    * 0.5
                ).bfloat16()
                got = out.cpu()
                bad = (got != expected).sum(1)
                if bool(bad.any()):
                    rows = torch.where(bad != 0)[0]
                    print(
                        "DECODE_MISMATCH",
                        case,
                        "rank",
                        rank,
                        "rows",
                        rows.tolist(),
                        "counts",
                        bad[rows].tolist(),
                        "got_amax",
                        got[rows].float().abs().amax(1).tolist(),
                        "ref_amax",
                        expected[rows].float().abs().amax(1).tolist(),
                        flush=True,
                    )
                torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=str(case))
                assert status.cpu().item() == 0

        collective_check(verify)
        if rank == 0:
            print(
                "DECODE_REPLAY_PASS",
                world,
                args.activation,
                args.activation_beta,
                args.activation_linear_beta,
                "cases",
                [(t, p, m) for t, p, m in cases],
                flush=True,
            )
    finally:
        s.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
