"""Strict CPU checks for the final FP4 prefill stages and diagnostics."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

import pytest
import torch

from mate.mega_moe_fp4_fp4 import (
    FP4FP4SymmBuffer,
    transform_weights_for_fp4_fp4_mega_moe,
    fp4_fp4_stage1,
    fp4_fp4_stage2,
    fp4_fp4_mega_moe,
    dequantize_fp4_reference,
    quantize_fp4_reference,
)
from mega_moe_fp4_fp4_reference import (
    rank_inputs,
    rank_weights,
    routes_for,
    activation_reference,
    reference_partials,
)


def collective_check(function):
    import torch.distributed as dist

    error = None
    try:
        function()
    except Exception:
        error = traceback.format_exc()
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), errors


def check_fused(
    s, logical, inputs, tokens, pattern, *, clamp=10.0, **activation_options
):
    c = s.config
    counts = s.counts.cpu().tolist()
    metadata = s.metadata.cpu()
    assert s.status.cpu().item() == 0
    pool_row = 0
    for e, rows in enumerate(counts):
        expected = []
        for source in range(c.ranks):
            ids = routes_for(source, tokens, c.topk, c.experts, pattern)
            tr, slots = torch.where(ids == s.rank * (c.experts // c.ranks) + e)
            expected.extend(((source * c.tokens + tr) * c.topk + slots).tolist())
        assert sorted(metadata[e, :rows].tolist()) == sorted(expected)
        assert rows == len(expected)
        if rows == 0:
            continue
        x = []
        sf = []
        for code in metadata[e, :rows].tolist():
            source = (code // c.topk) // c.tokens
            token = (code // c.topk) % c.tokens
            x.append(inputs[source][0][token])
            sf.append(inputs[source][1][token])
        x, sf = torch.stack(x), torch.stack(sf)
        torch.testing.assert_close(
            s.stage1_l1[pool_row : pool_row + rows].cpu(), x, rtol=0, atol=0
        )
        torch.testing.assert_close(
            s.stage1_l1_sf[:, pool_row : pool_row + rows].T.cpu(), sf, rtol=0, atol=0
        )
        a = dequantize_fp4_reference(x, sf)
        b = dequantize_fp4_reference(logical[0][e], logical[1][e])
        first = (a @ b.T).bfloat16()
        q, qs = quantize_fp4_reference(
            activation_reference(first, clamp, **activation_options)
        )
        torch.testing.assert_close(
            s.stage2_l2[pool_row : pool_row + rows].cpu(), q, rtol=0, atol=0
        )
        torch.testing.assert_close(
            s.stage2_l2_sf[:, pool_row : pool_row + rows].T.cpu(), qs, rtol=0, atol=0
        )
        pool_row += (rows + 255) // 256 * 256


@pytest.mark.parametrize("ranks", [1, 8])
@pytest.mark.parametrize(
    "activation,linear_beta", [("swiglu", None), ("situ", None), ("situ", 25.0)]
)
def test_prefill_pipeline(ranks, activation, linear_beta):
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available() or torch.musa.device_count() < ranks:
        pytest.skip("requires MP32")
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
        timeout=600,
    )


def main():
    import torch_musa  # noqa: F401
    import torch.distributed as dist
    from mate.jit.mega_moe_fp4_fp4 import (
        get_fp4_prefill_stage_module,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--activation-beta", type=float, default=4.0)
    parser.add_argument(
        "--activation-linear-beta",
        type=lambda value: None if value.lower() == "none" else float(value),
        default=25.0,
    )
    args = parser.parse_args()
    activation_options = {}
    reference_options = {}
    jit_activation_options = {}
    if args.activation == "situ":
        activation_options = dict(
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
        jit_activation_options = dict(
            activation="situ", use_situ_linear=args.activation_linear_beta is not None
        )
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    dist.init_process_group("mccl", device_id=torch.device("musa", local))
    s = FP4FP4SymmBuffer(dist.group.WORLD, world * 4, 256, 2, 256, 256)
    s.allocate_workspace()
    logical = rank_weights(rank, 4, 512, 256, 1, logical_only=True)
    logical_w2 = rank_weights(rank, 4, 256, 256, 2, logical_only=True)
    w1, w2 = transform_weights_for_fp4_fp4_mega_moe(
        tuple(t.to("musa") for t in logical),
        tuple(t.to("musa") for t in logical_w2),
    )
    for mode in ("full", "compute", "dispatch"):
        name, module = get_fp4_prefill_stage_module(
            s.config, 1, mode=mode, num_mps=s.common_mps, **jit_activation_options
        )
        if rank == 0:
            print(
                "COMPILED",
                mode,
                "shared",
                module.get_function(name + "_shared")(),
                flush=True,
            )
    get_fp4_prefill_stage_module(s.config, 2, num_mps=s.common_mps)
    name, mod3 = get_fp4_prefill_stage_module(s.config, 3)
    combine = mod3.get_function(name)
    inputs = [rank_inputs(r, 256, 256) for r in range(world)]
    out = torch.empty((256, 256), dtype=torch.bfloat16, device="musa")
    try:
        for phase, (pattern, tokens) in enumerate(
            [
                ("balanced", 256),
                ("skew", 129),
                ("disabled", 251),
                ("all_disabled", 256),
                ("balanced", 256),
            ]
        ):
            # Change input bytes on every phase to expose stale TME visibility.
            for x, _ in inputs:
                x[:, 0] ^= 1
            s.x.copy_(inputs[rank][0])
            s.x_sf.copy_(inputs[rank][1])
            s.topk_idx.copy_(routes_for(rank, 256, 2, world * 4, pattern))
            s.topk_weights.fill_(0.5)
            fp4_fp4_stage1(s, w1, tokens, **activation_options)
            collective_check(
                lambda: check_fused(
                    s, logical, inputs, tokens, pattern, **reference_options
                )
            )
            # Reusing a plan with another MP count would wait forever for the
            # wrong expert-ready tag in the unchanged prefill scheduler.
            with pytest.raises(ValueError, match="same tokens and MP count"):
                fp4_fp4_stage1(
                    s,
                    w1,
                    tokens,
                    mode="compute",
                    num_mps=s.common_mps - 4,
                    **activation_options,
                )
            for mode in ("compute", "dispatch", "compute"):
                fp4_fp4_stage1(s, w1, tokens, mode=mode, **activation_options)
                if mode != "dispatch":
                    collective_check(
                        lambda: check_fused(
                            s, logical, inputs, tokens, pattern, **reference_options
                        )
                    )
            fp4_fp4_stage2(s, w2, num_mps=s.common_mps)

            def verify_workspace_retirement():
                start = s.layout.offsets["bytes"]
                control = s.buffer[
                    start + s.dispatch_barrier_bytes : start + s.dispatch_control_bytes
                ]
                assert control.cpu().count_nonzero().item() == 0
                assert s.counts.cpu().count_nonzero().item() == 0

            collective_check(verify_workspace_retirement)
            combine(out[:tokens], s.buffer, tokens, s.common_mps)

            def verify_output():
                ids = routes_for(rank, tokens, 2, world * 4, pattern)
                partials = reference_partials(
                    inputs[rank][0][:tokens],
                    inputs[rank][1][:tokens],
                    ids,
                    256,
                    **reference_options,
                )
                expected = (partials.float().sum(1) * 0.5).bfloat16()
                torch.testing.assert_close(out[:tokens].cpu(), expected, rtol=0, atol=0)
                assert s.status.cpu().item() == 0

            collective_check(verify_output)
            if phase == 2:
                # Exercise negative/nonuniform route weights and disabled
                # routes independently of all upstream GEMM/quantization work.
                weights = torch.linspace(-1.25, 1.25, 256 * 2).view(256, 2)
                s.topk_weights.copy_(weights)
                combine(out[:tokens], s.buffer, tokens, s.common_mps)

                def verify_weighted_combine():
                    ids = routes_for(rank, tokens, 2, world * 4, pattern)
                    partials = reference_partials(
                        inputs[rank][0][:tokens],
                        inputs[rank][1][:tokens],
                        ids,
                        256,
                        **reference_options,
                    )
                    acc = torch.zeros((tokens, 256), dtype=torch.float32)
                    for slot in range(2):
                        # BF16 x bounded FP32 + FP32 is exact in FP64 here;
                        # round once per FMA, then BF16 only at final output.
                        acc = (
                            acc.double()
                            + partials[:, slot].double()
                            * weights[:tokens, slot, None].double()
                        ).float()
                    torch.testing.assert_close(
                        out[:tokens].cpu(),
                        acc.bfloat16(),
                        rtol=0,
                        atol=0,
                        msg="native vs CPU",
                    )

                collective_check(verify_weighted_combine)
                s.topk_weights.fill_(0.5)
                combine(out[:tokens], s.buffer, tokens, s.common_mps)
            expected_output = out[:tokens].clone()
            fp4_fp4_mega_moe(out[:tokens], w1, w2, s, **activation_options)
            collective_check(
                lambda: torch.testing.assert_close(
                    out[:tokens], expected_output, rtol=0, atol=0
                )
            )
            if rank == 0:
                print(
                    "FP4_FUSION_PASS",
                    world,
                    phase,
                    pattern,
                    tokens,
                    args.activation,
                    args.activation_linear_beta,
                    flush=True,
                )
    finally:
        s.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
