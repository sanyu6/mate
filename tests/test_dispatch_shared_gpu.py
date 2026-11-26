"""Exact dispatch + same-native GEMM contract; independent CPU and muDNN diagnostics."""

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from mate.ops.moe import Fp4DispatchBuffer, Fp4DispatchSharedGemm
from test_fp4_dispatch_gpu import fixture, verify

try:
    import torch_musa
except ImportError:
    torch_musa = None


def q8(x):
    f = x.float().reshape(x.shape[0], -1, 32)
    exp = torch.ceil(torch.log2((f.abs().amax(-1) / 448).clamp_min(2**-126))).clamp(
        -126, 127
    )
    return (f / torch.pow(2.0, exp)[:, :, None]).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).reshape_as(x).contiguous(), (exp + 127).to(torch.uint8).t().contiguous()


def dequant(q, sf):
    return (
        q.float().reshape(q.shape[0], -1, 32)
        * torch.pow(2.0, sf.t().float() - 127)[:, :, None]
    ).reshape_as(q)


@pytest.mark.parametrize("ranks", [1, 4])
def test_dispatch_shared_graph(ranks):
    if (
        torch_musa is None
        or not torch.musa.is_available()
        or torch.musa.device_count() < ranks
    ):
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
    torch.set_num_threads(2)
    torch.musa.set_device(local)
    dist.init_process_group("gloo")
    device = torch.device("musa", local)

    def gather(v):
        a = [None] * world
        dist.all_gather_object(a, v)
        return a

    def checked(fn):
        err = None
        try:
            fn()
        except Exception as e:
            err = repr(e)
        errors = gather(err)
        if any(errors):
            raise AssertionError(errors)

    records = []
    try:
        for b, h, n, e, k in (
            (1, 256, 128, 32, 2),
            (24, 256, 128, 32, 2),
            (24, 5120, 4608, 384, 6),
        ):
            for padded in (False, True):
                ctx = Fp4DispatchBuffer(
                    dist.group.WORLD, e, h, k, line_padded_scales=padded
                )
                graphs = []
                try:
                    cpu = [fixture(rank, b, h, e, k, p) for p in range(4)]
                    peers_by_phase = []
                    for fc in cpu:
                        peers_by_phase.append(
                            gather(
                                dict(
                                    payload=fc[0].tolist(),
                                    scales=fc[1].tolist(),
                                    ids=fc[2].tolist(),
                                    weights=fc[3].tolist(),
                                )
                            )
                        )
                    rng = torch.Generator().manual_seed(1234 + rank)
                    base_x = torch.randn(b, h, generator=rng).bfloat16()
                    base_w = (torch.randn(n, h, generator=rng) * 0.02).bfloat16()
                    fixtures = []
                    expected = []
                    for p in range(4):
                        x = base_x * (-1 if p % 2 else 1) * (p + 1)
                        w = base_w * (1 if p < 2 else -0.5)
                        if p == 2:
                            x.zero_()
                        xq, xs = q8(x)
                        wq, ws = q8(w)
                        expected.append(dequant(xq, xs) @ dequant(wq, ws).t())
                        fixtures.append(
                            tuple(t.to(device) for t in (*cpu[p], xq, xs, wq, ws))
                        )
                    x, xs, w, ws = (torch.empty_like(t) for t in fixtures[0][4:])
                    out = torch.empty(b, n, dtype=torch.bfloat16, device=device)

                    def prep(p):
                        fc = fixtures[p]
                        ctx.prepare_fp4(*fc[:4])
                        for dst, src in zip((x, xs, w, ws), fc[4:]):
                            dst.copy_(src)

                    references = []
                    for p in range(4):
                        prep(p)
                        Fp4DispatchSharedGemm(
                            ctx,
                            x,
                            xs.t(),
                            w,
                            ws.t(),
                            out,
                            override_backend="mp32_serial",
                        )
                        torch.musa.synchronize()
                        reference = out.cpu()
                        references.append(reference)
                        Fp4DispatchSharedGemm(ctx, x, xs.t(), w, ws.t(), out)
                        torch.musa.synchronize()

                        def check_values(p=p, reference=reference):
                            verify(ctx.result, peers_by_phase[p], rank, e)
                            torch.testing.assert_close(
                                out.cpu(), reference, rtol=0, atol=0
                            )
                            d = out.cpu().float() - expected[p]
                            assert float(d.abs().max()) <= 0.02 + 0.004 * float(
                                expected[p].abs().max()
                            )
                            assert (
                                float(
                                    d.square().mean().sqrt()
                                    / expected[p]
                                    .square()
                                    .mean()
                                    .sqrt()
                                    .clamp_min(1e-30)
                                )
                                <= 0.004
                            )

                        checked(check_values)
                    prep(0)
                    torch.musa.synchronize()
                    dist.barrier()
                    graph = torch.musa.MUSAGraph()
                    snapshots = []
                    with torch.musa.graph(graph):
                        for p in range(4):
                            prep(p)
                            result, _ = Fp4DispatchSharedGemm(
                                ctx, x, xs.t(), w, ws.t(), out
                            )
                            snapshots.append((result.clone(), out.clone()))
                    graphs.append(graph)
                    for repeat in range(4):
                        graph.replay()
                        torch.musa.synchronize()
                        for p, (dispatch, gateup) in enumerate(snapshots):
                            checked(
                                lambda p=p, dispatch=dispatch, gateup=gateup: (
                                    verify(dispatch, peers_by_phase[p], rank, e),
                                    torch.testing.assert_close(
                                        gateup.cpu(), references[p], rtol=0, atol=0
                                    ),
                                )
                            )
                    record = dict(
                        b=b,
                        h=h,
                        n=n,
                        padded=padded,
                        queued_changed_input_invocations=16,
                        native_fused_bitwise_exact=True,
                        dispatch_byte_exact=True,
                        cpu_reference_bf16_nrmse_limit=0.004,
                    )
                    records.append(record)
                    if rank == 0:
                        print("PASS", record, flush=True)
                finally:
                    for g in graphs:
                        g.reset()
                    torch.musa.synchronize()
                    ctx.close()
        if rank == 0:
            Path(
                os.environ.get("RESULT_PATH", "/tmp/dispatch-shared-correctness.json")
            ).write_text(json.dumps(dict(world=world, records=records), indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
