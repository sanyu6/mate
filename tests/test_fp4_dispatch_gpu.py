"""EP1/EP4 dispatch oracle and graph input-lifetime regression; no profiler."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from mate.ops.moe import Fp4Dispatch, Fp4DispatchBuffer
from fp4_dispatch_reference import dispatch_reference

try:
    import torch_musa
except ImportError:
    torch_musa = None


@pytest.mark.parametrize("ranks", [1, 4])
def test_dispatch_graph_updates(ranks):
    if (
        torch_musa is None
        or not torch.musa.is_available()
        or torch.musa.device_count() < ranks
    ):
        pytest.skip("requires available MP32 GPUs")
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
        timeout=600,
    )


def fixture(rank, b, h, e, k, phase):
    x = (
        torch.arange(b * (h // 2), dtype=torch.int64).reshape(b, h // 2)
        + rank * 19
        + phase * 37
    ).to(torch.uint8)
    sf = (
        torch.arange(b * (h // 32), dtype=torch.int64).reshape(b, h // 32)
        + rank * 13
        + phase * 7
    ).to(torch.uint8)
    ids = (
        (
            torch.arange(b)[:, None] * 7
            + torch.arange(k)[None, :] * 11
            + rank * 5
            + phase * 3
        )
        % e
    ).long()
    if phase == 1:
        ids = torch.arange(k).expand(b, k).clone()
    elif phase == 2:
        ids.fill_(-1)
    elif phase == 3:
        ids[:, ::2] = -1
    weights = (torch.arange(b * k).reshape(b, k).float() + rank + phase + 1) / 31
    if phase == 1:
        weights = -weights
    return x, sf, ids, weights


def verify(result, peers, rank, experts, local_invalid=False):
    rows, offsets, _ = dispatch_reference(peers, rank, experts)
    assert result.status.cpu().item() == int(local_invalid)
    assert result.counts.cpu().tolist() == [len(v) for v in rows]
    assert result.offsets.cpu().tolist() == offsets
    metadata = result.metadata.cpu().tolist()
    valid = [offsets[e] + j for e, rs in enumerate(rows) for j in range(len(rs))]
    if not valid:
        return
    idx = torch.tensor(valid, device=result.payload.device)
    payload = result.payload.index_select(0, idx).cpu().tolist()
    sidx = idx // 32 * 128 + idx % 32 if result.line_padded_scales else idx
    scales = result.scales.index_select(1, sidx).t().cpu().tolist()
    pos = 0
    for e, expected in enumerate(rows):
        want = {tuple(meta): (x, sf) for meta, x, sf in expected}
        seen = set()
        source_order = []
        for j in range(len(expected)):
            meta = tuple(metadata[offsets[e] + j])
            assert meta in want and meta not in seen
            assert (payload[pos], scales[pos]) == want[meta]
            seen.add(meta)
            source_order.append(meta[0])
            pos += 1
        assert len(seen) == len(expected)
        assert source_order == sorted(source_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--output", default="/tmp/dispatch-result.json")
    parser.add_argument("--backend", default=None, choices=("mp32", "mp32_pull"))
    parser.add_argument("--num-ctas", type=int, default=None)
    args = parser.parse_args()

    def dispatch(buf, tokens):
        return Fp4Dispatch(
            buf, tokens, override_backend=args.backend, num_ctas=args.num_ctas
        )

    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.set_num_threads(2)
    torch.musa.set_device(local)
    dist.init_process_group("gloo")
    records = []

    def checked(fn):
        err = None
        try:
            fn()
        except Exception as exc:
            err = repr(exc)
        errors = [None] * world
        dist.all_gather_object(errors, err)
        if any(errors):
            raise AssertionError(errors)

    try:
        for b, h, e, k in (
            (1, 256, 32, 2),
            (24, 256, 32, 2),
            (128, 256, 32, 2),
            (24, 5120, 384, 6),
        ):
            for padded in (False, True):
                buf = Fp4DispatchBuffer(
                    dist.group.WORLD, e, h, k, line_padded_scales=padded
                )
                s = buf._sym
                graphs = []
                try:
                    cpu = [fixture(rank, b, h, e, k, p) for p in range(4)]
                    local_peers = [
                        dict(
                            payload=x.tolist(),
                            scales=sf.tolist(),
                            ids=ids.tolist(),
                            weights=w.tolist(),
                        )
                        for x, sf, ids, w in cpu
                    ]
                    all_peers = [None] * world
                    dist.all_gather_object(all_peers, local_peers)
                    peers = [[all_peers[r][p] for r in range(world)] for p in range(4)]
                    gpu = [tuple(t.to(s.buffer.device) for t in v) for v in cpu]

                    def prep(phase):
                        buf.prepare_fp4(*gpu[phase])

                    prep(0)
                    got = dispatch(buf, b)
                    torch.musa.synchronize()
                    checked(lambda: verify(got, peers[0], rank, e))
                    graph = torch.musa.MUSAGraph()
                    pending = []
                    torch.musa.synchronize()
                    dist.barrier()
                    with torch.musa.graph(graph):
                        for step in range(8):
                            prep(step % 4)
                            pending.append(dispatch(buf, b).clone())
                    graphs.append(graph)
                    for repeat in range(4):
                        graph.replay()
                        torch.musa.synchronize()
                        for step, output in enumerate(pending):
                            checked(
                                lambda output=output, step=step: verify(
                                    output, peers[step % 4], rank, e
                                )
                            )
                    # Invalid IDs are reported without abandoning collective synchronization.
                    s.topk_idx[:b].fill_(e + 1)
                    got = dispatch(buf, b)
                    torch.musa.synchronize()
                    checked(
                        lambda: (
                            torch.testing.assert_close(
                                got.status.cpu(), torch.ones(1, dtype=torch.int32)
                            ),
                            torch.testing.assert_close(
                                got.counts.cpu(),
                                torch.zeros(e // world, dtype=torch.int32),
                            ),
                        )
                    )
                    prep(0)
                    got = dispatch(buf, b)
                    torch.musa.synchronize()
                    checked(lambda: verify(got, peers[0], rank, e))
                    if b == 24 and h == 5120:
                        # Seed the per-CTA epoch near uint32 wrap, then cross it
                        # without host synchronization between dispatch calls.
                        control_bytes = s.common_mps * (world + 1) * 128
                        begin = s.layout.offsets["combine"]
                        s.buffer[begin : begin + control_bytes].view(torch.int32).fill_(
                            -4
                        )
                        torch.musa.synchronize()
                        dist.barrier()
                        wrapped = []
                        for _ in range(4):
                            wrapped.append(dispatch(buf, b).clone())
                        torch.musa.synchronize()
                        for output in wrapped:
                            checked(
                                lambda output=output: verify(output, peers[0], rank, e)
                            )
                        # Active batch and default CTA count change within one
                        # captured sequence, while the IPC allocation stays fixed.
                        changes = []
                        for step, active in enumerate((24, 1, 128, 24)):
                            fc = fixture(rank, active, h, e, k, step % 2)
                            peer = dict(
                                payload=fc[0].tolist(),
                                scales=fc[1].tolist(),
                                ids=fc[2].tolist(),
                                weights=fc[3].tolist(),
                            )
                            expected = [None] * world
                            dist.all_gather_object(expected, peer)
                            device_fc = tuple(t.to(s.buffer.device) for t in fc)
                            buf.prepare_fp4(*device_fc)
                            dispatch(buf, active)
                            changes.append((active, device_fc, expected))
                        torch.musa.synchronize()
                        dist.barrier()
                        mixed = torch.musa.MUSAGraph()
                        snapshots = []
                        with torch.musa.graph(mixed):
                            for active, fc, expected in changes:
                                buf.prepare_fp4(*fc)
                                snapshots.append(dispatch(buf, active).clone())
                        graphs.append(mixed)
                        for _ in range(4):
                            mixed.replay()
                            torch.musa.synchronize()
                            for output, (_, _, expected) in zip(snapshots, changes):
                                checked(
                                    lambda output=output, expected=expected: verify(
                                        output, expected, rank, e
                                    )
                                )
                        prep(0)
                        got = dispatch(buf, b)
                        torch.musa.synchronize()
                        checked(lambda: verify(got, peers[0], rank, e))
                    rec = dict(
                        b=b,
                        h=h,
                        experts=e,
                        topk=k,
                        padded=padded,
                        queued_graph_invocations=32,
                        exact=True,
                    )
                    if b == 24 and h == 5120:
                        rec.update(
                            epoch_wrap_checked=True, mixed_batch_graph_invocations=16
                        )
                    if args.bench and b == 24 and h == 5120:
                        timing_graph = torch.musa.MUSAGraph()
                        with torch.musa.graph(timing_graph):
                            for _ in range(8):
                                dispatch(buf, b)
                        graphs.append(timing_graph)
                        flush = torch.empty(
                            256 * 1024 * 1024, dtype=torch.uint8, device=s.buffer.device
                        )
                        start, end = (
                            torch.musa.Event(enable_timing=True) for _ in range(2)
                        )
                        samples = []
                        for j in range(25):
                            torch.musa.synchronize()
                            dist.barrier()
                            flush.zero_()
                            start.record()
                            timing_graph.replay()
                            end.record()
                            end.synchronize()
                            if j >= 5:
                                samples.append(start.elapsed_time(end) / 8)
                        times = [None] * world
                        dist.all_gather_object(times, samples)
                        rec["dispatch_only_median_ms"] = statistics.median(
                            max(t[j] for t in times) for j in range(20)
                        )
                        rec["rank_samples_ms"] = times
                        rec["scope"] = (
                            "prepared FP4 IPC -> local expert pools, including retirement; input preparation excluded"
                        )
                    records.append(rec)
                    if rank == 0:
                        print("PASS", rec, flush=True)
                finally:
                    for graph in graphs:
                        graph.reset()
                    torch.musa.synchronize()
                    buf.close()
        if rank == 0:
            Path(args.output).write_text(
                json.dumps(dict(world=world, records=records), indent=2)
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
