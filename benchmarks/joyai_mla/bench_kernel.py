import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torch_musa  # noqa: F401
import tvm_ffi
import mate
import mate.mha_interface as mi

parser = argparse.ArgumentParser(
    description="Original MLA+combine vs optimized MLA+combine"
)
parser.add_argument("--baseline-library", required=True)
parser.add_argument("--optimized-library", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5])
parser.add_argument(
    "--lengths", type=int, nargs="+", default=[6000, 7000, 8000, 9000, 10000]
)
parser.add_argument("--heads", type=int, default=16)
parser.add_argument("--samples", type=int, default=12)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=False)
modules = {
    "original": tvm_ffi.load_module(args.baseline_library),
    "optimized": tvm_ffi.load_module(args.optimized_library),
}
labels = list(modules)
results = []
print(
    json.dumps(
        {
            "mate": mate.__file__,
            "device": str(torch.musa.get_device_properties(0)),
            "labels": labels,
            "baseline": "original M128 MLA and original combine at bbcd8b8",
            "metadata": "generated once with original module and reused unchanged",
            "candidate": "M32 two-squad pipelined MLA and parallel combine",
            "samples": args.samples,
            "rotating_measurement_order": True,
        }
    ),
    flush=True,
)

for batch in args.batches:
    for length in args.lengths:
        torch.manual_seed(42)
        heads, page = args.heads, 64
        pages = (length + page - 1) // page
        ql = torch.randn(batch, 1, heads, 512, device="musa", dtype=torch.bfloat16)
        qr = torch.randn(batch, 1, heads, 64, device="musa", dtype=torch.bfloat16)
        kv = torch.randn(
            batch * pages, page, 1, 512, device="musa", dtype=torch.bfloat16
        )
        kr = torch.randn(
            batch * pages, page, 1, 64, device="musa", dtype=torch.bfloat16
        )
        lens = torch.full((batch,), length, device="musa", dtype=torch.int32)
        table = torch.arange(batch * pages, device="musa", dtype=torch.int32).view(
            batch, pages
        )
        workspace = torch.empty(16 * 1024 * 1024, device="musa", dtype=torch.uint8)
        latent = kv.view(batch, pages * page, 512)[:, :length].float()
        rope = kr.view(batch, pages * page, 64)[:, :length].float()
        logits = (
            ql[:, 0].float() @ latent.transpose(1, 2)
            + qr[:, 0].float() @ rope.transpose(1, 2)
        ) * 576**-0.5
        ref = (logits.softmax(-1) @ latent).to(torch.bfloat16).unsqueeze(1)
        ref_lse = logits.logsumexp(-1).unsqueeze(1)

        def run(label, initialized=True):
            mi._get_mla_ops = lambda: modules[label]
            return mate.flash_attn_with_kvcache(
                qr,
                kr,
                kv,
                qv=ql,
                cache_seqlens=lens,
                page_table=table,
                max_seqlen_q=1,
                softmax_scale=576**-0.5,
                causal=False,
                scheduler_metadata=(workspace, initialized),
                return_softmax_lse=True,
            )[:2]

        run("original", False)
        _, splits = mi._prepare_mla_scheduler_metadata_from_workspace(
            workspace, True, ql, kv, lens, None, 1
        )
        split_counts = splits.cpu().tolist()
        graphs, held = {}, {}
        outputs = {}
        for label in labels:
            out, lse = run(label)
            torch.testing.assert_close(out, ref, atol=1.5e-2, rtol=1e-2)
            torch.testing.assert_close(lse, ref_lse, atol=1.5e-2, rtol=1e-2)
            outputs[label] = out.clone()
            for _ in range(5):
                run(label)
            torch.musa.synchronize()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph):
                held[label] = [run(label) for _ in range(20)]
            graphs[label] = graph

        samples = {label: [] for label in labels}
        for iteration in range(args.samples):
            order = (
                labels[iteration % len(labels) :] + labels[: iteration % len(labels)]
            )
            for label in order:
                graph = graphs[label]
                for _ in range(3):
                    graph.replay()
                start, end = (
                    torch.musa.Event(enable_timing=True),
                    torch.musa.Event(enable_timing=True),
                )
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[label].append(start.elapsed_time(end) * 1000 / 200)
        row = {
            "batch": batch,
            "kv_length": length,
            "graph_us": {label: statistics.median(samples[label]) for label in labels},
            "samples_us": samples,
            "ab_max_abs": (outputs["optimized"].float() - outputs["original"].float())
            .abs()
            .max()
            .item(),
            "split_offsets": split_counts,
            "speedup": statistics.median(samples["original"])
            / statistics.median(samples["optimized"]),
        }
        for label in labels:
            for out, lse in held[label]:
                torch.testing.assert_close(out, ref, atol=1.5e-2, rtol=1e-2)
                torch.testing.assert_close(lse, ref_lse, atol=1.5e-2, rtol=1e-2)
        if length == 10000:
            row["kernels"] = {}
            for label in labels:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.MUSA]
                ) as profiler:
                    for _ in range(20):
                        run(label)
                    torch.musa.synchronize()
                path = OUT / f"{label}_B{batch}_S{length}.json"
                profiler.export_chrome_trace(str(path))
                events = json.loads(path.read_text())["traceEvents"]
                grouped = {}
                for event in events:
                    if event.get("cat") == "kernel":
                        key = "combine" if "combine" in event["name"] else "main"
                        grouped.setdefault(key, []).append(event["dur"])
                row["kernels"][label] = {
                    key: statistics.mean(values) for key, values in grouped.items()
                }
        results.append(row)
        print(json.dumps(row), flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

with (OUT / "results.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "batch",
            "kv_length",
            "original_us",
            "optimized_us",
            "speedup",
            "latency_reduction_pct",
        ]
    )
    for row in results:
        values = row["graph_us"]
        writer.writerow(
            [
                row["batch"],
                row["kv_length"],
                values["original"],
                values["optimized"],
                row["speedup"],
                (1 - values["optimized"] / values["original"]) * 100,
            ]
        )
print("COMPLETE", len(results), flush=True)
