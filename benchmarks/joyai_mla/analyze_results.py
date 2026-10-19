"""Validate paired inputs and summarize only the final fresh-process runs."""

import argparse
import gzip
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--requests", type=Path, required=True)
parser.add_argument("--rounds", type=int, default=3)
args = parser.parse_args()
ROOT = args.results
canonical = json.loads(args.requests.read_text())
input_keys = ("prompt", "prompt_len", "output_len")
canonical_inputs = [{k: r[k] for k in input_keys} for r in canonical]


def percentile(values, fraction):
    a = sorted(values)
    offset = (len(a) - 1) * fraction
    lo = int(offset)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (offset - lo)


summary = {}
for mode in ["default_final", "optimized_final"]:
    directory = ROOT / mode
    records = []
    rounds = []
    for iteration in range(1, args.rounds + 1):
        case = directory / f"round{iteration}"
        row = json.loads(next(case.glob("openai-chat*.json")).read_text())
        assert row["completed"] == 5
        request_records = [
            json.loads(line)
            for line in (case / "requests.jsonl").read_text().splitlines()
        ]
        assert [
            {k: r["request"][k] for k in input_keys} for r in request_records
        ] == canonical_inputs, mode
        assert all(r["result"]["success"] for r in request_records)
        tpots = [
            (r["result"]["latency"] - r["result"]["ttft"])
            / len(r["result"]["itl"])
            * 1000
            for r in request_records
        ]
        assert abs(statistics.mean(tpots) - row["mean_tpot_ms"]) < 1e-8
        records.extend(request_records)
        rounds.append(row)
    tpots = [
        (r["result"]["latency"] - r["result"]["ttft"]) / len(r["result"]["itl"]) * 1000
        for r in records
    ]
    profiles = []
    for path in sorted((directory / "profile").glob("*DECODE*")):
        events = json.load(gzip.open(path, "rt"))["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        main = [e for e in kernels if "MlaKernelTmeWarpSpecialized" in e["name"]]
        combine = [e for e in kernels if "mla_combine" in e["name"]]
        assert main and combine, path
        steps = len(main) / 40
        assert steps == 8 and len(combine) == len(main), (path, steps, len(combine))
        assert any("combine_parallel_kernel" in e["name"] for e in combine) == (
            mode == "optimized_final"
        )
        profiles.append(
            {
                "file": str(path.relative_to(ROOT)),
                "steps": steps,
                "main_ms_per_step": sum(e["dur"] for e in main) / steps / 1000,
                "combine_ms_per_step": sum(e["dur"] for e in combine) / steps / 1000,
                "main_kernel": main[0]["name"],
                "combine_kernel": combine[0]["name"],
            }
        )
    assert len(profiles) == 2, profiles
    summary[mode] = {
        "requests": len(tpots),
        "round_mean_tpot_ms": [r["mean_tpot_ms"] for r in rounds],
        "mean_tpot_ms": statistics.mean(tpots),
        "median_tpot_ms": statistics.median(tpots),
        "p90_tpot_ms": percentile(tpots, 0.9),
        "p99_tpot_ms": percentile(tpots, 0.99),
        "mean_ttft_ms": statistics.mean(r["result"]["ttft"] * 1000 for r in records),
        "output_throughput": sum(r["total_output_tokens"] for r in rounds)
        / sum(r["duration"] for r in rounds),
        "streamed_output_counts": [len(r["result"]["itl"]) + 1 for r in records],
        "profiles": profiles,
    }

before, after = summary["default_final"], summary["optimized_final"]
summary["speedup"] = before["mean_tpot_ms"] / after["mean_tpot_ms"]
summary["latency_reduction_pct"] = (
    1 - after["mean_tpot_ms"] / before["mean_tpot_ms"]
) * 100
old = json.loads((ROOT / "default_final/greedy.json").read_text())
new = json.loads((ROOT / "optimized_final/greedy.json").read_text())
a, b = (
    old["meta_info"]["output_token_logprobs"],
    new["meta_info"]["output_token_logprobs"],
)
summary["greedy_smoke"] = {
    "prompt_tokens": [
        old["meta_info"]["prompt_tokens"],
        new["meta_info"]["prompt_tokens"],
    ],
    "output_tokens": [
        old["meta_info"]["completion_tokens"],
        new["meta_info"]["completion_tokens"],
    ],
    "text_equal": old["text"] == new["text"],
    "token_ids_equal": [x[1] for x in a] == [x[1] for x in b],
    "interpretation": "generation consistency probe, not a numerical accuracy verdict",
}
shared_prefix = 0
for x, y in zip(a, b):
    if x[1] != y[1]:
        break
    shared_prefix += 1
summary["greedy_smoke"]["shared_token_prefix"] = shared_prefix
summary["greedy_smoke"]["max_logprob_difference_shared_prefix"] = max(
    (abs(x[0] - y[0]) for x, y in zip(a[:shared_prefix], b[:shared_prefix])),
    default=0.0,
)
default_repeat = json.loads((ROOT / "default_final/profile_request.json").read_text())
repeat_probs = default_repeat["meta_info"]["output_token_logprobs"]
repeat_prefix = 0
for x, y in zip(a, repeat_probs):
    if x[1] != y[1]:
        break
    repeat_prefix += 1
summary["default_repeat_observation"] = {
    "shared_token_prefix": repeat_prefix,
    "cached_tokens": [
        old["meta_info"]["cached_tokens"],
        default_repeat["meta_info"]["cached_tokens"],
    ],
    "controlled_aa": False,
    "reason": "the repeat had a different cache hit and profiler enabled",
}
(ROOT / "final_results.json").write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {
            k: {a: b for a, b in v.items() if a != "profiles"}
            if isinstance(v, dict)
            else v
            for k, v in summary.items()
        },
        indent=2,
    )
)
