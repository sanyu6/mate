"""Check recorded real-input A/A and FP32-reference attention comparisons."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    paths = sorted(args.input.glob("attention-numerics-*.jsonl"))
    if not paths:
        parser.error("No recorded attention comparisons found")
    for p in paths:
        rows.extend(json.loads(line) for line in p.read_text().splitlines())
    result = {
        "cases": len(rows),
        "ranks": len(paths),
        "aa_exact": sum(r["aa_exact"] for r in rows),
        "ab_exact": sum(r["ab"]["max_abs"] == 0 for r in rows),
        "default_fp32_passes": sum(r["close_default_ref"] for r in rows),
        "optimized_fp32_passes": sum(r["close_optimized_ref"] for r in rows),
        "kv_lengths": sorted({r["kv_length"] for r in rows}),
        "scope": "real-input attention, baseline output propagated; not model task accuracy",
    }
    for key in ["aa", "ab", "default_fp32", "optimized_fp32", "lse_fp32"]:
        result[key] = {
            m: max(r[key][m] for r in rows)
            for m in ["max_abs", "relative_l2", "mean_abs"]
        }
        assert all(r[key]["finite"] for r in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert (
        result["aa_exact"]
        == result["default_fp32_passes"]
        == result["optimized_fp32_passes"]
        == len(rows)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
