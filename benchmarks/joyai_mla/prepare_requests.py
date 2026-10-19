"""Generate the historical random requests once, then freeze them for A/B."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from run_benchmark import load_historical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=940105)
    args = parser.parse_args()
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    historical = load_historical(args.historical_dir)
    rows = historical.sample_rand_requests_with_prefix_cache_rate(
        input_len=20000,
        num_requests=5,
        max_output_len=300,
        tokenizer=tokenizer,
        prefix_cache_rate=0.6,
    )
    requests = [dict(prompt=p, prompt_len=n, output_len=o) for p, n, o in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(requests, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
