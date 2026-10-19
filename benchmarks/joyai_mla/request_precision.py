"""Send a short baseline-driven request to a server in validation mode."""

import argparse
import json
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:38642")
    parser.add_argument("--tokens", type=int, default=4)
    args = parser.parse_args()
    for _ in range(600):
        try:
            if requests.get(args.url + "/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Server did not become healthy")
    fixture = json.loads(args.requests.read_text())
    payload = {
        "text": fixture[0]["prompt"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.tokens,
            "ignore_eos": True,
            "sampling_seed": 0,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    response = requests.post(args.url + "/generate", json=payload, timeout=600)
    response.raise_for_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        "Request complete; use check_numerics.py on the server validation output directory."
    )


if __name__ == "__main__":
    main()
