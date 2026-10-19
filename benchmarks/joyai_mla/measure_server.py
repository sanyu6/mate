"""Warm and measure one server with the user-selected historical protocol."""

import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--requests", type=Path, required=True)
parser.add_argument("--historical-dir", type=Path, required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--model", default="joyai-tp2")
parser.add_argument("--url", default="http://127.0.0.1:38642")
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--supplemental", action="store_true")
args = parser.parse_args()
MODE = args.output.name
OUT = args.output
OUT.mkdir(parents=True, exist_ok=args.supplemental)
URL = args.url.rstrip("/")
PYTHON = sys.executable

for _ in range(600):
    try:
        if requests.get(URL + "/health", timeout=2).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(2)
else:
    raise RuntimeError("Server did not become healthy")

info = requests.get(URL + "/get_server_info", timeout=30)
info.raise_for_status()
(OUT / "server_info.json").write_text(json.dumps(info.json(), indent=2))

labels = (
    []
    if args.supplemental
    else ["warmup", *[f"round{i}" for i in range(1, args.rounds + 1)]]
)
for label in labels:
    if label != "warmup":
        response = requests.post(URL + "/flush_cache", timeout=30)
        response.raise_for_status()
    dest = OUT / label
    command = [
        PYTHON,
        str(ROOT / "run_benchmark.py"),
        "--historical-dir",
        str(args.historical_dir),
        "--requests",
        str(args.requests),
        "--output",
        str(dest),
        "--url",
        URL,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
    ]
    # The child creates its result directory; keep its log alongside it.
    with (OUT / f"{label}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (dest / "command.json").write_text(json.dumps(command, indent=2))
    results = list(dest.glob("openai-chat*.json"))
    assert len(results) == 1, results
    result = json.loads(results[0].read_text())
    assert result["completed"] == 5, result
    print(
        MODE,
        label,
        json.dumps(
            {
                k: result[k]
                for k in [
                    "mean_tpot_ms",
                    "median_tpot_ms",
                    "p90_tpot_ms",
                    "mean_ttft_ms",
                    "output_throughput",
                    "total_output_tokens",
                ]
            }
        ),
        flush=True,
    )

record = json.loads((OUT / "round1/requests.jsonl").read_text().splitlines()[0])
prompt = record["request"]["prompt"]
payload = {
    "text": prompt,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 32,
        "ignore_eos": True,
        "sampling_seed": 0,
    },
    "return_logprob": True,
    "logprob_start_len": -1,
}
response = requests.post(URL + "/generate", json=payload, timeout=300)
response.raise_for_status()
(OUT / "greedy.json").write_text(
    json.dumps(response.json(), ensure_ascii=False, indent=2)
)

profile = requests.post(
    URL + "/start_profile",
    json={
        "output_dir": str(OUT / "profile"),
        "num_steps": 8,
        "activities": ["CPU", "MUSA"],
        "profile_by_stage": True,
        "with_stack": False,
        "record_shapes": False,
        "profile_id": MODE,
    },
    timeout=60,
)
profile.raise_for_status()
response = requests.post(URL + "/generate", json=payload, timeout=300)
response.raise_for_status()
(OUT / "profile_request.json").write_text(
    json.dumps(response.json(), ensure_ascii=False, indent=2)
)
print(MODE, "COMPLETE", flush=True)
