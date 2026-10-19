"""Replay fixed inputs through the historical fixed-concurrency HTTP benchmark."""

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys


def load_historical(directory):
    script = directory.resolve() / "benchmark_serving_parallel.py"
    # Import the existing benchmark as a module instead of editing/executing a
    # rewritten source string. Its HTTP client and metric functions stay intact.
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("joyai_historical_benchmark", script)
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    return benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:38642")
    parser.add_argument("--model", default="joyai-tp2")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prefix-cache-rate", type=float, default=0.6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(args.requests.read_text())
    assert fixture and len({r["prompt_len"] for r in fixture}) == 1
    assert len({r["output_len"] for r in fixture}) == 1
    benchmark = load_historical(args.historical_dir)
    original = benchmark.ASYNC_REQUEST_FUNCS["openai-chat"]

    def replay(**kwargs):
        assert kwargs["num_requests"] == len(fixture)
        return [(r["prompt"], r["prompt_len"], r["output_len"]) for r in fixture]

    def record(*a, **kw):
        result = original(*a, **kw)
        request = a[0] if a else kw["request_func_input"]
        with (args.output / "requests.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "request": dataclasses.asdict(request),
                        "result": dataclasses.asdict(result),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return result

    benchmark.sample_rand_requests_with_prefix_cache_rate = replay
    benchmark.ASYNC_REQUEST_FUNCS["openai-chat"] = record
    values = dict(
        backend="openai-chat",
        base_url=args.url.rstrip("/"),
        host="127.0.0.1",
        port=38642,
        endpoint="/v1/chat/completions",
        dataset=None,
        dataset_name="random",
        dataset_path=None,
        model=args.model,
        tokenizer=args.tokenizer,
        best_of=1,
        use_beam_search=False,
        num_prompts=len(fixture),
        min_input_len=256,
        max_input_len=256,
        min_output_len=256,
        max_output_len=256,
        input_len=fixture[0]["prompt_len"],
        output_len=fixture[0]["output_len"],
        prefix_len=200,
        prefix_cache_rate=args.prefix_cache_rate,
        request_rate=args.concurrency,
        seed=940105,
        trust_remote_code=True,
        disable_tqdm=True,
        save_result=True,
        metadata=None,
        result_dir=str(args.output),
    )
    (args.output / "config.json").write_text(json.dumps(values, indent=2))
    benchmark.main(argparse.Namespace(**values))
    records = [
        json.loads(line)
        for line in (args.output / "requests.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(fixture)
    assert all(r["result"]["success"] and r["result"]["itl"] for r in records), (
        "Failed/empty streaming responses"
    )


if __name__ == "__main__":
    main()
