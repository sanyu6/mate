---
name: add-mate-benchmark
description: Add, port, review, and validate public MATE benchmark routines under benchmarks/routines. Use when Codex needs to add an operator benchmark, register a new -R routine, define MUSA inputs/backends, add logical load/store/FLOP metrics in mate.testing.metrics, wire optional Kineto profiling, or produce Docker-tested benchmark evidence for this framework.
---

# Add MATE Benchmark

Create benchmark routines that fit the unified MATE runner instead of adding one-off scripts. A good routine exposes a reproducible workload through `benchmarks/mate_benchmark.py -R ...`, validates arguments early, constructs representative MUSA tensors, reports logical metrics through `mate.testing.metrics`, and prints the established `[PERF]` line.

## First Reads

1. Read `AGENTS.md` and any nearer `AGENTS.md` before editing.
2. Read the target public MATE API, its tests, and the closest existing routine in `benchmarks/routines/`.
3. Read `benchmarks/mate_benchmark.py`, `benchmarks/routines/mate_benchmark_utils.py`, `mate/testing/metrics.py`, and `mate/testing/utils.py`.
4. If porting from another project, use that benchmark for semantic coverage only. Keep this repository's runner, parser, output, and timing conventions.

## Routine Shape

1. Add or extend one routine family file under `benchmarks/routines/`, such as `gemm.py`, `attention.py`, `mla.py`, `gdn.py`, or `kda.py`.
2. Register the routine name in `benchmark_apis` and its supported MP architecture/backends in `routine_arch_to_supported_backends`.
3. Route the routine from the family `run_*_test(args)` function. Add a family import and route in `benchmarks/mate_benchmark.py` only when creating a new family.
4. Add routine-specific parser branches through the family `parse_*_args(line, parser)` function. Keep choices, defaults, dtype names, and shape constraints explicit.
5. Filter requested backends with `filter_backends_by_arch([args.backend], args.routine, device)` before constructing backend-specific timing loops.

## Inputs And Correctness

1. Construct tensors on `torch.device("musa")`; do not silently benchmark CUDA or CPU fallback paths.
2. Use the shared random seed from the runner. Prefer nonzero random inputs for active paths so optional branches are not accidentally masked.
3. Build real metadata for the workload: sequence lengths, `cu_seqlens`, page tables, cache indices, expert masks, grouped scales, and packed layouts should match the API contract.
4. Resolve actual workload sizes from the generated data. For variable length or sparse cases, count actual tokens, valid rows, valid indices, or touched state slots.
5. Run a warmup call before timing to trigger JIT/source generation, compilation, and module loading. Synchronize before starting steady-state timing.
6. Add a reference check when practical. Compare all semantically meaningful outputs, including auxiliary outputs such as states, LSE, max logits, indices, or packed outputs. Choose tolerances by dtype and quantization path.

## Metrics

1. Put reusable metric formulas in `mate/testing/metrics.py`, not only in a routine file.
2. Return `KernelMetrics(load_byte, store_byte, flop)` and call `metrics.to_perf(median_time)` before printing.
3. Separate logical algorithmic traffic from implementation traffic. Count logical reads and writes by semantic tensors and actual workload; document any deliberate implementation-specific addition.
4. Pass shape and granularity values explicitly when tensor shape alone is ambiguous, such as valid rows, total tokens, grouped scale granularity, retrieved KV bytes per token, or touched state slots.
5. Count load and store bytes separately. State tensors that are read and updated usually contribute to both sides; optional final-state tensors should only count when present.
6. Keep FLOP formulas tied to the mathematical workload, not temporary buffers or split-kernel implementation details.

## Timing And Output

1. Time the steady-state call with `mate.testing.utils.bench_gpu_time`, using the local routine convention for `repeat_iters` or `num_tests`.
2. Use `l2_flush=True`, `l2_flush_device="musa"`, and `sleep_after_run=True` unless the surrounding routine has a documented reason to differ.
3. Print only through `print_perf_metrics(backend, median_ms, std_ms, tflops, bandwidth_gb_per_sec)` so logs stay parseable.
4. Support the shared `--mupti-kernel-name STR` flag by calling `maybe_print_kineto_metrics(args, backend, run_fn, median_ms, tflops, bandwidth)` after the event-based `[PERF]` line.
5. Treat event timing and Kineto timing as separate evidence. Do not claim single-kernel Kineto performance unless a matching kernel name was actually profiled.

## Validation

Run the narrowest useful checks before handoff, then broaden when the routine touches shared code:

1. `python -m py_compile benchmarks/mate_benchmark.py benchmarks/routines/<family>.py mate/testing/metrics.py`
2. `python benchmarks/mate_benchmark.py --help` and one parser-only/help command for the new routine.
3. A small Docker smoke case in the `mate-benchmark` environment, from the repository checkout inside the container, with `PYTHONPATH` set if needed.
4. At least one representative backend/default performance run for every new routine path. Include a `--mupti-kernel-name` run when changing Kineto behavior.
5. Targeted pytest or reference-check coverage if the routine adds correctness logic or shared metric helpers.
6. `git diff --check` and the repository pre-commit command requested by `AGENTS.md`. Report unavailable Docker, MUSA, Kineto, or pre-commit checks as blockers instead of implying they passed.

## Handoff Notes

Report the routine name, changed files, exact validation commands, representative shapes/backends, and the observed `[PERF]` lines. Keep static formula review, parser smoke, event timing, Kineto timing, and correctness evidence clearly separated.
