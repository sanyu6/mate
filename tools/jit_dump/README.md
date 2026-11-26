# JIT dump and replay

MATE can dump the final TVM-FFI call of a supported JIT kernel without launching
the target function. Each dump is a standalone artifact that can be replayed
without Torch or MATE. Replay requires a compatible TVM-FFI Python package.

FMHA forward is the first supported integration.

## Dump with pytest

Select one or more pytest items and provide an artifact root:

```bash
pytest -s \
  --jit-dump /tmp/fmha-artifacts \
  'tests/test_fmha.py::test_varlen_func_bshd[param-1]' \
  'tests/test_fmha.py::test_varlen_func_bshd[param-2]'
```

The option sets `MATE_JIT_DUMP_DIR`. Setting the environment variable directly
is equivalent:

```bash
MATE_JIT_DUMP_DIR=/tmp/fmha-artifacts \
pytest -s 'tests/test_fmha.py::test_varlen_func_bshd[param-1]'
```

Set `MATE_MUSA_ARCH_LIST` to select the architecture of the dumped kernel:

```bash
MATE_MUSA_ARCH_LIST=3.2 pytest -s \
  --jit-dump /tmp/fmha-artifacts \
  'tests/test_fmha.py::test_varlen_func_bshd[param-1]'
```

JIT dump requires exactly one `MATE_MUSA_ARCH_LIST` target when the variable is
set. If it is unset, the dump defaults to the current device architecture. The
architecture fields have distinct meanings:

- `dump_arch` is selected by `MATE_MUSA_ARCH_LIST` and defaults to the current
  device architecture when the variable is unset.
- `config_arch` selects the operator configuration and dispatch path.
- `binary_arch` is the instruction-set architecture compiled into `kernel.so`.

When `dump_arch` differs from the current device architecture, an operator must
declare `config_arch` and `binary_arch` explicitly. FMHA metadata selects its
configuration using `dump_arch` while its helper binary keeps the current
compilation architecture. FMHA forward uses `dump_arch` for both its
configuration and kernel binary.

Set the environment variable before pytest collection. FMHA selects its dump
wrapper when the module is imported.

Every item that reaches a supported target call produces the next numbered
artifact. Items that do not produce a dump emit a warning and retain their
normal pytest result.

```text
/tmp/fmha-artifacts/
├── manifest.json
├── 0000/
│   ├── DUMP_COMPLETE
│   ├── kernel.so
│   ├── loader.so
│   ├── runtime/
│   ├── tensor_*.bin
│   ├── replay.py
│   ├── benchmark.py
│   ├── compare.py
│   └── run.sh
└── 0001/
    └── ...
```

`manifest.json` records `dump_arch` and maps each numbered directory to its
pytest node ID, FFI function, `config_arch`, and `binary_arch`. An existing
output root is accepted with a warning. Known generated files for the current
case are overwritten; unrelated files are left unchanged.

JIT dump does not support active pytest-xdist workers. Multiple items can be
selected in one ordinary pytest process.

## Dump references

Add `--jit-dump-reference` or set `MATE_JIT_DUMP_REFERENCE=1` to save reference
outputs:

```bash
pytest -s \
  --jit-dump /tmp/fmha-artifacts \
  --jit-dump-reference \
  'tests/test_fmha.py::test_varlen_func_bshd[param-1]'
```

The test must calculate and register each reference before calling the target
function:

```python
from mate.jit.dump import register_dump_reference

out_ref, lse_ref = compute_reference_before_target_call()
register_dump_reference("out", out_ref, atol=1.5e-2, rtol=1e-2)
register_dump_reference("lse", lse_ref, atol=1.5e-2, rtol=1e-2)
call_target_kernel()
```

Tests that compare against the error of another implementation can register a
baseline:

```python
from mate.jit.dump import register_dump_reference_with_baseline

register_dump_reference_with_baseline(
    "out",
    out_ref,
    baseline,
    max_error_factor=4,
    mean_error_factor=3,
)
```

Reference names must match the direct output names configured by the JIT
integration. References and baselines must have the same shapes as those direct
outputs. A later high-level combine step cannot be compared with a single
dumped FFI call.

## Replay one case

Install a compatible TVM-FFI package. The version used to verify this workflow
is:

```bash
python3 -m pip install \
  'apache_tvm_ffi==0.1.9.post3+musa.1' \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

Use the same TVM-FFI version as the dump environment to keep the Python
extension and `loader.so` ABI compatible. Enter one numbered directory and run
that case:

```bash
cd /path/to/fmha-artifacts/0000
./run.sh replay
./run.sh compare
```

`./run.sh` runs both operations. `replay` loads `kernel.so`, invokes the dumped
function, synchronizes, and writes `output_*.bin`. `compare` reads existing
outputs and `references.json` without invoking the kernel again. A zero replay
exit code means execution completed; a zero compare exit code means every
registered comparison passed.

Artifacts are intentionally single-case. Run and diagnose one numbered
directory at a time.

## Benchmark one case

The artifact can benchmark the dumped TVM-FFI call without Torch or MATE. The
warmup calls are completed before timing. Each profiled call is measured with a
pair of MUSA events, and the device is synchronized once after all profiled
calls have been submitted:

```bash
cd /path/to/fmha-artifacts/0000
./run.sh bench --warmup-iters 10 --profile-iters 100
```

`--warmup-iters` defaults to 10 and accepts zero; `--profile-iters` defaults to
100 and must be positive. The command prints one JSON object containing the
minimum, median, mean, p90, and maximum device latency in microseconds.

Pass the theoretical floating-point operation count for one call to report the
per-iteration throughput distribution as well:

```bash
./run.sh bench --warmup-iters 10 --profile-iters 100 --flops 26388279066624
```

The JSON then also contains `workload_flops` and `throughput_tflops`. TFLOPS
uses decimal teraFLOPS (`FLOP/s / 10^12`). The caller is responsible for using
the FLOP convention appropriate for the dumped operator.

Benchmark mode loads inputs once and reuses the same tensors for every call. It
does not save outputs or run reference comparison. The dumped function must be
safe to invoke repeatedly with the same arguments. L2 cache flushing is not
currently performed.

## Loader and runtime

The dump process builds `tools/jit_dump/loader.cpp` through the MATE JIT system
and bundles the active MUSA runtime library. The loader provides tensor I/O,
device synchronization, and MUSA event timing. The automatic build requires a
MATE source checkout. Installed packages can provide prebuilt files with:

```bash
MATE_JIT_DUMP_LOADER_SO=/path/to/loader.so
MATE_JIT_DUMP_MUSART_SO=/path/to/libmusart.so.4
```

The loader can also be built explicitly:

```bash
tools/jit_dump/build_loader.sh /tmp/loader.so
```

JIT dump currently supports contiguous tensor arguments on one MUSA device.
The target FFI function is not invoked while producing the dump.
