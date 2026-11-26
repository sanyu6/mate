---
name: mate-jit-dump
description: Dump, inspect, replay, compare, and diagnose standalone MATE JIT artifacts produced from supported TVM-FFI calls. Use for pytest-driven JIT dumps, architecture selection, reference or baseline comparison, per-case replay, timeouts, load failures, and kernel-output mismatches.
---

# MATE JIT Dump

Use the repository workflow to dump selected pytest cases without invoking the
target FFI function. Replay and diagnose one standalone artifact at a time.

Read [`tools/jit_dump/README.md`](../../tools/jit_dump/README.md) before
acting. Treat it and the current source as the contract; do not reconstruct old
smoke scripts or assume fixed tensor ordinals.

## Establish the boundary

Before producing a dump:

- Use a MATE source checkout with the required kernel toolchain.
- Select concrete pytest node IDs. Multiple items may run serially in one
  pytest process.
- Do not enable pytest-xdist.
- Decide whether inputs alone are sufficient or registered references are
  required.
- Keep the target FFI function as the first supported dump point in each item.

Calculate references before the target call using an independent implementation,
then register them through `mate.jit.dump`.

## Produce artifacts

Prefer pytest options for interactive use:

```bash
pytest -s --jit-dump /tmp/jit-dumps TEST_NODE [TEST_NODE ...]
```

Set `MATE_MUSA_ARCH_LIST=ARCH` before pytest collection when the dumped kernel
needs a different architecture. JIT dump requires exactly one target in the
architecture list.

Add `--jit-dump-reference` only when the selected tests register every required
direct output. Environment variables are equivalent:

```bash
MATE_JIT_DUMP_DIR=/tmp/jit-dumps \
MATE_JIT_DUMP_REFERENCE=1 \
pytest -s TEST_NODE
```

Set options or variables before pytest collection. Each dumped item produces
`0000/`, `0001/`, and so on. A warning that an item did not produce a dump is an
integration or selection signal.

An existing output root is valid. The tool warns and overwrites known generated
files for the current case; do not delete unrelated files.

## Verify a dump

Read `manifest.json`. Verify `dump_arch`, `config_arch`, and `binary_arch`, and
map each numbered directory to its pytest node ID. Every completed case contains:

```text
DUMP_COMPLETE
kernel.so
loader.so
runtime/
replay.py
compare.py
run.sh
tensor_*.bin
```

When references are enabled, also verify `REFERENCE_COMPLETE`,
`references.json`, and every referenced `reference_*.bin` or `baseline_*.bin`.
No `output_*.bin` should exist before replay.

Use `replay.py` as the authoritative argument mapping. Optional arguments change
tensor numbering, so never infer semantics from names such as `tensor_5.bin`.

## Replay one case

Install the same TVM-FFI version used to produce the dump. The verified package
can be installed with:

```bash
python3 -m pip install \
  'apache_tvm_ffi==0.1.9.post3+musa.1' \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

Enter one numbered directory and run only that case:

```bash
./run.sh replay
./run.sh compare
```

Use `./run.sh` when replay and comparison need not be separated. Do not create
or invoke a root-level loop over every artifact. Finish diagnosis for one case
before proceeding to the next.

Interpret results in layers:

- `replay` returns zero: load, invocation, synchronization, and output saving
  completed.
- `compare` returns zero: every registered numerical comparison passed.
- `compare` returns one: execution completed but at least one output mismatched.
- timeout: locate whether progress stopped during load, invocation, or
  synchronization before attributing the failure to kernel logic.
- invalid-device or load failure: verify branch, `binary_arch`, toolchain,
  TVM-FFI ABI, loader, and bundled runtime before debugging kernel math.

The generated replay intentionally uses TVM-FFI and the bundled loader only; it
does not require Torch.

## Preserve evidence

Record:

- MATE branch and commit used for the dump.
- Selected pytest node ID and its manifest entry.
- Numbered artifact directory and hashes of relevant shared libraries.
- Replay and comparison commands with separate exit codes.
- Replay stdout/stderr and timeout duration.
- First mismatch, maximum error, and configured tolerance or baseline factors.

Minimize a failing case while preserving the same parameter semantics and kernel
specialization. Changing only a scalar in generated `replay.py` may be invalid
when the FFI wrapper derives dimensions from tensor shapes.

Do not start, restart, or stop shared execution infrastructure without explicit
user authorization. Do not delete user artifact roots or logs while diagnosing
them.
