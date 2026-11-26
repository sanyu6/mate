# Agent Instructions for MATE

This file provides guidance to AI-assisted contributions to MATE.


## Quick Reference

| Task | Command |
|------|---------|
| Install for development | `pip install --no-build-isolation -e . -v` |
| Initialize submodules | `git submodule update --init --recursive` |
| Run all tests | `pytest tests/` |
| Run specific test | `pytest tests/path/test_file.py::test_function` |
| Run multi-GPU test | `mpirun -np 4 pytest tests/comm/test_allreduce_unified_api.py` |
| Run linting | `pre-commit run -a` |
| Install pre-commit hooks | `pre-commit install` |
| Clear JIT cache | `rm -rf ~/.cache/mate/` |
| Enable API logging (basic) | `export MATE_LOGLEVEL=1` |
| Enable API logging (detailed) | `export MATE_LOGLEVEL=3` |
| Enable API logging (with stats) | `export MATE_LOGLEVEL=5` |
| Set API log destination | `export MATE_LOGDEST=mylog.txt` |
| Enable verbose JIT logging | `export MATE_JIT_VERBOSE=1` |
| Set target architectures | `export MATE_MUSA_ARCH_LIST="3.1"` |


## Quick Start for Development

### Installation

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
pip install --no-build-isolation -e . -v
```

**Important**: The `--recursive` flag is required to initialize submodules in `3rdparty/` (mutlass...).

If you forgot `--recursive` when cloning:
```bash
git submodule update --init --recursive
```

The `--no-build-isolation` flag prevents pip from pulling incompatible PyTorch versions from PyPI.


## External Integrations

### TVM-FFI: Cross-Language Unified ABI

MATE uses **TVM-FFI** (Apache TVM's Foreign Function Interface) for bindings, which provides a **cross-language unified ABI**. This means:

- **Not limited to PyTorch**: The same compiled kernels can be used from multiple frameworks
- **Language agnostic**: Bindings can be created for Python, C++, Rust, etc.
- **Type-safe marshaling**: Automatic tensor/array conversion between languages
- **Export syntax**: Use `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, func)` to expose C++ functions

### TVM-FFI Kernel Binding Rules

When exporting MATE kernels through TVM-FFI, follow these rules:

- TVM-FFI functions do **not** need to be `void` by default. Return values are allowed when they represent real metadata or status produced by C++.
- For kernel-style bindings, if an output tensor's shape, dtype, and device are already known on the Python side, allocate it in Python and pass it into C++ as a tensor/tensor view.
- Do **not** use `alloc_tensor()` inside C++ FFI bindings for ordinary kernel outputs that can be preallocated by the caller.
- Exception: if an output tensor's padding, row stride, or storage layout is tightly coupled to the kernel implementation and easy to misuse from Python, allocate it inside C++ and return the logical tensor/view from FFI.
- Keep non-`void` returns only for values that are not ordinary output tensors, such as dispatch metadata that must be surfaced from C++.
- When updating FFI bindings, prefer the kernel-library guidance from TVM-FFI over generic PyTorch extension patterns.


## External Documentation Resources

When working with MATE's dependencies and tools, refer to these official documentation sources:

### Core Dependencies

- **TVM-FFI**: Apache TVM's Foreign Function Interface
  - Documentation: <https://tvm.apache.org/ffi/>
  - Package: `apache-tvm-ffi` (<https://pypi.org/project/apache-tvm-ffi/>)
  - Use for: Understanding FFI export syntax, cross-language bindings
  - MUSA Fork: https://github.com/MooreThreads/tvm-ffi/
    - When working on MUSA-related features, always refer to the musa-specific fork and its tags (e.g., v0.1.11.post1+musa.1) instead of upstream TVM-FFI.


### When to Consult These Docs

- **Working on FFI bindings** → Check TVM-FFI docs for export patterns and type marshaling
- **Working on kernel bindings** → Check the TVM-FFI kernel library guide for output-allocation patterns and `TensorView` usage


## Operator Development

- All concrete operator classes must inherit `mate.ops.Op`, and all concrete
  implementations must inherit `mate.ops.Impl`. Place concrete operators and
  their implementations under `mate.ops`.
- Before developing or migrating an operator, read the
  [operator design guide](mate/ops/README.md)
  ([中文对照](mate/ops/README_cn.md)).
- Use the same Op for implementations with the same computation, outputs, and
  rules for modifying caller tensors. Algorithms, backends, and target
  architectures may differ. Forward and backward use separate Ops.
- Export callable Op classes such as `mate.ops.attention.FaFwd`. The base
  metaclass forwards `FaFwd(...)` to its `run(...)` class method without
  creating an Op instance. List all public parameters, defaults, and types on
  `run`, with parameter docstrings. Explain the computation on the class. Set
  up package exports so users can call the Op after `import mate`.
- Each `run(...)` creates a new internal `_XxxInput` from `_input.py`, selects
  an Impl, and calls it. Users pass ordinary arguments. Impl instances hold
  reusable resources such as compiled modules and caches.
- Each Op owns implementation selection in `resolve_backend`; preserve
  `override_backend: str | None = None` where supported. Each Impl owns its
  complete checks and input normalization in `sanitize`, building in `build`,
  and execution through `__call__`, invoked as `impl(inputs)`. Document
  compatibility for implementation-specific prepared data such as metadata.
- Keep Python imports at module scope. Existing API modules can adapt arguments
  and call the corresponding Op. They can call `FaFwd.run(...)` to retain
  argument and return type checking.
- Migrate existing operators incrementally within the requested scope and
  preserve their behavior. Shared kernel and build helpers do not each need a
  public Op.
- Shared AOT, trace, profiling, and build artifact APIs are still being designed.
  Operator development and migration do not need to implement them. Preserve
  existing build and loading behavior.


## Repository Skills

Public repository skills live under `skills/<skill-name>/SKILL.md`.

- Before planning a non-trivial task, inspect the `name` and `description`
  frontmatter in `skills/*/SKILL.md`.
- If the user names a skill, or a skill description clearly matches the task,
  read that `SKILL.md` completely before acting.
- Load only the selected skill body and the references needed for the current
  task. Do not bulk-load every skill or every bundled resource.
- Resolve skill-relative paths from the directory containing `SKILL.md`.
- When multiple skills are relevant, use the smallest complementary set and
  follow any explicit baseline order declared by those skills.


## Behavior Rules

- For TileLang kernels, follow the
  [TileLang development rules](docs/tilelang_development_rules.md).

- Always read relevant files before modifying code
- For non-trivial tasks, create a plan first
- Validate changes by running tests
- After completing a task, always run `pre-commit run -a` to format/check code and fix any reported issues before handing off
- Prefer minimal, localized changes
- When running as `root` in the container, restore edited tracked files to the
  workspace owner/group and keep them user/group writable before handing off.
  A typical fix is `chown <workspace_uid>:<workspace_gid> <files>` followed by
  `chmod ug+rw <files>`, using the owner of `/workspace/mate` as the target.
