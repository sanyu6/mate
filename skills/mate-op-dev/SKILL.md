---
name: mate-op-dev
description: Migrate existing MATE operators to callable mate.ops.Op classes and Impl packages. Use when porting a legacy operator or completing a partial Op/Impl migration while preserving API behavior.
---

# MATE Operator Migration

This is a temporary skill for migrating existing operators. Long-term design
and development conventions live in the
[operator design guide](../../mate/ops/README.md)
([中文对照](../../mate/ops/README_cn.md)). Read it before planning a migration,
along with [AGENTS.md](../../AGENTS.md) and the current
[Op](../../mate/ops/op.py) / [Impl](../../mate/ops/impl.py) interfaces.

The guide includes planned migrations. Check the current code before editing
and state what still needs to be migrated. Shared AOT, trace, profiling, and
build artifact APIs are still being designed; migrations do not need to
implement them.

## Check the existing behavior

Identify the computation, outputs, and rules for modifying caller tensors.
Use an existing Op when these match. Add an Impl for a different algorithm,
backend, or target architecture. For example, FA3 and a future FA4 implementation
belong to `FaFwd`. Forward and backward use separate Ops.

Follow the requested operator from its public API through input preparation,
backend/target selection, kernel configuration, source generation, artifact
loading, and launches. Locate auxiliary stages, AOT enumeration, prewarming,
and existing tests. Search for callers before moving or deleting a module.

Record the behavior that must survive the move: signatures and overrides,
defaults, return ordering, output dtype/layout, tensor preparation, errors,
device/stream handling, compiler options, and supported configurations. Pay
particular attention to optional outputs and paths such as variable-length
inputs or cache updates when the existing operator supports them.

Keep the migration scoped to the requested operator and direction; migrating
forward does not require migrating backward or implementing a future algorithm.
Discuss any public API change separately from moving the implementation.

## Assign responsibilities

- Export a public Op class with a `run(...)` class method declaring explicit
  arguments, defaults, type annotations, and parameter docstrings. Document
  what the operator computes on the class. The base metaclass forwards `Xxx(...)`
  directly to `Xxx.run(...)` without creating an Op instance. Set up
  package exports so users can call it after `import mate`. Users pass ordinary
  arguments and do not import input types.
- The Op's `run(...)` constructs a new internal `_XxxInput` from `_input.py`
  and owns target resolution and `resolve_backend`. Existing API modules
  preserve their signatures, adapt arguments, and call the same Op. They can
  call `Xxx.run(...)` to retain argument and return type checking.
- Return Impl instances from selection. Keep implementation-specific
  checks and normalization in each Impl's `sanitize`, with building in `build`
  and execution through `__call__`, invoked as `impl(inputs)`. Selection can inspect
  target and input properties, but must not sanitize, compile, launch, or trial-run
  candidates. Preserve explicit overrides and report unsupported requests.
- An algorithm can use different compilers or kernel languages and run on
  different target architectures. When several Impls share a backend, make
  implementation selection clear and document automatic priority. Keep complete
  validation in each Impl; do not turn validation failures into silent fallback.
- Move the operator-specific configuration, source-generation choices, compiler
  flags, artifact loading, and launches into its implementation package. Reuse
  shared `mate.jit` infrastructure rather than copying the compiler or cache.
- Give each separate implementation its own directory. Share code across
  targets where useful; use the common `Arch` definition. Selection rules and
  support information stay with the Op, without imposing a registry layout.
- Split internal Impls when their inputs or execution responsibilities differ.
  A complete Impl composes these stages; an internal stage need not be a public
  Op. Prepare reusable metadata explicitly when the API requires it.

Use the operator's existing configuration and kernel sources as the starting
point. The migration should carry real execution into the new structure; avoid
substituting mock compilation, probe output, or demonstration-only launches.

## Preserve input preparation and execution

Use a fresh mutable input object per call. `sanitize` may replace fields, and
later configuration and execution must use those updated fields. Reusable
Impl state holds resources and caches, not one call's inputs or buffers.

Before moving input fields, distinguish computation parameters, tuning options,
and data prepared for a particular Impl. For example, when migrating attention
forward, check which FA3 fields belong in the shared FA interface. For metadata
and intermediate buffers, check which Impl
produces them, their configuration and layout, whether they can be modified,
and how long they remain valid. Select a compatible Impl or report an error.
Define compatibility rules before reusing or rebuilding data for another Impl.

Preserve the exact checks and transformations while relocating them. For
example, a full `is_contiguous()` requirement is not equivalent to checking
only the last stride. Rejecting an input is not equivalent to silently copying
it. Different implementations may legitimately require different checks.

Keep imports at module scope, preserve supported overrides such as
`override_backend`, and preserve output values and ordering when assembling results
from multiple stages.

`build` prepares artifacts without launching operation kernels. Normal calls
perform their required preparation without requiring a separate caller-side
build. When building through dry-run execution, ensure every stage suppresses
launches; keep fake conversion and build-only field replacement local. A shallow
input copy does not protect tensors from in-place writes. Build-time detachment
must not leak into normal execution or alter autograd behavior.

## Connect existing build paths

Preserve the operator's existing build, loading, prewarming, and packaging
consumers where applicable. If it already supports AOT, retain that capability
and its current entry points within the migration scope. Verify compatible
source, flags, names, and artifact paths for the existing consumers.

Shared APIs for AOT, target selection during builds, build artifact storage,
and tracing need a separate design. Report unfinished migration work without
adding those APIs. Existing `aot` build parameters do not require every Impl
to support AOT; report unsupported build modes instead of silently ignoring them.

## Validate and decide what remains

Use relevant existing numerical and interface tests to compare the new path
with the original behavior, including the meaningful optional paths found
above. Check selection/overrides, input preparation, returns, building without
launches, metadata compatibility, and existing artifact reuse where supported.
Compare each Impl with the Op's reference results and tensor modification rules using suitable
tolerances. Verify `import mate` followed by the public Op class call without
additional submodule imports. Follow the task's instructions on permanent tests;
temporary migration checks can live outside the worktree.

Update incremental-test ownership when implementation paths move, if the
repository has a test catalog. Run repository checks required by `AGENTS.md`.
State which targets were executed and which were checked only by compilation.

A completed migration has operator-specific preparation, configuration,
building, and launches owned by the new Impls, with callers using the new path.
Public API adapters and shared JIT helpers can remain. A bridge into the old
operator implementation is partial progress; identify the remaining work.
Remove old operator-specific code after its consumers have migrated, preserving
any compatibility required by the task.

Retire this skill once the existing operator migration is complete across the
repository. Keep lasting design rules in the operator guide and `AGENTS.md`.
