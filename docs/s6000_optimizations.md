# S6000 optimizations on the public baseline

This branch starts from public MATE `5ab49c437638caf499b6f8b2480e4a1fa0da7dfe`
and ports the operator changes used by the S6000 performance snapshot. The
unchanged snapshot remains on branch `s6000-best-baseline-20261125`.

The port includes MP32 W4A4/W4A8 MoE, dispatch/shared-expert/mHC fusion,
scheduled MP32 V4/V4.1 sparse MLA, FlashMLA RoPE argument forwarding, and
architecture-specific JIT generation. Public baseline implementations remain
available for other paths.

## Dependency preparation

The public MUTLASS revision lacks several MP32/FP4 definitions used by these
kernels. Keep it as a submodule and apply the separate header patch:

```bash
git submodule update --init 3rdparty/mutlass
python scripts/prepare_s6000_mutlass.py --check
python scripts/prepare_s6000_mutlass.py
```

The helper checks the pinned revision, preserves unrelated local changes,
and accepts repeated invocations after the patch has been applied. The
patched include tree matches the original snapshot, including TME cache hints.
The dependency delta is tracked in `patches/mutlass-mp32-fp4.patch`; it is
separate from the MATE operator changes for review.

## Validation scope

The snapshot's performance result does not transfer automatically to this
new baseline. CPU contract tests and import/source-generation checks establish
the port's host-side integration. GPU correctness, graph replay, end-to-end
accuracy, and performance must be revalidated in the matching MUSA environment
before replacing the measured deployment.

Publication checks completed in a MUSA 5.3.0 / PyTorch 2.11 environment:

- 124 existing FP4/MoE CPU contract cases and 5 sparse-decode ABI cases passed.
- FlashMLA imports retain both RoPE arguments through the wrapper chain.
- Dispatch/shared/mHC fusion and FP4 decode Stage1/Stage2 compiled for MP32.
- All repository pre-commit hooks passed.

The compile checks did not launch kernels on a GPU. Reproduce the CPU tests with
`MATE_MUSA_ARCH_LIST=3.2 PYTHONPATH=.:tests python -m pytest` and the six
`test_*contract.py` / `test_s6000_sparse_decode_dispatch.py` files added here.
