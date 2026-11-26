from __future__ import annotations

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
    jit_env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
    jit_env.MUTLASS_INCLUDE_DIR.parent / "experimental" / "fmha",
]


def gen_mla_ops_spec() -> JitSpec:
    # The MP32 sparse decode metadata path does not need the legacy dense
    # MP31 MLA kernels.  Keeping those templates in the same MP32 translation
    # unit triggers MCC's dependency-graph pass before metadata compilation.
    # Use a metadata-only binding for MP32; retain the full binding for MP31.
    mla_pybind_source = (
        jit_env.MATE_CSRC_DIR / "mla_pybind_mp32.mu"
        if "mp32" in jit_env.MATE_MUSA_ARCH_LIST
        and "mp31" not in jit_env.MATE_MUSA_ARCH_LIST
        else jit_env.MATE_CSRC_DIR / "mla_pybind.mu"
    )
    sources = [
        jit_env.MATE_CSRC_DIR / "attention_scheduler.mu",
        jit_env.MATE_CSRC_DIR / "attention_combine.mu",
        mla_pybind_source,
    ]
    return gen_jit_spec(
        "mla_ops",
        sources,
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


def gen_mla_ops_aot() -> list[JitSpec]:
    return [gen_mla_ops_spec()]


@functools.cache
def get_mla_ops_module():
    return gen_mla_ops_spec().build_and_load()
