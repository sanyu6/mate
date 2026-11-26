"""Standalone dispatch-only JIT; uses no GEMM specialization or weights."""

from functools import cache

from jinja2 import Environment, FileSystemLoader

from . import env
from .core import gen_jit_spec
from .utils import TVM_HEADER


def gen_fp4_dispatch_spec(
    config,
    num_mps,
    line_padded_scales,
    *,
    pull=False,
    num_ctas=None,
    flag_words=32,
    compact=False,
):
    if (
        isinstance(num_mps, bool)
        or not isinstance(num_mps, int)
        or num_mps < 2
        or num_mps % 2
    ):
        raise ValueError(
            "num_mps must be a positive even resident MP count, at least 2"
        )
    if not isinstance(line_padded_scales, bool):
        raise TypeError("line_padded_scales must be bool")
    name = (
        f"mate_mp32_fp4_dispatch_v1_mp{num_mps}_sf{int(line_padded_scales)}_"
        + "_".join(str(v) for v in config.__dict__.values())
    )
    if not isinstance(compact, bool):
        raise TypeError("compact must be bool")
    if not isinstance(pull, bool):
        raise TypeError("pull must be bool")
    if flag_words != 32 or isinstance(flag_words, bool):
        raise ValueError("pull control flags require 128-byte spacing")
    if pull:
        if (
            isinstance(num_ctas, bool)
            or not isinstance(num_ctas, int)
            or not 1 <= num_ctas <= num_mps
        ):
            raise ValueError("num_ctas must be within the common MP count")
        name += f"_pull_final_c{num_ctas}_flags{flag_words}"
    if compact:
        if not pull:
            raise ValueError("compact queue requires pull")
        name += "_owner_queue24"
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    source = TVM_HEADER + templates.get_template("fp4_dispatch_only.j2").render(
        func_name=name,
        pull=pull,
        compact=compact,
        num_ctas=num_ctas,
        flag_words=flag_words,
        num_mps=num_mps,
        line_padded_scales=line_padded_scales,
        **config.__dict__,
    )
    spec = gen_jit_spec(
        name,
        [path],
        generated_sources={path: source},
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-Od3", "-fno-strict-aliasing", "-std=c++17"],
        target_arch=arch,
        extra_include_paths=[
            env.MATE_INCLUDE_DIR,
            env.MATE_CSRC_DIR,
            env.MUTLASS_INCLUDE_DIR,
            env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        ],
    )
    return name, spec


@cache
def get_fp4_dispatch_module(
    config,
    num_mps,
    line_padded_scales,
    *,
    pull=False,
    num_ctas=None,
    flag_words=32,
    compact=False,
):
    name, spec = gen_fp4_dispatch_spec(
        config,
        num_mps,
        line_padded_scales,
        pull=pull,
        compact=compact,
        num_ctas=num_ctas,
        flag_words=flag_words,
    )
    return name, spec.build_and_load()


@cache
def get_fp4_dispatch_launcher(
    config,
    num_mps,
    line_padded_scales,
    *,
    pull=False,
    num_ctas=None,
    flag_words=32,
    compact=False,
):
    name, module = get_fp4_dispatch_module(
        config,
        num_mps,
        line_padded_scales,
        pull=pull,
        compact=compact,
        num_ctas=num_ctas,
        flag_words=flag_words,
    )
    return module.get_function(name)
