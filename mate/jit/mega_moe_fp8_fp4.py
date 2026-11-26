"""Independent MP32 FP8/FP4 prefill JIT; stage helpers remain private."""

from dataclasses import dataclass
from functools import cache

from jinja2 import Environment, FileSystemLoader

from . import env
from .core import gen_jit_spec
from .utils import EXPORT_FUNC, TVM_HEADER


@dataclass(frozen=True)
class PrefillConfig:
    num_ranks: int
    num_experts: int
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_max_pool_tokens: int
    num_mps: int = 40
    fast_math: bool = True
    use_situ: bool = False
    use_situ_linear: bool = False
    profile_mode: str = "full"
    stage1_stages: int = 4
    owner_ctas: int = 4


@cache
def _get_stage_module(config, stage):
    if config.profile_mode not in ("full", "dispatch_only", "compute_only"):
        raise ValueError("invalid Stage1 profile mode")
    if stage != 1 and config.profile_mode != "full":
        raise ValueError("profile modes apply to Stage1 only")
    name = (
        "mate_mp32_fp8_fp4_prefill_s"
        + str(stage)
        + "_"
        + "_".join(str(v) for v in config.__dict__.values())
    )
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    rendering = {
        **config.__dict__,
        "func_name": name,
        "num_experts_per_rank": config.num_experts // config.num_ranks,
        "num_padded_sf_pool_tokens": config.num_max_pool_tokens,
        "num_l1_ring_tokens": config.num_max_pool_tokens,
    }
    if stage == 3:
        rendering.update(num_threads=128, num_blocks=config.num_mps * 24)
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    source = (
        TVM_HEADER
        + templates.get_template(f"fp8_fp4_prefill_stage{stage}_kern.j2").render(
            rendering
        )
        + EXPORT_FUNC.render(rendering)
    )
    flags = ["-Od3", "-fno-strict-aliasing", "-std=c++17", "-DMARCH_TYPE=320"]
    if config.fast_math:
        flags += ["-ffast-math"]
    if stage == 1:
        flags += [
            "-mllvm",
            "-mtgpu-load-store-opt",
            "-mllvm",
            "-mtgpu-enchanced-minreg-schedule=false",
            "-mllvm",
            "-mtgpu-enable-max-ilp-scheduling-strategy=true",
        ]
    if stage == 2:
        flags += ["-mllvm", "-mtgpu-enchanced-minreg-schedule=false"]
    spec = gen_jit_spec(
        name,
        [path],
        generated_sources={path: source},
        extra_cflags=["-O3"],
        extra_cuda_cflags=flags,
        target_arch=arch,
        extra_include_paths=[
            env.MATE_INCLUDE_DIR,
            env.MATE_CSRC_DIR,
            env.MUTLASS_INCLUDE_DIR,
        ],
    )
    return name, spec.build_and_load()
