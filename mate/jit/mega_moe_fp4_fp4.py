"""Independent MP32 native SS FP4 x FP4 JIT entrypoints."""

from dataclasses import dataclass
from functools import cache

from jinja2 import Environment, FileSystemLoader

from . import env
from .core import gen_jit_spec
from .utils import EXPORT_FUNC, TVM_HEADER
from .mega_moe_common import _normalize_activation


@dataclass(frozen=True)
class FP4GemmConfig:
    tile_m: int = 256
    tile_n: int = 256

    def __post_init__(self):
        if self.tile_m not in (128, 256) or self.tile_n not in (128, 256):
            raise ValueError("native SS tiles must have M/N in {128,256}")


@cache
def get_fp4_grouped_module(config=FP4GemmConfig()):
    kind = "grouped"
    name = f"mate_mp32_fp4_fp4_{kind}_{config.tile_m}x{config.tile_n}"
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    rendering = {"func_name": name, **config.__dict__}
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    source = (
        TVM_HEADER
        + templates.get_template(f"fp4_fp4_{kind}_gemm.j2").render(rendering)
        + EXPORT_FUNC.render(rendering)
    )
    spec = gen_jit_spec(
        name,
        [path],
        generated_sources={path: source},
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-Od3",
            "-fno-strict-aliasing",
            "-std=c++17",
            "-mllvm",
            "-mtgpu-load-store-2d=1",
            "-mllvm",
            "-mtgpu-enable-max-ilp-scheduling-strategy=true",
        ],
        target_arch=arch,
        extra_include_paths=[
            env.MATE_INCLUDE_DIR,
            env.MATE_CSRC_DIR,
            env.MUTLASS_INCLUDE_DIR,
            env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        ],
    )
    return name, spec.build_and_load()


@dataclass(frozen=True)
class FP4PrefillConfig:
    ranks: int
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int

    def __post_init__(self):
        if self.ranks not in (1, 2, 4, 8):
            raise ValueError("FP4 prefill supports EP1/2/4/8")
        if not 0 < self.tokens <= 65536 or self.tokens % 256:
            raise ValueError("token capacity must be 256 aligned within [1,65536]")
        if (
            min(self.hidden, self.intermediate) <= 0
            or self.hidden % 256
            or self.intermediate % 256
        ):
            raise ValueError("hidden/intermediate must be positive and 256 aligned")
        if (
            self.experts <= 0
            or self.experts % self.ranks
            or self.experts // self.ranks > 1024
        ):
            raise ValueError("invalid expert count")
        if not 0 < self.topk <= min(self.experts, 32):
            raise ValueError("invalid top-k")


@cache
def _build_fp4_prefill_module(
    config, stage, mode="full", num_mps=0, activation="swiglu", use_situ_linear=False
):
    kind = "layout" if stage == 0 else f"prefill_stage{stage}"
    suffix = (
        f"_{mode}_mp{num_mps}" if stage == 1 else f"_mp{num_mps}" if stage == 2 else ""
    )
    if stage == 1 and activation == "situ":
        suffix += f"_actsitu_lin{int(use_situ_linear)}"
    name = (
        "mate_mp32_fp4_fp4_"
        + kind
        + suffix
        + "_"
        + "_".join(str(v) for v in config.__dict__.values())
    )
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    template = f"fp4_fp4_{kind}.j2"
    source = TVM_HEADER + templates.get_template(template).render(
        {
            "func_name": name,
            "mode": mode.capitalize(),
            "num_mps": num_mps,
            "use_situ": stage == 1 and activation == "situ",
            "use_situ_linear": use_situ_linear,
            **config.__dict__,
        }
    )
    spec = gen_jit_spec(
        name,
        [path],
        generated_sources={path: source},
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-Od3",
            "-fno-strict-aliasing",
            "-std=c++17",
            "-mllvm",
            "-mtgpu-load-store-2d=1",
            "-mllvm",
            "-mtgpu-enable-max-ilp-scheduling-strategy=true",
        ],
        target_arch=arch,
        extra_include_paths=[
            env.MATE_INCLUDE_DIR,
            env.MATE_CSRC_DIR,
            env.MUTLASS_INCLUDE_DIR,
            env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        ],
    )
    return name, spec.build_and_load()


def get_fp4_layout_module(config):
    return _build_fp4_prefill_module(config, 0)


def get_fp4_prefill_stage_module(
    config,
    stage,
    *,
    num_mps=None,
    mode="full",
    activation="swiglu",
    use_situ_linear=False,
):
    """Build a final prefill stage; Stage1/2 require a resident MP count."""
    activation = _normalize_activation(activation)
    if stage != 1 and (activation != "swiglu" or use_situ_linear):
        raise ValueError("activation specialization is only valid for Stage1")
    if activation != "situ" and use_situ_linear:
        raise ValueError("linear softcap requires SiTU")
    if stage not in (1, 2, 3):
        raise ValueError("prefill stage must be 1/2/3")
    if mode not in ("full", "compute", "dispatch") or (stage != 1 and mode != "full"):
        raise ValueError("only Stage1 supports compute/dispatch diagnostic modes")
    if stage in (1, 2):
        if (
            isinstance(num_mps, bool)
            or not isinstance(num_mps, int)
            or num_mps <= 0
            or num_mps % 4
        ):
            raise ValueError(
                "resident prefill MP count must be a positive multiple of four"
            )
    elif num_mps is not None:
        raise ValueError("Stage3 uses a runtime grid configuration")
    if activation == "swiglu":
        return _build_fp4_prefill_module(config, stage, mode, num_mps or 0)
    return _build_fp4_prefill_module(
        config, stage, mode, num_mps or 0, activation, bool(use_situ_linear)
    )
