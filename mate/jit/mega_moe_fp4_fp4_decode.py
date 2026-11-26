"""Experimental M32 MP32 FP4 decode; generating specs never launches kernels."""

from functools import cache

from jinja2 import Environment, FileSystemLoader

from . import env
from .core import gen_jit_spec
from .mega_moe_common import _normalize_activation
from .utils import TVM_HEADER

FP4_DECODE_MAX_TOKENS = 128
FP4_DECODE_BLOCK_M = 32


def gen_fp4_decode_spec(
    config,
    stage,
    *,
    num_mps=0,
    activation="swiglu",
    use_situ_linear=False,
    swap_ab=False,
    pack_b=False,
    sglang_numerics=False,
    line_padded_scales=False,
    double_buffered=False,
    feature_tile=256,
):
    """Generate the unified-layout query or one of the two decode stages."""
    for value in (sglang_numerics, line_padded_scales, double_buffered):
        if not isinstance(value, bool):
            raise TypeError("decode specialization flags must be bool")
    if double_buffered and stage == 0:
        raise ValueError("double buffering specializes compute stages only")
    if double_buffered and (swap_ab or activation != "swiglu"):
        raise ValueError("double buffering requires token-A N256")
    if line_padded_scales and (stage != 1 or swap_ab or activation != "swiglu"):
        raise ValueError("coherent scales specialize Stage1 only")
    if sglang_numerics and (stage == 0 or swap_ab or activation != "swiglu"):
        raise ValueError("SGLang numerics require M32 SwiGLU decode")
    if not isinstance(pack_b, bool) or (pack_b and swap_ab):
        raise ValueError("pack_b and swap_ab are exclusive boolean decode backends")
    if pack_b and (stage == 0 or activation != "swiglu"):
        raise ValueError("pack_b supports decode SwiGLU only")
    if feature_tile not in (64, 128, 256):
        raise ValueError("swap-AB feature tile must be 64, 128 or 256")
    if not isinstance(swap_ab, bool):
        raise TypeError("swap_ab must be bool")
    if swap_ab and (stage == 0 or activation != "swiglu" or use_situ_linear):
        raise ValueError("swap-AB is a decode SwiGLU compute specialization")
    activation = _normalize_activation(activation)
    if stage != 1 and (activation != "swiglu" or use_situ_linear):
        raise ValueError("activation specialization is only valid for decode Stage1")
    if activation != "situ" and use_situ_linear:
        raise ValueError("linear softcap requires SiTU")
    if stage not in (0, 1, 2):
        raise ValueError("decode stage must be 0 (layout), 1 or 2")
    if stage and (
        isinstance(num_mps, bool)
        or not isinstance(num_mps, int)
        or num_mps < 2
        or num_mps % 2
    ):
        raise ValueError(
            "decode resident MP count must be positive and even, at least two"
        )
    if not stage and num_mps != 0:
        raise ValueError("decode layout does not use an MP count")
    kind = "layout" if not stage else f"stage{stage}"
    name = f"mate_mp32_fp4_fp4_decode_{kind}_m32_mp{num_mps}_" + "_".join(
        str(v) for v in config.__dict__.values()
    )
    if activation == "situ":
        name += f"_actsitu_lin{int(use_situ_linear)}"
    if swap_ab:
        name += f"_swapab_f{feature_tile}_p3_c{512 // feature_tile}_regact_mperm4"
    if pack_b:
        name += "_packb_f256_p2" if double_buffered else "_packb_f256_p3"
    if sglang_numerics:
        name += "_sgnum_v1"
    if line_padded_scales:
        name += "_sfline128_final"
    if double_buffered:
        name += "_p2c3_bound"
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    source = TVM_HEADER + templates.get_template(f"fp4_fp4_decode_{kind}.j2").render(
        {
            "func_name": name,
            "num_mps": num_mps,
            "swap_ab": swap_ab,
            "pack_b": pack_b,
            "sglang_numerics": sglang_numerics,
            "line_padded_scales": line_padded_scales,
            "double_buffered": double_buffered,
            "feature_tile": feature_tile,
            "use_situ": activation == "situ",
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
    return name, spec


@cache
def get_fp4_decode_module(
    config,
    stage,
    *,
    num_mps=0,
    activation="swiglu",
    use_situ_linear=False,
    swap_ab=False,
    pack_b=False,
    sglang_numerics=False,
    line_padded_scales=False,
    double_buffered=False,
    feature_tile=256,
):
    name, spec = gen_fp4_decode_spec(
        config,
        stage,
        num_mps=num_mps,
        activation=activation,
        use_situ_linear=use_situ_linear,
        swap_ab=swap_ab,
        pack_b=pack_b,
        sglang_numerics=sglang_numerics,
        line_padded_scales=line_padded_scales,
        double_buffered=double_buffered,
        feature_tile=feature_tile,
    )
    return name, spec.build_and_load()


@cache
def get_fp4_decode_input_module():
    """BF16 -> row-major E2M1/UE8M0 plus routing into symmetric storage."""
    spec = gen_jit_spec(
        "mate_mp32_fp4_decode_input_sg_v1",
        [env.MATE_CSRC_DIR / "fp4_decode_input.mu"],
        target_arch=env.MusaArch.parse("3.2"),
        extra_cuda_cflags=["-Od3", "-std=c++17", "-mllvm", "-mtgpu-load-store-2d=1"],
        extra_include_paths=[
            env.MATE_CSRC_DIR,
            env.MATE_INCLUDE_DIR,
            env.MUTLASS_INCLUDE_DIR,
        ],
    )
    return spec.build_and_load()
