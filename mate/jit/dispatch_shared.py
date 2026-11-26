from functools import cache
from jinja2 import Environment, FileSystemLoader
from . import env
from .core import gen_jit_spec
from .utils import TVM_HEADER


def gen_dispatch_shared_spec(
    config,
    shared_features=4608,
    num_ctas=24,
    padded=True,
    fused=True,
    feature_tile=128,
    token_tile=32,
    stages=2,
    gemm_ctas=None,
):
    if (
        isinstance(num_ctas, bool)
        or not isinstance(num_ctas, int)
        or not 1 <= num_ctas <= 128
    ):
        raise ValueError("num_ctas must be an integer within 1..128")
    if not isinstance(padded, bool) or not isinstance(fused, bool):
        raise TypeError("padded/fused must be bool")
    if feature_tile not in (64, 128) or token_tile != 32 or stages not in (2, 3, 4):
        raise ValueError("unsupported shared GEMM tile/pipeline")
    if (
        not isinstance(shared_features, int)
        or shared_features <= 0
        or shared_features % feature_tile
    ):
        raise ValueError("invalid shared output features")
    gemm_ctas = num_ctas if gemm_ctas is None else gemm_ctas
    if (
        isinstance(gemm_ctas, bool)
        or not isinstance(gemm_ctas, int)
        or not 1 <= gemm_ctas <= 128
    ):
        raise ValueError("invalid GEMM CTA count")
    name = "mate_dispatch_shared_v4_grid_" + "_".join(
        map(
            str,
            (
                *config.__dict__.values(),
                shared_features,
                num_ctas,
                int(padded),
                int(fused),
                feature_tile,
                token_tile,
                stages,
                gemm_ctas,
            ),
        )
    )
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    source = TVM_HEADER + templates.get_template("dispatch_shared.j2").render(
        func_name=name,
        shared_features=shared_features,
        num_ctas=num_ctas,
        padded=padded,
        fused=fused,
        feature_tile=feature_tile,
        token_tile=token_tile,
        stages=stages,
        gemm_ctas=gemm_ctas,
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
def get_dispatch_shared(
    config,
    shared_features=4608,
    num_ctas=24,
    padded=True,
    fused=True,
    feature_tile=128,
    token_tile=32,
    stages=2,
    gemm_ctas=None,
):
    name, spec = gen_dispatch_shared_spec(
        config,
        shared_features,
        num_ctas,
        padded,
        fused,
        feature_tile,
        token_tile,
        stages,
        gemm_ctas,
    )
    module = spec.build_and_load()
    return module.get_function(name)
