"""Boundary modes 0..6 are unchanged; mode7 writes post-only updated residual.

Mode7 retains merged for diagnostics, but does not consume pre/gamma or write
normalized/q/sf. It uses the same caller-owned buffers and ordered IPC protocol.
"""

from functools import cache

from jinja2 import Environment, FileSystemLoader

from . import env
from .core import gen_jit_spec
from .utils import TVM_HEADER


def gen_combine_shared_mhc_spec(
    config,
    shared_features=4608,
    num_ctas=24,
    padded=True,
    fused=True,
    feature_tile=128,
    token_tile=32,
    stages=2,
    gemm_ctas=None,
    producer_ctas=24,
    ring2=False,
    fast_quant=False,
    comm_threads=128,
    gemm_stages=2,
    local_merge=False,
):
    if not isinstance(local_merge, bool):
        raise TypeError("local_merge must be bool")
    if (
        producer_ctas not in (24, 32, 40)
        or comm_threads not in (128, 256, 512, 768)
        or gemm_stages not in (2, 3, 4)
    ):
        raise ValueError("unsupported producer geometry")
    if not isinstance(ring2, bool) or not isinstance(fast_quant, bool):
        raise TypeError("ring2/fast_quant must be bool")
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
    name = "mate_combine_shared_mhc_v20_local_merge_" + "_".join(
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
                producer_ctas,
                int(ring2),
                int(fast_quant),
                comm_threads,
                gemm_stages,
                int(local_merge),
            ),
        )
    )
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / "mega_moe" / (name + ".mu")
    templates = Environment(
        loader=FileSystemLoader(str(env.MATE_TEMPLATE_DIR / "mega_moe"))
    )
    source = TVM_HEADER + templates.get_template("combine_shared_mhc.j2").render(
        func_name=name,
        shared_features=shared_features,
        num_ctas=num_ctas,
        padded=padded,
        fused=fused,
        feature_tile=feature_tile,
        token_tile=token_tile,
        stages=stages,
        gemm_ctas=gemm_ctas,
        producer_ctas=producer_ctas,
        ring2=ring2,
        fast_quant=fast_quant,
        comm_threads=comm_threads,
        gemm_stages=gemm_stages,
        local_merge=local_merge,
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
def get_combine_shared_mhc(
    config,
    shared_features=4608,
    num_ctas=24,
    padded=True,
    fused=True,
    feature_tile=128,
    token_tile=32,
    stages=2,
    gemm_ctas=None,
    producer_ctas=24,
    ring2=False,
    fast_quant=False,
    comm_threads=128,
    gemm_stages=2,
    local_merge=False,
):
    name, spec = gen_combine_shared_mhc_spec(
        config,
        shared_features,
        num_ctas,
        padded,
        fused,
        feature_tile,
        token_tile,
        stages,
        gemm_ctas,
        producer_ctas,
        ring2,
        fast_quant,
        comm_threads,
        gemm_stages,
        local_merge,
    )
    module = spec.build_and_load()
    return module.get_function(name)
