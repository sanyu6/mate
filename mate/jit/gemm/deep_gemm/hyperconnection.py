from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...cpp_ext import get_mudnn_ldflags
from ...utils import EXPORT_FUNC, TVM_HEADER
from ....utils import ceil_div
from .. import gemm_utils
from .deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS


@dataclass(frozen=True)
class HcPrenormGemmJitConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    num_mma_warp_squads: int


HC_PRENORM_TILE_M_OPTIONS = (32, 64)
HC_PRENORM_STAGE_OPTIONS = (3, 2)
HC_PRENORM_SMEM_BYTES_PER_MP = 192 * 1024
HC_PRENORM_BF16_BYTES = 2
HC_PRENORM_FP32_BYTES = 4
HC_PRENORM_TILE_N = 32

# Tuning space on top of the staging configs above. A config is usable only when
# every GEMM squad covers a 32/64/128-row M slice, which is what the MP32 SQMMA
# atom selector requires.
HC_PRENORM_TILE_M_SPACE = (32, 64, 128)
HC_PRENORM_TILE_K_SPACE = (32, 64)
HC_PRENORM_STAGE_SPACE = (2, 3, 4, 6, 8)
HC_PRENORM_MMA_WARP_SQUAD_SPACE = (1, 2, 4)
HC_PRENORM_FORCE_CONFIG_ENV = "MATE_HC_PRENORM_FORCE_CONFIG"
HC_PRENORM_SMALL_M_LIMIT = 32

_HC_PRENORM_IS_MP32 = "mp32" in jit_env.MATE_MUSA_ARCH_LIST

# The staging scheme is the pre-promotion one and MP31 stays on it: the MP32 prefill box below is
# separate and MP31 runs a different kernel. The staging configs carry two GEMM squads on MP31 and one
# on MP32, where a squad has to cover a whole 32-row M slice.
HC_PRENORM_STAGING_MMA_WARP_SQUADS = 1 if _HC_PRENORM_IS_MP32 else 2

# Prefill box (m > 32): BlockM=128 split across four GEMM squads, so each squad covers a 32-row M slice
# and the SQMMA atom selector keeps the M32N32K64 RS atom, at four stages. A CTA stages
# 4*(2*128 + 128)*64 + 128*32*4 = 114688 B, so two CTAs need 229376 B of this box's 256 KB smem per
# MP. That also puts the old 3-CTA/MP launch bound out of reach (270336 B > 256 KB), which is why the
# kernel's MinBlocksPerMultiprocessor is 2 rather than 3 - at 640 threads two CTAs also leave just 96
# registers per thread. The fourth stage is a latency-hiding choice rather than an arithmetic one, and
# it is the deepest stage count that stays inside that register budget; a fifth spills.
#
# tile_k is K=64 on MP32, matching the M32N32K64 RS atom. MP31 keeps K=32: the K=64 box was never
# revalidated there and that kernel stages an extra fp32 copy of A per stage.
HC_PRENORM_PREFILL_TILE_M_MP32 = 128
HC_PRENORM_PREFILL_TILE_K_MP32 = 64
HC_PRENORM_PREFILL_STAGES_MP32 = 4
HC_PRENORM_PREFILL_MMA_WARP_SQUADS_MP32 = 4


def hc_prenorm_prefill_box() -> Tuple[int, int, int, int]:
    """``(tile_m, tile_k, stages, num_mma_warp_squads)`` of the configured MP32 prefill box."""
    return (
        HC_PRENORM_PREFILL_TILE_M_MP32,
        HC_PRENORM_PREFILL_TILE_K_MP32,
        HC_PRENORM_PREFILL_STAGES_MP32,
        HC_PRENORM_PREFILL_MMA_WARP_SQUADS_MP32,
    )


def _hc_prenorm_arch() -> str:
    archs = jit_env.MATE_MUSA_ARCH_LIST
    if len(archs) != 1 or archs[0] not in {"mp31", "mp32"}:
        raise ValueError(
            "TF32 HC prenorm JIT requires exactly one target architecture: mp31 or mp32"
        )
    return archs[0]


HC_PRENORM_CONFIGS: List[HcPrenormGemmJitConfig] = [
    HcPrenormGemmJitConfig(
        tile_m=tile_m,
        tile_n=HC_PRENORM_TILE_N,
        tile_k=32,
        stages=stages,
        num_mma_warp_squads=HC_PRENORM_STAGING_MMA_WARP_SQUADS,
    )
    for stages in HC_PRENORM_STAGE_OPTIONS
    for tile_m in HC_PRENORM_TILE_M_OPTIONS
]
if _HC_PRENORM_IS_MP32:
    HC_PRENORM_CONFIGS.append(
        HcPrenormGemmJitConfig(
            tile_m=HC_PRENORM_PREFILL_TILE_M_MP32,
            tile_n=HC_PRENORM_TILE_N,
            tile_k=HC_PRENORM_PREFILL_TILE_K_MP32,
            stages=HC_PRENORM_PREFILL_STAGES_MP32,
            num_mma_warp_squads=HC_PRENORM_PREFILL_MMA_WARP_SQUADS_MP32,
        )
    )

_HC_PRENORM_CONFIGS_BY_TILE_M_AND_STAGES = {
    (cfg.tile_m, cfg.stages): cfg for cfg in HC_PRENORM_CONFIGS
}


def hc_prenorm_squad_tile_m(tile_m: int, num_mma_warp_squads: int) -> int:
    return tile_m // num_mma_warp_squads


def hc_prenorm_config_is_valid(
    tile_m: int, tile_k: int, stages: int, num_mma_warp_squads: int
) -> bool:
    if num_mma_warp_squads < 1 or tile_m % num_mma_warp_squads:
        return False
    squad_tile_m = hc_prenorm_squad_tile_m(tile_m, num_mma_warp_squads)
    if squad_tile_m % 32 or squad_tile_m > 128:
        return False
    return tile_k % 32 == 0 and stages >= 2


@functools.cache
def hc_prenorm_config(
    tile_m: int, tile_k: int, stages: int, num_mma_warp_squads: int
) -> HcPrenormGemmJitConfig:
    if not hc_prenorm_config_is_valid(tile_m, tile_k, stages, num_mma_warp_squads):
        raise ValueError(
            f"invalid MP32 HC prenorm config: tile_m={tile_m} tile_k={tile_k} "
            f"stages={stages} squads={num_mma_warp_squads}"
        )
    return HcPrenormGemmJitConfig(
        tile_m=tile_m,
        tile_n=HC_PRENORM_TILE_N,
        tile_k=tile_k,
        stages=stages,
        num_mma_warp_squads=num_mma_warp_squads,
    )


def _hc_prenorm_config_by_tile_m_and_stages(
    tile_m: int, stages: int
) -> HcPrenormGemmJitConfig:
    staged = _HC_PRENORM_CONFIGS_BY_TILE_M_AND_STAGES.get((tile_m, stages))
    if staged is not None:
        return staged
    return hc_prenorm_config(tile_m, HC_PRENORM_TILE_K_SPACE[0], stages, 1)


def _hc_prenorm_forced_config() -> Optional[HcPrenormGemmJitConfig]:
    spec = os.environ.get(HC_PRENORM_FORCE_CONFIG_ENV, "").strip()
    if not spec:
        return None
    parts = [
        int(part)
        for part in spec.replace("x", ",").replace(":", ",").split(",")
        if part.strip()
    ]
    if len(parts) != 4:
        raise ValueError(
            f"{HC_PRENORM_FORCE_CONFIG_ENV} expects 'tile_m,tile_k,stages,squads', got {spec!r}"
        )
    return hc_prenorm_config(*parts)


_hc_prenorm_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def get_hc_prenorm_template(name: str):
    return _hc_prenorm_template_env.get_template(name)


def _generated_source_path(dispatch_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"


def _hyperconnection_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def _hc_prenorm_waves(
    m: int,
    n: int,
    num_splits: int,
    num_mps: int,
    cfg: HcPrenormGemmJitConfig,
) -> int:
    smem_per_stage = (
        cfg.tile_m * cfg.tile_k * HC_PRENORM_BF16_BYTES
        + cfg.tile_n * cfg.tile_k * HC_PRENORM_FP32_BYTES
    )
    # The MP32 kernel always allocates the epilogue staging buffer (one BlockM x BlockN fp32 tile) on
    # top of the pipeline; the MP31 kernel writes D straight from registers.
    epilogue_bytes = 0
    if _hc_prenorm_arch() == "mp31":
        smem_per_stage += cfg.tile_m * cfg.tile_k * HC_PRENORM_FP32_BYTES
    else:
        epilogue_bytes = cfg.tile_m * cfg.tile_n * HC_PRENORM_FP32_BYTES
    ctas_per_mp = max(
        1,
        HC_PRENORM_SMEM_BYTES_PER_MP // (cfg.stages * smem_per_stage + epilogue_bytes),
    )
    total_ctas = max(num_splits, 1) * ceil_div(m, cfg.tile_m) * ceil_div(n, cfg.tile_n)
    return ceil_div(total_ctas, max(num_mps, 1) * ctas_per_mp)


@functools.cache
def select_hc_prenorm_config(
    m: int,
    n: int,
    num_splits: int = 1,
    num_mps: int = 1,
) -> HcPrenormGemmJitConfig:
    forced = _hc_prenorm_forced_config()
    if forced is not None:
        return forced

    if _HC_PRENORM_IS_MP32 and m > HC_PRENORM_SMALL_M_LIMIT:
        return hc_prenorm_config(*hc_prenorm_prefill_box())

    target_tile_m = 32 if m <= HC_PRENORM_SMALL_M_LIMIT else 64
    cfg_2stage = _hc_prenorm_config_by_tile_m_and_stages(target_tile_m, 2)
    cfg_3stage = _hc_prenorm_config_by_tile_m_and_stages(target_tile_m, 3)
    target_stages = (
        2
        if _hc_prenorm_waves(m, n, num_splits, num_mps, cfg_2stage)
        < _hc_prenorm_waves(m, n, num_splits, num_mps, cfg_3stage)
        else 3
    )

    return _hc_prenorm_config_by_tile_m_and_stages(target_tile_m, target_stages)


def hc_prenorm_config_dict(cfg: HcPrenormGemmJitConfig, arch: str) -> Dict[str, object]:
    return {
        "kind": "tf32_hc_prenorm",
        "arch": arch,
        "tile_m": cfg.tile_m,
        "tile_n": cfg.tile_n,
        "tile_k": cfg.tile_k,
        "stages": cfg.stages,
        "num_mma_warp_squads": cfg.num_mma_warp_squads,
    }


def hc_prenorm_dispatch_name(config: Mapping[str, object]) -> str:
    squads = config["num_mma_warp_squads"]
    assert isinstance(squads, int)
    num_mma_warp_squads = int(squads)
    squad_suffix = "" if num_mma_warp_squads == 1 else f"_{num_mma_warp_squads}q"
    return (
        "mate_jit_tf32_hc_prenorm_"
        f"{config['tile_m']}x"
        f"{config['tile_n']}x"
        f"{config['tile_k']}_"
        f"{config['stages']}stages"
        f"{squad_suffix}"
    )


def render_hc_prenorm_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "tf32_hc_prenorm_gemm"
    return (
        TVM_HEADER
        + get_hc_prenorm_template("hc_prenorm_gemm_kern.j2").render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_hyperconnection_spec(config: HcPrenormGemmJitConfig) -> JitSpec:
    arch = _hc_prenorm_arch()
    render_config = hc_prenorm_config_dict(config, arch)
    dispatch_name = hc_prenorm_dispatch_name(render_config)
    source_path = _generated_source_path(dispatch_name)
    cuda_flags = list(DEEP_GEMM_CUDA_FLAGS)
    if arch == "mp32":
        # Flag set for this kernel. The max-ILP scheduler has to be requested target-scoped: mcc
        # forwards every `-mllvm` to the x86_64 host pass as well, where the global
        # `-misched=mtgpu-max-ilp` form segfaults.
        cuda_flags.extend(["-mllvm", "--slp-max-reg-size=128"])
        cuda_flags.extend(["-mllvm", "-mtgpu-load-store-opt"])
        cuda_flags.extend(["-mllvm", "-mtgpu-enable-max-ilp-scheduling-strategy"])
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=list(get_mudnn_ldflags()),
        extra_include_paths=_hyperconnection_include_paths(),
        generated_sources={source_path: render_hc_prenorm_source(render_config)},
    )


def gen_hyperconnection_aot() -> list[JitSpec]:
    return [gen_hyperconnection_spec(config) for config in HC_PRENORM_CONFIGS]


@functools.cache
def _load_hyperconnection_module(frozen_config: tuple[int, int, int, int, int]):
    config = HcPrenormGemmJitConfig(*frozen_config)
    return gen_hyperconnection_spec(config).build_and_load()


@functools.cache
def get_hyperconnection_module(m: int, n: int, num_splits: int = 1, num_mps: int = 1):
    config = select_hc_prenorm_config(m, n, num_splits, num_mps)
    frozen_config = (
        config.tile_m,
        config.tile_n,
        config.tile_k,
        config.stages,
        config.num_mma_warp_squads,
    )
    return _load_hyperconnection_module(frozen_config)


__all__ = [
    "HcPrenormGemmJitConfig",
    "HC_PRENORM_CONFIGS",
    "gen_hyperconnection_aot",
    "gen_hyperconnection_spec",
]
