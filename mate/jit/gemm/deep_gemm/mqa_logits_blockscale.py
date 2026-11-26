from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import EXPORT_FUNC, TVM_HEADER
from ....mate_runtime import resolve_num_mps
from ....utils import round_up
from .. import gemm_utils


_MXFP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)


MP32_MQA_LOGITS_BLOCKSCALE_FLAG = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
    "-std=c++17",
]


@dataclass(frozen=True)
class MQALogitsBlockScaleJITConfig:
    tile_q: int
    tile_qh: int
    tile_kv: int
    stages_q: int
    stages_kv: int
    num_heads: int
    headdim: int
    compressed_logits: bool
    qkv_dtype: torch.dtype = torch.float8_e4m3fn
    weights_dtype: torch.dtype = torch.float32
    logits_dtype: torch.dtype = torch.float32
    use_schedule: bool = False
    num_mps: int = 0


def _get_mqa_logits_blockscale_tile(num_heads: int) -> tuple[int, int, int, int, int]:
    tile_qh = 64 if num_heads == 8 else 128
    tile_q = tile_qh // num_heads
    tile_kv = 512
    stage_q = 2
    stage_kv = 3
    return tile_q, tile_qh, tile_kv, stage_q, stage_kv


def _get_mqa_logit_blockscale_alignment(
    seqlen_q: int, seqlen_kv: int, num_heads: int, compressed_logits: bool
) -> tuple[int, int]:
    tile_q, _, tile_kv, _, _ = _get_mqa_logits_blockscale_tile(num_heads)
    seqlen_kv_alignment = 1024 // torch.float32.itemsize
    padded_seqlen_kv = (
        round_up(seqlen_kv, tile_kv) if compressed_logits else seqlen_kv + tile_kv
    )
    return round_up(seqlen_q, tile_q), round_up(padded_seqlen_kv, seqlen_kv_alignment)


MQA_LOGITS_BLOCKSCALE_CONFIGS: List[MQALogitsBlockScaleJITConfig] = [
    MQALogitsBlockScaleJITConfig(
        tile_q=tile_config[0],
        tile_qh=tile_config[1],
        tile_kv=tile_config[2],
        stages_q=tile_config[3],
        stages_kv=tile_config[4],
        num_heads=num_heads,
        headdim=headdim,
        compressed_logits=compressed_logits,
        qkv_dtype=qkv_dtype,
    )
    for compressed_logits in (False, True)
    for num_heads in (8, 16, 32, 64)
    for qkv_dtype, headdims in (
        (torch.float8_e4m3fn, (32, 64, 128)),
        *(((_MXFP4_DTYPE, (32, 64, 128)),) if _MXFP4_DTYPE is not None else ()),
    )
    for headdim in headdims
    for tile_config in (_get_mqa_logits_blockscale_tile(num_heads),)
]


_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def _validate_config(config: MQALogitsBlockScaleJITConfig) -> None:
    if config.tile_q * config.num_heads != config.tile_qh:
        raise ValueError("blockscale MQA logits requires tile_q * num_heads == tile_qh")
    supported_qkv_dtypes = (torch.float8_e4m3fn,) + (
        (_MXFP4_DTYPE,) if _MXFP4_DTYPE is not None else ()
    )
    if config.qkv_dtype not in supported_qkv_dtypes:
        raise ValueError(
            f"Unsupported blockscale MQA logits qkv_dtype={config.qkv_dtype}"
        )
    is_supported_headdim = (
        config.headdim in (32, 64, 128)
        if config.qkv_dtype == torch.float8_e4m3fn
        else 0 < config.headdim <= 128 and config.headdim % 32 == 0
    )
    if not is_supported_headdim:
        raise ValueError(
            f"Unsupported {config.qkv_dtype} blockscale MQA logits headdim={config.headdim}"
        )
    if config.weights_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported blockscale MQA logits weights_dtype={config.weights_dtype}"
        )
    if config.logits_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported blockscale MQA logits logits_dtype={config.logits_dtype}"
        )
    if config.weights_dtype == torch.bfloat16 and config.logits_dtype != torch.bfloat16:
        raise ValueError("bfloat16 weights require bfloat16 logits")
    if config.use_schedule and config.num_mps <= 0:
        raise ValueError("scheduled blockscale MQA logits requires a positive num_mps")


def _config_dict(cfg: MQALogitsBlockScaleJITConfig) -> Dict[str, object]:
    is_mxfp4 = _MXFP4_DTYPE is not None and cfg.qkv_dtype == _MXFP4_DTYPE
    is_wbf16 = cfg.weights_dtype == torch.bfloat16
    is_obf16 = cfg.logits_dtype == torch.bfloat16
    return {
        "kind": "mqa_logits_blockscale",
        "tile_q": cfg.tile_q,
        "tile_qh": cfg.tile_qh,
        "tile_kv": cfg.tile_kv,
        "stages_q": cfg.stages_q,
        "stages_kv": cfg.stages_kv,
        "num_heads": cfg.num_heads,
        "headdim": cfg.headdim,
        "compressed_logits": cfg.compressed_logits,
        "compressed_logits_literal": "true" if cfg.compressed_logits else "false",
        "use_schedule": cfg.use_schedule,
        "use_schedule_literal": "true" if cfg.use_schedule else "false",
        "num_mps": cfg.num_mps,
        "qkv_format": "mxfp4" if is_mxfp4 else "mxfp8",
        "qkv_element_type": (
            "mutlass::float_e2m1_t" if is_mxfp4 else "mutlass::float_e4m3_t"
        ),
        "qkv_input_dtype": ("dl_float4_e2m1fn_x2" if is_mxfp4 else "dl_float8_e4m3fn"),
        "qkv_pack_factor": 2 if is_mxfp4 else 1,
        "element_w_literal": "mutlass::bfloat16_t" if is_wbf16 else "float",
        "out_element_literal": "mutlass::bfloat16_t" if is_obf16 else "float",
        "weights_input_dtype": "dl_bfloat16" if is_wbf16 else "dl_float32",
        "out_input_dtype": "dl_bfloat16" if is_obf16 else "dl_float32",
        "dtype_suffix": ("_wbf16" if is_wbf16 else "") + ("_obf16" if is_obf16 else ""),
    }


def _dispatch_name(config: Mapping[str, object]) -> str:
    mode = "comp" if config["compressed_logits"] else "full"
    schedule_suffix = f"_sched_mp{config['num_mps']}" if config["use_schedule"] else ""
    return (
        "mate_jit_mqa_logits_blockscale_"
        f"{config['qkv_format']}_"
        f"h{config['num_heads']}_"
        f"d{config['headdim']}_"
        f"q{config['tile_q']}_"
        f"kv{config['tile_kv']}_"
        f"sq{config['stages_q']}_"
        f"skv{config['stages_kv']}_"
        f"{mode}"
        f"{config['dtype_suffix']}"
        f"{schedule_suffix}"
    )


def _render_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "mqa_logits_blockscale"
    return (
        TVM_HEADER
        + _template_env.get_template("mqa_logits_blockscale_kernel.j2").render(
            render_config
        )
        + EXPORT_FUNC.render(render_config)
    )


def gen_mqa_logits_blockscale_spec(
    config: MQALogitsBlockScaleJITConfig,
) -> JitSpec:
    _validate_config(config)
    render_config = _config_dict(config)
    dispatch_name = _dispatch_name(render_config)
    source_path = jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=list(MP32_MQA_LOGITS_BLOCKSCALE_FLAG),
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={source_path: _render_source(render_config)},
    )


def gen_mqa_logits_blockscale_aot() -> list[JitSpec]:
    return [
        gen_mqa_logits_blockscale_spec(config)
        for config in MQA_LOGITS_BLOCKSCALE_CONFIGS
    ]


@functools.cache
def _load_module(
    frozen_config: tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        bool,
        torch.dtype,
        torch.dtype,
        torch.dtype,
        bool,
        int,
    ],
):
    config = MQALogitsBlockScaleJITConfig(*frozen_config)
    return gen_mqa_logits_blockscale_spec(config).build_and_load()


@functools.cache
def get_mqa_logits_blockscale_module(
    num_heads: int,
    headdim: int,
    compressed_logits: bool,
    qkv_dtype: torch.dtype = torch.float8_e4m3fn,
    weights_dtype: torch.dtype = torch.float32,
    logits_dtype: torch.dtype = torch.float32,
    use_schedule: bool = False,
    num_mps: Optional[int] = None,
):
    if num_mps is None:
        num_mps = resolve_num_mps()
    tile_q, tile_qh, tile_kv, stages_q, stages_kv = _get_mqa_logits_blockscale_tile(
        num_heads
    )
    config = MQALogitsBlockScaleJITConfig(
        tile_q,
        tile_qh,
        tile_kv,
        stages_q,
        stages_kv,
        num_heads,
        headdim,
        compressed_logits,
        qkv_dtype,
        weights_dtype,
        logits_dtype,
        use_schedule,
        num_mps,
    )
    return _load_module(
        (
            config.tile_q,
            config.tile_qh,
            config.tile_kv,
            config.stages_q,
            config.stages_kv,
            config.num_heads,
            config.headdim,
            config.compressed_logits,
            config.qkv_dtype,
            config.weights_dtype,
            config.logits_dtype,
            config.use_schedule,
            config.num_mps,
        )
    )


__all__ = [
    "MQALogitsBlockScaleJITConfig",
    "MQA_LOGITS_BLOCKSCALE_CONFIGS",
    "gen_mqa_logits_blockscale_aot",
    "gen_mqa_logits_blockscale_spec",
    "get_mqa_logits_blockscale_module",
]
