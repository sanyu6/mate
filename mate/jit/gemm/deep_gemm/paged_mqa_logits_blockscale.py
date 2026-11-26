from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import EXPORT_FUNC, TVM_HEADER
from .. import gemm_utils


_MXFP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)


MP32_PAGED_MQA_LOGITS_BLOCKSCALE_FLAG = [
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


def _get_paged_mqa_logits_blockscale_tile(
    num_heads: int,
    next_n: int = 1,
) -> tuple[int, int, int, int, int]:
    tile_qh = next(
        (
            qh
            for qh in (32, 64, 128)
            if qh % num_heads == 0 and qh >= next_n * num_heads
        ),
        128,
    )
    tile_q = tile_qh // num_heads
    tile_kv = 512
    stages_q = 2
    stages_kv = 3
    return tile_q, tile_qh, tile_kv, stages_q, stages_kv


@dataclass(frozen=True)
class PagedMQALogitsBlockScaleJITConfig:
    num_heads: int
    headdim: int
    page_size: int
    next_n: int
    is_context_lens_2d: bool
    stages_q: int = 2
    stages_kv: int = 3
    is_varlen: bool = False
    clean_logits: bool = False
    qkv_dtype: torch.dtype = torch.float8_e4m3fn
    tile_qh: int = 128
    is_kv_scale_m_major: bool = False
    weights_dtype: torch.dtype = torch.float32
    logits_dtype: torch.dtype = torch.float32


PAGED_MQA_LOGITS_BLOCKSCALE_NUM_HEADS_OPTIONS = (8, 16, 32, 64)
PAGED_MQA_LOGITS_BLOCKSCALE_HEADDIM_OPTIONS = (32, 64, 128)
PAGED_MQA_LOGITS_BLOCKSCALE_PAGE_SIZE_OPTIONS = (128, 32, 64)
PAGED_MQA_LOGITS_BLOCKSCALE_AOT_NEXT_N_OPTIONS = (1, 6)

PAGED_MQA_LOGITS_BLOCKSCALE_CONFIGS: List[PagedMQALogitsBlockScaleJITConfig] = [
    PagedMQALogitsBlockScaleJITConfig(
        num_heads=num_heads,
        headdim=headdim,
        page_size=page_size,
        next_n=next_n,
        is_context_lens_2d=True,
        stages_q=tile_config[3],
        stages_kv=tile_config[4],
        clean_logits=clean_logits,
        qkv_dtype=qkv_dtype,
        tile_qh=tile_config[1],
        is_kv_scale_m_major=is_kv_scale_m_major,
    )
    for clean_logits in (False, True)
    for qkv_dtype in (
        (torch.float8_e4m3fn,) + ((_MXFP4_DTYPE,) if _MXFP4_DTYPE is not None else ())
    )
    for is_kv_scale_m_major in (False, True)
    for num_heads in PAGED_MQA_LOGITS_BLOCKSCALE_NUM_HEADS_OPTIONS
    for headdim in PAGED_MQA_LOGITS_BLOCKSCALE_HEADDIM_OPTIONS
    for page_size in PAGED_MQA_LOGITS_BLOCKSCALE_PAGE_SIZE_OPTIONS
    for next_n in PAGED_MQA_LOGITS_BLOCKSCALE_AOT_NEXT_N_OPTIONS
    for tile_config in (_get_paged_mqa_logits_blockscale_tile(num_heads, next_n),)
] + [
    PagedMQALogitsBlockScaleJITConfig(
        num_heads=num_heads,
        headdim=headdim,
        page_size=page_size,
        next_n=1,
        is_context_lens_2d=True,
        stages_q=tile_config[3],
        stages_kv=tile_config[4],
        is_varlen=True,
        clean_logits=clean_logits,
        qkv_dtype=qkv_dtype,
        tile_qh=tile_config[1],
        is_kv_scale_m_major=is_kv_scale_m_major,
    )
    for clean_logits in (False, True)
    for qkv_dtype in (
        (torch.float8_e4m3fn,) + ((_MXFP4_DTYPE,) if _MXFP4_DTYPE is not None else ())
    )
    for is_kv_scale_m_major in (False, True)
    for num_heads in PAGED_MQA_LOGITS_BLOCKSCALE_NUM_HEADS_OPTIONS
    for headdim in PAGED_MQA_LOGITS_BLOCKSCALE_HEADDIM_OPTIONS
    for page_size in PAGED_MQA_LOGITS_BLOCKSCALE_PAGE_SIZE_OPTIONS
    for tile_config in (_get_paged_mqa_logits_blockscale_tile(num_heads),)
]


_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def _validate_config(config: PagedMQALogitsBlockScaleJITConfig) -> None:
    if config.tile_qh not in (32, 64, 128):
        raise ValueError(
            f"Unsupported paged blockscale MQA logits tile_qh={config.tile_qh}"
        )
    if config.tile_qh % config.num_heads != 0:
        raise ValueError(
            "Paged blockscale MQA logits requires tile_qh divisible by num_heads "
            f"(tile_qh={config.tile_qh}, num_heads={config.num_heads})"
        )
    if config.num_heads not in PAGED_MQA_LOGITS_BLOCKSCALE_NUM_HEADS_OPTIONS:
        raise ValueError(
            f"Unsupported paged blockscale MQA logits num_heads={config.num_heads}"
        )
    if config.headdim not in PAGED_MQA_LOGITS_BLOCKSCALE_HEADDIM_OPTIONS:
        raise ValueError(
            f"Unsupported paged blockscale MQA logits headdim={config.headdim}"
        )
    if config.page_size not in PAGED_MQA_LOGITS_BLOCKSCALE_PAGE_SIZE_OPTIONS:
        raise ValueError(
            f"Unsupported paged blockscale MQA logits page_size={config.page_size}"
        )
    if config.next_n < 1:
        raise ValueError("Paged blockscale MQA logits requires next_n >= 1")
    if config.is_varlen and config.next_n != 1:
        raise ValueError("Varlen paged blockscale MQA logits requires next_n=1")
    _, _, _, stages_q, stages_kv = _get_paged_mqa_logits_blockscale_tile(
        config.num_heads, config.next_n
    )
    if config.stages_q != stages_q or config.stages_kv != stages_kv:
        raise ValueError(
            "Paged blockscale MQA logits currently requires stages_q=2 and stages_kv=3"
        )
    supported_qkv_dtypes = (torch.float8_e4m3fn,) + (
        (_MXFP4_DTYPE,) if _MXFP4_DTYPE is not None else ()
    )
    if config.qkv_dtype not in supported_qkv_dtypes:
        raise ValueError(
            f"Unsupported paged blockscale MQA logits qkv_dtype={config.qkv_dtype}"
        )
    if config.weights_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported paged blockscale MQA logits weights_dtype={config.weights_dtype}"
        )
    if config.logits_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported paged blockscale MQA logits logits_dtype={config.logits_dtype}"
        )
    if config.weights_dtype == torch.bfloat16 and config.logits_dtype != torch.bfloat16:
        raise ValueError("bfloat16 weights require bfloat16 logits")


def _config_dict(
    config: PagedMQALogitsBlockScaleJITConfig,
) -> Dict[str, object]:
    is_mxfp4 = _MXFP4_DTYPE is not None and config.qkv_dtype == _MXFP4_DTYPE
    is_wbf16 = config.weights_dtype == torch.bfloat16
    is_obf16 = config.logits_dtype == torch.bfloat16
    _, _, tile_kv, _, _ = _get_paged_mqa_logits_blockscale_tile(config.num_heads)
    return {
        "kind": "paged_mqa_logits_blockscale",
        "num_heads": config.num_heads,
        "headdim": config.headdim,
        "page_size": config.page_size,
        "tile_kv": tile_kv,
        "tile_q": config.tile_qh // config.num_heads,
        "tile_qh": config.tile_qh,
        "is_kv_scale_m_major": config.is_kv_scale_m_major,
        "is_kv_scale_m_major_literal": "true"
        if config.is_kv_scale_m_major
        else "false",
        "next_n": config.next_n,
        "is_context_lens_2d": config.is_context_lens_2d,
        "is_context_lens_2d_literal": (
            "true" if config.is_context_lens_2d else "false"
        ),
        "stages_q": config.stages_q,
        "stages_kv": config.stages_kv,
        "is_varlen": config.is_varlen,
        "is_varlen_literal": "true" if config.is_varlen else "false",
        "clean_logits": config.clean_logits,
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
    context_layout = "ctx2d" if config["is_context_lens_2d"] else "ctx1d"
    dispatch_name = (
        "mate_jit_paged_mqa_logits_blockscale_"
        f"{config['qkv_format']}_"
        f"h{config['num_heads']}_"
        f"d{config['headdim']}_"
        f"bn{config['tile_qh']}_"
        f"p{config['page_size']}_"
        f"n{config['next_n']}_"
        f"{context_layout}_"
        f"sq{config['stages_q']}_"
        f"skv{config['stages_kv']}_"
        f"{'clean' if config['clean_logits'] else 'noclean'}_"
        f"{'kvsm1' if config['is_kv_scale_m_major'] else 'kvsm0'}"
        f"{config['dtype_suffix']}"
    )
    return f"{dispatch_name}_varlen" if config["is_varlen"] else dispatch_name


def _render_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "paged_mqa_logits_blockscale"
    return (
        TVM_HEADER
        + _template_env.get_template("paged_mqa_logits_blockscale_kernel.j2").render(
            render_config
        )
        + EXPORT_FUNC.render(render_config)
    )


def gen_paged_mqa_logits_blockscale_spec(
    config: PagedMQALogitsBlockScaleJITConfig,
) -> JitSpec:
    _validate_config(config)
    render_config = _config_dict(config)
    dispatch_name = _dispatch_name(render_config)
    source_path = jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=list(MP32_PAGED_MQA_LOGITS_BLOCKSCALE_FLAG),
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={source_path: _render_source(render_config)},
    )


def gen_paged_mqa_logits_blockscale_aot() -> list[JitSpec]:
    return [
        gen_paged_mqa_logits_blockscale_spec(config)
        for config in PAGED_MQA_LOGITS_BLOCKSCALE_CONFIGS
    ]


@functools.cache
def _load_module(
    frozen_config: tuple[
        int,
        int,
        int,
        int,
        bool,
        int,
        int,
        bool,
        bool,
        torch.dtype,
        int,
        bool,
        torch.dtype,
        torch.dtype,
    ],
):
    config = PagedMQALogitsBlockScaleJITConfig(*frozen_config)
    return gen_paged_mqa_logits_blockscale_spec(config).build_and_load()


def get_paged_mqa_logits_blockscale_module(
    num_heads: int,
    headdim: int,
    page_size: int,
    next_n: int,
    is_context_lens_2d: bool,
    is_varlen: bool = False,
    clean_logits: bool = False,
    qkv_dtype: torch.dtype = torch.float8_e4m3fn,
    tile_qh: Optional[int] = None,
    is_kv_scale_m_major: bool = False,
    weights_dtype: torch.dtype = torch.float32,
    logits_dtype: torch.dtype = torch.float32,
):
    _, selected_tile_qh, _, stages_q, stages_kv = _get_paged_mqa_logits_blockscale_tile(
        num_heads, next_n
    )
    if tile_qh is None:
        tile_qh = selected_tile_qh
    config = PagedMQALogitsBlockScaleJITConfig(
        num_heads=num_heads,
        headdim=headdim,
        page_size=page_size,
        next_n=next_n,
        is_context_lens_2d=is_context_lens_2d,
        stages_q=stages_q,
        stages_kv=stages_kv,
        is_varlen=is_varlen,
        clean_logits=clean_logits,
        qkv_dtype=qkv_dtype,
        tile_qh=tile_qh,
        is_kv_scale_m_major=is_kv_scale_m_major,
        weights_dtype=weights_dtype,
        logits_dtype=logits_dtype,
    )
    _validate_config(config)
    return _load_module(
        (
            config.num_heads,
            config.headdim,
            config.page_size,
            config.next_n,
            config.is_context_lens_2d,
            config.stages_q,
            config.stages_kv,
            config.is_varlen,
            config.clean_logits,
            config.qkv_dtype,
            config.tile_qh,
            config.is_kv_scale_m_major,
            config.weights_dtype,
            config.logits_dtype,
        )
    )


__all__ = [
    "PagedMQALogitsBlockScaleJITConfig",
    "PAGED_MQA_LOGITS_BLOCKSCALE_CONFIGS",
    "gen_paged_mqa_logits_blockscale_aot",
    "gen_paged_mqa_logits_blockscale_spec",
    "get_paged_mqa_logits_blockscale_module",
]
