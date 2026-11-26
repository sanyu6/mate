from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, List, Mapping

from jinja2 import Environment, FileSystemLoader

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import EXPORT_FUNC, TVM_HEADER
from .. import gemm_utils


MP32_MQA_LOGITS_BLOCKSCALE_METADATA_FLAG = [
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

PAGED_MQA_LOGITS_BLOCKSCALE_TILE_KV = 512


@dataclass(frozen=True)
class PagedMQALogitsBlockScaleMetadataJITConfig:
    next_n: int
    is_context_lens_2d: bool
    tile_kv: int
    num_mps: int
    is_varlen: bool = False


PAGED_MQA_LOGITS_BLOCKSCALE_METADATA_CONFIGS: List[
    PagedMQALogitsBlockScaleMetadataJITConfig
] = [
    PagedMQALogitsBlockScaleMetadataJITConfig(
        next_n=next_n,
        is_context_lens_2d=is_context_lens_2d,
        tile_kv=PAGED_MQA_LOGITS_BLOCKSCALE_TILE_KV,
        num_mps=8,
        is_varlen=is_varlen,
    )
    for is_varlen in (False, True)
    for is_context_lens_2d in (False, True)
    for next_n in (1, 2, 4, 6)
    if not is_varlen or next_n == 1
    if is_context_lens_2d or next_n == 1
]


_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def _config_dict(
    config: PagedMQALogitsBlockScaleMetadataJITConfig,
) -> Dict[str, object]:
    return {
        "kind": "paged_mqa_logits_blockscale_metadata",
        "next_n": config.next_n,
        "is_context_lens_2d": config.is_context_lens_2d,
        "is_context_lens_2d_literal": (
            "true" if config.is_context_lens_2d else "false"
        ),
        "tile_kv": config.tile_kv,
        "num_mps": config.num_mps,
        "is_varlen": config.is_varlen,
        "is_varlen_literal": "true" if config.is_varlen else "false",
    }


def _dispatch_name(config: Mapping[str, object]) -> str:
    context_layout = "ctx2d" if config["is_context_lens_2d"] else "ctx1d"
    request_layout = "varlen" if config["is_varlen"] else "regular"
    return (
        "mate_jit_paged_mqa_logits_blockscale_metadata_"
        f"n{config['next_n']}_"
        f"kv{config['tile_kv']}_"
        f"mp{config['num_mps']}_"
        f"{context_layout}_"
        f"{request_layout}"
    )


def _render_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "get_paged_mqa_logits_blockscale_metadata"
    return (
        TVM_HEADER
        + _template_env.get_template(
            "paged_mqa_logits_blockscale_metadata_kernel.j2"
        ).render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_paged_mqa_logits_blockscale_metadata_spec(
    config: PagedMQALogitsBlockScaleMetadataJITConfig,
) -> JitSpec:
    render_config = _config_dict(config)
    dispatch_name = _dispatch_name(render_config)
    source_path = jit_env.MATE_GEN_SRC_DIR / "deep_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cuda_cflags=list(MP32_MQA_LOGITS_BLOCKSCALE_METADATA_FLAG),
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={source_path: _render_source(render_config)},
    )


def gen_paged_mqa_logits_blockscale_metadata_aot() -> list[JitSpec]:
    return [
        gen_paged_mqa_logits_blockscale_metadata_spec(config)
        for config in PAGED_MQA_LOGITS_BLOCKSCALE_METADATA_CONFIGS
    ]


@functools.cache
def _load_module(frozen_config: tuple[int, bool, int, int, bool]):
    config = PagedMQALogitsBlockScaleMetadataJITConfig(*frozen_config)
    return gen_paged_mqa_logits_blockscale_metadata_spec(config).build_and_load()


def get_paged_mqa_logits_blockscale_metadata_module(
    next_n: int,
    is_context_lens_2d: bool,
    tile_kv: int,
    num_mps: int,
    is_varlen: bool = False,
):
    del tile_kv
    return _load_module(
        (
            next_n,
            is_context_lens_2d,
            PAGED_MQA_LOGITS_BLOCKSCALE_TILE_KV,
            num_mps,
            is_varlen,
        )
    )


__all__ = [
    "PagedMQALogitsBlockScaleMetadataJITConfig",
    "PAGED_MQA_LOGITS_BLOCKSCALE_METADATA_CONFIGS",
    "PAGED_MQA_LOGITS_BLOCKSCALE_TILE_KV",
    "gen_paged_mqa_logits_blockscale_metadata_aot",
    "gen_paged_mqa_logits_blockscale_metadata_spec",
    "get_paged_mqa_logits_blockscale_metadata_module",
]
