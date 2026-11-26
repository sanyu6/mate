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

MQA_LOGITS_BLOCKSCALE_METADATA_TILE_KV = 512


@dataclass(frozen=True)
class MQALogitsBlockScaleMetadataJITConfig:
    tile_q: int
    tile_kv: int
    num_mps: int


MQA_LOGITS_BLOCKSCALE_METADATA_CONFIGS: List[MQALogitsBlockScaleMetadataJITConfig] = [
    MQALogitsBlockScaleMetadataJITConfig(
        tile_q=tile_q,
        tile_kv=MQA_LOGITS_BLOCKSCALE_METADATA_TILE_KV,
        num_mps=48,
    )
    for tile_q in (2, 4, 8)
]


_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def _config_dict(
    config: MQALogitsBlockScaleMetadataJITConfig,
) -> Dict[str, object]:
    return {
        "kind": "mqa_logits_blockscale_metadata",
        "tile_q": config.tile_q,
        "tile_kv": config.tile_kv,
        "num_mps": config.num_mps,
    }


def _dispatch_name(config: Mapping[str, object]) -> str:
    return (
        "mate_jit_mqa_logits_blockscale_metadata_"
        f"q{config['tile_q']}_"
        f"kv{config['tile_kv']}_"
        f"mp{config['num_mps']}"
    )


def _render_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = "get_mqa_logits_blockscale_metadata"
    return (
        TVM_HEADER
        + _template_env.get_template("mqa_logits_blockscale_metadata_kernel.j2").render(
            render_config
        )
        + EXPORT_FUNC.render(render_config)
    )


def gen_mqa_logits_blockscale_metadata_spec(
    config: MQALogitsBlockScaleMetadataJITConfig,
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


def gen_mqa_logits_blockscale_metadata_aot() -> list[JitSpec]:
    return [
        gen_mqa_logits_blockscale_metadata_spec(config)
        for config in MQA_LOGITS_BLOCKSCALE_METADATA_CONFIGS
    ]


@functools.cache
def _load_module(frozen_config: tuple[int, int, int]):
    config = MQALogitsBlockScaleMetadataJITConfig(*frozen_config)
    return gen_mqa_logits_blockscale_metadata_spec(config).build_and_load()


def get_mqa_logits_blockscale_metadata_module(
    tile_q: int,
    tile_kv: int,
    num_mps: int,
):
    del tile_kv
    return _load_module(
        (
            tile_q,
            MQA_LOGITS_BLOCKSCALE_METADATA_TILE_KV,
            num_mps,
        )
    )


__all__ = [
    "MQALogitsBlockScaleMetadataJITConfig",
    "MQA_LOGITS_BLOCKSCALE_METADATA_CONFIGS",
    "MQA_LOGITS_BLOCKSCALE_METADATA_TILE_KV",
    "gen_mqa_logits_blockscale_metadata_aot",
    "gen_mqa_logits_blockscale_metadata_spec",
    "get_mqa_logits_blockscale_metadata_module",
]
