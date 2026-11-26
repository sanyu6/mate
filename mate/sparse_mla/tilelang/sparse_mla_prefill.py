# ruff: noqa
"""Unified TileLang sparse prefill entrypoint."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...mate_runtime import (
    MP31_ARCH,
    MP32_ARCH,
    require_musa_arch,
)

from .mp31.sparse_mla_v32_fwd_pipelined import (
    tilelang_sparse_mla_prefill_fwd_interface as _prefill_v32_mp31,
)
from .mp32.sparse_mla_v32_fwd_pipelined import (
    tilelang_sparse_mla_prefill_fwd_interface as _prefill_v32_mp32,
)
from .mp31.sparse_mla_v4_fwd_generic import (
    sparse_mla_fwd_interface_v4_generic as _prefill_v4_generic,
)
from .mp32.sparse_mla_v4_mp32_fwd_pipelined import (
    sparse_mla_fwd_interface_v4_mp32_optimized as _prefill_v4_mp32_optimized,
)


_V4_PERSISTENT_MAX_QUERY_TOKENS = 65535


def _use_v4_persistence(
    seq_len_q: int,
    num_heads_q: int,
    topk_length: Optional[torch.Tensor],
) -> bool:
    # Long H64 variable-work loops can stall while carrying barrier phases.
    return not (
        num_heads_q == 64
        and seq_len_q > _V4_PERSISTENT_MAX_QUERY_TOKENS
        and topk_length is not None
    )


def sparse_mla_prefill_fwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch sparse prefill by compile-time layout variant."""

    assert d_v == 512, "sparse prefill only supports d_v == 512"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.bfloat16, "kv must be bfloat16"
    assert indices.dtype == torch.int32, "indices must be int32"
    seq_len_q, num_heads_q, head_dim_q = q.shape
    _, h_k, _ = kv.shape
    assert q.stride(-1) == 1, "q last dimension must be contiguous"
    assert kv.stride(-1) == 1, "kv last dimension must be contiguous"
    assert indices.stride(-1) == 1, "indices last dimension must be contiguous"
    assert head_dim_q == kv.shape[-1]
    assert seq_len_q == indices.shape[0]
    assert h_k == indices.shape[1]
    assert h_k == 1

    arch = require_musa_arch(
        q,
        (MP31_ARCH, MP32_ARCH),
        "sparse MLA prefill",
    )
    is_mp32 = arch == MP32_ARCH

    if head_dim_q == 512:
        topk = indices.shape[-1]
        assert topk % 64 == 0, "V4 sparse prefill requires topk to be a multiple of 64"

        # MP32 has a different shared-memory/layout contract (SG16 and a
        # two-stage KV buffer).  Do not silently route an unsupported MP32
        # request through an MP31/V4 kernel; that would make the device
        # selection depend on incidental wrapper shapes.
        if is_mp32:
            # The MP32 optimized factory has two validated compile-time QK
            # tile variants.  H64 benefits from M32N128K128 at large TOPK;
            # smaller TOPK keeps the lower launch/pipeline-fill variant.
            block_i = 128 if num_heads_q == 64 and topk >= 4096 else 64
            mp32_out = _prefill_v4_mp32_optimized(
                q,
                kv,
                indices,
                topk_length=topk_length,
                sm_scale=sm_scale,
                attn_sink=attn_sink,
                d_v=d_v,
                return_max_logits=True,
                is_persistence=_use_v4_persistence(
                    seq_len_q,
                    num_heads_q,
                    topk_length,
                ),
                block_i=block_i,
            )
            if mp32_out is None:
                raise RuntimeError(
                    "MP32 V4 BF16 prefill does not support this shape or ABI"
                )
            return mp32_out

        return _prefill_v4_generic(
            q,
            kv,
            indices,
            topk_length=topk_length,
            sm_scale=sm_scale,
            attn_sink=attn_sink,
            d_v=d_v,
            return_max_logits=True,
            is_persistence=_use_v4_persistence(
                seq_len_q,
                num_heads_q,
                topk_length,
            ),
        )

    if head_dim_q == 576:
        prefill_v32 = _prefill_v32_mp32 if is_mp32 else _prefill_v32_mp31
        return prefill_v32(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            return_max_logits=True,
        )

    raise AssertionError(f"Unsupported sparse prefill q.shape[-1]: {head_dim_q}")
