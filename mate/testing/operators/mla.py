"""Structured operator lifecycle types for the public MLA APIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Callable, Literal

import torch

from mate.testing.sparse_mla import (
    dequantize_sparse_mla_cache,
    ref_sparse_mla_attention,
)

from .operator import (
    Operator,
    OperatorInputs,
    OperatorOutputs,
    OperatorReference,
    OperatorWorkload,
)


class MlaApi(Enum):
    SPARSE_FWD = "sparse_fwd"
    SPARSE_FWD_PACK8 = "sparse_fwd_pack8"
    KVCACHE = "kvcache"


@dataclass(frozen=True, kw_only=True)
class MlaWorkload(OperatorWorkload):
    """Description from which concrete MLA inputs are generated.

    Legacy callables still execute through ``call``. To verify such a callable,
    supply an independent ``reference_call``; the operator never self-compares.
    """

    api: MlaApi | None = None
    call: Callable[[], Any] | None = None
    reference_call: Callable[[], Any] | None = None
    batch_size: int = 1
    seqlen_q: int = 1
    seqlen_kv: int = 1
    num_head_q: int = 128
    num_head_kv: int = 1
    headdim_latent: int = 512
    headdim_rope: int = 64
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype | None = None
    device: str | torch.device = "musa"
    seed: int | None = 666
    page_size: int = 64
    topk: int = 512
    topk_length_mode: str = "full"
    seqlen_mode_kv: str = "full"
    causal: bool = False
    causal_window: int = 0
    is_fp8_kvcache: bool = False
    attn_sink: bool = False
    softmax_scale: float | None = None
    head_dim_v: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.call is not None:
            if self.api is not None:
                raise ValueError("provide api or a legacy call, not both")
            return
        if not isinstance(self.api, MlaApi):
            raise ValueError("structured MLA workloads require a MlaApi")
        for name in ("batch_size", "seqlen_q", "seqlen_kv", "num_head_q", "page_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_head_kv != 1 or self.resolved_head_dim_v != 512:
            raise ValueError("MLA requires num_head_kv=1 and head_dim_v=512")
        if self.headdim_latent != 512 or self.headdim_rope not in (0, 64):
            raise ValueError("MLA requires latent=512 and rope=0 or 64")
        if self.causal_window < 0:
            raise ValueError("causal_window must be non-negative")
        if self.topk_length_mode not in ("full", "random"):
            raise ValueError("topk_length_mode must be 'full' or 'random'")
        if self.seqlen_mode_kv not in ("full", "random"):
            raise ValueError("seqlen_mode_kv must be 'full' or 'random'")
        kv_dtype = self.dtype if self.kv_dtype is None else self.kv_dtype
        if kv_dtype != self.dtype:
            raise ValueError("generated Q and KV must use matching dtypes")
        if self.api is MlaApi.KVCACHE:
            if self.head_dim_qk != 576 or self.page_size != 64:
                raise ValueError(
                    "generated KV-cache workloads require dim=576 and page_size=64"
                )
            if self.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("dense MLA requires bfloat16 or float16")
            if self.causal and self.seqlen_q > self.seqlen_kv:
                raise ValueError("causal MLA requires seqlen_q <= seqlen_kv")
            if self.is_fp8_kvcache and (self.dtype != torch.bfloat16 or self.causal):
                raise ValueError("FP8 sparse MLA requires bfloat16 Q and causal=False")
        else:
            if self.dtype != torch.bfloat16:
                raise ValueError("sparse prefill requires bfloat16 Q and KV")
            if self.is_fp8_kvcache:
                raise ValueError("is_fp8_kvcache is only valid for KV-cache workloads")
        if self.api is MlaApi.SPARSE_FWD_PACK8:
            if self.num_head_q != 8 or self.head_dim_qk != 512 or self.seqlen_q % 8:
                raise ValueError(
                    "pack8 MLA requires Hq=8, dim=512, and seqlen_q divisible by 8"
                )
        elif self.api is MlaApi.SPARSE_FWD or self.is_fp8_kvcache:
            if self.topk <= 0 or self.topk % 64:
                raise ValueError(
                    "sparse MLA requires topk to be a positive multiple of 64"
                )

    @property
    def head_dim_qk(self) -> int:
        return self.headdim_latent + self.headdim_rope

    @property
    def resolved_head_dim_v(self) -> int:
        return self.headdim_latent if self.head_dim_v is None else self.head_dim_v


@dataclass(kw_only=True)
class MlaInputs(OperatorInputs):
    """Concrete public-API arguments, usable without a generated workload.

    Prefill Q is [query, head, dim]; cache Q is [batch, query, head, dim].
    Pack8 indices are per eight queries; decode topk lengths are per batch.
    """

    api: MlaApi | None = None
    workload: MlaWorkload | None = None

    q: torch.Tensor | None = None
    # V3.2 prefill needs physical storage for at least 64 heads. Generated
    # logical tensors are views into these buffers, avoiding per-call copies.
    q_padded: torch.Tensor | None = None
    attn_sink_padded: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    k_cache: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    topk_length: torch.Tensor | None = None
    attn_sink: torch.Tensor | None = None
    tile_scheduler_metadata: torch.Tensor | None = None
    num_splits: torch.Tensor | None = None
    extra_k_cache: torch.Tensor | None = None
    extra_indices_in_kvcache: torch.Tensor | None = None
    extra_topk_length: torch.Tensor | None = None

    head_dim_v: int = 512
    softmax_scale: float | None = None
    causal: bool = False
    is_fp8_kvcache: bool = False
    causal_window: int = 0
    row_masks: torch.Tensor | None = None
    compressed_kv_len: int = 0
    compress_ratio: int = 1
    pack_metadata: object | None = None
    prepare_scheduler_metadata: bool = True

    # Compatibility escape hatch for callers that supplied a complete callable.
    legacy_call: Callable[[], Any] | None = None
    legacy_reference: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        if self.workload is not None:
            if self.api is None:
                self.api = self.workload.api
            if self.legacy_call is None:
                self.legacy_call = self.workload.call
            if self.legacy_reference is None:
                self.legacy_reference = self.workload.reference_call


@dataclass(frozen=True, kw_only=True)
class MlaOutputs(OperatorOutputs):
    """Named MLA outputs; value is retained for legacy callable consumers.

    Dense cache LSE is [batch, query, head], sparse cache LSE is
    [batch, head, query], and prefill LSE/max_logits are [query, head].
    """

    out: torch.Tensor | None = None
    max_logits: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    value: Any = None


@dataclass(frozen=True, kw_only=True)
class MlaReference(OperatorReference):
    out: torch.Tensor | None = None
    max_logits: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    value: Any = None


def _split_result(value: Any) -> MlaOutputs:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, torch.Tensor) for item in value
    ):
        raise TypeError("MLA APIs must return a tuple of tensors")
    if len(value) == 3:
        return MlaOutputs(out=value[0], max_logits=value[1], lse=value[2], value=value)
    if len(value) == 2:
        return MlaOutputs(out=value[0], lse=value[1], value=value)
    raise ValueError("MLA APIs must return two or three tensors")


def _make_indices(
    rows: int, topk: int, seqlen_kv: int, device: torch.device
) -> torch.Tensor:
    indices = torch.full((rows, 1, topk), -1, dtype=torch.int32, device=device)
    valid = min(topk, seqlen_kv)
    for row in range(rows):
        indices[row, 0, :valid] = torch.randperm(
            seqlen_kv, device=device, dtype=torch.int32
        )[:valid]
    return indices


def _make_topk_length(
    workload: MlaWorkload,
    rows: int,
    device: torch.device,
    *,
    topk: int | None = None,
) -> torch.Tensor:
    topk = workload.topk if topk is None else topk
    if workload.topk_length_mode == "full":
        return torch.full((rows,), topk, dtype=torch.int32, device=device)
    return torch.randint(1, topk + 1, (rows,), dtype=torch.int32, device=device)


def _make_pack8_indices(
    workload: MlaWorkload, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_rows = workload.seqlen_q // 8
    max_pack_kv = (
        min(workload.seqlen_kv, workload.causal_window + 7)
        if workload.causal_window > 0
        else min(workload.seqlen_q, workload.seqlen_kv)
    )
    topk = max(64, math.ceil(max_pack_kv / 64) * 64)
    offsets = torch.arange(topk, dtype=torch.int32, device=device)
    pack_starts = torch.arange(packed_rows, dtype=torch.int32, device=device) * 8
    pack_ends = torch.clamp(pack_starts + 7, max=workload.seqlen_kv - 1)
    starts = (
        torch.clamp(pack_starts - workload.causal_window + 1, min=0)
        if workload.causal_window > 0
        else torch.zeros_like(pack_starts)
    )
    lengths = torch.clamp(pack_ends - starts + 1, min=0, max=topk)
    indices = starts[:, None] + offsets[None, :]
    indices = torch.where(
        offsets[None, :] < lengths[:, None],
        indices,
        torch.full_like(indices, -1),
    )[:, None, :].contiguous()
    return indices, lengths


def _quantize_v32_cache(kv: torch.Tensor) -> torch.Tensor:
    """Create the 656-byte/token V3.2 sparse cache layout."""
    blocks, page_size, heads, dim = kv.shape
    if heads != 1 or dim != 576:
        raise ValueError("V3.2 sparse cache requires [blocks,page_size,1,576]")
    source = kv.squeeze(2)
    out = torch.empty((blocks, page_size, 656), dtype=torch.uint8, device=kv.device)
    nope = out[..., :512].view(torch.float8_e4m3fn)
    scales = out[..., 512:528].view(torch.float32)
    rope = out[..., 528:].view(torch.bfloat16)
    rope.copy_(source[..., 512:])
    for tile in range(4):
        part = source[..., tile * 128 : (tile + 1) * 128].float()
        scale = torch.pow(
            2.0, torch.clamp_min(part.abs().amax(-1) / 448.0, 1e-4).log2().ceil()
        )
        scales[..., tile] = scale
        nope[..., tile * 128 : (tile + 1) * 128] = (part / scale.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )
    return out.view(blocks, page_size, 1, 656)


def _dense_reference(inputs: MlaInputs) -> tuple[torch.Tensor, torch.Tensor]:
    assert inputs.q is not None and inputs.k_cache is not None
    assert inputs.block_table is not None and inputs.cache_seqlens is not None
    q = inputs.q.float()
    cache = inputs.k_cache
    if inputs.is_fp8_kvcache:
        cache = dequantize_sparse_mla_cache(cache, "v32")
    cache = cache[..., 0, :].float()
    batch, q_len, heads, dim = q.shape
    out = torch.empty(
        (batch, q_len, heads, inputs.head_dim_v), dtype=torch.float32, device=q.device
    )
    # Dense public API returns [batch, query, head], unlike sparse decode.
    lse = torch.empty((batch, q_len, heads), dtype=torch.float32, device=q.device)
    scale = dim**-0.5 if inputs.softmax_scale is None else inputs.softmax_scale
    for b in range(batch):
        length = int(inputs.cache_seqlens[b].item())
        pages = math.ceil(length / inputs.k_cache.shape[1])
        page_ids = inputs.block_table[b, :pages].long()
        selected = cache.index_select(0, page_ids).reshape(-1, dim)[:length]
        k = selected[:, None, :].expand(-1, heads, -1)
        v = k[..., : inputs.head_dim_v]
        logits = torch.einsum("qhd,khd->hqk", q[b], k) * scale
        if inputs.causal:
            mask = torch.triu(
                torch.ones(q_len, length, dtype=torch.bool, device=q.device),
                diagonal=length - q_len + 1,
            )
            logits = logits.masked_fill(mask[None], float("-inf"))
        lse[b] = torch.logsumexp(logits, dim=-1).transpose(0, 1)
        out[b] = torch.einsum(
            "hqk,khd->qhd", torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0), v
        )
    return out.to(inputs.q.dtype), lse


def _pack8_reference_indices(
    inputs: MlaInputs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expand pack selections, then apply the mask for each of the eight queries."""
    assert inputs.q is not None and inputs.kv is not None
    rows = inputs.q.shape[0]
    lengths = inputs.topk_length
    if inputs.indices is None:
        if inputs.row_masks is not None:
            raise ValueError("row_masks requires explicit indices")
        indices = torch.arange(
            inputs.kv.shape[0], dtype=torch.int32, device=inputs.q.device
        )
        indices = indices.view(1, 1, -1).expand(rows, 1, -1)
        lengths = None
    else:
        indices = inputs.indices.repeat_interleave(8, dim=0)
        if lengths is not None:
            lengths = lengths.repeat_interleave(8)
    row = torch.arange(rows, device=inputs.q.device).view(-1, 1, 1)
    if inputs.row_masks is not None:
        masks = inputs.row_masks.repeat_interleave(8, dim=0)
        valid = ((masks >> (row % 8)) & 1).bool()
    else:
        logical_pos = indices
        if inputs.compressed_kv_len:
            logical_pos = torch.where(
                indices < inputs.compressed_kv_len,
                (indices + 1) * inputs.compress_ratio - 1,
                indices - inputs.compressed_kv_len,
            )
        valid = logical_pos <= row
        if inputs.causal_window:
            in_window = logical_pos > row - inputs.causal_window
            if inputs.compressed_kv_len:
                in_window |= indices < inputs.compressed_kv_len
            valid &= in_window
    return indices.masked_fill(~valid, -1), lengths


def _sparse_cache_reference(inputs: MlaInputs, scale: float) -> MlaReference:
    assert (
        inputs.q is not None
        and inputs.k_cache is not None
        and inputs.indices is not None
    )
    batch, rows, heads, dim = inputs.q.shape
    variant: Literal["v32", "v4"] = "v32" if dim == 576 else "v4"
    cache = dequantize_sparse_mla_cache(inputs.k_cache, variant).reshape(-1, 1, dim)
    extra_kv = extra_indices = extra_lengths = None
    if inputs.extra_k_cache is not None:
        if variant == "v32":
            raise ValueError("V3.2 sparse decode does not support extra KV")
        if inputs.extra_indices_in_kvcache is None:
            raise ValueError("extra_k_cache requires extra_indices_in_kvcache")
        extra_kv = dequantize_sparse_mla_cache(inputs.extra_k_cache, variant).reshape(
            -1, 1, dim
        )
        extra_indices = inputs.extra_indices_in_kvcache.reshape(batch * rows, 1, -1)
        if inputs.extra_topk_length is not None:
            extra_lengths = inputs.extra_topk_length.repeat_interleave(rows)
    lengths = (
        inputs.topk_length.repeat_interleave(rows)
        if inputs.topk_length is not None
        else None
    )
    out, _, lse = ref_sparse_mla_attention(
        inputs.q.reshape(batch * rows, heads, dim),
        cache,
        inputs.indices.reshape(batch * rows, 1, -1),
        sm_scale=scale,
        d_v=inputs.head_dim_v,
        topk_length=lengths,
        attn_sink=inputs.attn_sink,
        extra_kv=extra_kv,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
    )
    return MlaReference(
        out=out.reshape(batch, rows, heads, inputs.head_dim_v),
        lse=lse.reshape(batch, rows, heads).transpose(1, 2),
    )


class MlaOperator(Operator):
    """Generate, invoke, reference, and verify structured MLA workloads."""

    def generate(self, workload: OperatorWorkload) -> MlaInputs:
        if not isinstance(workload, MlaWorkload):
            raise TypeError("expected MlaWorkload")
        if workload.call is not None and workload.api is None:
            return MlaInputs(
                workload=workload,
                legacy_call=workload.call,
                legacy_reference=workload.reference_call,
            )
        assert workload.api is not None
        if workload.seed is not None:
            torch.manual_seed(workload.seed)
        device = torch.device(workload.device)
        dtype = workload.dtype
        kv_dtype = workload.kv_dtype or dtype
        inputs = MlaInputs(
            api=workload.api,
            workload=workload,
            head_dim_v=workload.resolved_head_dim_v,
            softmax_scale=(
                workload.head_dim_qk**-0.5
                if workload.softmax_scale is None
                else workload.softmax_scale
            ),
            causal=workload.causal,
            is_fp8_kvcache=workload.is_fp8_kvcache,
            causal_window=workload.causal_window,
        )
        if workload.api in (MlaApi.SPARSE_FWD, MlaApi.SPARSE_FWD_PACK8):
            q_rows = workload.seqlen_q
            q = torch.randn(
                q_rows,
                workload.num_head_q,
                workload.head_dim_qk,
                dtype=dtype,
                device=device,
            )
            kv = torch.randn(
                workload.seqlen_kv,
                workload.num_head_kv,
                workload.head_dim_qk,
                dtype=kv_dtype,
                device=device,
            )
            inputs.q, inputs.kv = q, kv
            if workload.api is MlaApi.SPARSE_FWD_PACK8:
                inputs.indices, inputs.topk_length = _make_pack8_indices(
                    workload, device
                )
            else:
                inputs.indices = _make_indices(
                    q_rows, workload.topk, workload.seqlen_kv, device
                )
                inputs.topk_length = _make_topk_length(workload, q_rows, device)
            if workload.attn_sink:
                inputs.attn_sink = torch.randn(
                    workload.num_head_q, dtype=torch.float32, device=device
                )
            if (
                workload.api is MlaApi.SPARSE_FWD
                and workload.head_dim_qk == 576
                and workload.num_head_q < 64
            ):
                heads = workload.num_head_q
                inputs.q_padded = torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
                inputs.q = inputs.q_padded[:, :heads, :]
                if inputs.attn_sink is not None:
                    inputs.attn_sink_padded = torch.nn.functional.pad(
                        inputs.attn_sink, (0, 64 - heads)
                    )
                    inputs.attn_sink = inputs.attn_sink_padded[:heads]
            return inputs
        q = torch.randn(
            workload.batch_size,
            workload.seqlen_q,
            workload.num_head_q,
            workload.head_dim_qk,
            dtype=dtype,
            device=device,
        )
        pages = math.ceil(workload.seqlen_kv / workload.page_size)
        raw = torch.randn(
            pages * (1 if workload.is_fp8_kvcache else workload.batch_size),
            workload.page_size,
            1,
            workload.head_dim_qk,
            dtype=kv_dtype,
            device=device,
        )
        cache = _quantize_v32_cache(raw) if workload.is_fp8_kvcache else raw
        table = torch.arange(
            pages * workload.batch_size, dtype=torch.int32, device=device
        ).view(workload.batch_size, pages)
        lengths = torch.full(
            (workload.batch_size,), workload.seqlen_kv, dtype=torch.int32, device=device
        )
        if workload.seqlen_mode_kv == "random":
            lengths = torch.randint(
                workload.seqlen_q if workload.causal else 1,
                workload.seqlen_kv + 1,
                (workload.batch_size,),
                dtype=torch.int32,
                device=device,
            )
        for batch_idx in range(workload.batch_size):
            table[batch_idx] = (
                torch.randperm(pages, dtype=torch.int32, device=device)
                + batch_idx * pages
            )
        inputs.q, inputs.k_cache, inputs.block_table, inputs.cache_seqlens = (
            q,
            cache,
            table,
            lengths,
        )
        if workload.is_fp8_kvcache:
            inputs.block_table = None
            inputs.cache_seqlens = None
            inputs.indices = _make_indices(
                workload.batch_size * workload.seqlen_q,
                workload.topk,
                workload.seqlen_kv,
                device,
            ).view(workload.batch_size, workload.seqlen_q, 1, workload.topk)
            inputs.topk_length = _make_topk_length(
                workload, workload.batch_size, device
            )
            if workload.attn_sink:
                inputs.attn_sink = torch.randn(
                    workload.num_head_q, dtype=torch.float32, device=device
                )
        # CPU generation is useful for reference tests; do not fabricate scheduler data.
        if device.type == "musa":
            self._prepare_metadata(inputs)
        return inputs

    @staticmethod
    def _prepare_metadata(inputs: MlaInputs) -> None:
        if (inputs.tile_scheduler_metadata is None) != (inputs.num_splits is None):
            raise ValueError(
                "tile_scheduler_metadata and num_splits must be supplied together"
            )
        if inputs.tile_scheduler_metadata is not None:
            return
        if not inputs.prepare_scheduler_metadata:
            raise ValueError("KV-cache calls require scheduler metadata")
        assert inputs.q is not None and inputs.k_cache is not None
        q = inputs.q
        api = import_module("mate.flashmla")
        inputs.tile_scheduler_metadata, inputs.num_splits = api.get_mla_metadata(
            cache_seqlens=inputs.cache_seqlens,
            num_q_tokens_per_head_k=q.shape[1] * q.shape[2] // inputs.k_cache.shape[2],
            num_heads_k=inputs.k_cache.shape[2],
            num_heads_q=q.shape[2],
            is_fp8_kvcache=inputs.is_fp8_kvcache,
            topk=inputs.indices.shape[-1] if inputs.indices is not None else None,
            extra_topk=inputs.extra_indices_in_kvcache.shape[-1]
            if inputs.extra_indices_in_kvcache is not None
            else None,
            q=q if inputs.indices is not None else None,
            bs=q.shape[0] if inputs.indices is not None else None,
            topk_length=inputs.topk_length,
            extra_topk_length=inputs.extra_topk_length,
        )

    def call(self, inputs: OperatorInputs) -> MlaOutputs:
        if not isinstance(inputs, MlaInputs):
            raise TypeError("expected MlaInputs")
        if inputs.legacy_call is not None:
            return MlaOutputs(value=inputs.legacy_call())
        self._validate_inputs(inputs)
        api = import_module("mate.flashmla")
        if inputs.api is MlaApi.SPARSE_FWD:
            assert (
                inputs.q is not None
                and inputs.kv is not None
                and inputs.indices is not None
            )
            q, sink = inputs.q, inputs.attn_sink
            heads = q.shape[1]
            pad_heads = q.shape[-1] == 576 and heads < 64
            if pad_heads:
                # Reuse generated storage only while the logical input is its
                # view. Direct/replaced/cloned inputs are padded from their
                # current values, so cached buffers can never go stale.
                q = (
                    inputs.q_padded
                    if inputs.q_padded is not None and q._base is inputs.q_padded
                    else torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
                )
                if sink is not None:
                    sink = (
                        inputs.attn_sink_padded
                        if inputs.attn_sink_padded is not None
                        and sink._base is inputs.attn_sink_padded
                        else torch.nn.functional.pad(sink, (0, 64 - heads))
                    )
            result = api.flash_mla_sparse_fwd(
                q=q,
                kv=inputs.kv,
                indices=inputs.indices,
                sm_scale=inputs.q.shape[-1] ** -0.5
                if inputs.softmax_scale is None
                else inputs.softmax_scale,
                d_v=inputs.head_dim_v,
                attn_sink=sink,
                topk_length=inputs.topk_length,
            )
            if pad_heads and result[0].shape[1] > heads:
                out, max_logits, lse = result
                result = (out[:, :heads, :], max_logits[:, :heads], lse[:, :heads])
            return _split_result(result)
        if inputs.api is MlaApi.SPARSE_FWD_PACK8:
            assert inputs.q is not None and inputs.kv is not None
            return _split_result(
                api.flash_mla_sparse_fwd_pack8(
                    q=inputs.q,
                    kv=inputs.kv,
                    indices=inputs.indices,
                    sm_scale=inputs.softmax_scale,
                    d_v=inputs.head_dim_v,
                    attn_sink=inputs.attn_sink,
                    topk_length=inputs.topk_length,
                    causal_window=inputs.causal_window,
                    row_masks=inputs.row_masks,
                    compressed_kv_len=inputs.compressed_kv_len,
                    compress_ratio=inputs.compress_ratio,
                    pack_metadata=inputs.pack_metadata,
                )
            )
        assert inputs.q is not None and inputs.k_cache is not None
        self._prepare_metadata(inputs)
        return _split_result(
            api.flash_mla_with_kvcache(
                q=inputs.q,
                k_cache=inputs.k_cache,
                block_table=inputs.block_table,
                cache_seqlens=inputs.cache_seqlens,
                head_dim_v=inputs.head_dim_v,
                tile_scheduler_metadata=inputs.tile_scheduler_metadata,
                num_splits=inputs.num_splits,
                softmax_scale=inputs.softmax_scale,
                causal=inputs.causal,
                is_fp8_kvcache=inputs.is_fp8_kvcache,
                indices=inputs.indices,
                attn_sink=inputs.attn_sink,
                extra_k_cache=inputs.extra_k_cache,
                extra_indices_in_kvcache=inputs.extra_indices_in_kvcache,
                topk_length=inputs.topk_length,
                extra_topk_length=inputs.extra_topk_length,
            )
        )

    @staticmethod
    def _validate_inputs(inputs: MlaInputs) -> None:
        if not isinstance(inputs.api, MlaApi):
            raise ValueError("structured MLA inputs require a MlaApi")
        if inputs.q is None:
            raise ValueError("MLA requires q")
        if inputs.api is MlaApi.KVCACHE:
            if inputs.q.ndim != 4 or inputs.k_cache is None:
                raise ValueError("KV-cache requires BSHD q and k_cache")
            if inputs.indices is None:
                if inputs.is_fp8_kvcache:
                    raise ValueError("FP8 KV-cache requires sparse indices")
                if inputs.block_table is None or inputs.cache_seqlens is None:
                    raise ValueError(
                        "dense KV-cache requires block_table and cache_seqlens"
                    )
                if any(
                    value is not None
                    for value in (
                        inputs.attn_sink,
                        inputs.topk_length,
                        inputs.extra_k_cache,
                        inputs.extra_indices_in_kvcache,
                        inputs.extra_topk_length,
                    )
                ):
                    raise ValueError(
                        "sparse-only parameters cannot be used with dense KV-cache"
                    )
            elif not inputs.is_fp8_kvcache or inputs.causal:
                raise ValueError(
                    "sparse KV-cache requires is_fp8_kvcache=True and causal=False"
                )
            if (inputs.extra_k_cache is None) != (
                inputs.extra_indices_in_kvcache is None
            ):
                raise ValueError(
                    "extra cache and extra indices must be supplied together"
                )
            if inputs.extra_topk_length is not None and inputs.extra_k_cache is None:
                raise ValueError("extra_topk_length requires extra cache")
            if inputs.q.shape[-1] == 576 and inputs.extra_k_cache is not None:
                raise ValueError("V3.2 sparse decode does not support extra KV")
        else:
            if inputs.q.ndim != 3 or inputs.kv is None:
                raise ValueError("prefill requires SHD q and kv")
            if inputs.api is MlaApi.SPARSE_FWD and inputs.indices is None:
                raise ValueError("sparse prefill requires indices")
            if inputs.api is MlaApi.SPARSE_FWD_PACK8:
                if inputs.q.shape[0] % 8 or inputs.q.shape[1:] != (8, 512):
                    raise ValueError("pack8 requires q shape [8*n, 8, 512]")
                if inputs.pack_metadata is not None:
                    raise ValueError("pack_metadata is reserved by the public API")
                if inputs.compressed_kv_len < 0 or inputs.compress_ratio < 1:
                    raise ValueError("invalid pack8 compression parameters")

    def reference(self, inputs: OperatorInputs) -> MlaReference:
        if not isinstance(inputs, MlaInputs):
            raise TypeError("expected MlaInputs")
        if inputs.legacy_call is not None:
            if inputs.legacy_reference is None:
                raise NotImplementedError(
                    "legacy call requires an independent reference_call for verification"
                )
            return MlaReference(value=inputs.legacy_reference())
        self._validate_inputs(inputs)
        assert inputs.q is not None
        scale = (
            inputs.q.shape[-1] ** -0.5
            if inputs.softmax_scale is None
            else inputs.softmax_scale
        )
        if inputs.api in (MlaApi.SPARSE_FWD, MlaApi.SPARSE_FWD_PACK8):
            assert inputs.kv is not None
            indices, lengths = inputs.indices, inputs.topk_length
            if inputs.api is MlaApi.SPARSE_FWD_PACK8:
                indices, lengths = _pack8_reference_indices(inputs)
            assert indices is not None
            out, max_logits, lse = ref_sparse_mla_attention(
                inputs.q,
                inputs.kv,
                indices,
                sm_scale=scale,
                d_v=inputs.head_dim_v,
                topk_length=lengths,
                attn_sink=inputs.attn_sink,
            )
            return MlaReference(out=out, max_logits=max_logits, lse=lse)
        if inputs.indices is not None:
            return _sparse_cache_reference(inputs, scale)
        out, lse = _dense_reference(inputs)
        return MlaReference(out=out, lse=lse)

    def verify(
        self,
        inputs: OperatorInputs,
        outputs: OperatorOutputs,
        reference: OperatorReference,
    ) -> None:
        if (
            not isinstance(inputs, MlaInputs)
            or not isinstance(outputs, MlaOutputs)
            or not isinstance(reference, MlaReference)
        ):
            raise TypeError("expected MLA inputs, outputs, and reference")
        if inputs.legacy_call is not None:
            torch.testing.assert_close(outputs.value, reference.value)
            return
        sparse = inputs.api is not MlaApi.KVCACHE or inputs.indices is not None
        for name in ("out", "lse", "max_logits"):
            actual, expected = getattr(outputs, name), getattr(reference, name)
            required = name != "max_logits" or inputs.api is not MlaApi.KVCACHE
            if required and (actual is None or expected is None):
                raise AssertionError(f"MLA verification requires {name}")
            if (actual is None) != (expected is None):
                raise AssertionError(f"MLA {name} presence differs from reference")
            if expected is not None:
                atol, rtol = (1.5e-2, 1e-2)
                if sparse:
                    atol, rtol = (0.01, 0.02) if name == "out" else (0.02, 0.002)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


__all__ = [
    "MlaApi",
    "MlaInputs",
    "MlaOperator",
    "MlaOutputs",
    "MlaReference",
    "MlaWorkload",
]
