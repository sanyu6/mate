"""Logical performance metric helpers for MATE kernels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch


class KernelMetrics(NamedTuple):
    """Logical bytes and FLOPs for one kernel workload."""

    load_byte: int
    store_byte: int
    flop: int

    @property
    def total_byte(self) -> int:
        return self.load_byte + self.store_byte

    def to_perf(self, time_ms: float) -> tuple[float, float]:
        """Convert counts into TFLOP/s and decimal GB/s."""
        return self.flop / (1e9 * time_ms), self.total_byte / (1e6 * time_ms)


def tensor_bytes(*tensors: torch.Tensor | None) -> int:
    """Return the dense storage span counted by existing benchmarks."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if tensor is not None
    )


def get_bmm_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    k: int | None = None,
) -> KernelMetrics:
    """Return logical metrics for batched matrix multiplication."""
    batch = int(a.shape[0])
    m = int(out.shape[-2])
    n = int(out.shape[-1])
    k = int(a.shape[-1] if k is None else k)
    return KernelMetrics(
        tensor_bytes(a, b),
        tensor_bytes(out),
        2 * batch * m * n * k,
    )


def get_gemm_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    extra_load_tensors: Sequence[torch.Tensor | None] = (),
    extra_store_tensors: Sequence[torch.Tensor | None] = (),
) -> KernelMetrics:
    """Return logical metrics for one GEMM-like operation."""
    m = int(out.shape[-2] if m is None else m)
    n = int(out.shape[-1] if n is None else n)
    k = int(a.shape[-1] if k is None else k)
    return KernelMetrics(
        tensor_bytes(a, b, *extra_load_tensors),
        tensor_bytes(out, *extra_store_tensors),
        2 * m * n * k,
    )


def get_masked_moe_gemm_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    *,
    valid_m: int,
    n: int,
    k: int,
    a_scale: torch.Tensor | None = None,
    granu_k: int | None = None,
    b_scale: torch.Tensor | None = None,
    packed_b: torch.Tensor | None = None,
    b_scale_bytes: int | None = None,
) -> KernelMetrics:
    """Return logical metrics for masked grouped MoE GEMM."""
    load_byte = valid_m * k * a.element_size() + tensor_bytes(
        b if packed_b is None else packed_b, mask
    )
    if a_scale is not None:
        if granu_k is None:
            raise ValueError("granu_k is required when counting A scale bytes")
        if granu_k <= 0 or k % granu_k != 0:
            raise ValueError(
                f"Expected granu_k to divide k, got k={k}, granu_k={granu_k}"
            )
        load_byte += valid_m * (k // granu_k) * a_scale.element_size()
    if b_scale_bytes is None:
        load_byte += tensor_bytes(b_scale)
    else:
        load_byte += b_scale_bytes
    return KernelMetrics(
        load_byte,
        valid_m * n * out.element_size(),
        2 * valid_m * n * k,
    )


def get_ragged_m_moe_gemm_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    valid_m: int,
    n: int,
    k: int,
    a_scale: torch.Tensor | None = None,
    granu_k: int | None = None,
    extra_load_tensors: Sequence[torch.Tensor | None] = (),
) -> KernelMetrics:
    """Return logical metrics for ragged-M grouped MoE GEMM."""
    load_byte = tensor_bytes(a, b, m_indices, *extra_load_tensors)
    if a_scale is not None:
        if granu_k is None:
            raise ValueError("granu_k is required when counting A scale bytes")
        if granu_k <= 0 or k % granu_k != 0:
            raise ValueError(
                f"Expected granu_k to divide k, got k={k}, granu_k={granu_k}"
            )
        load_byte += valid_m * (k // granu_k) * a_scale.element_size()
    return KernelMetrics(
        load_byte,
        tensor_bytes(out),
        2 * valid_m * n * k,
    )


def get_ragged_k_moe_gemm_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_idx: torch.Tensor,
    out: torch.Tensor,
    *,
    total_k: int,
    m: int,
    n: int,
    extra_load_tensors: Sequence[torch.Tensor | None] = (),
) -> KernelMetrics:
    """Return logical metrics for ragged-K grouped MoE GEMM."""
    return KernelMetrics(
        tensor_bytes(a, b, group_k_idx, *extra_load_tensors),
        2 * tensor_bytes(out),
        2 * m * n * total_k,
    )


def get_mixed_dtype_moe_gemm_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    mixed_dtype: str,
    valid_m: int,
    n: int,
    k: int,
    masked_m: torch.Tensor | None = None,
    m_indices: torch.Tensor | None = None,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    residual_e8m0: torch.Tensor | None = None,
    epilogue_fp32: torch.Tensor | None = None,
) -> KernelMetrics:
    """Count packed weights and format metadata for mixed-dtype MoE GEMM."""
    metadata: tuple[torch.Tensor | None, ...]
    if mixed_dtype == "s4fp8":
        metadata = (b_scale,)
    elif mixed_dtype == "fp4fp8":
        metadata = (residual_e8m0, epilogue_fp32)
    else:
        raise ValueError(f"Unsupported mixed dtype: {mixed_dtype}")
    if masked_m is not None:
        return get_masked_moe_gemm_metrics(
            a,
            b,
            masked_m,
            out,
            valid_m=valid_m,
            n=n,
            k=k,
            a_scale=a_scale,
            granu_k=k,
            b_scale_bytes=tensor_bytes(*metadata),
        )
    if m_indices is None:
        raise ValueError("ragged mixed-dtype GEMM requires m_indices")
    return get_ragged_m_moe_gemm_metrics(
        a,
        b,
        m_indices,
        out,
        valid_m=valid_m,
        n=n,
        k=k,
        a_scale=a_scale,
        granu_k=k,
        extra_load_tensors=metadata,
    )


def get_effective_kv_lengths(
    query_lengths: Sequence[int],
    kv_lengths: Sequence[int],
    causal: bool,
    window_size: tuple[int, int],
) -> list[float]:
    """Return the effective KV length used by FlashAttention metrics."""
    window_left, window_right = window_size
    if window_left >= 0 and window_right >= 0:
        if causal:
            window_right = 0
        result: list[float] = []
        for query_length, kv_length in zip(query_lengths, kv_lengths, strict=True):
            total = 0
            for row_idx in range(query_length):
                col_left = max(0, row_idx + kv_length - query_length - window_left)
                col_right = min(
                    kv_length - 1, row_idx + kv_length - query_length + window_right
                )
                total += max(0, col_right - col_left + 1)
            result.append(total / query_length if query_length > 0 else 0.0)
        return result

    if window_size != (-1, -1):
        return [
            min(query_length + window_left, kv_length, window_left + window_right + 1)
            for query_length, kv_length in zip(query_lengths, kv_lengths, strict=True)
        ]

    if causal:
        return [
            (max(0, kv_length - query_length) + kv_length) / 2
            for query_length, kv_length in zip(query_lengths, kv_lengths, strict=True)
        ]

    return [float(kv_length) for kv_length in kv_lengths]


def get_flash_attention_metrics(
    query_lengths: Sequence[int],
    kv_lengths: Sequence[int],
    *,
    num_head_q: int,
    num_head_kv: int,
    headdim_qk: int,
    headdim_v: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    causal: bool,
    window_size: tuple[int, int],
) -> KernelMetrics:
    """Return logical metrics for FlashAttention forward."""
    effective_kv_lengths = get_effective_kv_lengths(
        query_lengths, kv_lengths, causal, window_size
    )
    pairs = sum(
        query_length * kv_length
        for query_length, kv_length in zip(
            query_lengths, effective_kv_lengths, strict=True
        )
    )
    total_query_length = sum(query_lengths)
    total_kv_length = sum(effective_kv_lengths)
    load_byte = (
        total_query_length * num_head_q * headdim_qk * q_dtype.itemsize
        + total_kv_length * num_head_kv * headdim_qk * kv_dtype.itemsize
        + total_kv_length * num_head_kv * headdim_v * kv_dtype.itemsize
    )
    store_byte = total_query_length * num_head_q * headdim_v * out_dtype.itemsize
    flop = int(2 * pairs * num_head_q * (headdim_qk + headdim_v))
    return KernelMetrics(int(load_byte), int(store_byte), flop)


def get_flash_attention_bwd_metrics(
    tensors: Sequence[torch.Tensor | None],
    *,
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    num_head_q: int,
    headdim_qk: int,
    headdim_v: int,
    terms: int,
    causal: bool,
    window_size: tuple[int, int],
) -> KernelMetrics:
    """Return logical metrics for FlashAttention backward."""
    effective_kv_lengths = get_effective_kv_lengths(
        q_lens, kv_lens, causal, window_size
    )
    pairs = sum(
        q_len * kv_len
        for q_len, kv_len in zip(q_lens, effective_kv_lengths, strict=True)
    )
    flop = int(pairs * num_head_q * terms * (headdim_qk + headdim_v))
    return KernelMetrics(tensor_bytes(*tensors), 0, flop)


def get_dense_mla_metrics(
    q: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    seqlen_q: int,
    num_head_q: int,
    headdim_latent: int,
    headdim_rope: int,
    causal: bool,
) -> KernelMetrics:
    """Return logical metrics for dense MLA."""
    kv_tokens = int(cache_seqlens.sum().item())
    pairs = seqlen_q * kv_tokens
    if causal:
        pairs = sum(
            seqlen_q * (2 * int(length) - seqlen_q + 1) // 2
            for length in cache_seqlens.tolist()
        )
    head_dim_qk = headdim_latent + headdim_rope
    output_elements = q.numel() // q.shape[-1] * headdim_latent
    load_byte = (
        q.numel() * q.element_size() + kv_tokens * head_dim_qk * q.element_size()
    )
    store_byte = output_elements * q.element_size()
    flop = 2 * pairs * num_head_q * (head_dim_qk + headdim_latent)
    return KernelMetrics(load_byte, store_byte, flop)


def get_pack8_mla_metrics(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    head_dim_v: int = 512,
) -> KernelMetrics:
    """Preserve pack8's scheduled-slot FLOPs and logical payload byte counts.

    Each index slot schedules eight queries, including padded/masked slots.
    Count one KV vector per packed slot, as in the original benchmark; this
    is not a measurement of executed FLOPs or physical DRAM traffic.
    """
    scheduled_slots = indices.numel() * 8
    flop = 2 * scheduled_slots * q.shape[-2] * (q.shape[-1] + head_dim_v)
    load_byte = tensor_bytes(q) + indices.numel() * kv.shape[-1] * kv.element_size()
    store_byte = q.numel() // q.shape[-1] * head_dim_v * q.element_size()
    return KernelMetrics(load_byte, store_byte, flop)


def get_sparse_mla_metrics(
    q: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_kv: int,
    num_head_q: int,
    headdim_latent: int,
    headdim_rope: int,
    kv_bytes_per_token: int,
) -> KernelMetrics:
    """Return logical metrics for sparse MLA."""
    head_dim_qk = headdim_latent + headdim_rope
    output_bytes = (
        q.shape[0]
        * (q.shape[1] if q.ndim == 4 else 1)
        * num_head_q
        * headdim_latent
        * q.element_size()
    )

    if indices.ndim == 3:
        total_topk = (
            q.shape[0] * indices.shape[-1]
            if topk_length is None
            else int(topk_length.sum().item())
        )
        valid_mask = (indices >= 0) & (indices < seqlen_kv)
        if topk_length is not None:
            pos = torch.arange(indices.shape[-1], device=indices.device).view(1, 1, -1)
            valid_mask &= pos < topk_length.view(q.shape[0], 1, 1)
        num_valid_indices = int(valid_mask.sum().item())
        flop = 2 * total_topk * num_head_q * (head_dim_qk + headdim_latent)
        load_byte = (
            q.numel() * q.element_size() + num_valid_indices * kv_bytes_per_token
        )
    else:
        if topk_length is None:
            num_attended = batch_size * seqlen_q * indices.shape[-1]
            masked_indices = indices
        else:
            num_attended = int(topk_length.sum().item()) * seqlen_q
            pos = torch.arange(indices.shape[-1], device=indices.device).view(
                1, 1, 1, -1
            )
            masked_indices = torch.where(
                pos < topk_length.view(-1, 1, 1, 1),
                indices,
                indices.new_full((), -1),
            )
        num_retrieved = int(masked_indices.unique().numel())
        flop = 2 * num_head_q * num_attended * (head_dim_qk + headdim_latent)
        load_byte = q.numel() * q.element_size() + num_retrieved * kv_bytes_per_token

    return KernelMetrics(load_byte, output_bytes, flop)


def get_gdn_decode_metrics(
    tensors: Sequence[torch.Tensor | None],
    *,
    batch_size: int,
    head_o: int,
    head_size: int,
) -> KernelMetrics:
    """Return logical metrics for GDN decode.

    ``tensors`` follows the GDN decode call order used by the benchmark:
    ``(q, k, v, state, A_log, a, dt_bias, b, output, state_indices)``.
    The recurrent state is both read and updated. When ``state`` is a larger
    indexed pool, only the ``batch_size`` state slots touched by this decode
    step contribute to the logical traffic.
    """
    (
        q,
        k,
        v,
        state,
        a_log,
        a,
        dt_bias,
        b,
        output,
        state_indices,
    ) = tensors

    state_byte = 0
    if state is not None:
        if state.ndim == 0 or state.shape[0] == 0:
            raise ValueError("GDN decode state must contain at least one state slot")
        state_byte = (
            batch_size * (state.numel() // int(state.shape[0])) * state.element_size()
        )

    return KernelMetrics(
        tensor_bytes(q, k, v, a_log, a, dt_bias, b, state_indices) + state_byte,
        tensor_bytes(output) + state_byte,
        6 * batch_size * head_o * head_size * head_size,
    )


def get_gdn_prefill_metrics(
    tensors: Sequence[torch.Tensor | None],
    *,
    total_tokens: int,
    head_o: int,
    head_size: int,
) -> KernelMetrics:
    """Return logical metrics for GDN prefill.

    ``tensors`` must be ``(q, k, v, g, beta, initial_state, output,
    output_state, cu_seqlens)``. Initial state and sequence metadata are
    inputs; the attention result and optional final state are outputs.
    """
    (
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output,
        output_state,
        cu_seqlens,
    ) = tensors

    return KernelMetrics(
        tensor_bytes(q, k, v, g, beta, initial_state, cu_seqlens),
        tensor_bytes(output, output_state),
        4 * total_tokens * head_o * head_size * head_size,
    )


def get_kda_metrics(
    tensors: Sequence[torch.Tensor | None],
    *,
    total_tokens: int,
    num_head_v: int,
    head_dim: int,
) -> KernelMetrics:
    """Return logical metrics for chunk KDA.

    ``tensors`` must be ``(q, k, v, g, beta, output, initial_state,
    final_state, cu_seqlens, A_log, dt_bias)``. This reports the logical
    recurrence workload; optional input and output state traffic is counted
    only when the corresponding tensor is present.
    """
    (
        q,
        k,
        v,
        g,
        beta,
        output,
        initial_state,
        final_state,
        cu_seqlens,
        a_log,
        dt_bias,
    ) = tensors

    return KernelMetrics(
        tensor_bytes(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            a_log,
            dt_bias,
        ),
        tensor_bytes(output, final_state),
        7 * total_tokens * num_head_v * head_dim * head_dim,
    )
