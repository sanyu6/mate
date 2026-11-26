# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors
"""MP32 backward framework for dense BSHD and packed THD attention.

D128, D192/V128 and D256 use separate WS kernel factories with shared
preprocessing and gradient packing.

LSE is natural-log [B,Hq,Sq] or [Hq,total_q]. Packed offsets must delimit
valid sequences and max_seqlen arguments must bound their lengths, as in the
forward API. Inspecting offset values here would synchronize the device.
"""

import math
from numbers import Integral

import torch

from ....execution_context import raise_complete_if_dry_run
from .._flash_attention_bwd_common import (
    _contiguous_cosize_bytes,
    _jit_for_index_type_promotion,
    _needs_index_type_promotion,
    _tensor_cosize_bytes,
)
from .._flash_attention_bwd_post import (
    compute_delta_ws,
    pack_dq_from_accum_ws,
    reduce_kv_grads_ws,
)
from ._flash_attention_bwd_d128 import flashattn_bwd_d128
from ._flash_attention_bwd_d192_v128 import flashattn_bwd_d192_v128
from ._flash_attention_bwd_d256 import flashattn_bwd_d256
from ._flash_attention_fwd_common import (
    _check_attention_strides,
    _current_musa_cc,
    _to_tilelang_dtype,
)


def _select_kernel_config(actual_dim_qk, actual_dim_v):
    if (actual_dim_qk, actual_dim_v) == (192, 128):
        return flashattn_bwd_d192_v128, 192, 128, {"block_M": 64}
    if (
        actual_dim_qk == actual_dim_v
        and 8 <= actual_dim_qk <= 256
        and actual_dim_qk % 8 == 0
    ):
        if actual_dim_qk <= 128:
            return flashattn_bwd_d128, 128, 128, {"block_M": 64}
        return flashattn_bwd_d256, 256, 256, {"block_M": 64, "block_N": 64}
    raise NotImplementedError(
        "MP32 backward expects equal dims 8..256 (multiples of 8), or (192,128)"
    )


def _validate_inputs(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q,
    cu_seqlens_k,
):
    if q.device.type != "musa":
        raise ValueError("MP32 TileLang backward requires MUSA tensors")
    if _current_musa_cc(q) != 32:
        raise ValueError("MP32 TileLang backward requires compute capability 3.2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("MP32 TileLang backward supports float16 and bfloat16")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out), ("dout", dout)):
        if tensor.device != q.device or tensor.dtype != q.dtype:
            raise ValueError(f"{name} must have q's device and dtype")
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
        if any(x.ndim != 3 for x in (q, k, v)):
            raise ValueError("packed inputs must have shape [total_seq, heads, dim]")
        for name, cu in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if (
                cu.device != q.device
                or cu.dtype != torch.int32
                or cu.ndim != 1
                or not cu.is_contiguous()
                or cu.numel() < 2
            ):
                raise ValueError(
                    f"{name} must be contiguous int32 [batch+1] on q.device"
                )
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("query and key offsets must describe the same batch")
        for name, value, total in (
            ("max_seqlen_q", max_seqlen_q, q.shape[0]),
            ("max_seqlen_k", max_seqlen_k, k.shape[0]),
        ):
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or value < 0
                or (total > 0 and value == 0)
            ):
                raise ValueError(f"{name} must bound the packed sequence lengths")
        lse_shape = (q.shape[1], q.shape[0])
    else:
        if any(x.ndim != 4 for x in (q, k, v)):
            raise ValueError("dense inputs must have shape [batch, seq, heads, dim]")
        if q.shape[0] != k.shape[0]:
            raise ValueError("q and k must have the same batch size")
        max_seqlen_q, max_seqlen_k = q.shape[1], k.shape[1]
        lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    if k.shape[:-1] != v.shape[:-1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("k/v layouts and q/k head dimensions must match")
    if q.shape[-2] <= 0 or k.shape[-2] <= 0:
        raise ValueError("head counts must be positive")
    if q.shape[-2] % k.shape[-2] != 0:
        raise ValueError("q head count must be divisible by k/v head count")
    expected_out = (*q.shape[:-1], v.shape[-1])
    if out.shape != expected_out or dout.shape != expected_out:
        raise ValueError(f"out and dout must have shape {expected_out}")
    if (
        softmax_lse.device != q.device
        or softmax_lse.dtype != torch.float32
        or softmax_lse.shape != lse_shape
        or not softmax_lse.is_contiguous()
    ):
        raise ValueError(f"LSE must be contiguous float32 {lse_shape} on q.device")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out), ("dout", dout)):
        _check_attention_strides(name, tensor)
    return is_varlen, int(max_seqlen_q), int(max_seqlen_k)


def flashattn_varlen_bwd_interface(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    is_causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    smscale=None,
    dtype=None,
    is_bhsd=False,
    deterministic=False,
):
    if deterministic:
        raise NotImplementedError("MP32 backward deterministic mode is not implemented")
    if is_bhsd:
        raise NotImplementedError("MP32 backward expects dense BSHD inputs")
    if seqused_q is not None or seqused_k is not None:
        raise NotImplementedError("MP32 backward does not support seqused")
    if window_size is not None and tuple(window_size) != (-1, -1):
        raise NotImplementedError("MP32 backward does not support windowed attention")
    if softcap != 0.0:
        raise NotImplementedError("MP32 backward does not support softcap")
    # Autograd may supply an expanded or sliced gradient. Preserve conforming
    # strided tensors and materialize only dO layouts the TME ABI cannot express.
    if dout.stride(-1) != 1 or any(
        stride <= 0 or stride % 8 != 0 for stride in dout.stride()[:-1]
    ):
        dout = dout.contiguous()
    is_varlen, max_seqlen_q, max_seqlen_k = _validate_inputs(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
    )
    main_factory, dim_qk, dim_v, main_config = _select_kernel_config(
        q.shape[-1], v.shape[-1]
    )
    kernel_dtype = str(_to_tilelang_dtype(q.dtype))
    if dtype is not None and str(dtype) != str(kernel_dtype):
        raise ValueError("TileLang dtype must match the input tensor dtype")
    scale = q.shape[-1] ** -0.5 if smscale is None else float(smscale)
    if not math.isfinite(scale):
        raise ValueError("softmax scale must be finite")

    batch = cu_seqlens_q.numel() - 1 if is_varlen else q.shape[0]
    heads = q.shape[-2]
    heads_kv = k.shape[-2]
    heads_q_eq_heads_kv = heads == heads_kv
    total_q = q.shape[0] if is_varlen else batch * q.shape[1]
    delta_head_major = dim_qk == 128
    delta_shape = softmax_lse.shape if delta_head_major else (total_q, heads)
    # LSE and Delta are contiguous FP32 tensors with the same element count.
    metadata_bytes = softmax_lse.numel() * softmax_lse.element_size()
    # The same selected tile controls main and workspace padding. WS atomics
    # cover whole Q tiles; postprocess discards padded sequence/features.
    block_M = main_config.get("block_M", 1)
    q_capacity = (max_seqlen_q + block_M - 1) // block_M * block_M
    dq_accum_shape = (batch, heads, q_capacity, dim_qk)
    # Both FP32 partials use the same storage dimension, including D192/V128.
    # Main-kernel views and the reduction only access the actual dimensions.
    kv_partial_shape = (*k.shape[:-2], heads, dim_qk)
    kv_partial_bytes = (
        0
        if heads_q_eq_heads_kv
        else _contiguous_cosize_bytes(kv_partial_shape, torch.float32)
    )
    promote_index = _needs_index_type_promotion(
        *(_tensor_cosize_bytes(x) for x in (q, k, v, out, dout, softmax_lse)),
        *(
            _tensor_cosize_bytes(x)
            for x in (cu_seqlens_q, cu_seqlens_k)
            if x is not None
        ),
        *(_contiguous_cosize_bytes(x.shape, x.dtype) for x in (q, k, v)),
        _contiguous_cosize_bytes(delta_shape, torch.float32),
        _contiguous_cosize_bytes(dq_accum_shape, torch.float32),
        kv_partial_bytes,
    )

    # Resolve all kernels before allocating workspaces or launching device work.
    main_kernel = _jit_for_index_type_promotion(main_factory, promote_index)(
        is_varlen=is_varlen,
        is_causal=bool(is_causal),
        dtype=kernel_dtype,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        **main_config,
    )
    delta_kernel = _jit_for_index_type_promotion(compute_delta_ws, promote_index)(
        dim_v, is_varlen, dtype=kernel_dtype, head_major=delta_head_major
    )
    pack_dq_kernel = _jit_for_index_type_promotion(
        pack_dq_from_accum_ws, promote_index
    )(dim_qk, is_varlen, dtype=kernel_dtype, dq_accum_swizzle=True)
    reduce_kv_kernel = None
    if not heads_q_eq_heads_kv:
        reduce_kv_kernel = _jit_for_index_type_promotion(
            reduce_kv_grads_ws, promote_index
        )(heads // heads_kv, dim_qk, is_varlen, output_dtype=kernel_dtype)
    raise_complete_if_dry_run()

    dq, dk, dv = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in (q, k, v)
    )
    delta = torch.empty(delta_shape, device=q.device, dtype=torch.float32)
    dq_accum = torch.empty(dq_accum_shape, device=q.device, dtype=torch.float32)
    dk_main, dv_main = dk, dv
    if not heads_q_eq_heads_kv:
        dk_partial = torch.empty(kv_partial_shape, device=q.device, dtype=torch.float32)
        dv_partial = torch.empty(kv_partial_shape, device=q.device, dtype=torch.float32)
        dk_main = dk_partial[..., : q.shape[-1]]
        dv_main = dv_partial[..., : v.shape[-1]]

    # Product flow: Delta -> zero dQ workspace -> main -> cast/pack dQ.
    # Main owns dK/dV per Q head, including rows with no contributing queries.
    # Grouped attention reduces those FP32 partials after the main kernel.
    delta_kernel(out, dout, cu_seqlens_q, None, delta, max_seqlen_q)
    dq_accum.zero_()
    main_kernel(
        q,
        k,
        v,
        dout,
        softmax_lse,
        delta,
        cu_seqlens_q,
        cu_seqlens_k,
        dq_accum,
        dk_main,
        dv_main,
        max_seqlen_k,
        metadata_bytes,
        scale,
        scale * 1.4426950408889634,
    )
    pack_dq_kernel(dq_accum, cu_seqlens_q, None, dq)
    if reduce_kv_kernel is not None:
        reduce_kv_kernel(dk_partial, dv_partial, dk, dv)
    return dq, dk, dv


__all__ = ["flashattn_varlen_bwd_interface"]
