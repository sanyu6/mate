# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors
"""Shared runtime tensor definitions for the three MP32 backward backends."""

import tilelang.language as T


def _make_kernel_types(dim_qk, dim_v, is_varlen, dtype, heads_q_eq_heads_kv=True):
    """Storage dimensions/layout/dtype and grouped-head mode are compile-time.

    Logical dimensions and every non-feature stride remain runtime symbols.
    Tensor signatures use the public BSHD/THD layout; a kernel's TME aliases
    must preserve these bounds and strides when changing dimension order.
    """
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    heads_kv = heads if heads_q_eq_heads_kv else T.dynamic("heads_kv")
    kv_grad_dtype = dtype if heads_q_eq_heads_kv else T.float32
    seqlen_q = T.dynamic("seqlen_q")
    seqlen_kv = T.dynamic("seqlen_kv")
    total_seq_q = T.dynamic("total_seq_q") if is_varlen else batch * seqlen_q
    total_seq_kv = T.dynamic("total_seq_kv") if is_varlen else batch * seqlen_kv
    actual_dim_qk = T.dynamic("actual_dim_qk")
    actual_dim_v = T.dynamic("actual_dim_v")
    q_capacity = T.dynamic("q_capacity")

    q_shape = (
        (total_seq_q, heads, actual_dim_qk)
        if is_varlen
        else (batch, seqlen_q, heads, actual_dim_qk)
    )
    k_shape = (
        (total_seq_kv, heads_kv, actual_dim_qk)
        if is_varlen
        else (batch, seqlen_kv, heads_kv, actual_dim_qk)
    )
    v_shape = (
        (total_seq_kv, heads_kv, actual_dim_v)
        if is_varlen
        else (batch, seqlen_kv, heads_kv, actual_dim_v)
    )
    # Grouped attention writes one FP32 partial per Q head. The host keeps the
    # padded storage strides while exposing only actual feature dimensions.
    dk_shape = (
        (total_seq_kv, heads, actual_dim_qk)
        if is_varlen
        else (batch, seqlen_kv, heads, actual_dim_qk)
    )
    dv_shape = (
        (total_seq_kv, heads, actual_dim_v)
        if is_varlen
        else (batch, seqlen_kv, heads, actual_dim_v)
    )
    do_shape = (
        (total_seq_q, heads, actual_dim_v)
        if is_varlen
        else (batch, seqlen_q, heads, actual_dim_v)
    )
    lse_shape = (heads, total_seq_q) if is_varlen else (batch, heads, seqlen_q)
    kernel_types = {
        "dim_qk": dim_qk,
        "dim_v": dim_v,
        "batch": batch,
        "heads": heads,
        "heads_kv": heads_kv,
        "kv_grad_dtype": kv_grad_dtype,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "total_seq_q": total_seq_q,
        "total_seq_kv": total_seq_kv,
        "actual_dim_qk": actual_dim_qk,
        "actual_dim_v": actual_dim_v,
        "q_capacity": q_capacity,
        "lse_type": T.Tensor(lse_shape, T.float32),
        "delta_type": T.Tensor((total_seq_q, heads), T.float32),
        # T.Tensor creates a Buffer: distinct inputs must not share the object.
        "cu_q_type": T.Tensor((batch + 1,), T.int32),
        "cu_kv_type": T.Tensor((batch + 1,), T.int32),
        "dq_accum_type": T.Tensor((batch, heads, q_capacity, dim_qk), T.float32),
    }
    for name, shape in (
        ("q", q_shape),
        ("k", k_shape),
        ("v", v_shape),
        ("do", do_shape),
        ("dk", dk_shape),
        ("dv", dv_shape),
    ):
        axes = ("s", "h") if is_varlen else ("b", "s", "h")
        stride_symbols = {
            f"{name}_stride_{axis}": T.dynamic(f"{name}_stride_{axis}") for axis in axes
        }
        strides = (*stride_symbols.values(), 1)
        kernel_types.update(stride_symbols)
        kernel_types[f"{name}_strides"] = strides
        tensor_dtype = kv_grad_dtype if name in ("dk", "dv") else dtype
        kernel_types[f"{name}_type"] = T.StridedTensor(shape, strides, tensor_dtype)
    return kernel_types
