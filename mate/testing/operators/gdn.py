"""Structured recurrent operator with independent Python references."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .operator import (
    Operator,
    OperatorInputs,
    OperatorOutputs,
    OperatorReference,
    OperatorWorkload,
)


class GdnApi(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True, kw_only=True)
class GdnWorkload(OperatorWorkload):
    api: GdnApi = GdnApi.PREFILL
    batch_size: int = 1
    seqlen: int = 64
    seq_lens: tuple[int, ...] | None = None
    num_head_q: int = 4
    num_head_v: int = 4
    head_size: int = 128
    dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "musa"
    seed: int | None = 666
    cache_indices: str = "none"
    initial_state: bool = False
    output_final_state: bool = False
    qk_l2norm: bool = True
    is_log_space: bool = True
    chunk_size: int = 64
    scale: float | None = None
    call: Callable[[], Any] | None = None
    reference_call: Callable[[], Any] | None = None

    def __post_init__(self):
        if self.call is not None:
            return
        if not isinstance(self.api, GdnApi):
            raise ValueError("expected GdnApi")
        if (
            min(
                self.batch_size,
                self.seqlen,
                self.num_head_q,
                self.num_head_v,
                self.head_size,
            )
            <= 0
        ):
            raise ValueError("GDN dimensions must be positive")
        if self.num_head_v % self.num_head_q:
            raise ValueError("num_head_v must be divisible by num_head_q")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("GDN requires float16 or bfloat16")
        if self.seq_lens is not None and (
            len(self.seq_lens) != self.batch_size or min(self.seq_lens) <= 0
        ):
            raise ValueError("seq_lens must contain a positive length per sequence")
        if self.api is GdnApi.DECODE:
            if self.head_size % 32 or self.seqlen != 1 or self.seq_lens is not None:
                raise ValueError(
                    "generated decode requires T=1 and head_size divisible by 32"
                )
            if self.cache_indices not in ("none", "identity"):
                raise ValueError("decode cache_indices must be none or identity")
        elif self.head_size != 128 or self.chunk_size != 64:
            raise ValueError("GDN prefill requires head_size=128 and chunk_size=64")

    @property
    def lengths(self) -> tuple[int, ...]:
        return (
            self.seq_lens
            if self.seq_lens is not None
            else (self.seqlen,) * self.batch_size
        )


@dataclass(kw_only=True)
class GdnInputs(OperatorInputs):
    api: GdnApi = GdnApi.PREFILL
    workload: GdnWorkload | None = None
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    g: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    state: torch.Tensor | None = None
    A_log: torch.Tensor | None = None
    a: torch.Tensor | None = None
    dt_bias: torch.Tensor | None = None
    b: torch.Tensor | None = None
    state_indices: torch.Tensor | None = None
    initial_state: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    output: torch.Tensor | None = None
    output_state: torch.Tensor | None = None
    intermediate_states_buffer: torch.Tensor | None = None
    scale: float | None = None
    state_layout: str = "VK"
    disable_state_update: bool = False
    use_qk_l2norm: bool = True
    use_qk_l2norm_in_kernel: bool = False
    output_final_state: bool = False
    chunk_size: int = 64
    is_log_space: bool = True


@dataclass(frozen=True, kw_only=True)
class GdnOutputs(OperatorOutputs):
    out: torch.Tensor | None = None
    state: torch.Tensor | None = None
    intermediate_states: torch.Tensor | None = None
    value: Any = None


@dataclass(frozen=True, kw_only=True)
class GdnReference(OperatorReference):
    out: torch.Tensor | None = None
    state: torch.Tensor | None = None
    intermediate_states: torch.Tensor | None = None
    unchanged_initial_state: torch.Tensor | None = None
    value: Any = None


class GdnOperator(Operator):
    def generate(self, workload: OperatorWorkload) -> GdnInputs:
        if not isinstance(workload, GdnWorkload):
            raise TypeError("expected GdnWorkload")
        w = workload
        if w.call is not None:
            return GdnInputs(workload=w)
        if w.seed is not None:
            torch.manual_seed(w.seed)
        shape = (
            (sum(w.lengths),) if w.seq_lens is not None else (w.batch_size, w.seqlen)
        )
        device, dtype = w.device, w.dtype
        if w.api is GdnApi.PREFILL:
            mixed = torch.randn(
                (*shape, (2 * w.num_head_q + w.num_head_v) * w.head_size),
                device=device,
                dtype=dtype,
            )
            q, k, v = torch.split(
                mixed,
                [w.num_head_q * w.head_size] * 2 + [w.num_head_v * w.head_size],
                dim=-1,
            )
            q = q.view(*shape, w.num_head_q, w.head_size)
            k = k.view_as(q)
            v = v.view(*shape, w.num_head_v, w.head_size)
        else:
            q = torch.randn(
                *shape, w.num_head_q, w.head_size, device=device, dtype=dtype
            )
            k = torch.randn_like(q)
            v = torch.randn(
                *shape, w.num_head_v, w.head_size, device=device, dtype=dtype
            )
        x = GdnInputs(
            api=w.api,
            workload=w,
            q=q,
            k=k,
            v=v,
            output=torch.empty_like(v),
            scale=w.scale,
            use_qk_l2norm=w.qk_l2norm,
            use_qk_l2norm_in_kernel=w.qk_l2norm,
            output_final_state=w.output_final_state,
            chunk_size=w.chunk_size,
            is_log_space=w.is_log_space,
        )
        state_shape = (w.batch_size, w.num_head_v, w.head_size, w.head_size)
        if w.api is GdnApi.DECODE:
            x.state = torch.randn(state_shape, device=device, dtype=torch.float32)
            x.A_log = torch.randn(w.num_head_v, device=device) * 0.1
            x.a = torch.randn(*shape, w.num_head_v, device=device, dtype=dtype) * 0.1
            x.dt_bias = torch.randn(w.num_head_v, device=device) * 0.1
            x.b = torch.randn(*shape, w.num_head_v, device=device, dtype=dtype)
            if w.cache_indices == "identity":
                x.state_indices = torch.arange(
                    w.batch_size, dtype=torch.int32, device=device
                )
        else:
            alpha = torch.exp(-0.02 * torch.rand(*shape, w.num_head_v, device=device))
            x.g = alpha.log() if w.is_log_space else alpha
            x.beta = torch.sigmoid(torch.randn(*shape, w.num_head_v, device=device))
            if w.initial_state:
                x.initial_state = torch.randn(state_shape, device=device) * 0.05
            if w.output_final_state:
                x.output_state = torch.empty(state_shape, device=device)
            if w.seq_lens is not None:
                x.cu_seqlens = torch.tensor(
                    exclusive_cumsum(list(w.lengths)), dtype=torch.int32, device=device
                )
        return x

    def call(self, inputs: OperatorInputs) -> GdnOutputs:
        if not isinstance(inputs, GdnInputs):
            raise TypeError("expected GdnInputs")
        x = inputs
        if x.workload is not None and x.workload.call is not None:
            return GdnOutputs(value=x.workload.call())
        if x.q is None or x.k is None or x.v is None:
            raise ValueError("GDN requires q, k, v")
        if x.api is GdnApi.DECODE:
            result = import_module("mate.gdn_decode").gated_delta_rule_decode(
                q=x.q,
                k=x.k,
                v=x.v,
                state=x.state,
                A_log=x.A_log,
                a=x.a,
                dt_bias=x.dt_bias,
                b=x.b,
                state_layout=x.state_layout,
                state_indices=x.state_indices,
                scale=x.scale,
                output=x.output,
                intermediate_states_buffer=x.intermediate_states_buffer,
                disable_state_update=x.disable_state_update,
                use_qk_l2norm=x.use_qk_l2norm,
            )
            return GdnOutputs(
                out=result[0],
                state=result[1],
                intermediate_states=x.intermediate_states_buffer,
                value=result,
            )
        if x.api is not GdnApi.PREFILL:
            raise ValueError("unsupported GDN API")
        result = import_module("mate.gdn_prefill").chunk_gated_delta_rule(
            q=x.q,
            k=x.k,
            v=x.v,
            g=x.g,
            beta=x.beta,
            scale=x.scale,
            initial_state=x.initial_state,
            output_final_state=x.output_final_state,
            cu_seqlens=x.cu_seqlens,
            use_qk_l2norm_in_kernel=x.use_qk_l2norm_in_kernel,
            chunk_size=x.chunk_size,
            is_log_space=x.is_log_space,
            output=x.output,
            output_state=x.output_state,
        )
        return GdnOutputs(
            out=result[0] if isinstance(result, tuple) else result,
            state=result[1] if isinstance(result, tuple) else None,
            value=result,
        )

    def reference(self, inputs: OperatorInputs) -> GdnReference:
        if not isinstance(inputs, GdnInputs):
            raise TypeError("expected GdnInputs")
        x = inputs
        if x.workload is not None and x.workload.call is not None:
            if x.workload.reference_call is None:
                raise NotImplementedError(
                    "legacy call requires independent reference_call"
                )
            return GdnReference(value=x.workload.reference_call())
        if x.q is None or x.k is None or x.v is None:
            raise ValueError("GDN requires q, k, v")
        scale = x.q.shape[-1] ** -0.5 if x.scale is None else x.scale
        if x.api is GdnApi.DECODE:
            return self._reference_decode(x, scale)
        if x.api is not GdnApi.PREFILL:
            raise ValueError("unsupported GDN API")
        q, k, v = (t.reshape(-1, *t.shape[-2:]) for t in (x.q, x.k, x.v))
        lengths = (
            (x.cu_seqlens[1:] - x.cu_seqlens[:-1]).tolist()
            if x.cu_seqlens is not None
            else [x.q.shape[1]] * x.q.shape[0]
        )
        if x.use_qk_l2norm_in_kernel:
            q = (
                q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(q.dtype)
            k = (
                k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(k.dtype)
        initial = (
            x.initial_state
            if x.initial_state is not None
            else torch.zeros(
                len(lengths), v.shape[1], v.shape[2], q.shape[2], device=q.device
            )
        )
        g = (
            torch.ones(v.shape[:2], device=v.device)
            if x.g is None
            else x.g.reshape(v.shape[:2]).float()
        )
        if x.g is not None and x.is_log_space:
            g = g.exp()
        beta = torch.ones_like(g) if x.beta is None else x.beta.reshape(g.shape).float()
        out, state = blockwise_gdn_prefill_reference(
            q,
            k,
            v,
            g,
            beta,
            lengths,
            initial_state=initial,
            chunk_size=x.chunk_size,
            scale=scale,
        )
        return GdnReference(
            out=out.reshape(x.v.shape).to(x.v.dtype),
            state=state if x.output_final_state else None,
            unchanged_initial_state=x.initial_state.clone()
            if x.initial_state is not None
            else None,
        )

    @staticmethod
    def _reference_decode(x: GdnInputs, scale: float) -> GdnReference:
        if x.state_layout != "VK":
            raise ValueError("only VK decode state is implemented")
        assert x.q is not None and x.k is not None and x.v is not None
        if any(t is None for t in (x.state, x.A_log, x.a, x.dt_bias, x.b)):
            raise ValueError("decode requires state and all gate tensors")
        assert (
            x.state is not None
            and x.A_log is not None
            and x.a is not None
            and x.dt_bias is not None
            and x.b is not None
        )
        state = x.state.clone()
        output = torch.zeros_like(x.v)
        intermediate = (
            x.intermediate_states_buffer.clone()
            if x.intermediate_states_buffer is not None
            else None
        )
        indices = (
            list(range(x.q.shape[0]))
            if x.state_indices is None
            else x.state_indices.tolist()
        )
        valid = [i for i in indices if i >= 0]
        if len(set(valid)) != len(valid) or any(i >= state.shape[0] for i in valid):
            raise ValueError(
                "state indices must be distinct valid pool rows or negative padding"
            )
        for batch, idx in enumerate(indices):
            if idx < 0:
                if intermediate is not None:
                    intermediate[batch, : x.q.shape[1]].zero_()
                continue
            current = state[idx : idx + 1].transpose(-1, -2).contiguous()
            for step in range(x.q.shape[1]):
                out, current = decode_delta_rule(
                    x.q[batch : batch + 1, step],
                    x.k[batch : batch + 1, step],
                    x.v[batch : batch + 1, step],
                    current,
                    x.A_log,
                    x.a[batch : batch + 1, step],
                    x.dt_bias,
                    x.b[batch : batch + 1, step],
                    scale_factor=scale,
                    use_l2_norm=x.use_qk_l2norm,
                    state_dtype=state.dtype,
                )
                output[batch, step] = out[0].to(output.dtype)
                if intermediate is not None:
                    intermediate[batch, step] = current[0].transpose(-1, -2)
            if not x.disable_state_update:
                state[idx] = current[0].transpose(-1, -2)
        return GdnReference(out=output, state=state, intermediate_states=intermediate)

    def verify(
        self,
        inputs: OperatorInputs,
        outputs: OperatorOutputs,
        reference: OperatorReference,
    ) -> None:
        if (
            not isinstance(inputs, GdnInputs)
            or not isinstance(outputs, GdnOutputs)
            or not isinstance(reference, GdnReference)
        ):
            raise TypeError("expected GDN lifecycle types")
        x = inputs
        if x.workload is not None and x.workload.call is not None:
            torch.testing.assert_close(outputs.value, reference.value)
            return
        assert x.q is not None
        if x.api is GdnApi.DECODE:
            tol = 1e-3 if x.q.dtype == torch.float16 else 5e-3
            if x.state is not None and x.state.dtype == torch.bfloat16:
                tol = 2e-2
            atol, rtol = tol, tol
        else:
            atol, rtol = (2e-2, 2e-2) if x.q.dtype == torch.bfloat16 else (1e-2, 1e-2)
        for name in ("out", "state", "intermediate_states"):
            actual, expected = getattr(outputs, name), getattr(reference, name)
            if (actual is None) != (expected is None) or (
                name == "out" and actual is None
            ):
                raise AssertionError(f"GDN {name} presence mismatch")
            if expected is not None:
                if (
                    name == "out"
                    and x.api is GdnApi.DECODE
                    and x.q.shape[1] == 1
                    and actual.ndim == 3
                ):
                    actual = actual.unsqueeze(1)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        if (
            x.output is not None
            and outputs.out is not None
            and outputs.out.data_ptr() != x.output.data_ptr()
        ):
            raise AssertionError("GDN output buffer was not reused")
        target = x.state if x.api is GdnApi.DECODE else x.output_state
        if (
            target is not None
            and outputs.state is not None
            and target.data_ptr() != outputs.state.data_ptr()
        ):
            raise AssertionError("GDN state buffer was not reused")
        if reference.unchanged_initial_state is not None:
            torch.testing.assert_close(
                x.initial_state, reference.unchanged_initial_state, atol=0, rtol=0
            )
        if (
            x.api is GdnApi.DECODE
            and x.state is not None
            and reference.state is not None
        ):
            torch.testing.assert_close(x.state, reference.state, atol=atol, rtol=rtol)
            if x.state_indices is not None:
                mask = torch.ones(
                    x.state.shape[0], dtype=torch.bool, device=x.state.device
                )
                indices = x.state_indices[x.state_indices >= 0].long()
                mask[indices] = False
                torch.testing.assert_close(
                    x.state[mask], reference.state[mask], atol=0, rtol=0
                )


__all__ = [
    "GdnApi",
    "GdnWorkload",
    "GdnInputs",
    "GdnOutputs",
    "GdnReference",
    "GdnOperator",
]


# Independent reference implementations shared with the correctness tests.


@torch.inference_mode
def decode_delta_rule(
    q: torch.Tensor,  # [B, num_q_heads, K]
    k: torch.Tensor,  # [B, num_k_heads, K]
    v: torch.Tensor,  # [B, num_v_heads, V]
    state: torch.Tensor,  # [B, num_heads, K, V]
    A_log: torch.Tensor,  # [num_heads] - log decay parameter
    a: torch.Tensor,  # [B, num_heads] - input-dependent decay
    dt_bias: torch.Tensor,  # [num_heads] - decay bias
    b: torch.Tensor,  # [B, num_heads] - update gate input
    scale_factor: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_l2_norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for single-step decode with GDN formula.

    Strictly follows the Triton kernel logic from fused_sigmoid_gating_recurrent.py:
        1. Compute g = -exp(A_log) * softplus(a + dt_bias)
        2. Compute beta = sigmoid(b)
        3. Apply L2 norm to q and k (if enabled)
        4. h *= exp(g)                    # Apply decay to state
        5. v_new = v - k^T @ h            # Delta rule (h is [K,V], k is [K])
        6. v_new *= beta                  # Apply update gate
        7. h += k @ v_new^T               # Update state (outer product)
        8. o = q^T @ h                    # Compute output (h is [K,V], q is [K])

    Args:
        q: Query [B, num_q_heads, K]
        k: Key [B, num_k_heads, K]
        v: Value [B, num_v_heads, V]
        state: Input state [B, num_heads, K, V], where num_heads = num_v_heads
        A_log: Log decay parameter [num_heads]
        a: Input-dependent decay [B, num_heads]
        dt_bias: Decay bias [num_heads]
        b: Update gate input [B, num_heads]
        scale_factor: Scale factor for q
        softplus_beta: Beta parameter for softplus activation
        softplus_threshold: Threshold for softplus numerical stability
        use_l2_norm: Whether to apply L2 normalization to q and k
        state_dtype: Storage dtype for the hidden state (read in fp32, stored in this dtype)

    Returns:
        output: [B, num_heads, V]
        new_state: [B, num_heads, K, V]
    """
    B = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    K = q.size(2)
    V = v.size(2)

    # State and output are always based on num_v_heads (matches kernel's HV dimension)
    num_heads = num_v_heads

    device = q.device
    dtype = torch.float32

    # Convert to float32 for computation
    A_log = A_log.to(dtype).to(device)
    a = a.to(dtype).to(device)
    dt_bias = dt_bias.to(dtype).to(device)
    b = b.to(dtype).to(device)

    # ============================================
    # Compute gating values (following Triton kernel exactly)
    # ============================================

    # Step 1: Compute g = -exp(A_log) * softplus(a + dt_bias)
    # Triton kernel lines 100-109
    x = a + dt_bias  # [B, num_heads]
    beta_x = softplus_beta * x

    # Apply softplus with numerical stability
    # softplus(x) = (1/beta) * log(1 + exp(beta*x)) if beta*x <= threshold, else x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )

    # Compute g (log-space decay gate)
    # Triton kernel line 109: b_g = -tl.exp(b_A_log) * softplus_x
    g = -torch.exp(A_log) * softplus_x  # [B, num_heads]

    # Step 2: Compute beta = sigmoid(b)
    # Triton kernel line 112: b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    beta = 1.0 / (1.0 + torch.exp(-b))  # [B, num_heads]

    # Expand heads if needed (for GQA/GVA)
    # The reference works at v_heads level
    # For GQA (num_q_heads > num_v_heads): k and q need to be averaged/pooled per v_head
    # For GVA (num_v_heads > num_q_heads): q and k need to be repeated
    if num_k_heads < num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    if num_q_heads < num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    elif num_q_heads > num_v_heads:
        # GQA: multiple q_heads per v_head, reshape and average
        # [B, num_q_heads, K] -> [B, num_v_heads, num_q_heads//num_v_heads, K]
        q = q.reshape(B, num_v_heads, num_q_heads // num_v_heads, K).mean(dim=2)
        if num_k_heads == num_q_heads:
            k = k.reshape(B, num_v_heads, num_k_heads // num_v_heads, K).mean(dim=2)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    state = state.to(dtype)

    # Apply L2 normalization if requested
    if use_l2_norm:
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

    # Apply scale to q
    q = q * scale_factor

    # ============================================
    # Process each batch and head
    # ============================================
    new_state = torch.zeros(B, num_heads, K, V, device=device, dtype=state_dtype)
    output = torch.zeros(B, num_heads, V, device=device, dtype=dtype)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            # Get current vectors
            q_h = q[b_idx, h_idx]  # [K]
            k_h = k[b_idx, h_idx]  # [K]
            v_h = v[b_idx, h_idx]  # [V]
            h_state = (
                state[b_idx, h_idx].clone().to(torch.float32)
            )  # [K, V] read as fp32

            # Get gating values for this batch and head
            g_val = g[b_idx, h_idx]  # scalar
            beta_val = beta[b_idx, h_idx]  # scalar

            # ============================================
            # Recurrent update (following Triton kernel lines 121-134)
            # ============================================

            # Step 1: Apply gating to hidden state: h *= exp(g)
            # Triton kernel line 122: b_h *= tl.exp(b_g)
            h_state = h_state * torch.exp(g_val)

            # Step 2: Delta rule: v -= sum(h * k, dim=0)
            # Triton kernel line 125: b_v -= tl.sum(b_h * b_k[:, None], 0)
            # Triton: b_h is [BK, BV], b_k is [BK]
            # b_k[:, None] makes it [BK, 1]
            # b_h * b_k[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: k^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            # Match the kernel's explicit FP32 reduction for k * state rather
            # than dispatching a backend GEMV with an implementation-defined
            # reduction order.
            v_new = v_h - torch.sum(h_state * k_h[:, None], dim=0)

            # Step 3: Apply beta gating: v *= beta
            # Triton kernel line 128: b_v *= b_beta
            v_new = v_new * beta_val

            # Step 4: Update hidden state: h += k[:, None] * v[None, :]
            # Triton kernel line 131: b_h += b_k[:, None] * b_v[None, :]
            # Triton: [BK, BV] += [BK, 1] * [1, BV]
            # This is outer product: k @ v^T
            # [K, V] += [K, 1] @ [1, V]
            # Keep the reference as element-wise FP32 arithmetic.  Expressing
            # this outer product with ``@`` dispatches to MUBLAS, whose TF32
            # override changes the reference values.
            h_state = h_state + k_h[:, None] * v_new[None, :]

            # Step 5: Compute output: o = sum(h * q, dim=0)
            # Triton kernel line 134: b_o = tl.sum(b_h * b_q[:, None], 0)
            # Triton: b_h is [BK, BV], b_q is [BK]
            # b_q[:, None] makes it [BK, 1]
            # b_h * b_q[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: q^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            # Keep the reference reduction explicit instead of dispatching a
            # MUSA GEMV.  The kernel accumulates q * state in a fixed FP32
            # warp-reduction order; using ``q_h @ h_state`` can select a
            # backend reduction with a different order near low-precision
            # output rounding boundaries.
            output[b_idx, h_idx] = torch.sum(h_state * q_h[:, None], dim=0)

            # Store updated state (cast back to state_dtype)
            new_state[b_idx, h_idx] = h_state.to(state_dtype)

    return output, new_state


def exclusive_cumsum(seq_lens: list[int]) -> list[int]:
    out = [0]
    for seq_len in seq_lens:
        out.append(out[-1] + seq_len)
    return out


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    # Callers mask causality by token order (tril). A device cumsum is a parallel
    # scan and need not be monotone, so a rounded-positive gate difference means
    # "no decay", not "masked"; NaN (from -inf - -inf) still maps to zero weight.
    no_decay = torch.zeros_like(x)
    masked = torch.full_like(x, -float("inf"))
    return torch.exp(torch.where(x <= 0, x, torch.where(x < 1.0, no_decay, masked)))


@contextlib.contextmanager
def _exact_fp32_matmul():
    """The reference runs on the input device; keep its fp32 matmuls out of TF32.

    ``torch.backends.musa.matmul.allow_tf32`` is a separate switch from
    ``torch.backends.mudnn.allow_tf32`` and may be on, and TF32's error exceeds the
    output tolerance. (float64 is not an option: muDNN lacks some fp64 unary ops.)
    """
    switches = [
        getattr(getattr(torch.backends, "musa", None), "matmul", None),
        getattr(torch.backends, "mudnn", None),
    ]
    saved = [(s, s.allow_tf32) for s in switches if s is not None]
    try:
        for switch, _ in saved:
            switch.allow_tf32 = False
        yield
    finally:
        for switch, value in saved:
            switch.allow_tf32 = value


def _inverse_from_neg_strict_lower(p: torch.Tensor) -> torch.Tensor:
    length = p.shape[0]
    p = p.to(torch.float32)
    s = torch.eye(length, device=p.device, dtype=torch.float32)
    for i in range(1, length):
        s[i, :i] = p[i, :i] @ s[:i, :i]
    return s


@torch.inference_mode()
@_exact_fp32_matmul()
def blockwise_gdn_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    *,
    initial_state: torch.Tensor,
    chunk_size: int,
    scale: float,
    chunk_states: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """If ``chunk_states`` is a list, the state before every chunk is appended to it
    (same layout as ``initial_state``), sequence after sequence."""
    total_seq_len, num_qk_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    assert k.shape == (total_seq_len, num_qk_heads, head_size)
    assert v.shape == (total_seq_len, num_v_heads, head_size)
    assert g.shape == beta.shape == (total_seq_len, num_v_heads)
    assert num_v_heads % num_qk_heads == 0
    assert initial_state.shape == (
        len(seq_lens),
        num_v_heads,
        head_size,
        head_size,
    )

    group_size = num_v_heads // num_qk_heads
    output = torch.empty(
        total_seq_len, num_v_heads, head_size, device=q.device, dtype=torch.float32
    )
    final_state = torch.empty_like(initial_state, dtype=torch.float32)

    cu_seqlens = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(cu_seqlens[:-1]):
        seq_end = cu_seqlens[seq_idx + 1]
        state = torch.transpose(initial_state[seq_idx].to(torch.float32).clone(), 1, 2)

        for left in range(seq_start, seq_end, chunk_size):
            right = min(left + chunk_size, seq_end)
            if chunk_states is not None:
                chunk_states.append(torch.transpose(state, 1, 2).clone())
            q_blk = q[left:right].to(torch.float32)
            k_blk = k[left:right].to(torch.float32)
            v_blk = v[left:right].to(torch.float32)
            g_blk = torch.cumsum(torch.log(g[left:right].to(torch.float32)), dim=0)
            beta_blk = beta[left:right].to(torch.float32)

            next_state = torch.empty_like(state)
            for v_head in range(num_v_heads):
                qk_head = v_head // group_size
                q_h = q_blk[:, qk_head, :]
                k_h = k_blk[:, qk_head, :]
                v_h = v_blk[:, v_head, :]
                g_h = g_blk[:, v_head]
                beta_h = beta_blk[:, v_head]

                gamma = torch.tril(_safe_exp(g_h[:, None] - g_h[None, :]))
                gram = k_h @ k_h.transpose(0, 1)
                transition = -gram * beta_h[:, None] * torch.tril(gamma, diagonal=-1)
                solve = _inverse_from_neg_strict_lower(transition)

                old_state = state[v_head]
                g_exp = _safe_exp(g_h)
                w = v_h - g_exp.unsqueeze(-1) * (k_h @ old_state)
                new_v = (solve * beta_h[None, :]) @ w

                o_inter = g_exp.unsqueeze(-1) * (q_h @ old_state)
                attn = (q_h @ k_h.transpose(0, 1)) * gamma
                output[left:right, v_head] = scale * (o_inter + attn @ new_v)

                g_last = g_h[-1]
                new_v_scaled = _safe_exp(g_last - g_h).unsqueeze(-1) * new_v
                next_state[v_head] = (
                    _safe_exp(g_last) * old_state + k_h.transpose(0, 1) @ new_v_scaled
                )

            state = next_state

        final_state[seq_idx] = torch.transpose(state, 1, 2)

    return output, final_state
