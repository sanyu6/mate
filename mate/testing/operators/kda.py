"""Structured recurrent operator with independent Python references."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Callable
from functools import wraps

import torch

from .operator import (
    Operator,
    OperatorInputs,
    OperatorOutputs,
    OperatorReference,
    OperatorWorkload,
)

CHUNK_SIZE = 32
HEAD_SIZE = 128


class KdaApi(Enum):
    CHUNK = "chunk_kda"


@dataclass(frozen=True, kw_only=True)
class KdaWorkload(OperatorWorkload):
    api: KdaApi = KdaApi.CHUNK
    batch_size: int = 1
    seqlen: int = 64
    seq_lens: tuple[int, ...] | None = None
    num_head_qk: int = 4
    num_head_v: int = 4
    headdim: int = 128
    dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "musa"
    seed: int | None = 666
    state_mode: str = "none"
    state_in: bool = False
    state_out: bool = False
    gate_params: bool = True
    qk_l2norm: bool = True
    lower_bound: float = -5.0
    scale: float | None = None
    call: Callable[[], Any] | None = None
    reference_call: Callable[[], Any] | None = None

    def __post_init__(self):
        if self.call is not None:
            return
        if self.api is not KdaApi.CHUNK:
            raise ValueError("unsupported KDA API")
        if min(self.batch_size, self.seqlen, self.num_head_qk, self.num_head_v) <= 0:
            raise ValueError("KDA dimensions must be positive")
        if self.headdim != 128 or self.num_head_v % self.num_head_qk:
            raise ValueError("KDA requires D=128 and value heads divisible by QK heads")
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("KDA requires bfloat16 or float16")
        if self.seq_lens is not None and (
            len(self.seq_lens) != self.batch_size or min(self.seq_lens) <= 0
        ):
            raise ValueError("seq_lens must contain a positive length per sequence")
        if self.state_mode not in ("none", "bfloat16", "fp32"):
            raise ValueError("invalid state_mode")
        if (self.state_mode == "none") == (self.state_in or self.state_out):
            raise ValueError("state_mode must agree with state_in/state_out")
        if self.state_mode == "bfloat16" and self.dtype != torch.bfloat16:
            raise ValueError("bfloat16 state requires bfloat16 inputs")

    @property
    def lengths(self) -> tuple[int, ...]:
        return (
            self.seq_lens
            if self.seq_lens is not None
            else (self.seqlen,) * self.batch_size
        )


@dataclass(kw_only=True)
class KdaInputs(OperatorInputs):
    api: KdaApi = KdaApi.CHUNK
    workload: KdaWorkload | None = None
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    g: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    initial_state: torch.Tensor | None = None
    initial_state_indices: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    A_log: torch.Tensor | None = None
    dt_bias: torch.Tensor | None = None
    output: torch.Tensor | None = None
    final_state: torch.Tensor | None = None
    scale: float | None = None
    output_final_state: bool = False
    lower_bound: float = -5.0
    use_qk_l2norm_in_kernel: bool = True


@dataclass(frozen=True, kw_only=True)
class KdaOutputs(OperatorOutputs):
    out: torch.Tensor | None = None
    final_state: torch.Tensor | None = None
    value: Any = None


@dataclass(frozen=True, kw_only=True)
class KdaReference(OperatorReference):
    out: torch.Tensor | None = None
    final_state: torch.Tensor | None = None
    expected_initial_state: torch.Tensor | None = None
    value: Any = None


class KdaOperator(Operator):
    def generate(self, workload: OperatorWorkload) -> KdaInputs:
        if not isinstance(workload, KdaWorkload):
            raise TypeError("expected KdaWorkload")
        w = workload
        if w.call is not None:
            return KdaInputs(workload=w)
        if w.seed is not None:
            torch.manual_seed(w.seed)
        shape = (
            (sum(w.lengths),) if w.seq_lens is not None else (w.batch_size, w.seqlen)
        )
        q = torch.randn(
            *shape, w.num_head_qk, w.headdim, dtype=w.dtype, device=w.device
        )
        k = torch.randn_like(q)
        if not w.qk_l2norm:
            q, k = (_normalize_last_dim(t.float()).to(w.dtype) for t in (q, k))
        v = torch.randn(*shape, w.num_head_v, w.headdim, dtype=w.dtype, device=w.device)
        # Without gate parameters g is already log2 decay, not a sigmoid input.
        g = torch.randn_like(v) if w.gate_params else -0.02 * torch.rand_like(v)
        x = KdaInputs(
            workload=w,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=torch.randn(*shape, w.num_head_v, dtype=w.dtype, device=w.device),
            output=torch.empty_like(v),
            scale=w.scale,
            output_final_state=w.state_out,
            lower_bound=w.lower_bound,
            use_qk_l2norm_in_kernel=w.qk_l2norm,
        )
        if w.gate_params:
            x.A_log = torch.randn(w.num_head_v, device=w.device)
            x.dt_bias = torch.randn(w.num_head_v, w.headdim, device=w.device)
        dtype = torch.bfloat16 if w.state_mode == "bfloat16" else torch.float32
        state_shape = (w.batch_size, w.num_head_v, w.headdim, w.headdim)
        if w.state_in:
            x.initial_state = (
                torch.randn(state_shape, dtype=dtype, device=w.device) * 0.05
            )
        if w.state_out:
            x.final_state = torch.empty(state_shape, dtype=dtype, device=w.device)
        if w.seq_lens is not None:
            lengths = torch.tensor((0, *w.lengths), dtype=torch.int32, device=w.device)
            x.cu_seqlens = lengths.cumsum(0).to(torch.int32)
        return x

    def call(self, inputs: OperatorInputs) -> KdaOutputs:
        if not isinstance(inputs, KdaInputs):
            raise TypeError("expected KdaInputs")
        x = inputs
        if x.workload is not None and x.workload.call is not None:
            return KdaOutputs(value=x.workload.call())
        if x.api is not KdaApi.CHUNK:
            raise ValueError("unsupported KDA API")
        result = import_module("mate.kda").chunk_kda(
            q=x.q,
            k=x.k,
            v=x.v,
            g=x.g,
            beta=x.beta,
            scale=x.scale,
            initial_state=x.initial_state,
            output_final_state=x.output_final_state,
            cu_seqlens=x.cu_seqlens,
            A_log=x.A_log,
            dt_bias=x.dt_bias,
            lower_bound=x.lower_bound,
            use_qk_l2norm_in_kernel=x.use_qk_l2norm_in_kernel,
            output=x.output,
            final_state=x.final_state,
            initial_state_indices=x.initial_state_indices,
        )
        return KdaOutputs(
            out=result[0] if isinstance(result, tuple) else result,
            final_state=result[1] if isinstance(result, tuple) else None,
            value=result,
        )

    def reference(self, inputs: OperatorInputs) -> KdaReference:
        if not isinstance(inputs, KdaInputs):
            raise TypeError("expected KdaInputs")
        x = inputs
        if x.workload is not None and x.workload.call is not None:
            if x.workload.reference_call is None:
                raise NotImplementedError(
                    "legacy call requires independent reference_call"
                )
            return KdaReference(value=x.workload.reference_call())
        if any(t is None for t in (x.q, x.k, x.v, x.g, x.beta)):
            raise ValueError("KDA requires q, k, v, g, beta")
        assert (
            x.q is not None
            and x.k is not None
            and x.v is not None
            and x.g is not None
            and x.beta is not None
        )
        if (x.A_log is None) != (x.dt_bias is None):
            raise ValueError("A_log and dt_bias must be supplied together")
        nseq = x.cu_seqlens.numel() - 1 if x.cu_seqlens is not None else x.q.shape[0]
        initial = x.initial_state
        pool = initial.clone() if initial is not None else None
        if x.initial_state_indices is not None:
            if initial is None:
                raise ValueError("state indices require initial_state")
            ids = x.initial_state_indices.long()
            if (
                ids.numel() != nseq
                or ids.unique().numel() != nseq
                or (ids < 0).any()
                or (ids >= initial.shape[0]).any()
            ):
                raise ValueError("state indices require distinct valid pool rows")
            initial = initial.index_select(0, ids)
        if initial is None:
            dtype = x.final_state.dtype if x.final_state is not None else x.q.dtype
            initial = torch.zeros(
                nseq,
                x.v.shape[-2],
                x.v.shape[-1],
                x.q.shape[-1],
                device=x.q.device,
                dtype=dtype,
            )
        tensors = [x.q, x.k, x.v, x.g, x.beta]
        if x.cu_seqlens is not None and x.q.ndim == 4:
            tensors = [t.squeeze(0) for t in tensors]
        out, state = blockwise_kda_reference(
            *tensors,
            scale=x.q.shape[-1] ** -0.5 if x.scale is None else x.scale,
            initial_state=initial,
            A_log=x.A_log,
            dt_bias=x.dt_bias,
            lower_bound=x.lower_bound,
            use_qk_l2norm_in_kernel=x.use_qk_l2norm_in_kernel,
            cu_seqlens=x.cu_seqlens,
        )
        final = state
        if x.initial_state_indices is not None:
            assert pool is not None
            pool[x.initial_state_indices.long()] = state
            final = pool
        return KdaReference(
            out=out.reshape(x.v.shape),
            final_state=final if x.output_final_state else None,
            expected_initial_state=pool,
        )

    def verify(
        self,
        inputs: OperatorInputs,
        outputs: OperatorOutputs,
        reference: OperatorReference,
    ) -> None:
        if (
            not isinstance(inputs, KdaInputs)
            or not isinstance(outputs, KdaOutputs)
            or not isinstance(reference, KdaReference)
        ):
            raise TypeError("expected KDA lifecycle types")
        x = inputs
        if x.workload is not None and x.workload.call is not None:
            torch.testing.assert_close(outputs.value, reference.value)
            return
        assert x.q is not None
        for name in ("out", "final_state"):
            actual, expected = getattr(outputs, name), getattr(reference, name)
            if (actual is None) != (expected is None) or (
                name == "out" and actual is None
            ):
                raise AssertionError(f"KDA {name} presence mismatch")
            if expected is not None:
                compare_bf16 = (
                    x.q.dtype if name == "out" else expected.dtype
                ) == torch.bfloat16
                atol = (
                    (2e-2 if name == "out" else 8e-2)
                    if compare_bf16
                    else (1.25e-2 if name == "out" else 6e-2)
                )
                torch.testing.assert_close(
                    actual, expected, atol=atol, rtol=2e-2 if compare_bf16 else 1.5e-2
                )
        if (
            x.output is not None
            and outputs.out is not None
            and x.output.data_ptr() != outputs.out.data_ptr()
        ):
            raise AssertionError("KDA output buffer was not reused")
        target = (
            x.initial_state if x.initial_state_indices is not None else x.final_state
        )
        if (
            target is not None
            and outputs.final_state is not None
            and target.data_ptr() != outputs.final_state.data_ptr()
        ):
            raise AssertionError("KDA final state buffer was not reused")
        if reference.expected_initial_state is not None:
            if x.initial_state_indices is None:
                torch.testing.assert_close(
                    x.initial_state, reference.expected_initial_state, atol=0, rtol=0
                )
            else:
                assert x.initial_state is not None
                bf16 = x.initial_state.dtype == torch.bfloat16
                torch.testing.assert_close(
                    x.initial_state,
                    reference.expected_initial_state,
                    atol=8e-2 if bf16 else 6e-2,
                    rtol=2e-2 if bf16 else 1.5e-2,
                )
                mask = torch.ones(
                    x.initial_state.shape[0],
                    dtype=torch.bool,
                    device=x.initial_state.device,
                )
                mask[x.initial_state_indices.long()] = False
                torch.testing.assert_close(
                    x.initial_state[mask],
                    reference.expected_initial_state[mask],
                    atol=0,
                    rtol=0,
                )


__all__ = [
    "KdaApi",
    "KdaWorkload",
    "KdaInputs",
    "KdaOutputs",
    "KdaReference",
    "KdaOperator",
]


# Independent reference implementations shared with the correctness tests.


def _exp2_ftz_like(x: torch.Tensor) -> torch.Tensor:
    y = torch.exp2(torch.clamp(x, max=127.0))
    return torch.where(x < -126.0, torch.zeros_like(y), y)


def _full_precision_musa_matmul(fn):
    @wraps(fn)
    def wrapped(q: torch.Tensor, *args, **kwargs):
        if q.device.type != "musa":
            return fn(q, *args, **kwargs)
        allow_tf32 = torch.backends.mudnn.allow_tf32
        torch.backends.mudnn.allow_tf32 = False
        try:
            return fn(q, *args, **kwargs)
        finally:
            torch.backends.mudnn.allow_tf32 = allow_tf32

    return wrapped


def _normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    # Match K1's normalization exactly. The epsilon is observable for the
    # all-zero Q/K vectors used by the single-token residual-layout probe:
    # without it the reference computes 0 * inf, which is NaN on CI devices.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1.0e-6)


@torch.inference_mode()
@_full_precision_musa_matmul
def blockwise_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = q.shape[-1]
    Hqk = q.shape[-2]
    H = v.shape[-2]
    assert D == HEAD_SIZE
    assert k.shape[-2] == Hqk
    assert g.shape[-2] == H
    assert beta.shape[-1] == H
    assert H % Hqk == 0
    qk_group = H // Hqk

    if cu_seqlens is None and q.ndim == 4:
        batch_size, num_tokens = q.shape[:2]
        flat_q = q.reshape(-1, Hqk, D)
        flat_k = k.reshape(-1, Hqk, D)
        flat_v = v.reshape(-1, H, D)
        flat_g = g.reshape(-1, H, D)
        flat_beta = beta.reshape(-1, H)
        seq_offsets = [i * num_tokens for i in range(batch_size + 1)]
        out_shape = v.shape
    elif cu_seqlens is None:
        if q.ndim != 3:
            raise AssertionError(
                "Dense reference inputs must be [B, T, H, D] or flattened to [B*T, H, D]."
            )
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_len = flat_q.shape[0] // initial_state.shape[0]
        seq_offsets = [i * seq_len for i in range(initial_state.shape[0] + 1)]
        out_shape = v.shape
    else:
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_offsets = [int(x) for x in cu_seqlens.tolist()]
        out_shape = v.shape

    if use_qk_l2norm_in_kernel:
        qn = _normalize_last_dim(flat_q.float()).to(q.dtype).float()
        kn = _normalize_last_dim(flat_k.float()).to(k.dtype).float()
    else:
        qn = flat_q.float()
        kn = flat_k.float()
    out = torch.empty_like(flat_v)
    state = initial_state.clone()
    log2e = 1.4426950408889634
    qk_head_indices = torch.arange(H, device=q.device, dtype=torch.long) // qk_group

    def round_workspace(x: torch.Tensor) -> torch.Tensor:
        return x.to(q.dtype).float()

    for seq_idx, bos in enumerate(seq_offsets[:-1]):
        eos = seq_offsets[seq_idx + 1]
        s = state[seq_idx].float().transpose(-1, -2)
        for start in range(bos, eos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, eos)
            actual = end - start
            q_blk = torch.zeros(
                (H, CHUNK_SIZE, D), device=q.device, dtype=torch.float32
            )
            k_blk = torch.zeros_like(q_blk)
            v_blk = torch.zeros_like(q_blk)
            g_blk = torch.zeros_like(q_blk)
            beta_blk = torch.full(
                (H, CHUNK_SIZE), -80.0, device=q.device, dtype=torch.float32
            )

            q_blk[:, :actual] = (
                qn[start:end].index_select(1, qk_head_indices).permute(1, 0, 2)
            )
            k_blk[:, :actual] = (
                kn[start:end].index_select(1, qk_head_indices).permute(1, 0, 2)
            )
            v_blk[:, :actual] = flat_v[start:end].permute(1, 0, 2).float()
            g_blk[:, :actual] = flat_g[start:end].permute(1, 0, 2).float()
            beta_blk[:, :actual] = flat_beta[start:end].transpose(0, 1).float()

            gate = (
                g_blk
                if A_log is None
                else (
                    lower_bound
                    * log2e
                    * torch.sigmoid(
                        torch.exp(A_log)[:, None, None] * (g_blk + dt_bias[:, None, :])
                    )
                )
            )
            if actual < CHUNK_SIZE:
                gate[:, actual:] = 0.0
            g_cumsum = gate.cumsum(dim=1)
            g_total = g_cumsum[:, -1]
            shift = 0.5 * (g_cumsum[:, 0] + g_total)
            k_decayed = round_workspace(k_blk * _exp2_ftz_like(g_cumsum))
            q_decayed = round_workspace(q_blk * _exp2_ftz_like(g_cumsum) * scale)
            k_decayed_for_tri = round_workspace(
                k_blk * _exp2_ftz_like(g_cumsum - shift[:, None])
            )
            q_decayed_for_p = round_workspace(
                q_blk * _exp2_ftz_like(g_cumsum - shift[:, None]) * scale
            )
            k_inv = round_workspace(k_blk * _exp2_ftz_like(shift[:, None] - g_cumsum))
            beta_sigmoid = torch.sigmoid(beta_blk)
            l_mat = round_workspace(
                torch.bmm(k_decayed_for_tri, k_inv.transpose(1, 2)).tril(-1)
                * beta_sigmoid[:, :, None]
            )

            a = torch.zeros(
                (H, CHUNK_SIZE, CHUNK_SIZE), device=q.device, dtype=torch.float32
            )
            a[:, 0, 0] = beta_sigmoid[:, 0]
            for i in range(1, CHUNK_SIZE):
                a[:, i, i] = beta_sigmoid[:, i]
                a[:, i, :i] = -torch.bmm(l_mat[:, i : i + 1, :i], a[:, :i, :i]).squeeze(
                    1
                )
            a = round_workspace(a)
            p = round_workspace(
                torch.bmm(q_decayed_for_p, k_inv.transpose(1, 2)).tril()
            )
            s_committed = round_workspace(s)
            tmp = round_workspace(v_blk - torch.bmm(k_decayed, s_committed))
            u = round_workspace(torch.bmm(a, tmp))
            out_chunk = round_workspace(
                torch.bmm(q_decayed, s_committed) + torch.bmm(p, u)
            )
            out[start:end] = out_chunk[:, :actual].permute(1, 0, 2)

            k_restored = round_workspace(
                k_blk * _exp2_ftz_like(g_total[:, None] - g_cumsum)
            )
            # s is [head, K, V]; each gate channel scales a K row.
            s = s * _exp2_ftz_like(g_total)[:, :, None] + torch.bmm(
                k_restored.transpose(1, 2), u
            )
        state[seq_idx].copy_(s.transpose(-1, -2).to(initial_state.dtype))

    return out.reshape(out_shape), state
