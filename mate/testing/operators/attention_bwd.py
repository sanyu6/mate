"""FlashAttention backward lifecycle with consistent forward data and autograd reference."""

from __future__ import annotations

import argparse
import random
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

import torch
import torch.nn.functional as F

from mate.execution_context import is_fake_mode
from .operator import (
    Operator,
    OperatorInputs,
    OperatorOutputs,
    OperatorReference,
    OperatorWorkload,
)


@dataclass
class BwdInputData:
    """Compatibility record for legacy input factories."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    dout: torch.Tensor
    lse: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    q_lens: torch.Tensor
    kv_lens: torch.Tensor
    seqused_q: torch.Tensor | None
    seqused_k: torch.Tensor | None
    max_q: int
    max_kv: int
    headdim_qk: int
    headdim_v: int
    is_bshd: bool


@dataclass(frozen=True, kw_only=True)
class AttentionBwdWorkload(OperatorWorkload):
    batch_size: int = 1
    seqlen_q: int = 128
    seqlen_kv: int = 128
    num_head_q: int = 8
    num_head_kv: int = 8
    headdim_qk: int = 64
    headdim_v: int | None = None
    dtype: torch.dtype = torch.bfloat16
    is_bshd: bool = False
    q_length_mode: str = "full"
    kv_length_mode: str = "full"
    q_lengths: tuple[int, ...] | None = None
    kv_lengths: tuple[int, ...] | None = None
    seqused_q: bool = False
    seqused_k: bool = False
    causal: bool = False
    window_size: tuple[int, int] = (-1, -1)
    softcap: float = 0.0
    softmax_scale: float | None = None
    deterministic: bool = False
    device: str | torch.device = "musa"
    seed: int | None = 42
    # Explicit compatibility only; new workloads do not carry argparse state.
    args: argparse.Namespace | None = None
    input_factory: Callable[[torch.device], BwdInputData] | None = None

    def __post_init__(self):
        if self.input_factory is not None:
            return
        for name in (
            "batch_size",
            "seqlen_q",
            "seqlen_kv",
            "num_head_q",
            "num_head_kv",
            "headdim_qk",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_head_q % self.num_head_kv:
            raise ValueError("num_head_q must be divisible by num_head_kv")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("backward supports FP16/BF16 inputs")
        if self.headdim_qk < 8 or (self.headdim_v is not None and self.headdim_v < 8):
            raise ValueError("backward head dimensions must be at least 8")
        if max(self.headdim_qk, self.headdim_v or self.headdim_qk) > 256:
            raise ValueError("backward supports head dimensions up to 256")
        left, right = self.window_size
        if min(left, right) < -1 or (left < 0) != (right < 0):
            raise ValueError("window dimensions must both be -1 or both non-negative")
        if self.softcap < 0:
            raise ValueError("softcap must be non-negative")
        if self.causal and self.seqlen_q > self.seqlen_kv:
            raise ValueError("causal benchmark requires seqlen_q <= seqlen_kv")
        for name, mode, lengths, maximum in (
            ("q", self.q_length_mode, self.q_lengths, self.seqlen_q),
            ("kv", self.kv_length_mode, self.kv_lengths, self.seqlen_kv),
        ):
            if mode not in ("full", "random"):
                raise ValueError(f"invalid {name} length mode")
            if lengths is not None and (
                len(lengths) != self.batch_size
                or any(n < 0 or n > maximum for n in lengths)
            ):
                raise ValueError(f"{name} lengths must fit each sequence capacity")


@dataclass(kw_only=True)
class AttentionBwdOperatorInputs(OperatorInputs):
    workload: AttentionBwdWorkload | None = None
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    out: torch.Tensor | None = None
    dout: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    cu_q: torch.Tensor | None = None
    cu_kv: torch.Tensor | None = None
    q_lens: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None
    seqused_q: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    max_q: int | None = None
    max_kv: int | None = None
    headdim_qk: int | None = None
    headdim_v: int | None = None
    is_bshd: bool = False
    causal: bool = False
    window_size: tuple[int, int] = (-1, -1)
    softcap: float = 0.0
    softmax_scale: float | None = None
    deterministic: bool = False
    kernel_dtype: str | None = None
    q_lengths: tuple[int, ...] | None = None
    kv_lengths: tuple[int, ...] | None = None


AttentionBwdInputs = AttentionBwdOperatorInputs


@dataclass(frozen=True, kw_only=True)
class AttentionBwdOutputs(OperatorOutputs):
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class AttentionBwdReference(OperatorReference):
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor

    @property
    def outputs(self) -> AttentionBwdOutputs:
        return AttentionBwdOutputs(dq=self.dq, dk=self.dk, dv=self.dv)


class AttentionBwdOperator(Operator):
    def generate(self, workload: OperatorWorkload) -> AttentionBwdInputs:
        if not isinstance(workload, AttentionBwdWorkload):
            raise TypeError("expected AttentionBwdWorkload")
        w = workload
        if w.input_factory is not None:
            data = w.input_factory(torch.device(w.device))
            x = AttentionBwdInputs(workload=w, **vars(data))
            if w.args is not None:
                x.causal = w.args.causal
                x.window_size = (w.args.window_left, w.args.window_right)
                x.softcap = w.args.softcap
                x.deterministic = w.args.deterministic
            return x
        if w.seed is not None:
            torch.manual_seed(w.seed)
        rng = random.Random(w.seed) if w.seed is not None else random
        q_len = w.q_lengths or tuple(
            rng.randint(1, w.seqlen_q) if w.q_length_mode == "random" else w.seqlen_q
            for _ in range(w.batch_size)
        )
        kv_len = w.kv_lengths or tuple(
            rng.randint(1, w.seqlen_kv) if w.kv_length_mode == "random" else w.seqlen_kv
            for _ in range(w.batch_size)
        )
        if w.causal:
            kv_len = tuple(max(q, k) for q, k in zip(q_len, kv_len, strict=True))
        physical_q = (w.seqlen_q,) * w.batch_size if w.is_bshd else q_len
        physical_k = (w.seqlen_kv,) * w.batch_size if w.is_bshd else kv_len
        q_shape = (w.batch_size, w.seqlen_q) if w.is_bshd else (sum(q_len),)
        kv_shape = (w.batch_size, w.seqlen_kv) if w.is_bshd else (sum(kv_len),)
        q = torch.randn(
            *q_shape, w.num_head_q, w.headdim_qk, device=w.device, dtype=w.dtype
        )
        k = torch.randn(
            *kv_shape, w.num_head_kv, w.headdim_qk, device=w.device, dtype=w.dtype
        )
        v = torch.randn(
            *kv_shape,
            w.num_head_kv,
            w.headdim_v or w.headdim_qk,
            device=w.device,
            dtype=w.dtype,
        )
        q_lens = torch.tensor(
            q_len if w.seqused_q else physical_q, device=w.device, dtype=torch.int32
        )
        kv_lens = torch.tensor(
            kv_len if w.seqused_k else physical_k, device=w.device, dtype=torch.int32
        )
        x = AttentionBwdInputs(
            workload=w,
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            kv_lens=kv_lens,
            cu_q=_cu(physical_q, w.device),
            cu_kv=_cu(physical_k, w.device),
            seqused_q=q_lens if w.seqused_q else None,
            seqused_k=kv_lens if w.seqused_k else None,
            max_q=max(physical_q),
            max_kv=max(physical_k),
            headdim_qk=w.headdim_qk,
            headdim_v=v.shape[-1],
            is_bshd=w.is_bshd,
            causal=w.causal,
            window_size=w.window_size,
            softcap=w.softcap,
            softmax_scale=w.softmax_scale,
            deterministic=w.deterministic,
            q_lengths=q_len if w.seqused_q else physical_q,
            kv_lengths=kv_len if w.seqused_k else physical_k,
        )
        x.dout = torch.randn(
            *q_shape, w.num_head_q, v.shape[-1], device=w.device, dtype=w.dtype
        )
        if is_fake_mode():
            x.out = torch.empty_like(x.dout)
            x.lse = torch.empty(
                (w.batch_size, w.num_head_q, w.seqlen_q)
                if w.is_bshd
                else (w.num_head_q, sum(q_len)),
                device=w.device,
                dtype=torch.float32,
            )
        else:
            with torch.no_grad(), _full_precision(q):
                out, lse = _forward(x, q.float(), k.float(), v.float())
                x.out = out.to(q.dtype).contiguous()
                x.lse = lse.contiguous()
        return x

    def call(self, inputs: OperatorInputs) -> AttentionBwdOutputs:
        if not isinstance(inputs, AttentionBwdInputs):
            raise TypeError("expected AttentionBwdInputs")
        x = inputs
        if any(t is None for t in (x.q, x.k, x.v, x.out, x.dout, x.lse)):
            raise ValueError("backward requires q, k, v, out, dout and lse")
        api = import_module("mate.flash_attention.tilelang.flash_attention_varlen_bwd")
        dq, dk, dv = api.flashattn_varlen_bwd_interface(
            x.q,
            x.k,
            x.v,
            x.out,
            x.dout,
            x.lse,
            x.max_q,
            x.max_kv,
            cu_seqlens_q=None if x.is_bshd else x.cu_q,
            cu_seqlens_k=None if x.is_bshd else x.cu_kv,
            seqused_q=x.seqused_q,
            seqused_k=x.seqused_k,
            is_causal=x.causal,
            window_size=x.window_size,
            softcap=x.softcap,
            smscale=x.softmax_scale,
            dtype=x.kernel_dtype,
            is_bhsd=x.is_bshd,
            deterministic=x.deterministic,
        )
        return AttentionBwdOutputs(dq=dq, dk=dk, dv=dv)

    def reference(self, inputs: OperatorInputs) -> AttentionBwdReference:
        if not isinstance(inputs, AttentionBwdInputs):
            raise TypeError("expected AttentionBwdInputs")
        x = inputs
        if x.q is None or x.k is None or x.v is None or x.dout is None:
            raise ValueError("reference requires q, k, v and dout")
        with torch.inference_mode(False), torch.enable_grad(), _full_precision(x.q):
            q, k, v = (
                t.detach().float().clone().requires_grad_(True) for t in (x.q, x.k, x.v)
            )
            out, _ = _forward(x, q, k, v)
            grads = torch.autograd.grad(
                out, (q, k, v), x.dout.float(), allow_unused=False
            )
        return AttentionBwdReference(
            dq=grads[0].to(x.q.dtype),
            dk=grads[1].to(x.k.dtype),
            dv=grads[2].to(x.v.dtype),
        )

    def verify(
        self,
        inputs: OperatorInputs,
        outputs: OperatorOutputs,
        reference: OperatorReference,
    ) -> None:
        if (
            not isinstance(inputs, AttentionBwdInputs)
            or not isinstance(outputs, AttentionBwdOutputs)
            or not isinstance(reference, AttentionBwdReference)
        ):
            raise TypeError("expected attention backward lifecycle types")
        for name in ("dq", "dk", "dv"):
            actual, expected = getattr(outputs, name), getattr(reference, name)
            bf16 = expected.dtype == torch.bfloat16
            torch.testing.assert_close(
                actual,
                expected,
                atol=3e-2 if bf16 else 5e-3,
                rtol=5e-2 if bf16 else 1e-2,
            )


def _cu(lengths, device):
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    return torch.tensor(offsets, device=device, dtype=torch.int32)


@contextmanager
def _full_precision(q):
    backend = (
        getattr(torch.backends, "mudnn", None) if q.device.type == "musa" else None
    )
    before = backend.allow_tf32 if backend is not None else None
    try:
        if backend is not None:
            backend.allow_tf32 = False
        yield
    finally:
        if backend is not None:
            backend.allow_tf32 = before


def _forward(x: AttentionBwdInputs, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Independent per-sequence attention, including GQA and bottom-right masks."""
    if x.is_bshd:
        q_offsets = [i * q.shape[1] for i in range(q.shape[0] + 1)]
        k_offsets = [i * k.shape[1] for i in range(k.shape[0] + 1)]
    else:
        if x.cu_q is None or x.cu_kv is None:
            raise ValueError("packed inputs require both cumulative length tensors")
        q_offsets = x.cu_q.tolist()
        k_offsets = x.cu_kv.tolist()
    qflat = q.reshape(-1, *q.shape[-2:])
    kflat = k.reshape(-1, *k.shape[-2:])
    vflat = v.reshape(-1, *v.shape[-2:])
    used_q = (
        x.seqused_q.tolist()
        if x.seqused_q is not None
        else [b - a for a, b in zip(q_offsets, q_offsets[1:])]
    )
    used_k = (
        x.seqused_k.tolist()
        if x.seqused_k is not None
        else [b - a for a, b in zip(k_offsets, k_offsets[1:])]
    )
    if (
        len(q_offsets) != len(k_offsets)
        or len(used_q) != len(q_offsets) - 1
        or len(used_k) != len(used_q)
    ):
        raise ValueError("sequence metadata batch sizes differ")
    scale = q.shape[-1] ** -0.5 if x.softmax_scale is None else x.softmax_scale
    outputs, lses = [], []
    for batch, (qs, qe, ks, ke) in enumerate(
        zip(q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True)
    ):
        nq, nk = used_q[batch], used_k[batch]
        if not (0 <= nq <= qe - qs and 0 <= nk <= ke - ks):
            raise ValueError("used lengths exceed sequence capacity")
        qb = qflat[qs : qs + nq]
        kb = kflat[ks : ks + nk]
        vb = vflat[ks : ks + nk]
        if q.shape[-2] % k.shape[-2]:
            raise ValueError("Q heads must be divisible by KV heads")
        repeat = q.shape[-2] // k.shape[-2]
        kb = kb.repeat_interleave(repeat, dim=1)
        vb = vb.repeat_interleave(repeat, dim=1)
        if nq and nk:
            score = torch.einsum("qhd,khd->hqk", qb * scale, kb)
            if x.softcap > 0:
                score = x.softcap * torch.tanh(score / x.softcap)
            posq = torch.arange(nq, device=q.device)[:, None] + nk - nq
            posk = torch.arange(nk, device=q.device)[None, :]
            valid = torch.ones((nq, nk), device=q.device, dtype=torch.bool)
            left, right = x.window_size
            if left >= 0:
                valid &= posk >= posq - left
            if right >= 0:
                valid &= posk <= posq + right
            if x.causal:
                valid &= posk <= posq
            score = score.masked_fill(~valid[None], -torch.inf)
            active = valid.any(-1)
            safe = torch.where(active[None, :, None], score, torch.zeros_like(score))
            lse = torch.where(
                active[None],
                torch.logsumexp(safe, -1),
                torch.full_like(safe[..., 0], torch.inf),
            )
            probs = torch.softmax(safe, -1) * active[None, :, None]
            out = torch.einsum("hqk,khd->qhd", probs, vb)
        else:
            out = (
                qb.new_zeros((nq, q.shape[-2], v.shape[-1]))
                + (qb.sum() + kb.sum() + vb.sum()) * 0
            )
            lse = q.new_full((q.shape[-2], nq), torch.inf)
        outputs.append(F.pad(out, (0, 0, 0, 0, 0, qe - qs - nq)))
        lses.append(F.pad(lse, (0, qe - qs - nq), value=torch.inf))
    out = (
        torch.cat(outputs, dim=0)
        if outputs
        else qflat.new_empty((0, q.shape[-2], v.shape[-1]))
    )
    # Connect even wholly unused/empty input tensors to the gradient graph.
    out = out + (q.sum() + k.sum() + v.sum()) * 0
    lse = torch.cat(lses, dim=-1) if lses else q.new_empty((q.shape[-2], 0))
    if x.is_bshd:
        out = out.reshape(*q.shape[:-1], v.shape[-1])
        lse = (
            lse.reshape(q.shape[-2], q.shape[0], q.shape[1])
            .permute(1, 0, 2)
            .contiguous()
        )
    return out, lse


__all__ = [
    "AttentionBwdWorkload",
    "AttentionBwdInputs",
    "AttentionBwdOperatorInputs",
    "AttentionBwdOutputs",
    "AttentionBwdReference",
    "AttentionBwdOperator",
    "BwdInputData",
]
