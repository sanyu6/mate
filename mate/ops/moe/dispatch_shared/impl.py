from importlib import import_module
import torch

from mate.ops import Impl
from ..fp4_dispatch.op import Fp4Dispatch
from ..fp4_dispatch.buffer import Fp4DispatchBuffer
from ._input import _DispatchSharedInput


class DispatchSharedImpl(Impl):
    def __init__(self, fused=True):
        self.fused = fused

    def sanitize(self, inputs):
        if not isinstance(inputs, _DispatchSharedInput):
            raise TypeError("requires _DispatchSharedInput")
        b = inputs.buffer
        if not isinstance(b, Fp4DispatchBuffer) or b.closed:
            raise ValueError("requires a live dedicated Fp4DispatchBuffer")
        x, xs, w, ws, out = (
            inputs.x,
            inputs.x_scale,
            inputs.weight,
            inputs.weight_scale,
            inputs.output,
        )
        if (
            x.ndim != 2
            or w.ndim != 2
            or x.shape[1] != b._sym.config.hidden
            or w.shape[1] != x.shape[1]
        ):
            raise ValueError("invalid shared GEMM dimensions")
        rows, h = x.shape
        n = w.shape[0]
        if inputs.feature_tile not in (64, 128) or inputs.stages not in (2, 3, 4):
            raise ValueError("unsupported GEMM tile/pipeline configuration")
        if not 1 <= rows <= b.max_tokens or n <= 0 or n % inputs.feature_tile:
            raise ValueError("invalid rows or shared output feature count")
        if (
            x.dtype != torch.float8_e4m3fn
            or w.dtype != torch.float8_e4m3fn
            or not x.is_contiguous()
            or not w.is_contiguous()
        ):
            raise ValueError("shared input and weight must be contiguous E4M3")
        for sf, mn in ((xs, rows), (ws, n)):
            if (
                sf.shape != (mn, h // 32)
                or sf.dtype != torch.uint8
                or not sf.t().is_contiguous()
            ):
                raise ValueError("shared scales must be MN-major uint8 UE8M0")
        if (
            out.shape != (rows, n)
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            raise ValueError("shared output must be contiguous BF16 [rows,features]")
        if any(t.device != b.device for t in (x, xs, w, ws, out)):
            raise ValueError(
                "all shared operands must be on the dispatch buffer device"
            )
        ctas = self._ctas(inputs)
        if (
            isinstance(ctas, bool)
            or not isinstance(ctas, int)
            or not 1 <= ctas <= b._sym.common_mps
        ):
            raise ValueError("num_ctas must be within the common MP count")
        gemm_ctas = self._gemm_ctas(inputs)
        if (
            isinstance(gemm_ctas, bool)
            or not isinstance(gemm_ctas, int)
            or not 1 <= gemm_ctas <= b._sym.common_mps
        ):
            raise ValueError("gemm_ctas must be within the common MP count")

    def _ctas(self, inputs):
        return (
            inputs.num_ctas
            if inputs.num_ctas is not None
            else min(24, inputs.buffer._sym.common_mps)
        )

    def _gemm_ctas(self, inputs):
        tasks = (inputs.weight.shape[0] // inputs.feature_tile) * (
            (inputs.x.shape[0] + 31) // 32
        )
        return (
            inputs.gemm_ctas
            if inputs.gemm_ctas is not None
            else min(tasks, self._ctas(inputs), inputs.buffer._sym.common_mps)
        )

    def build(self, inputs=None, *, aot=False):
        if aot:
            raise ValueError("no predefined AOT configurations")
        self.sanitize(inputs)
        b = inputs.buffer
        jit = import_module("mate.jit.dispatch_shared")
        return jit.get_dispatch_shared(
            b._sym.config,
            shared_features=inputs.weight.shape[0],
            num_ctas=self._ctas(inputs),
            padded=b.line_padded_scales,
            fused=self.fused,
            feature_tile=inputs.feature_tile,
            stages=inputs.stages,
            gemm_ctas=self._gemm_ctas(inputs),
        )

    def __call__(self, inputs):
        launcher = self.build(inputs)
        b = inputs.buffer
        s = b._sym
        rows = inputs.x.shape[0]
        if not self.fused:
            Fp4Dispatch(b, rows, num_ctas=self._ctas(inputs))
        launcher(
            s.buffer,
            s.ptrs,
            *b.result.tensors,
            inputs.x.view(torch.uint8),
            inputs.x_scale.t(),
            inputs.weight.view(torch.uint8),
            inputs.weight_scale.t(),
            inputs.output,
            s.rank,
            rows,
        )
        return b.result, inputs.output
