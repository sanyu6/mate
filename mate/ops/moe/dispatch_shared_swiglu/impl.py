import math
import torch
from mate.ops import Impl
from mate.jit.dispatch_shared_swiglu_mhc import get_dispatch_shared_swiglu_mhc
from ._input import _Input, PairedSharedWeight


class SwiGluImpl(Impl):
    def sanitize(self, i):
        if not isinstance(i, _Input) or i.buffer.closed:
            raise ValueError("requires live dispatch state")
        if not isinstance(i.weight, PairedSharedWeight):
            raise TypeError("prepare paired shared weight first")
        if i.weight.feature_tile not in (64, 128) or i.stages not in (2, 3, 4):
            raise ValueError("unsupported GEMM geometry")
        if i.dispatch_tactic not in (0, 2):
            raise ValueError("unsupported public dispatch schedule")
        if not math.isfinite(i.limit):
            raise ValueError("SwiGLU limit must be finite")
        b = i.buffer
        if b._sym.config.hidden != 5120:
            raise ValueError("supports H5120 only")
        for c in (i.dispatch_ctas, i.gemm_ctas):
            if (
                isinstance(c, bool)
                or not isinstance(c, int)
                or not 1 <= c <= b._sym.common_mps
            ):
                raise ValueError("CTA count exceeds common MPs")
        if (
            isinstance(i.mhc_ctas, bool)
            or not isinstance(i.mhc_ctas, int)
            or not 1 <= i.mhc_ctas <= 128
        ):
            raise ValueError("invalid mHC CTA count")
        tensors = [
            (i.x, (24, 5120), torch.float8_e4m3fn, False),
            (i.xs, (24, 160), torch.uint8, True),
            (i.weight.data, (4608, 5120), torch.float8_e4m3fn, False),
            (i.weight.scale, (4608, 160), torch.uint8, True),
            (i.q, (24, 2304), torch.float8_e4m3fn, False),
            (i.sf, (24, 72), torch.uint8, True),
            (i.residual, (32, 20480), torch.bfloat16, False),
            (i.mhc_weight, (24, 20480), torch.float32, False),
            (i.mhc_scale, (3,), torch.float32, False),
            (i.mhc_base, (24,), torch.float32, False),
            (i.mhc_packed, (3, 32, 20480), torch.bfloat16, False),
            (i.partial_mix, (20, 24, 24), torch.float32, False),
            (i.partial_sq, (20, 24, 3), torch.float32, False),
            (i.completion, (1,), torch.int32, False),
            (i.mhc_out, (576,), torch.float32, False),
        ]
        for t, shape, dtype, mn in tensors:
            if (
                t.shape != shape
                or t.dtype != dtype
                or t.device != b.device
                or not (t.t().is_contiguous() if mn else t.is_contiguous())
            ):
                raise ValueError(
                    f"requires {shape}, {dtype}, MN-major={mn} on dispatch device"
                )

    def build(self, inputs=None, *, aot=False):
        if aot:
            raise ValueError("no AOT configurations")
        self.sanitize(inputs)
        i = inputs
        return get_dispatch_shared_swiglu_mhc(
            i.buffer._sym.config,
            num_ctas=i.dispatch_ctas,
            gemm_ctas=i.gemm_ctas,
            mhc_ctas=i.mhc_ctas,
            feature_tile=i.weight.feature_tile,
            stages=i.stages,
            tactic=3,
            dispatch_tactic=i.dispatch_tactic,
            padded=i.buffer.line_padded_scales,
        )

    def __call__(self, i):
        f = self.build(i)
        b = i.buffer
        s = b._sym
        f(
            s.buffer,
            s.ptrs,
            *b.result.tensors,
            i.x.view(torch.uint8),
            i.xs.t(),
            i.weight.data.view(torch.uint8),
            i.weight.scale.t(),
            s.rank,
            24,
            i.residual,
            i.mhc_weight,
            i.mhc_scale,
            i.mhc_base,
            i.mhc_out,
            1e-6,
            1e-6,
            0,
            i.mhc_packed,
            i.partial_mix,
            i.partial_sq,
            i.completion,
            i.q.view(torch.uint8),
            i.sf.t(),
            float(i.limit),
        )
        return (
            b.result,
            (i.q, i.sf),
            (
                i.mhc_out[:96].view(24, 4),
                i.mhc_out[96:192].view(24, 4),
                i.mhc_out[192:].view(24, 4, 4),
            ),
        )
