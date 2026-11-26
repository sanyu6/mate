import math
import torch
from mate.ops import Impl
from mate.jit.combine_shared_mhc import get_combine_shared_mhc
from importlib import import_module
from functools import cache


@cache
def legacy_tail():
    kernels = import_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.dsv4.kernels.mhc_post_combine_norm_musa"
    )
    stream = import_module(
        "sglang.srt.hardware_backend.musa.jit_kernel.csrc.dsv4.kernels.stream_utils"
    )
    return (
        kernels._mhc_post_combine_norm_module().mhc_post_combine_norm_quant,
        stream.current_musa_stream_ptr,
    )


class CombineImpl(Impl):
    def sanitize(self, i):
        if not math.isfinite(i.route_scale):
            raise ValueError("routed scaling factor must be finite")
        if i.state._sym.config.hidden != 5120 or i.state._sym.config.topk != 6:
            raise ValueError("requires H5120/topk6 state")
        if i.state.closed:
            raise ValueError("requires dedicated live symmetric state")
        if i.mode not in (0, 1, 2):
            raise ValueError("invalid boundary mode")
        d = i.state.device
        specs = [
            (i.routed, (i.routed.shape[0], 5120), torch.bfloat16, False),
            (i.x, (24, 2304), torch.float8_e4m3fn, False),
            (i.xs, (24, 72), torch.uint8, True),
            (i.w, (5120, 2304), torch.float8_e4m3fn, False),
            (i.ws, (5120, 72), torch.uint8, True),
            (i.shared, (24, 5120), torch.bfloat16, False),
            (i.reduced, (24, 5120), torch.bfloat16, False),
            (i.merged, (24, 5120), torch.bfloat16, False),
            (i.status, (1,), torch.int32, False),
            (i.residual, (24, 4, 5120), torch.bfloat16, False),
            (i.post, (24, 4), torch.float32, False),
            (i.pre, (24, 4), torch.float32, False),
            (i.comb, (24, 4, 4), torch.float32, False),
            (i.gamma, (5120,), torch.bfloat16, False),
            (i.updated, (24, 4, 5120), torch.bfloat16, False),
            (i.normalized, (24, 5120), torch.bfloat16, False),
            (i.q, (24, 5120), torch.float8_e4m3fn, False),
            (i.sf, (24, 160), torch.uint8, True),
        ]
        for t, shape, dtype, mn in specs:
            if (
                t.shape != shape
                or t.dtype != dtype
                or t.device != d
                or not (t.t().is_contiguous() if mn else t.is_contiguous())
            ):
                raise ValueError(f"invalid operand {shape}/{dtype}/MN={mn}")
        if i.routes.metadata.shape != (i.routed.shape[0], 4):
            raise ValueError("routed and metadata pool capacity differ")

    def build(self, inputs=None, *, aot=False):
        if aot:
            raise ValueError("no AOT configurations")
        self.sanitize(inputs)
        return get_combine_shared_mhc(
            inputs.state._sym.config,
            producer_ctas=40,
            comm_threads=512,
            gemm_stages=4,
            ring2=True,
            fast_quant=True,
        )

    def __call__(self, i):
        f = self.build(i)
        s = i.state._sym
        f(
            s.buffer,
            s.ptrs,
            s.rank,
            i.routed,
            i.routes.metadata,
            i.routes.counts,
            i.routes.offsets,
            i.weighted,
            i.x.view(torch.uint8),
            i.xs.t(),
            i.w.view(torch.uint8),
            i.ws.t(),
            i.shared,
            i.reduced,
            i.merged,
            i.status,
            i.residual,
            i.post,
            i.comb,
            i.pre,
            i.gamma,
            i.updated,
            i.normalized,
            i.q.view(torch.uint8),
            i.sf.t(),
            i.route_scale,
            6 if i.mode == 1 else i.mode,
        )
        if i.mode == 1:
            tail, stream = legacy_tail()
            tail(
                i.merged,
                i.residual,
                i.post,
                i.comb,
                i.pre,
                i.gamma,
                i.updated,
                i.normalized,
                i.q,
                i.sf.t(),
                1e-6,
                2,
                stream(),
            )
        return i.updated, i.normalized, (i.q, i.sf)
