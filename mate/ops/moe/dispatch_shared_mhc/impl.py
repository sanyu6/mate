import torch
from mate.ops import Impl
from mate.jit.dispatch_shared_mhc_tce import get_dispatch_shared_mhc_tce
from ..dispatch_shared.impl import DispatchSharedImpl


class _Impl(Impl):
    def sanitize(self, inputs):
        s = inputs.shared
        DispatchSharedImpl().sanitize(s)
        if s.x.shape != (24, 5120):
            raise ValueError("experimental mHC fusion supports rows=24,H=5120 only")
        if (
            not isinstance(inputs.mhc_ctas, int)
            or isinstance(inputs.mhc_ctas, bool)
            or not 1 <= inputs.mhc_ctas <= 128
        ):
            raise ValueError("mHC CTA count must be within 1..128")
        if inputs.tactic not in (0, 3):
            raise ValueError("unsupported public mHC tactic")
        if inputs.mode not in (0, 1):
            raise ValueError("mode must be fused or serial")
        for name, shape, dtype in (
            ("residual", (32, 20480), torch.bfloat16),
            ("weight", (24, 20480), torch.float32),
            ("scale", (3,), torch.float32),
            ("base", (24,), torch.float32),
            ("packed_weight", (3, 32, 20480), torch.bfloat16),
            ("partial_mix", (20, 24, 24), torch.float32),
            ("partial_sq", (20, 24, 3), torch.float32),
            ("completion", (1,), torch.int32),
            ("output", (576,), torch.float32),
        ):
            t = getattr(inputs, name)
            if (
                t.shape != shape
                or t.dtype != dtype
                or t.device != s.buffer.device
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"invalid {name}: requires contiguous {shape} {dtype} on dispatch device"
                )

    def build(self, inputs=None, *, aot=False):
        if aot:
            raise ValueError("no AOT configurations")
        self.sanitize(inputs)
        s = inputs.shared
        impl = DispatchSharedImpl()
        return get_dispatch_shared_mhc_tce(
            s.buffer._sym.config,
            shared_features=s.weight.shape[0],
            num_ctas=impl._ctas(s),
            gemm_ctas=impl._gemm_ctas(s),
            mhc_ctas=inputs.mhc_ctas,
            padded=s.buffer.line_padded_scales,
            feature_tile=s.feature_tile,
            stages=s.stages,
            tactic=inputs.tactic,
        )

    def __call__(self, inputs):
        f = self.build(inputs)
        s = inputs.shared
        b = s.buffer
        sym = b._sym
        f(
            sym.buffer,
            sym.ptrs,
            *b.result.tensors,
            s.x.view(torch.uint8),
            s.x_scale.t(),
            s.weight.view(torch.uint8),
            s.weight_scale.t(),
            s.output,
            sym.rank,
            24,
            inputs.residual,
            inputs.weight,
            inputs.scale,
            inputs.base,
            inputs.output,
            1e-6,
            1e-6,
            inputs.mode,
            inputs.packed_weight,
            inputs.partial_mix,
            inputs.partial_sq,
            inputs.completion,
        )
        return (
            b.result,
            s.output,
            (
                inputs.output[:96].view(24, 4),
                inputs.output[96:192].view(24, 4),
                inputs.output[192:].view(24, 4, 4),
            ),
        )
