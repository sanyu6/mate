import torch
from mate.arch import MP32_ARCH
from mate.ops import Op
from ..fp4_dispatch.buffer import Fp4DispatchBuffer
from ..dispatch_shared._input import _DispatchSharedInput
from ._input import _Input
from .impl import _Impl


class Fp4DispatchSharedMhc(Op):
    """Prepared dispatch + full local shared GEMM1 + independent mHC coefficients.

    Scratch and the dispatch buffer must be private to this stream, with completion
    initialized to zero. The residual's first 24 rows are valid; its storage is padded
    to 32 rows. packed_weight holds three BF16 components of immutable FP32 weight.
    Output is (dispatch,shared_gateup,(pre,post,comb)); no residual-post is applied.
    This experiment supports H5120, mHC factor4, b24, 20 Sinkhorn iterations and
    rms_eps=hc_eps=1e-6. Input preparation and weight decomposition are outside it.
    """

    @classmethod
    def resolve_backend(cls, target, *args, override_backend=None, **kwargs):
        if target != MP32_ARCH or override_backend not in (None, "fused", "serial"):
            raise ValueError("supports MP32 fused/serial only")
        return _Impl()

    @classmethod
    def run(
        cls,
        buffer: Fp4DispatchBuffer,
        shared_x: torch.Tensor,
        shared_x_scale: torch.Tensor,
        shared_weight: torch.Tensor,
        shared_weight_scale: torch.Tensor,
        shared_output: torch.Tensor,
        residual: torch.Tensor,
        mhc_weight: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        packed_weight: torch.Tensor,
        partial_mix: torch.Tensor,
        partial_sq: torch.Tensor,
        completion: torch.Tensor,
        mhc_output: torch.Tensor,
        *,
        num_ctas: int = 24,
        gemm_ctas: int | None = None,
        mhc_ctas: int = 20,
        mhc_tactic: str = "optimized",
        feature_tile: int = 64,
        stages: int = 2,
        override_backend: str | None = None,
    ):
        if mhc_tactic not in ("baseline", "optimized"):
            raise ValueError("mhc_tactic must be baseline or optimized")
        s = _DispatchSharedInput(
            buffer,
            shared_x,
            shared_x_scale,
            shared_weight,
            shared_weight_scale,
            shared_output,
            num_ctas,
            feature_tile,
            stages,
            gemm_ctas,
        )
        inputs = _Input(
            s,
            residual,
            mhc_weight,
            mhc_scale,
            mhc_base,
            packed_weight,
            partial_mix,
            partial_sq,
            completion,
            mhc_output,
            mhc_ctas,
            1 if override_backend == "serial" else 0,
            0 if mhc_tactic == "baseline" else 3,
        )
        return cls.resolve_backend(
            cls.resolve_target(device=buffer.device), override_backend=override_backend
        )(inputs)

    @classmethod
    def build(cls, inputs=None, *, aot=False):
        return _Impl().build(inputs, aot=aot)
