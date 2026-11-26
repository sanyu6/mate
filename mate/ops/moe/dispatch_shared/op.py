import torch
from mate.arch import MP32_ARCH
from mate.ops import Op
from ..fp4_dispatch.buffer import Fp4DispatchBuffer, Fp4DispatchResult
from ._input import _DispatchSharedInput
from .impl import DispatchSharedImpl


class Fp4DispatchSharedGemm(Op):
    """Dispatch routed FP4 inputs while computing the local full MXFP8 gate/up.

    Returns the same dispatch views as Fp4Dispatch and caller-owned BF16 shared
    output. No routed GEMM, SiLU, output quantizer, shared GEMM2, or combine is
    performed. Weights are unsharded on each rank. Inputs must already be
    quantized. Each call consumes one dispatch epoch; the buffer cannot be
    used concurrently from another stream. The native SS GEMM preserves MXFP8
    semantics but is not promised bitwise identical to muDNN's accumulation.
    """

    @classmethod
    def resolve_backend(cls, target, *args, override_backend=None, **kwargs):
        if target != MP32_ARCH or override_backend not in (
            None,
            "mp32_fused",
            "mp32_serial",
        ):
            raise ValueError("supports mp32_fused or mp32_serial on MP32 only")
        return DispatchSharedImpl(fused=override_backend != "mp32_serial")

    @classmethod
    def run(
        cls,
        buffer: Fp4DispatchBuffer,
        shared_x: torch.Tensor,
        shared_x_scale: torch.Tensor,
        shared_weight: torch.Tensor,
        shared_weight_scale: torch.Tensor,
        shared_output: torch.Tensor,
        *,
        num_ctas: int | None = None,
        gemm_ctas: int | None = None,
        feature_tile: int = 128,
        stages: int = 2,
        override_backend: str | None = None,
    ) -> tuple[Fp4DispatchResult, torch.Tensor]:
        """Execute on the current stream, with no implicit quantization.

        Parameters:
            buffer: Prepared dedicated dispatch state, same rows across ranks.
            shared_x: Contiguous E4M3 [rows,H] from the same local hidden input.
            shared_x_scale: UE8M0 [rows,H/32], MN-major (transpose contiguous).
            shared_weight: Contiguous E4M3 [features,H], full local gate/up weight.
            shared_weight_scale: UE8M0 [features,H/32], MN-major.
            shared_output: Caller-owned contiguous BF16 [rows,features].
            num_ctas: Same resident CTA count on every rank, default min(24,MPs).
            gemm_ctas: Optional compute CTA count; defaults to min(tile count, dispatch CTAs, common MPs).
            feature_tile: 64 or 128 output features per compute task.
            stages: 2, 3 or 4 TME pipeline stages.
            override_backend: None/mp32_fused or the two-kernel mp32_serial.
        """
        inputs = _DispatchSharedInput(
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
        DispatchSharedImpl().sanitize(inputs)
        target = cls.resolve_target(device=buffer.device)
        return cls.resolve_backend(target, override_backend=override_backend)(inputs)

    @classmethod
    def build(cls, inputs=None, *, aot=False):
        """Compile explicit _DispatchSharedInput without executing an epoch."""
        if aot:
            raise ValueError("no predefined AOT configurations")
        impl = DispatchSharedImpl()
        impl.sanitize(inputs)
        target = cls.resolve_target(device=inputs.buffer.device)
        return cls.resolve_backend(target).build(inputs)
