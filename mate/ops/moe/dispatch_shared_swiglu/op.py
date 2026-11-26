import torch
from mate.ops import Op
from mate.arch import MP32_ARCH
from ..fp4_dispatch.buffer import Fp4DispatchBuffer
from ._input import _Input, PairedSharedWeight
from .impl import SwiGluImpl


def prepare_shared_swiglu_weight(
    weight: torch.Tensor, scale: torch.Tensor, feature_tile: int = 128
) -> PairedSharedWeight:
    """Pack immutable gate/up halves once, outside capture and invocation timing.

    Input weight is E4M3 [4608,5120]; scale is UE8M0 [4608,160], MN-major.
    Each packed tile contains gate[feature_tile/2] followed by matching up rows.
    Quantized values and per-row scales are permuted without requantization.
    """
    if feature_tile not in (64, 128):
        raise ValueError("unsupported feature tile")
    if (
        weight.shape != (4608, 5120)
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        raise ValueError("invalid shared weight")
    if (
        scale.shape != (4608, 160)
        or scale.dtype != torch.uint8
        or not scale.t().is_contiguous()
        or scale.device != weight.device
    ):
        raise ValueError("invalid weight scale")
    perm = torch.arange(2304, device=weight.device).reshape(-1, feature_tile // 2)
    perm = torch.stack((perm, perm + 2304), dim=1).reshape(-1)
    data = (
        weight.view(torch.uint8)
        .index_select(0, perm)
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    sf = scale.index_select(0, perm).t().contiguous().t()
    return PairedSharedWeight(data, sf, feature_tile)


class Fp4DispatchSharedSwiGluMhc(Op):
    """Single-kernel dispatch/shared GEMM1/SwiGLU/MXFP8/mHC-coefficient path.

    Fixed b24,H5120,I2304. Output FP8/scale feed existing shared GEMM2 directly.
    No global BF16 gate/up output is allocated or written. Gate/up and SwiGLU
    BF16 rounding points are preserved; only the materializations are removed.
    Caller-owned dispatch and mHC scratch must be private to this stream;
    completion starts at zero. See Fp4DispatchSharedMhc for mHC contracts.
    """

    @classmethod
    def resolve_backend(cls, target, *args, override_backend=None, **kwargs):
        if target != MP32_ARCH or override_backend not in (None, "mp32_fused"):
            raise ValueError("requires MP32 fused backend")
        return SwiGluImpl()

    @classmethod
    def run(
        cls,
        buffer: Fp4DispatchBuffer,
        shared_x: torch.Tensor,
        shared_scale: torch.Tensor,
        paired_weight: PairedSharedWeight,
        output_q: torch.Tensor,
        output_scale: torch.Tensor,
        residual: torch.Tensor,
        mhc_weight: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_packed: torch.Tensor,
        partial_mix: torch.Tensor,
        partial_sq: torch.Tensor,
        completion: torch.Tensor,
        mhc_out: torch.Tensor,
        *,
        dispatch_ctas: int | None = None,
        dispatch_schedule: str = "owner_queue",
        gemm_ctas: int | None = None,
        mhc_ctas: int = 20,
        stages: int = 2,
        swiglu_limit: float = 10.0,
        override_backend: str | None = None,
    ):
        if dispatch_schedule not in ("legacy", "owner_queue"):
            raise ValueError("dispatch_schedule must be legacy or owner_queue")
        dispatch_ctas = (
            min(24 if dispatch_schedule == "legacy" else 36, buffer._sym.common_mps)
            if dispatch_ctas is None
            else dispatch_ctas
        )
        gemm_ctas = min(36, buffer._sym.common_mps) if gemm_ctas is None else gemm_ctas
        i = _Input(
            buffer,
            shared_x,
            shared_scale,
            paired_weight,
            output_q,
            output_scale,
            residual,
            mhc_weight,
            mhc_scale,
            mhc_base,
            mhc_packed,
            partial_mix,
            partial_sq,
            completion,
            mhc_out,
            dispatch_ctas,
            gemm_ctas,
            mhc_ctas,
            stages,
            float(swiglu_limit),
            0 if dispatch_schedule == "legacy" else 2,
        )
        return cls.resolve_backend(
            cls.resolve_target(device=buffer.device), override_backend=override_backend
        )(i)

    @classmethod
    def build(cls, inputs=None, *, aot=False):
        return SwiGluImpl().build(inputs, aot=aot)
