import torch
from mate.ops import Op
from mate.arch import MP32_ARCH
from ..fp4_dispatch.buffer import Fp4DispatchBuffer, Fp4DispatchResult
from ._input import _Input
from .impl import CombineImpl


class MoeCombineSharedMhc(Op):
    """Experimental b24 end boundary: combine + shared down + mHC post/norm/quant.

    routes describes valid M32 expert-pool rows of routed BF16 output. Source
    rank/token/topk identity and FP32 routing-weight bits are consumed directly.
    Input/output scale tensors use MN-major layout. All output and scratch tensors
    belong to the caller. Use dedicated symmetric state,one ordered stream,and no
    alias between inputs/outputs. The cooperative fused grid is occupancy-checked.
    apply_route_weights=False is required if routed GEMMs already applied weights.
    Norm epsilon is currently fixed1e-6. No model default is changed.
    Default two_kernel uses40 producer CTAs/512 communication threads/4 stages,
    then the installed SGLang post boundary (1024 threads,parts2). It requires
    matching SGLang MUSA kernels. fused is retained as a slower single-kernel
    control and leaves merged_out untouched; two_kernel materializes merged_out.
    Fixed8f7e2b1 remains the V1 baseline; do not alternate V1/V2 protocols on
    the same symmetric state without complete group synchronization.
    """

    @classmethod
    def resolve_backend(cls, target, *args, override_backend=None, **kwargs):
        if target != MP32_ARCH or override_backend not in (
            None,
            "fused",
            "two_kernel",
            "serial",
        ):
            raise ValueError("requires MP32 fused/two_kernel/serial")
        return CombineImpl()

    @classmethod
    def run(
        cls,
        state: Fp4DispatchBuffer,
        routed: torch.Tensor,
        routes: Fp4DispatchResult,
        shared_x: torch.Tensor,
        shared_scale: torch.Tensor,
        shared_weight: torch.Tensor,
        shared_weight_scale: torch.Tensor,
        shared_out: torch.Tensor,
        reduced_out: torch.Tensor,
        merged_out: torch.Tensor,
        status: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        gamma: torch.Tensor,
        updated: torch.Tensor,
        normalized: torch.Tensor,
        output_q: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        apply_route_weights: bool = True,
        routed_scaling_factor: float = 1.0,
        override_backend: str | None = None,
    ):
        mode = {"fused": 0, "two_kernel": 1, "serial": 2}.get(
            override_backend or "two_kernel", -1
        )
        i = _Input(
            state,
            routed,
            routes,
            shared_x,
            shared_scale,
            shared_weight,
            shared_weight_scale,
            shared_out,
            reduced_out,
            merged_out,
            status,
            residual,
            post,
            comb,
            pre,
            gamma,
            updated,
            normalized,
            output_q,
            output_scale,
            apply_route_weights,
            float(routed_scaling_factor),
            mode,
        )
        return cls.resolve_backend(
            cls.resolve_target(device=state.device), override_backend=override_backend
        )(i)

    @classmethod
    def build(cls, inputs=None, *, aot=False):
        return CombineImpl().build(inputs, aot=aot)
