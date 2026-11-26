from dataclasses import dataclass
import torch
from ..fp4_dispatch.buffer import Fp4DispatchBuffer


@dataclass(frozen=True)
class PairedSharedWeight:
    """Immutable gate/up row packing tied to a GEMM feature tile."""

    data: torch.Tensor
    scale: torch.Tensor
    feature_tile: int


@dataclass
class _Input:
    buffer: Fp4DispatchBuffer
    x: torch.Tensor
    xs: torch.Tensor
    weight: PairedSharedWeight
    q: torch.Tensor
    sf: torch.Tensor
    residual: torch.Tensor
    mhc_weight: torch.Tensor
    mhc_scale: torch.Tensor
    mhc_base: torch.Tensor
    mhc_packed: torch.Tensor
    partial_mix: torch.Tensor
    partial_sq: torch.Tensor
    completion: torch.Tensor
    mhc_out: torch.Tensor
    dispatch_ctas: int
    gemm_ctas: int
    mhc_ctas: int
    stages: int
    limit: float
    dispatch_tactic: int = 2
