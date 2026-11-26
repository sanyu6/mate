from dataclasses import dataclass
import torch
from ..fp4_dispatch.buffer import Fp4DispatchBuffer


@dataclass
class _DispatchSharedInput:
    buffer: Fp4DispatchBuffer
    x: torch.Tensor
    x_scale: torch.Tensor
    weight: torch.Tensor
    weight_scale: torch.Tensor
    output: torch.Tensor
    num_ctas: int | None = None
    feature_tile: int = 128
    stages: int = 2
    gemm_ctas: int | None = None
