from dataclasses import dataclass
import torch
from ..dispatch_shared._input import _DispatchSharedInput


@dataclass
class _Input:
    shared: _DispatchSharedInput
    residual: torch.Tensor
    weight: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    packed_weight: torch.Tensor
    partial_mix: torch.Tensor
    partial_sq: torch.Tensor
    completion: torch.Tensor
    output: torch.Tensor
    mhc_ctas: int
    mode: int
    tactic: int = 3
