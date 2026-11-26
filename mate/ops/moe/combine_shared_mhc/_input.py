from dataclasses import dataclass
import torch
from ..fp4_dispatch.buffer import Fp4DispatchBuffer, Fp4DispatchResult


@dataclass
class _Input:
    state: Fp4DispatchBuffer
    routed: torch.Tensor
    routes: Fp4DispatchResult
    x: torch.Tensor
    xs: torch.Tensor
    w: torch.Tensor
    ws: torch.Tensor
    shared: torch.Tensor
    reduced: torch.Tensor
    merged: torch.Tensor
    status: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    pre: torch.Tensor
    gamma: torch.Tensor
    updated: torch.Tensor
    normalized: torch.Tensor
    q: torch.Tensor
    sf: torch.Tensor
    weighted: bool
    route_scale: float
    mode: int
