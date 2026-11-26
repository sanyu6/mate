from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .buffer import Fp4DispatchBuffer


@dataclass
class _Fp4DispatchInput:
    buffer: "Fp4DispatchBuffer"
    tokens: int
    num_ctas: int | None = None
