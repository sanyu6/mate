"""Architecture values shared by operators, runtime and artifact selection.

This module describes targets without querying devices. Each operator and
implementation determines which targets it supports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    major: int
    minor: int

    def __repr__(self) -> str:
        return f"mp{self.major}{self.minor}"

    def __int__(self) -> int:
        return self.major * 10 + self.minor

    @classmethod
    def from_cc(cls, cc: int) -> Arch:
        """Parse a positive ``major * 10 + minor`` CC, e.g. 31 for MP31.

        This validates the encoding, not whether a kernel supports the target.
        """
        if isinstance(cc, bool) or not isinstance(cc, int):
            raise TypeError(f"compute capability must be an integer, got {cc!r}")
        if cc <= 0:
            raise ValueError(f"compute capability must be positive, got {cc!r}")
        return cls(*divmod(cc, 10))


MP31_ARCH = Arch(major=3, minor=1)
MP32_ARCH = Arch(major=3, minor=2)

__all__ = ["Arch", "MP31_ARCH", "MP32_ARCH"]
