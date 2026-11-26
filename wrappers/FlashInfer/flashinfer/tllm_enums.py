"""FlashInfer scale-layout names accepted by the MUSA compatibility layer."""

from enum import IntEnum


class SfLayout(IntEnum):
    """Compatibility values; MUSA quantizers always return native MN-major SF."""

    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2
