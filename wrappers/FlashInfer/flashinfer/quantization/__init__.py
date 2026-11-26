"""Quantization on MUSA through the public :mod:`mate.quantization` APIs.

Only FlashInfer API names are exported. SF layout arguments are accepted for
call compatibility, but all functions produce or consume MATE's native
MN-major scales. No NVIDIA scale-layout conversion is performed. Additional
recipes and fused activations are available directly from ``mate.quantization``.
"""

from mate.quantization import packbits, segment_packbits

from ..tllm_enums import SfLayout
from .fp4_quantization import fp4_quantize, mxfp4_dequantize, mxfp4_quantize
from .fp8_quantization import mxfp8_quantize

__all__ = [
    "SfLayout",
    "fp4_quantize",
    "mxfp4_dequantize",
    "mxfp4_quantize",
    "mxfp8_quantize",
    "packbits",
    "segment_packbits",
]
