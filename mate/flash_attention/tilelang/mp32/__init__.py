"""MP32 TileLang FlashAttention kernels."""

from .flash_attention_varlen_fwd import flashattn_varlen_fwd_interface
from .flash_attention_varlen_bwd import flashattn_varlen_bwd_interface

__all__ = ["flashattn_varlen_fwd_interface", "flashattn_varlen_bwd_interface"]
