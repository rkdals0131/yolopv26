"""torch.compile integration helpers."""

from .runner import _maybe_compile_model as maybe_compile_model
from .runner import _maybe_compile_seg_loss as maybe_compile_seg_loss

__all__ = ["maybe_compile_model", "maybe_compile_seg_loss"]

