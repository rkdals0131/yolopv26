from __future__ import annotations

from .aihub_debug import generate_debug_vis_outputs, select_debug_vis_summaries


def generate_debug_vis(*args, **kwargs):
    return generate_debug_vis_outputs(*args, **kwargs)


__all__ = [
    "generate_debug_vis",
    "generate_debug_vis_outputs",
    "select_debug_vis_summaries",
]
