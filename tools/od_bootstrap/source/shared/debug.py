from __future__ import annotations

from ..aihub.debug import build_debug_vis_manifest, generate_debug_vis_outputs, select_debug_vis_summaries


def generate_debug_vis(*args, **kwargs):
    return generate_debug_vis_outputs(*args, **kwargs)


__all__ = [
    "build_debug_vis_manifest",
    "generate_debug_vis",
    "generate_debug_vis_outputs",
    "select_debug_vis_summaries",
]
