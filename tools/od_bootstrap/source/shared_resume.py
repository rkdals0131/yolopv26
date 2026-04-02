from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .shared_summary import counter_to_dict


def count_held_annotation_reasons(
    held_annotations: Any,
    *,
    normalize_reason: Callable[[Any], str] | None = None,
) -> dict[str, int]:
    counter = Counter()
    if not isinstance(held_annotations, list):
        return {}
    for item in held_annotations:
        if not isinstance(item, dict):
            continue
        if normalize_reason is None:
            reason = str(item.get("reason") or "unknown").strip().lower()
        else:
            reason = normalize_reason(item.get("reason")) or "unknown"
        counter[reason] += 1
    return counter_to_dict(counter)


def load_existing_scene_output(
    *,
    output_root: Path,
    split: str,
    sample_id: str,
    image_suffix: str,
    load_json_fn: Callable[[Path], dict[str, Any]],
    scene_version: str | None = None,
) -> dict[str, Any] | None:
    image_path = output_root / "images" / split / f"{sample_id}{image_suffix}"
    scene_path = output_root / "labels_scene" / split / f"{sample_id}.json"
    det_path = output_root / "labels_det" / split / f"{sample_id}.txt"
    if not image_path.is_file() or not scene_path.is_file():
        return None

    try:
        scene = load_json_fn(scene_path)
    except Exception:
        return None

    if scene_version is not None and str(scene.get("version") or "").strip() != scene_version:
        return None

    return {
        "image_path": image_path,
        "scene_path": scene_path,
        "det_path": det_path,
        "scene": scene,
    }


__all__ = [
    "count_held_annotation_reasons",
    "load_existing_scene_output",
]
