from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from model.preprocess.aihub_standardize import OD_CLASSES


def _resolve_latest_root(root: Path) -> Path:
    if root.name != "latest":
        return root.resolve()
    parent = root.parent.resolve()
    candidates = [path for path in parent.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no exhaustive OD runs found under {parent}")
    return sorted(candidates, key=lambda item: item.name)[-1]


def _link_or_copy(source_path: Path, target_path: Path, *, copy_images: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if copy_images:
        shutil.copy2(source_path, target_path)
        return
    try:
        target_path.hardlink_to(source_path)
    except Exception:
        shutil.copy2(source_path, target_path)


def _copy_json(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _copy_optional(source_path: Path, target_path: Path) -> None:
    if source_path.is_file():
        _copy_json(source_path, target_path)


def build_pv26_exhaustive_od_lane_dataset(
    *,
    exhaustive_od_root: Path,
    aihub_canonical_root: Path,
    output_root: Path,
    copy_images: bool = False,
) -> dict[str, Any]:
    resolved_exhaustive_root = _resolve_latest_root(exhaustive_od_root)
    resolved_aihub_root = aihub_canonical_root.resolve()
    resolved_output_root = output_root.resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    copied_samples = 0
    dataset_counts = Counter()

    for scene_path in sorted((resolved_exhaustive_root / "labels_scene").rglob("*.json")):
        split = scene_path.parent.name
        sample_id = scene_path.stem
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        dataset_key = str(scene.get("source", {}).get("dataset") or "")
        image_name = str(scene.get("image", {}).get("file_name") or "")
        _copy_json(scene_path, resolved_output_root / "labels_scene" / split / scene_path.name)
        _copy_json(
            resolved_exhaustive_root / "labels_det" / split / f"{sample_id}.txt",
            resolved_output_root / "labels_det" / split / f"{sample_id}.txt",
        )
        _link_or_copy(
            resolved_exhaustive_root / "images" / split / image_name,
            resolved_output_root / "images" / split / image_name,
            copy_images=copy_images,
        )
        copied_samples += 1
        dataset_counts[dataset_key] += 1

    for scene_path in sorted((resolved_aihub_root / "labels_scene").rglob("*.json")):
        split = scene_path.parent.name
        sample_id = scene_path.stem
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        dataset_key = str(scene.get("source", {}).get("dataset") or "")
        if dataset_key != "aihub_lane_seoul":
            continue
        image_name = str(scene.get("image", {}).get("file_name") or "")
        _copy_json(scene_path, resolved_output_root / "labels_scene" / split / scene_path.name)
        _copy_optional(
            resolved_aihub_root / "labels_det" / split / f"{sample_id}.txt",
            resolved_output_root / "labels_det" / split / f"{sample_id}.txt",
        )
        _link_or_copy(
            resolved_aihub_root / "images" / split / image_name,
            resolved_output_root / "images" / split / image_name,
            copy_images=copy_images,
        )
        copied_samples += 1
        dataset_counts[dataset_key] += 1

    meta_root = resolved_output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "class_map_det.yaml").write_text(
        yaml.safe_dump({str(index): class_name for index, class_name in enumerate(OD_CLASSES)}, sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "version": "pv26-exhaustive-od-lane-v1",
        "exhaustive_od_root": str(resolved_exhaustive_root),
        "aihub_canonical_root": str(resolved_aihub_root),
        "output_root": str(resolved_output_root),
        "sample_count": copied_samples,
        "dataset_counts": dict(sorted(dataset_counts.items())),
    }
    summary_path = meta_root / "final_dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "output_root": str(resolved_output_root),
        "manifest_path": str(summary_path),
        "sample_count": copied_samples,
        "dataset_counts": dict(sorted(dataset_counts.items())),
    }
