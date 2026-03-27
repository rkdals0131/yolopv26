from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

@dataclass(frozen=True)
class TeacherDatasetLayout:
    source_root: Path
    staging_root: Path
    image_dir: str = "images"
    label_dir: str = "labels"
    train_split: str = "train"
    val_split: str = "val"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    _ensure_parent(destination)
    try:
        os.symlink(source, destination, target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)


def resolve_teacher_dataset_root(
    *,
    source_root: Path,
    image_dir: str = "images",
    label_dir: str = "labels",
    train_split: str = "train",
    val_split: str = "val",
) -> Path:
    resolved_root = source_root.resolve()
    image_root = resolved_root / image_dir
    label_root = resolved_root / label_dir
    if not image_root.is_dir():
        raise FileNotFoundError(f"teacher image root does not exist: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"teacher label root does not exist: {label_root}")
    for split in (train_split, val_split):
        image_split_root = image_root / split
        label_split_root = label_root / split
        if not image_split_root.is_dir():
            raise FileNotFoundError(f"teacher image split does not exist: {image_split_root}")
        if not label_split_root.is_dir():
            raise FileNotFoundError(f"teacher label split does not exist: {label_split_root}")
    return resolved_root


def stage_teacher_dataset_layout(layout: TeacherDatasetLayout) -> Path:
    source_root = resolve_teacher_dataset_root(
        source_root=layout.source_root,
        image_dir=layout.image_dir,
        label_dir=layout.label_dir,
        train_split=layout.train_split,
        val_split=layout.val_split,
    )
    staging_root = layout.staging_root.resolve()
    image_root = source_root / layout.image_dir
    label_root = source_root / layout.label_dir

    for split in (layout.train_split, layout.val_split):
        _link_or_copy_tree(image_root / split, staging_root / "images" / split)
        _link_or_copy_tree(label_root / split, staging_root / "labels" / split)
    return staging_root


def build_teacher_data_yaml(
    *,
    dataset_root: Path,
    class_names: tuple[str, ...],
    output_path: Path,
    train_split: str = "train",
    val_split: str = "val",
) -> Path:
    payload: dict[str, Any] = {
        "path": str(dataset_root),
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "nc": len(class_names),
        "names": list(class_names),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path
