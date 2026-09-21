from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common.io import iter_jsonl
from common.io import read_json
from common.io import write_jsonl
from common.paths import resolve_optional_path, resolve_path


@dataclass(frozen=True)
class ImageListEntry:
    sample_id: str
    sample_uid: str
    image_path: Path
    scene_path: Path
    dataset_root: Path
    dataset_key: str = ""
    split: str = ""
    det_path: Path | None = None
    source_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_uid": self.sample_uid,
            "image_path": str(self.image_path),
            "scene_path": str(self.scene_path),
            "dataset_root": str(self.dataset_root),
            "dataset_key": self.dataset_key,
            "split": self.split,
            "det_path": str(self.det_path) if self.det_path is not None else None,
            "source_name": self.source_name,
        }


def build_sample_uid(*, dataset_key: str, split: str, sample_id: str) -> str:
    dataset_token = _coerce_str(dataset_key, field_name="dataset_key")
    split_token = _coerce_str(split, field_name="split")
    sample_token = _coerce_str(sample_id, field_name="sample_id")
    return f"{dataset_token}__{split_token}__{sample_token}"


def _coerce_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _load_scene(scene_path: Path) -> dict[str, Any]:
    payload = read_json(scene_path)
    if not isinstance(payload, dict):
        raise TypeError(f"scene root must be an object: {scene_path}")
    return payload


def _det_label_state_from_tasks(scene: dict[str, Any]) -> bool | None:
    tasks = scene.get("tasks")
    if not isinstance(tasks, dict) or "has_det" not in tasks:
        return None
    return bool(tasks.get("has_det"))


def _validate_det_path_for_scene(
    *,
    scene: dict[str, Any],
    scene_path: Path,
    det_path: Path | None,
    default_det_path: Path,
) -> Path | None:
    det_required = _det_label_state_from_tasks(scene)
    candidate_det_path = det_path if det_path is not None else default_det_path
    if det_required is True and not candidate_det_path.is_file():
        raise FileNotFoundError(f"image_list det label not found: {candidate_det_path} ({scene_path})")
    if det_required is False and candidate_det_path.is_file():
        raise ValueError(f"image_list stale det label for non-det scene: {candidate_det_path} ({scene_path})")
    if det_path is not None and not det_path.is_file():
        raise FileNotFoundError(f"image_list det label not found: {det_path} ({scene_path})")
    return det_path if det_path is not None else (default_det_path if default_det_path.is_file() else None)


def _validate_manifest_scene_metadata(
    *,
    scene: dict[str, Any],
    scene_path: Path,
    dataset_root: Path,
    dataset_key: str,
    split: str,
    sample_id: str,
    sample_uid: str,
    image_path: Path,
) -> None:
    source = scene.get("source") if isinstance(scene.get("source"), dict) else {}
    scene_dataset_key = str(source.get("dataset") or "").strip()
    if not scene_dataset_key:
        raise ValueError(f"scene source.dataset missing: {scene_path}")
    if scene_dataset_key != dataset_key:
        raise ValueError(
            f"scene source.dataset must match image list dataset_key: "
            f"{scene_dataset_key} != {dataset_key} ({scene_path})"
        )

    scene_split = str(source.get("split") or "").strip()
    if not scene_split:
        raise ValueError(f"scene source.split missing: {scene_path}")
    if scene_split != split:
        raise ValueError(f"scene source.split must match image list split: {scene_split} != {split} ({scene_path})")

    expected_scene_path = (dataset_root / "labels_scene" / split / f"{sample_id}.json").resolve()
    if scene_path != expected_scene_path:
        raise ValueError(
            f"image_list scene_path must match dataset_root/split/sample_id: "
            f"{scene_path} != {expected_scene_path}"
        )

    image_file_name = _coerce_str(
        scene.get("image", {}).get("file_name"),
        field_name=f"{scene_path}.image.file_name",
    )
    if Path(image_file_name).is_absolute() or Path(image_file_name).name != image_file_name:
        raise ValueError(f"{scene_path}.image.file_name must be a file name, not a path")
    expected_image_path = (dataset_root / "images" / split / image_file_name).resolve()
    if image_path != expected_image_path:
        raise ValueError(
            f"image_list image_path must match scene image.file_name: "
            f"{image_path} != {expected_image_path} ({scene_path})"
        )

    expected_sample_uid = build_sample_uid(dataset_key=dataset_key, split=split, sample_id=sample_id)
    if sample_uid != expected_sample_uid:
        raise ValueError(
            f"image_list sample_uid must match dataset_key/split/sample_id: "
            f"{sample_uid} != {expected_sample_uid}"
        )


def load_image_list(path: str | Path) -> tuple[ImageListEntry, ...]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"image list manifest not found: {manifest_path}")

    entries: list[ImageListEntry] = []
    seen_paths: set[str] = set()
    seen_sample_uids: set[str] = set()
    for line_index, payload in iter_jsonl(manifest_path):
        if not isinstance(payload, dict):
            raise TypeError(f"image_list[{line_index}] must be a JSON object")

        image_path = resolve_path(
            _coerce_str(payload.get("image_path"), field_name=f"image_list[{line_index}].image_path"),
            base_dir=manifest_path.parent,
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"image_list image not found: {image_path}")
        dedupe_key = str(image_path)
        if dedupe_key in seen_paths:
            raise ValueError(f"duplicate image_path in image list manifest: {image_path}")
        seen_paths.add(dedupe_key)

        dataset_key = _coerce_str(payload.get("dataset_key"), field_name=f"image_list[{line_index}].dataset_key")
        split = _coerce_str(payload.get("split"), field_name=f"image_list[{line_index}].split")
        sample_id = str(payload.get("sample_id") or image_path.stem).strip()
        if not sample_id:
            raise ValueError(f"image_list[{line_index}].sample_id must not be empty")
        sample_uid = str(payload.get("sample_uid") or "").strip()
        if not sample_uid:
            raise ValueError(f"image_list[{line_index}].sample_uid must not be empty")
        if sample_uid in seen_sample_uids:
            raise ValueError(f"duplicate sample_uid in image list manifest: {sample_uid}")
        seen_sample_uids.add(sample_uid)
        raw_det_path = payload.get("det_path")
        det_path = resolve_optional_path(
            raw_det_path
            if raw_det_path in (None, "")
            else _coerce_str(raw_det_path, field_name=f"image_list[{line_index}].det_path"),
            base_dir=manifest_path.parent,
        )
        scene_path = resolve_path(
            _coerce_str(payload.get("scene_path"), field_name=f"image_list[{line_index}].scene_path"),
            base_dir=manifest_path.parent,
        )
        dataset_root = resolve_path(
            _coerce_str(payload.get("dataset_root"), field_name=f"image_list[{line_index}].dataset_root"),
            base_dir=manifest_path.parent,
        )
        scene = _load_scene(scene_path)
        _validate_manifest_scene_metadata(
            scene=scene,
            scene_path=scene_path,
            dataset_root=dataset_root,
            dataset_key=dataset_key,
            split=split,
            sample_id=sample_id,
            sample_uid=sample_uid,
            image_path=image_path,
        )
        default_det_path = dataset_root / "labels_det" / split / f"{sample_id}.txt"
        det_path = _validate_det_path_for_scene(
            scene=scene,
            scene_path=scene_path,
            det_path=det_path,
            default_det_path=default_det_path,
        )

        entries.append(
            ImageListEntry(
                sample_id=sample_id,
                sample_uid=sample_uid,
                image_path=image_path,
                scene_path=scene_path,
                dataset_root=dataset_root,
                dataset_key=dataset_key,
                split=split,
                det_path=det_path,
                source_name=str(payload.get("source_name", "")).strip(),
            )
        )

    entries.sort(key=lambda item: (item.sample_uid, str(item.image_path)))
    return tuple(entries)


def discover_image_list_entries(
    dataset_roots: Iterable[Path],
    *,
    allowed_dataset_keys: Iterable[str],
) -> tuple[ImageListEntry, ...]:
    allowed = set(allowed_dataset_keys)
    entries: list[ImageListEntry] = []
    for dataset_root in dataset_roots:
        resolved_root = Path(dataset_root).resolve()
        labels_scene_root = resolved_root / "labels_scene"
        if not labels_scene_root.is_dir():
            continue
        for scene_path in sorted(labels_scene_root.rglob("*.json"), key=lambda item: (item.parent.name, item.stem)):
            scene = _load_scene(scene_path)
            source = scene.get("source") if isinstance(scene.get("source"), dict) else {}
            dataset_key = str(source.get("dataset") or "").strip()
            if dataset_key not in allowed:
                continue
            labels_scene_split = scene_path.parent.name
            split = str(source.get("split") or labels_scene_split).strip()
            if split != labels_scene_split:
                raise ValueError(f"scene source.split must match labels_scene split: {scene_path}")
            image_file_name = _coerce_str(
                scene.get("image", {}).get("file_name"),
                field_name=f"{scene_path}.image.file_name",
            )
            if Path(image_file_name).is_absolute() or Path(image_file_name).name != image_file_name:
                raise ValueError(f"{scene_path}.image.file_name must be a file name, not a path")
            sample_id = scene_path.stem
            image_path = resolved_root / "images" / split / image_file_name
            if not image_path.is_file():
                raise FileNotFoundError(f"image_list image not found: {image_path}")
            det_path = resolved_root / "labels_det" / split / f"{sample_id}.txt"
            det_path = _validate_det_path_for_scene(
                scene=scene,
                scene_path=scene_path,
                det_path=None,
                default_det_path=det_path,
            )
            sample_uid = build_sample_uid(dataset_key=dataset_key, split=split, sample_id=sample_id)
            entries.append(
                ImageListEntry(
                    sample_id=sample_id,
                    sample_uid=sample_uid,
                    image_path=image_path,
                    scene_path=scene_path,
                    dataset_root=resolved_root,
                    dataset_key=dataset_key,
                    split=split,
                    det_path=det_path,
                    source_name=resolved_root.name,
                )
            )
    entries.sort(key=lambda item: (item.sample_uid, str(item.image_path)))
    return tuple(entries)


def write_image_list(path: Path, entries: Iterable[ImageListEntry]) -> Path:
    return write_jsonl(path, (entry.to_dict() for entry in entries))


__all__ = [
    "ImageListEntry",
    "build_sample_uid",
    "discover_image_list_entries",
    "load_image_list",
    "write_image_list",
]
