from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from tools.od_bootstrap.common import resolve_path


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


def _resolve_optional_path(value: Any, *, base_dir: Path, field_name: str) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(_coerce_str(value, field_name=field_name), base_dir=base_dir)


def load_image_list(path: str | Path) -> tuple[ImageListEntry, ...]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"image list manifest not found: {manifest_path}")

    entries: list[ImageListEntry] = []
    seen_paths: set[str] = set()
    seen_sample_uids: set[str] = set()
    for line_index, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"image_list[{line_index}] must be a JSON object")

        image_path = resolve_path(
            _coerce_str(payload.get("image_path"), field_name=f"image_list[{line_index}].image_path"),
            base_dir=manifest_path.parent,
        )
        dedupe_key = str(image_path)
        if dedupe_key in seen_paths:
            raise ValueError(f"duplicate image_path in image list manifest: {image_path}")
        seen_paths.add(dedupe_key)

        dataset_key = str(payload.get("dataset_key", "")).strip()
        split = str(payload.get("split", "")).strip()
        sample_id = str(payload.get("sample_id") or image_path.stem).strip()
        if not sample_id:
            raise ValueError(f"image_list[{line_index}].sample_id must not be empty")
        sample_uid = str(payload.get("sample_uid") or "").strip()
        if not sample_uid:
            raise ValueError(f"image_list[{line_index}].sample_uid must not be empty")
        if sample_uid in seen_sample_uids:
            raise ValueError(f"duplicate sample_uid in image list manifest: {sample_uid}")
        seen_sample_uids.add(sample_uid)

        entries.append(
            ImageListEntry(
                sample_id=sample_id,
                sample_uid=sample_uid,
                image_path=image_path,
                scene_path=resolve_path(
                    _coerce_str(payload.get("scene_path"), field_name=f"image_list[{line_index}].scene_path"),
                    base_dir=manifest_path.parent,
                ),
                dataset_root=resolve_path(
                    _coerce_str(payload.get("dataset_root"), field_name=f"image_list[{line_index}].dataset_root"),
                    base_dir=manifest_path.parent,
                ),
                dataset_key=dataset_key,
                split=split,
                det_path=_resolve_optional_path(
                    payload.get("det_path"),
                    base_dir=manifest_path.parent,
                    field_name=f"image_list[{line_index}].det_path",
                ),
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
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            dataset_key = str(scene.get("source", {}).get("dataset") or "").strip()
            if dataset_key not in allowed:
                continue
            split = str(scene.get("source", {}).get("split") or scene_path.parent.name).strip()
            image_file_name = _coerce_str(
                scene.get("image", {}).get("file_name"),
                field_name=f"{scene_path}.image.file_name",
            )
            sample_id = scene_path.stem
            det_path = resolved_root / "labels_det" / split / f"{sample_id}.txt"
            sample_uid = build_sample_uid(dataset_key=dataset_key, split=split, sample_id=sample_id)
            entries.append(
                ImageListEntry(
                    sample_id=sample_id,
                    sample_uid=sample_uid,
                    image_path=resolved_root / "images" / split / image_file_name,
                    scene_path=scene_path,
                    dataset_root=resolved_root,
                    dataset_key=dataset_key,
                    split=split,
                    det_path=det_path if det_path.is_file() else None,
                    source_name=resolved_root.name,
                )
            )
    entries.sort(key=lambda item: (item.sample_uid, str(item.image_path)))
    return tuple(entries)


def write_image_list(path: Path, entries: Iterable[ImageListEntry]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(entry.to_dict(), ensure_ascii=True) for entry in entries]
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
    return path
