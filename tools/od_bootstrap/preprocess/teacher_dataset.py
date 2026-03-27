from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from model.preprocess.aihub_standardize import OD_CLASSES

from .sources import CanonicalSourceBundle, BOOTSTRAP_SOURCE_KEYS


TEACHER_DATASET_SPECS = {
    "mobility": {
        "source_dataset_keys": ("bdd100k_det_100k",),
        "class_names": ("vehicle", "bike", "pedestrian"),
    },
    "signal": {
        "source_dataset_keys": ("aihub_traffic_seoul",),
        "class_names": ("traffic_light", "sign"),
    },
    "obstacle": {
        "source_dataset_keys": ("aihub_obstacle_seoul",),
        "class_names": ("traffic_cone", "obstacle"),
    },
}


@dataclass(frozen=True)
class TeacherDatasetSpec:
    name: str
    source_dataset_keys: tuple[str, ...]
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class TeacherDatasetBuildConfig:
    output_root: Path
    copy_images: bool = False


@dataclass(frozen=True)
class TeacherDatasetBuildResult:
    teacher_name: str
    output_root: Path
    dataset_root: Path
    manifest_path: Path
    data_yaml_path: Path
    sample_count: int
    detection_count: int
    class_counts: dict[str, int]
    source_dataset_keys: tuple[str, ...]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _link_or_copy(source_path: Path, target_path: Path, *, copy_images: bool) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return "existing"
    if copy_images:
        import shutil

        shutil.copy2(source_path, target_path)
        return "copy"
    try:
        target_path.hardlink_to(source_path)
        return "hardlink"
    except Exception:
        import shutil

        shutil.copy2(source_path, target_path)
        return "copy"


def _parse_det_row(line: str) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    class_id = int(parts[0])
    return class_id, [float(value) for value in parts[1:]]


def _global_class_name(class_id: int) -> str:
    try:
        return OD_CLASSES[class_id]
    except IndexError as exc:
        raise ValueError(f"unsupported OD class id: {class_id}") from exc


def _build_data_yaml(dataset_root: Path, spec: TeacherDatasetSpec) -> str:
    names_lines = "\n".join(f"  {index}: {name}" for index, name in enumerate(spec.class_names))
    return (
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(spec.class_names)}\n"
        "names:\n"
        f"{names_lines}\n"
    )


def _resolve_spec(spec: TeacherDatasetSpec | str) -> TeacherDatasetSpec:
    if isinstance(spec, TeacherDatasetSpec):
        return spec
    payload = TEACHER_DATASET_SPECS[spec]
    return TeacherDatasetSpec(
        name=spec,
        source_dataset_keys=tuple(payload["source_dataset_keys"]),
        class_names=tuple(payload["class_names"]),
    )


def _iter_scene_paths(source_root: Path) -> list[Path]:
    labels_scene_root = source_root / "labels_scene"
    if not labels_scene_root.is_dir():
        return []
    return sorted(labels_scene_root.rglob("*.json"), key=lambda item: (item.parent.name, item.stem))


def build_teacher_dataset(
    source_bundle: CanonicalSourceBundle,
    spec: TeacherDatasetSpec | str,
    config: TeacherDatasetBuildConfig,
) -> TeacherDatasetBuildResult:
    resolved_spec = _resolve_spec(spec)
    output_root = config.output_root.resolve()
    dataset_root = output_root / resolved_spec.name
    dataset_root.mkdir(parents=True, exist_ok=True)

    class_to_local_id = {class_name: index for index, class_name in enumerate(resolved_spec.class_names)}
    source_roots = [source_bundle.bdd_root, source_bundle.aihub_root]

    sample_count = 0
    detection_count = 0
    class_counts: dict[str, int] = {class_name: 0 for class_name in resolved_spec.class_names}
    seen_samples: set[tuple[str, str]] = set()
    manifest_rows: list[dict[str, Any]] = []

    for source_root in source_roots:
        for scene_path in _iter_scene_paths(source_root):
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            source_dataset_key = str(scene.get("source", {}).get("dataset") or "")
            if source_dataset_key not in resolved_spec.source_dataset_keys:
                continue
            split = str(scene.get("source", {}).get("split") or scene_path.parent.name)
            sample_id = scene_path.stem
            sample_key = (source_dataset_key, sample_id)
            if sample_key in seen_samples:
                continue
            seen_samples.add(sample_key)

            image_file_name = str(scene.get("image", {}).get("file_name") or "")
            image_src = source_root / "images" / split / image_file_name
            label_src = source_root / "labels_det" / split / f"{sample_id}.txt"
            if not image_src.is_file():
                raise FileNotFoundError(f"teacher dataset image missing: {image_src}")

            image_dst = dataset_root / "images" / split / image_file_name
            label_dst = dataset_root / "labels" / split / f"{sample_id}.txt"
            _link_or_copy(image_src, image_dst, copy_images=config.copy_images)

            filtered_rows: list[str] = []
            if label_src.is_file():
                for line in label_src.read_text(encoding="utf-8").splitlines():
                    parsed = _parse_det_row(line)
                    if parsed is None:
                        continue
                    class_id, values = parsed
                    class_name = _global_class_name(class_id)
                    if class_name not in class_to_local_id:
                        continue
                    filtered_rows.append(
                        f"{class_to_local_id[class_name]} " + " ".join(f"{value:.6f}" for value in values)
                    )
                    class_counts[class_name] += 1
                    detection_count += 1

            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.write_text(("\n".join(filtered_rows) + "\n") if filtered_rows else "", encoding="utf-8")
            sample_count += 1
            manifest_rows.append(
                {
                    "teacher_name": resolved_spec.name,
                    "source_dataset_key": source_dataset_key,
                    "split": split,
                    "sample_id": sample_id,
                    "source_scene_path": str(scene_path),
                    "source_image_path": str(image_src),
                    "output_image_path": str(image_dst),
                    "output_label_path": str(label_dst),
                    "detection_count": len(filtered_rows),
                }
            )

    data_yaml_path = _write_text(dataset_root / "data.yaml", _build_data_yaml(dataset_root, resolved_spec))
    manifest_path = _write_json(
        dataset_root / "meta" / "teacher_dataset_manifest.json",
        {
            "version": "od-bootstrap-teacher-dataset-v1",
            "generated_at": _now_iso(),
            "teacher_name": resolved_spec.name,
            "source_dataset_keys": list(resolved_spec.source_dataset_keys),
            "class_names": list(resolved_spec.class_names),
            "output_root": str(output_root),
            "dataset_root": str(dataset_root),
            "copy_images": bool(config.copy_images),
            "sample_count": sample_count,
            "detection_count": detection_count,
            "class_counts": class_counts,
            "bootstrap_source_keys": list(BOOTSTRAP_SOURCE_KEYS),
            "samples": manifest_rows,
        },
    )

    return TeacherDatasetBuildResult(
        teacher_name=resolved_spec.name,
        output_root=output_root,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        data_yaml_path=data_yaml_path,
        sample_count=sample_count,
        detection_count=detection_count,
        class_counts=class_counts,
        source_dataset_keys=resolved_spec.source_dataset_keys,
    )


def build_teacher_datasets(
    source_bundle: CanonicalSourceBundle,
    output_root: Path,
    *,
    copy_images: bool = False,
    teacher_specs: dict[str, TeacherDatasetSpec] | None = None,
) -> dict[str, TeacherDatasetBuildResult]:
    spec_map = teacher_specs or {
        name: TeacherDatasetSpec(
            name=name,
            source_dataset_keys=tuple(payload["source_dataset_keys"]),
            class_names=tuple(payload["class_names"]),
        )
        for name, payload in TEACHER_DATASET_SPECS.items()
    }
    config = TeacherDatasetBuildConfig(output_root=output_root, copy_images=copy_images)
    return {name: build_teacher_dataset(source_bundle, spec, config) for name, spec in spec_map.items()}
