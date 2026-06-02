from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import Dataset

from common.io import read_json as _read_common_json
from common.io import read_text as _read_common_text
from common.pv26_schema import (
    DET_SUPERVISION_BY_DATASET,
    LANE_CLASSES,
    LANE_TYPES,
    OD_CLASSES,
    SOURCE_MASK_BY_DATASET,
    TL_BITS,
)
from .transform import (
    NETWORK_HW,
    LetterboxTransform,
    TrainAugmentationConfig,
    apply_train_augmentations,
    clip_box_xyxy,
    clip_points,
    compute_letterbox_transform,
    load_letterboxed_image,
    transform_box_xyxy,
    transform_points,
    unique_point_count,
)


@dataclass(frozen=True)
class SampleRecord:
    dataset_root: Path
    dataset_key: str
    split: str
    sample_id: str
    scene_path: Path
    image_path: Path
    det_path: Path | None
    stop_line_count: int = 0
    manifest_path: Path | None = None
    source_kind: str | None = None
    source_scene_path: Path | None = None
    source_image_path: Path | None = None
    source_det_path: Path | None = None


OD_CLASS_TO_ID = {class_name: index for index, class_name in enumerate(OD_CLASSES)}
FINAL_DATASET_MANIFEST_NAME = "final_dataset_manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = _read_common_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"scene root must be an object: {path}")
    return payload


def _coerce_scene_image_file_name(scene: dict[str, Any], *, scene_path: Path) -> str:
    image = scene.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"scene image.file_name must not be empty: {scene_path}")
    image_file_name = str(image.get("file_name") or "").strip()
    if not image_file_name:
        raise ValueError(f"scene image.file_name must not be empty: {scene_path}")
    image_path = Path(image_file_name)
    if (
        image_path.is_absolute()
        or image_path.name != image_file_name
        or "/" in image_file_name
        or "\\" in image_file_name
    ):
        raise ValueError(f"scene image.file_name must be a basename: {scene_path}")
    return image_file_name


def _coerce_scene_image_hw(scene: dict[str, Any], *, scene_path: Path) -> tuple[int, int]:
    image = scene.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"scene image dimensions must be positive integers: {scene_path}")

    def coerce_dimension(key: str) -> int:
        value = image.get(key)
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, int):
            dimension = value
        elif isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError
            dimension = int(value)
        elif isinstance(value, str):
            dimension = int(value.strip())
        else:
            raise ValueError
        if dimension <= 0:
            raise ValueError
        return dimension

    try:
        raw_h = coerce_dimension("height")
        raw_w = coerce_dimension("width")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scene image dimensions must be positive integers: {scene_path}") from exc
    return raw_h, raw_w


def _coerce_scene_dataset_key(scene: dict[str, Any], *, scene_path: Path) -> str:
    source = scene.get("source")
    dataset_value = source.get("dataset") if isinstance(source, dict) else None
    dataset_key = str(dataset_value or "").strip()
    if not dataset_key:
        raise ValueError(f"scene source.dataset must not be empty: {scene_path}")
    if dataset_key not in SOURCE_MASK_BY_DATASET:
        raise KeyError(f"unsupported dataset key for loader: {dataset_key}")
    return dataset_key


def _coerce_scene_split(scene: dict[str, Any], *, scene_path: Path) -> str:
    split = scene_path.parent.name
    source = scene.get("source")
    source_split = str(source.get("split") if isinstance(source, dict) else "").strip()
    if source_split and source_split != split:
        raise ValueError(
            f"scene source.split must match labels_scene split: {source_split} != {split} ({scene_path})"
        )
    return split


def _coerce_scene_geometry_items(scene: dict[str, Any], key: str, *, scene_path: Path) -> list[dict[str, Any]]:
    if key not in scene:
        return []
    items = scene[key]
    if not isinstance(items, list):
        raise ValueError(f"scene {key} must be a list: {scene_path}")
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"scene {key}[{item_index}] must be an object: {scene_path}")
    return items


def _discover_records(
    dataset_root: Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
    progress_every: int = 2000,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    labels_scene_root = dataset_root / "labels_scene"
    if not labels_scene_root.is_dir():
        return records

    for scene_index, scene_path in enumerate(labels_scene_root.rglob("*.json"), start=1):
        scene = _load_json(scene_path)
        split = _coerce_scene_split(scene, scene_path=scene_path)
        dataset_key = _coerce_scene_dataset_key(scene, scene_path=scene_path)
        sample_id = scene_path.stem
        image_file_name = _coerce_scene_image_file_name(scene, scene_path=scene_path)
        image_path = dataset_root / "images" / split / image_file_name
        det_path = dataset_root / "labels_det" / split / f"{sample_id}.txt"
        stop_line_items = _coerce_scene_geometry_items(scene, "stop_lines", scene_path=scene_path)
        stop_line_count = len(stop_line_items)
        records.append(
            SampleRecord(
                dataset_root=dataset_root,
                dataset_key=dataset_key,
                split=split,
                sample_id=sample_id,
                scene_path=scene_path,
                image_path=image_path,
                det_path=det_path if det_path.is_file() else None,
                stop_line_count=int(stop_line_count),
            )
        )
        if progress_callback is not None and scene_index % max(1, int(progress_every)) == 0:
            progress_callback(
                f"indexed {scene_index} scene labels under {dataset_root}"
            )
    return sorted(records, key=lambda item: (item.dataset_key, item.split, item.sample_id))


def _coerce_manifest_string(row: dict[str, Any], key: str, *, manifest_path: Path) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"final dataset manifest sample {key} must not be empty: {manifest_path}")
    return value


def _coerce_manifest_path(
    row: dict[str, Any],
    key: str,
    *,
    manifest_path: Path,
    optional: bool = False,
) -> Path | None:
    value = row.get(key)
    if value is None and optional:
        return None
    path_text = str(value or "").strip()
    if not path_text:
        if optional:
            return None
        raise ValueError(f"final dataset manifest sample {key} must not be empty: {manifest_path}")
    path = Path(path_text).expanduser()
    return path.resolve()


def _apply_final_dataset_manifest(dataset_root: Path, records: list[SampleRecord]) -> list[SampleRecord]:
    manifest_path = dataset_root / "meta" / FINAL_DATASET_MANIFEST_NAME
    if not manifest_path.is_file():
        return records
    manifest = _read_common_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"final dataset manifest root must be an object: {manifest_path}")
    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise ValueError(f"final dataset manifest samples must be a list: {manifest_path}")
    sample_count = manifest.get("sample_count")
    if sample_count is not None and int(sample_count) != len(rows):
        raise ValueError(
            f"final dataset manifest sample_count must match samples length: {manifest_path}"
        )

    record_by_key = {(record.dataset_key, record.split, record.sample_id): record for record in records}
    manifest_keys: set[tuple[str, str, str]] = set()
    updated_records: list[SampleRecord] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"final dataset manifest samples[{row_index}] must be an object: {manifest_path}")
        sample_id = _coerce_manifest_string(row, "final_sample_id", manifest_path=manifest_path)
        dataset_key = _coerce_manifest_string(row, "source_dataset_key", manifest_path=manifest_path)
        split = _coerce_manifest_string(row, "split", manifest_path=manifest_path)
        manifest_key = (dataset_key, split, sample_id)
        if manifest_key in manifest_keys:
            raise ValueError(f"final dataset manifest samples must be unique: {manifest_path}")
        manifest_keys.add(manifest_key)
        record = record_by_key.get(manifest_key)
        if record is None:
            raise ValueError(
                "final dataset manifest samples must match discovered records: "
                f"missing={dataset_key}/{split}/{sample_id} ({manifest_path})"
            )

        scene_path = _coerce_manifest_path(row, "scene_path", manifest_path=manifest_path)
        image_path = _coerce_manifest_path(row, "image_path", manifest_path=manifest_path)
        det_path = _coerce_manifest_path(row, "det_path", manifest_path=manifest_path, optional=True)
        if scene_path != record.scene_path:
            raise ValueError(
                "final dataset manifest scene_path must match discovered record: "
                f"{scene_path} != {record.scene_path} ({manifest_path})"
            )
        if image_path != record.image_path:
            raise ValueError(
                "final dataset manifest image_path must match discovered record: "
                f"{image_path} != {record.image_path} ({manifest_path})"
            )
        if det_path != record.det_path:
            raise ValueError(
                "final dataset manifest det_path must match discovered record: "
                f"{det_path} != {record.det_path} ({manifest_path})"
            )
        updated_records.append(
            replace(
                record,
                manifest_path=manifest_path,
                source_kind=str(row.get("source_kind") or "").strip() or None,
                source_scene_path=_coerce_manifest_path(
                    row,
                    "source_scene_path",
                    manifest_path=manifest_path,
                    optional=True,
                ),
                source_image_path=_coerce_manifest_path(
                    row,
                    "source_image_path",
                    manifest_path=manifest_path,
                    optional=True,
                ),
                source_det_path=_coerce_manifest_path(
                    row,
                    "source_det_path",
                    manifest_path=manifest_path,
                    optional=True,
                ),
            )
        )

    discovered_keys = set(record_by_key)
    if manifest_keys != discovered_keys:
        missing = sorted(discovered_keys - manifest_keys)
        extra = sorted(manifest_keys - discovered_keys)
        raise ValueError(
            "final dataset manifest samples must match discovered records: "
            f"missing_in_manifest={missing[:3]} extra_in_manifest={extra[:3]} ({manifest_path})"
        )
    return sorted(updated_records, key=lambda item: (item.dataset_key, item.split, item.sample_id))


def _yolo_to_raw_box(
    line: str,
    raw_hw: tuple[int, int],
    *,
    det_path: Path,
    line_number: int,
) -> tuple[int, list[float]]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"invalid detection label row at {det_path}:{line_number}: expected 5 columns, got {len(parts)}"
        )
    raw_h, raw_w = raw_hw
    try:
        class_id = int(parts[0])
        center_x_norm = float(parts[1])
        center_y_norm = float(parts[2])
        width_norm = float(parts[3])
        height_norm = float(parts[4])
    except ValueError as exc:
        raise ValueError(f"invalid detection label row at {det_path}:{line_number}: {line.strip()}") from exc
    if class_id < 0 or class_id >= len(OD_CLASSES):
        raise ValueError(
            f"invalid detection class id at {det_path}:{line_number}: {class_id} not in [0, {len(OD_CLASSES) - 1}]"
        )
    normalized_values = {
        "center_x": center_x_norm,
        "center_y": center_y_norm,
        "width": width_norm,
        "height": height_norm,
    }
    for name, value in normalized_values.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite detection {name} at {det_path}:{line_number}: {value}")
    if not 0.0 <= center_x_norm <= 1.0 or not 0.0 <= center_y_norm <= 1.0:
        raise ValueError(
            f"invalid normalized detection center at {det_path}:{line_number}: "
            f"center=({center_x_norm}, {center_y_norm})"
        )
    if not 0.0 < width_norm <= 1.0 or not 0.0 < height_norm <= 1.0:
        raise ValueError(
            f"invalid normalized detection size at {det_path}:{line_number}: "
            f"size=({width_norm}, {height_norm})"
        )
    center_x = center_x_norm * raw_w
    center_y = center_y_norm * raw_h
    width = width_norm * raw_w
    height = height_norm * raw_h
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    if not (x2 > x1 and y2 > y1):
        raise ValueError(
            f"degenerate detection box at {det_path}:{line_number}: box=({x1}, {y1}, {x2}, {y2})"
        )
    return class_id, [x1, y1, x2, y2]


def _load_det_rows(det_path: Path | None, raw_hw: tuple[int, int]) -> list[tuple[int, list[float]]]:
    if det_path is None or not det_path.is_file():
        return []
    rows: list[tuple[int, list[float]]] = []
    for line_number, line in enumerate(_read_common_text(det_path).splitlines(), start=1):
        if not line.strip():
            continue
        parsed = _yolo_to_raw_box(
            line,
            raw_hw,
            det_path=det_path,
            line_number=line_number,
        )
        rows.append(parsed)
    return rows


def _coerce_traffic_light_detection_id(value: Any, *, field_name: str, scene_path: Path) -> int:
    if isinstance(value, bool):
        raise ValueError(f"scene {field_name} must be a non-negative integer: {scene_path}")
    try:
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError
            detection_id = int(value)
        else:
            detection_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scene {field_name} must be a non-negative integer: {scene_path}") from exc
    if detection_id < 0:
        raise ValueError(f"scene {field_name} must be a non-negative integer: {scene_path}")
    return detection_id


def _coerce_traffic_light_bit(value: Any, *, field_name: str, scene_path: Path) -> float:
    if value is None:
        return 0.0
    try:
        bit_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scene {field_name} must be finite 0/1: {scene_path}") from exc
    if not math.isfinite(bit_value) or bit_value not in (0.0, 1.0):
        raise ValueError(f"scene {field_name} must be finite 0/1: {scene_path}")
    return bit_value


def _traffic_light_lookup(scene: dict[str, Any], *, scene_path: Path) -> dict[int, dict[str, Any]]:
    raw_items = scene.get("traffic_lights", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"scene traffic_lights must be a list: {scene_path}")
    lookup: dict[int, dict[str, Any]] = {}
    for item_index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise TypeError(f"scene traffic_lights[{item_index}] must be an object: {scene_path}")
        if item.get("detection_id") is None:
            continue
        detection_id = _coerce_traffic_light_detection_id(
            item.get("detection_id"),
            field_name=f"traffic_lights[{item_index}].detection_id",
            scene_path=scene_path,
        )
        if detection_id in lookup:
            raise ValueError(
                f"scene traffic_lights detection_id must be unique: "
                f"traffic_lights[{item_index}].detection_id={detection_id} ({scene_path})"
            )
        raw_bits = item.get("tl_bits", {})
        if raw_bits is None:
            raw_bits = {}
        if not isinstance(raw_bits, dict):
            raise ValueError(f"scene traffic_lights[{item_index}].tl_bits must be an object: {scene_path}")
        bits = {
            bit: _coerce_traffic_light_bit(
                raw_bits.get(bit, 0),
                field_name=f"traffic_lights[{item_index}].tl_bits.{bit}",
                scene_path=scene_path,
            )
            for bit in TL_BITS
        }
        lookup[detection_id] = {**item, "detection_id": detection_id, "tl_bits": bits}
    return lookup


def _assert_traffic_light_detection_refs(
    lookup: dict[int, dict[str, Any]],
    *,
    det_count: int,
    scene_path: Path,
) -> None:
    for detection_id in sorted(lookup):
        if detection_id >= det_count:
            raise ValueError(
                f"scene traffic_lights detection_id must reference a detection row: "
                f"detection_id={detection_id} det_rows={det_count} ({scene_path})"
            )


def _lane_type_index(item: dict[str, Any]) -> int:
    value = str(item.get("source_style") or item.get("meta", {}).get("raw_type") or "").strip().lower()
    return LANE_TYPES.index(value) if value in LANE_TYPES else -1


def _lane_color_index(item: dict[str, Any]) -> int:
    class_name = str(item.get("class_name") or "")
    return LANE_CLASSES.index(class_name) if class_name in LANE_CLASSES else -1


def _lane_visibility_tensor(
    item: dict[str, Any],
    point_count: int,
    *,
    collection_key: str,
    item_index: int,
    scene_path: Path,
) -> torch.FloatTensor:
    raw_visibility = item.get("visibility")
    if raw_visibility is None:
        return torch.ones(point_count, dtype=torch.float32)
    if isinstance(raw_visibility, torch.Tensor):
        visibility = raw_visibility.to(dtype=torch.float32).reshape(-1)
    elif isinstance(raw_visibility, (list, tuple)):
        try:
            visibility = torch.tensor(raw_visibility, dtype=torch.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"scene {collection_key}[{item_index}].visibility must be finite values: {scene_path}"
            ) from exc
    else:
        raise ValueError(f"scene {collection_key}[{item_index}].visibility must be a list: {scene_path}")
    if visibility.numel() != point_count:
        raise ValueError(
            f"scene {collection_key}[{item_index}].visibility length must match points: {scene_path}"
        )
    if not bool(torch.isfinite(visibility).all()):
        raise ValueError(f"scene {collection_key}[{item_index}].visibility must be finite values: {scene_path}")
    return visibility.clamp(0.0, 1.0)


def _coerce_geometry_points(
    item: dict[str, Any],
    *,
    collection_key: str,
    item_index: int,
    scene_path: Path,
) -> list[list[float]]:
    raw_points = item.get("points")
    if raw_points is None:
        return []
    if not isinstance(raw_points, list):
        raise ValueError(f"scene {collection_key}[{item_index}].points must be a list: {scene_path}")
    points: list[list[float]] = []
    for point_index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ValueError(
                f"scene {collection_key}[{item_index}].points[{point_index}] must be [x, y]: {scene_path}"
            )
        try:
            x = float(raw_point[0])
            y = float(raw_point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"scene {collection_key}[{item_index}].points[{point_index}] coordinates must be finite: {scene_path}"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"scene {collection_key}[{item_index}].points[{point_index}] coordinates must be finite: {scene_path}"
            )
        points.append([x, y])
    return points


def _build_geometry_rows(
    items: list[dict[str, Any]],
    *,
    collection_key: str,
    scene_path: Path,
    transform: LetterboxTransform,
    min_unique_points: int,
    with_lane_attributes: bool,
) -> tuple[list[dict[str, Any]], torch.BoolTensor]:
    rows: list[dict[str, Any]] = []
    valid: list[bool] = []
    for item_index, item in enumerate(items):
        raw_points = _coerce_geometry_points(
            item,
            collection_key=collection_key,
            item_index=item_index,
            scene_path=scene_path,
        )
        transformed = clip_points(transform_points(raw_points, transform), transform.network_hw)
        row: dict[str, Any] = {"points_xy": torch.tensor(transformed, dtype=torch.float32)}
        if with_lane_attributes:
            row["color"] = _lane_color_index(item)
            row["lane_type"] = _lane_type_index(item)
            row["visibility"] = _lane_visibility_tensor(
                item,
                len(transformed),
                collection_key=collection_key,
                item_index=item_index,
                scene_path=scene_path,
            )
        rows.append(row)
        valid.append(unique_point_count(transformed) >= min_unique_points)
    return rows, torch.tensor(valid, dtype=torch.bool)


def _geometry_valid_mask(rows: list[dict[str, Any]], *, min_unique_points: int) -> torch.BoolTensor:
    valid: list[bool] = []
    for row in rows:
        points = row.get("points_xy")
        if isinstance(points, torch.Tensor):
            point_rows = points.detach().cpu().reshape(-1, 2).tolist()
        else:
            point_rows = []
        valid.append(unique_point_count(point_rows) >= int(min_unique_points))
    return torch.tensor(valid, dtype=torch.bool)


def _build_source_mask(dataset_key: str) -> dict[str, bool]:
    try:
        return dict(SOURCE_MASK_BY_DATASET[dataset_key])
    except KeyError as exc:
        raise KeyError(f"unsupported dataset key for loader: {dataset_key}") from exc


def _build_det_supervision_policy(dataset_key: str) -> dict[str, Any]:
    try:
        policy = DET_SUPERVISION_BY_DATASET[dataset_key]
    except KeyError as exc:
        raise KeyError(f"unsupported dataset key for det supervision policy: {dataset_key}") from exc
    class_names = [str(item) for item in policy["class_names"]]
    return {
        "class_names": class_names,
        "class_ids": [OD_CLASS_TO_ID[item] for item in class_names],
        "allow_objectness_negatives": bool(policy["allow_objectness_negatives"]),
        "allow_unmatched_class_negatives": bool(policy["allow_unmatched_class_negatives"]),
    }


class PV26CanonicalDataset(Dataset):
    def __init__(
        self,
        dataset_roots: Iterable[Path | str],
        *,
        train_augmentation: bool | TrainAugmentationConfig = False,
        train_augmentation_seed: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
        progress_every: int = 2000,
    ) -> None:
        roots = [Path(root).resolve() for root in dataset_roots]
        if isinstance(train_augmentation, TrainAugmentationConfig):
            self.train_augmentation = train_augmentation
        elif train_augmentation:
            self.train_augmentation = TrainAugmentationConfig()
        else:
            self.train_augmentation = None
        self.train_augmentation_seed = None if train_augmentation_seed is None else int(train_augmentation_seed)
        self.records: list[SampleRecord] = []
        for root in roots:
            if progress_callback is not None:
                progress_callback(f"scanning canonical dataset root: {root}")
            root_records = _discover_records(
                root,
                progress_callback=progress_callback,
                progress_every=progress_every,
            )
            root_records = _apply_final_dataset_manifest(root, root_records)
            self.records.extend(root_records)
            if progress_callback is not None:
                progress_callback(f"discovered {len(root_records)} records under {root}")
        self.records.sort(key=lambda item: (item.dataset_key, item.split, item.sample_id))
        if progress_callback is not None:
            progress_callback(
                f"loaded {len(self.records)} canonical records from {len(roots)} dataset roots"
            )
        self._stopline_copy_paste_donors = [
            record
            for record in self.records
            if record.split == "train" and int(record.stop_line_count) > 0
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _stopline_copy_paste_donor(
        self,
        *,
        current_record: SampleRecord,
        rng: random.Random,
    ) -> dict[str, object] | None:
        if not self._stopline_copy_paste_donors:
            return None
        for _ in range(min(8, len(self._stopline_copy_paste_donors))):
            donor_record = self._stopline_copy_paste_donors[
                rng.randrange(len(self._stopline_copy_paste_donors))
            ]
            if donor_record.scene_path == current_record.scene_path:
                continue
            donor_scene = _load_json(donor_record.scene_path)
            donor_raw_hw = _coerce_scene_image_hw(donor_scene, scene_path=donor_record.scene_path)
            donor_transform = compute_letterbox_transform(donor_raw_hw)
            donor_stop_lines, donor_valid = _build_geometry_rows(
                _coerce_scene_geometry_items(donor_scene, "stop_lines", scene_path=donor_record.scene_path),
                collection_key="stop_lines",
                scene_path=donor_record.scene_path,
                transform=donor_transform,
                min_unique_points=2,
                with_lane_attributes=False,
            )
            valid_stop_lines = [
                row
                for row, is_valid in zip(donor_stop_lines, donor_valid.tolist())
                if bool(is_valid)
            ]
            if not valid_stop_lines:
                continue
            return {
                "image": load_letterboxed_image(donor_record.image_path, donor_transform),
                "stop_lines": valid_stop_lines,
                "sample_id": donor_record.sample_id,
                "dataset_key": donor_record.dataset_key,
            }
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        scene = _load_json(record.scene_path)
        raw_hw = _coerce_scene_image_hw(scene, scene_path=record.scene_path)
        transform = compute_letterbox_transform(raw_hw)
        image = load_letterboxed_image(record.image_path, transform)

        source_mask = _build_source_mask(record.dataset_key)
        det_policy = _build_det_supervision_policy(record.dataset_key)
        if bool(source_mask.get("det")) and record.det_path is None:
            expected_det_path = record.dataset_root / "labels_det" / record.split / f"{record.sample_id}.txt"
            raise FileNotFoundError(
                f"det label file not found for detector-supervised sample: {expected_det_path}"
            )
        tl_lookup = _traffic_light_lookup(scene, scene_path=record.scene_path)
        det_rows = _load_det_rows(record.det_path, raw_hw)
        _assert_traffic_light_detection_refs(
            tl_lookup,
            det_count=len(det_rows),
            scene_path=record.scene_path,
        )

        det_boxes: list[list[float]] = []
        det_classes: list[int] = []
        tl_bits: list[list[float]] = []
        tl_is_light: list[bool] = []
        tl_collapse_reason: list[str] = []
        tl_valid: list[bool] = []

        for det_index, (class_id, raw_box) in enumerate(det_rows):
            transformed_box = clip_box_xyxy(transform_box_xyxy(raw_box, transform), transform.network_hw)
            if transformed_box is None:
                continue
            det_boxes.append(transformed_box)
            det_classes.append(class_id)
            tl_item = tl_lookup.get(det_index)
            if tl_item is None:
                tl_bits.append([0.0, 0.0, 0.0, 0.0])
                tl_is_light.append(False)
                tl_collapse_reason.append("not_traffic_light")
                tl_valid.append(False)
            else:
                bits = [float(tl_item.get("tl_bits", {}).get(bit, 0)) for bit in TL_BITS]
                tl_bits.append(bits)
                tl_is_light.append(True)
                tl_collapse_reason.append(str(tl_item.get("collapse_reason") or "unknown"))
                tl_valid.append(bool(tl_item.get("tl_attr_valid")))

        lane_items = _coerce_scene_geometry_items(scene, "lanes", scene_path=record.scene_path)
        stop_line_items = _coerce_scene_geometry_items(scene, "stop_lines", scene_path=record.scene_path)
        crosswalk_items = _coerce_scene_geometry_items(scene, "crosswalks", scene_path=record.scene_path)
        lanes, lane_valid = _build_geometry_rows(
            lane_items,
            collection_key="lanes",
            scene_path=record.scene_path,
            transform=transform,
            min_unique_points=2,
            with_lane_attributes=True,
        )
        stop_lines, stop_valid = _build_geometry_rows(
            stop_line_items,
            collection_key="stop_lines",
            scene_path=record.scene_path,
            transform=transform,
            min_unique_points=2,
            with_lane_attributes=False,
        )
        crosswalks, crosswalk_valid = _build_geometry_rows(
            crosswalk_items,
            collection_key="crosswalks",
            scene_path=record.scene_path,
            transform=transform,
            min_unique_points=3,
            with_lane_attributes=False,
        )
        augmentation_meta = None
        if record.split == "train" and self.train_augmentation is not None:
            rng = random.Random()
            if self.train_augmentation_seed is not None:
                key = f"{self.train_augmentation_seed}:{record.dataset_key}:{record.split}:{record.sample_id}".encode("utf-8")
                seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
                rng = random.Random(seed)
            stopline_copy_paste_donor = None
            if (
                not stop_lines
                and float(getattr(self.train_augmentation, "stopline_copy_paste_prob", 0.0)) > 0.0
            ):
                stopline_copy_paste_donor = self._stopline_copy_paste_donor(
                    current_record=record,
                    rng=rng,
                )
            image, det_boxes, lanes, stop_lines, crosswalks, augmentation_meta = apply_train_augmentations(
                image,
                det_boxes=det_boxes,
                lanes=lanes,
                stop_lines=stop_lines,
                crosswalks=crosswalks,
                network_hw=NETWORK_HW,
                config=self.train_augmentation,
                rng=rng,
                stopline_copy_paste_donor=stopline_copy_paste_donor,
            )
            lane_valid = _geometry_valid_mask(lanes, min_unique_points=2)
            stop_valid = _geometry_valid_mask(stop_lines, min_unique_points=2)
            crosswalk_valid = _geometry_valid_mask(crosswalks, min_unique_points=3)

        return {
            "image": image,
            "det_targets": {
                "boxes_xyxy": torch.tensor(det_boxes, dtype=torch.float32).reshape(-1, 4),
                "classes": torch.tensor(det_classes, dtype=torch.long),
            },
            "tl_attr_targets": {
                "bits": torch.tensor(tl_bits, dtype=torch.float32).reshape(-1, len(TL_BITS)),
                "is_traffic_light": torch.tensor(tl_is_light, dtype=torch.bool),
                "collapse_reason": tl_collapse_reason,
            },
            "lane_targets": {
                "lanes": lanes,
                "stop_lines": stop_lines,
                "crosswalks": crosswalks,
            },
            "source_mask": source_mask,
            "valid_mask": {
                "det": torch.ones(len(det_boxes), dtype=torch.bool),
                "tl_attr": torch.tensor(tl_valid, dtype=torch.bool),
                "lane": lane_valid,
                "stop_line": stop_valid,
                "crosswalk": crosswalk_valid,
            },
            "meta": {
                "sample_id": record.sample_id,
                "dataset_key": record.dataset_key,
                "split": record.split,
                "image_path": str(record.image_path),
                "raw_hw": raw_hw,
                "network_hw": NETWORK_HW,
                "transform": transform.as_meta(),
                "det_supervised_classes": list(det_policy["class_names"]),
                "det_supervised_class_ids": list(det_policy["class_ids"]),
                "det_allow_objectness_negatives": bool(det_policy["allow_objectness_negatives"]),
                "det_allow_unmatched_class_negatives": bool(det_policy["allow_unmatched_class_negatives"]),
                "augmentation": augmentation_meta,
                "final_manifest_path": str(record.manifest_path) if record.manifest_path is not None else None,
                "source_kind": record.source_kind,
                "source_scene_path": str(record.source_scene_path) if record.source_scene_path is not None else None,
                "source_image_path": str(record.source_image_path) if record.source_image_path is not None else None,
                "source_det_path": str(record.source_det_path) if record.source_det_path is not None else None,
            },
        }


def collate_pv26_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate zero PV26 samples")
    expected_shape = (3, int(NETWORK_HW[0]), int(NETWORK_HW[1]))
    for sample_index, sample in enumerate(samples):
        image = sample.get("image")
        image_invalid = (
            not isinstance(image, torch.Tensor)
            or image.dtype != torch.float32
            or tuple(image.shape) != expected_shape
        )
        if image_invalid:
            raise ValueError(
                "PV26 sample image must be float32 "
                f"{expected_shape}: sample_index={sample_index} "
                f"shape={tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__} "
                f"dtype={image.dtype if isinstance(image, torch.Tensor) else 'n/a'}"
            )
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "det_targets": [sample["det_targets"] for sample in samples],
        "tl_attr_targets": [sample["tl_attr_targets"] for sample in samples],
        "lane_targets": [sample["lane_targets"] for sample in samples],
        "source_mask": [sample["source_mask"] for sample in samples],
        "valid_mask": [sample["valid_mask"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }


def collate_pv26_encoded_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from .target_encoder import encode_pv26_batch

    return encode_pv26_batch(collate_pv26_samples(samples), include_lane_segfirst_targets=True)


def collate_pv26_encoded_eval_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from .target_encoder import encode_pv26_batch

    raw_batch = collate_pv26_samples(samples)
    encoded = encode_pv26_batch(raw_batch, include_lane_segfirst_targets=True)
    encoded["_raw_batch"] = {
        "det_targets": list(raw_batch["det_targets"]),
        "tl_attr_targets": list(raw_batch["tl_attr_targets"]),
        "lane_targets": list(raw_batch["lane_targets"]),
        "source_mask": list(raw_batch["source_mask"]),
        "valid_mask": list(raw_batch["valid_mask"]),
        "meta": list(raw_batch["meta"]),
    }
    return encoded
