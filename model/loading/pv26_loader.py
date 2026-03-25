from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from ..preprocess.aihub_standardize import LANE_CLASSES, LANE_TYPES, OD_CLASSES, TL_BITS
from .transform import (
    NETWORK_HW,
    LetterboxTransform,
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


SOURCE_MASK_BY_DATASET = {
    "aihub_traffic_seoul": {
        "det": True,
        "tl_attr": True,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    "aihub_obstacle_seoul": {
        "det": True,
        "tl_attr": False,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
    "aihub_lane_seoul": {
        "det": False,
        "tl_attr": False,
        "lane": True,
        "stop_line": True,
        "crosswalk": True,
    },
    "bdd100k_det_100k": {
        "det": True,
        "tl_attr": False,
        "lane": False,
        "stop_line": False,
        "crosswalk": False,
    },
}
DET_SUPERVISION_BY_DATASET = {
    "aihub_traffic_seoul": {
        "class_names": ("traffic_light", "sign"),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": True,
    },
    "aihub_obstacle_seoul": {
        "class_names": ("traffic_cone", "obstacle"),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": True,
    },
    "aihub_lane_seoul": {
        "class_names": (),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": False,
    },
    "bdd100k_det_100k": {
        "class_names": ("vehicle", "bike", "pedestrian"),
        "allow_objectness_negatives": False,
        "allow_unmatched_class_negatives": True,
    },
}
OD_CLASS_TO_ID = {class_name: index for index, class_name in enumerate(OD_CLASSES)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_records(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    labels_scene_root = dataset_root / "labels_scene"
    if not labels_scene_root.is_dir():
        return records

    for scene_path in sorted(labels_scene_root.rglob("*.json")):
        split = scene_path.parent.name
        scene = _load_json(scene_path)
        dataset_key = str(scene.get("source", {}).get("dataset") or "unknown_dataset")
        sample_id = scene_path.stem
        image_file_name = str(scene.get("image", {}).get("file_name"))
        image_path = dataset_root / "images" / split / image_file_name
        det_path = dataset_root / "labels_det" / split / f"{sample_id}.txt"
        records.append(
            SampleRecord(
                dataset_root=dataset_root,
                dataset_key=dataset_key,
                split=split,
                sample_id=sample_id,
                scene_path=scene_path,
                image_path=image_path,
                det_path=det_path if det_path.is_file() else None,
            )
        )
    return sorted(records, key=lambda item: (item.dataset_key, item.split, item.sample_id))


def _yolo_to_raw_box(line: str, raw_hw: tuple[int, int]) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    raw_h, raw_w = raw_hw
    class_id = int(parts[0])
    center_x = float(parts[1]) * raw_w
    center_y = float(parts[2]) * raw_h
    width = float(parts[3]) * raw_w
    height = float(parts[4]) * raw_h
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return class_id, [x1, y1, x2, y2]


def _load_det_rows(det_path: Path | None, raw_hw: tuple[int, int]) -> list[tuple[int, list[float]]]:
    if det_path is None or not det_path.is_file():
        return []
    rows: list[tuple[int, list[float]]] = []
    for line in det_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = _yolo_to_raw_box(line, raw_hw)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _traffic_light_lookup(scene: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["detection_id"]): item
        for item in scene.get("traffic_lights", [])
        if item.get("detection_id") is not None
    }


def _lane_type_index(item: dict[str, Any]) -> int:
    value = str(item.get("source_style") or item.get("meta", {}).get("raw_type") or "").strip().lower()
    return LANE_TYPES.index(value) if value in LANE_TYPES else -1


def _lane_color_index(item: dict[str, Any]) -> int:
    class_name = str(item.get("class_name") or "")
    return LANE_CLASSES.index(class_name) if class_name in LANE_CLASSES else -1


def _build_geometry_rows(
    items: list[dict[str, Any]],
    *,
    transform: LetterboxTransform,
    min_unique_points: int,
    with_lane_attributes: bool,
) -> tuple[list[dict[str, Any]], torch.BoolTensor]:
    rows: list[dict[str, Any]] = []
    valid: list[bool] = []
    for item in items:
        raw_points = item.get("points") or []
        transformed = clip_points(transform_points(raw_points, transform), transform.network_hw)
        row: dict[str, Any] = {"points_xy": torch.tensor(transformed, dtype=torch.float32)}
        if with_lane_attributes:
            row["color"] = _lane_color_index(item)
            row["lane_type"] = _lane_type_index(item)
        rows.append(row)
        valid.append(unique_point_count(transformed) >= min_unique_points)
    return rows, torch.tensor(valid, dtype=torch.bool)


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
    def __init__(self, dataset_roots: Iterable[Path | str]) -> None:
        roots = [Path(root).resolve() for root in dataset_roots]
        self.records: list[SampleRecord] = []
        for root in roots:
            self.records.extend(_discover_records(root))
        self.records.sort(key=lambda item: (item.dataset_key, item.split, item.sample_id))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        scene = _load_json(record.scene_path)
        raw_h = int(scene["image"]["height"])
        raw_w = int(scene["image"]["width"])
        raw_hw = (raw_h, raw_w)
        transform = compute_letterbox_transform(raw_hw)
        image = load_letterboxed_image(record.image_path, transform)

        source_mask = _build_source_mask(record.dataset_key)
        det_policy = _build_det_supervision_policy(record.dataset_key)
        tl_lookup = _traffic_light_lookup(scene)
        det_rows = _load_det_rows(record.det_path, raw_hw)

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

        lanes, lane_valid = _build_geometry_rows(
            scene.get("lanes", []),
            transform=transform,
            min_unique_points=2,
            with_lane_attributes=True,
        )
        stop_lines, stop_valid = _build_geometry_rows(
            scene.get("stop_lines", []),
            transform=transform,
            min_unique_points=2,
            with_lane_attributes=False,
        )
        crosswalks, crosswalk_valid = _build_geometry_rows(
            scene.get("crosswalks", []),
            transform=transform,
            min_unique_points=3,
            with_lane_attributes=False,
        )

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
            },
        }


def collate_pv26_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "det_targets": [sample["det_targets"] for sample in samples],
        "tl_attr_targets": [sample["tl_attr_targets"] for sample in samples],
        "lane_targets": [sample["lane_targets"] for sample in samples],
        "source_mask": [sample["source_mask"] for sample in samples],
        "valid_mask": [sample["valid_mask"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }
