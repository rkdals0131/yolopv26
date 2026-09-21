from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from common.pv26_schema import (
    AIHUB_LANE_DATASET_KEY,
    AIHUB_OBSTACLE_DATASET_KEY,
    AIHUB_TRAFFIC_DATASET_KEY,
    BDD100K_DATASET_KEY,
    EXHAUSTIVE_DATASET_KEY_BY_SOURCE,
    OD_CLASS_TO_ID,
)
from tools.pv26_train.config import (
    DatasetConfig,
    MetaTrainScenario,
    PhaseConfig,
    PreviewConfig,
    RunConfig,
    SelectionConfig,
    TrainDefaultsConfig,
)


EXHAUSTIVE_BDD_DATASET_KEY = EXHAUSTIVE_DATASET_KEY_BY_SOURCE[BDD100K_DATASET_KEY]
EXHAUSTIVE_TRAFFIC_DATASET_KEY = EXHAUSTIVE_DATASET_KEY_BY_SOURCE[AIHUB_TRAFFIC_DATASET_KEY]
EXHAUSTIVE_OBSTACLE_DATASET_KEY = EXHAUSTIVE_DATASET_KEY_BY_SOURCE[AIHUB_OBSTACLE_DATASET_KEY]
DEFAULT_PREPARED_DATASET_KEYS = (
    EXHAUSTIVE_BDD_DATASET_KEY,
    EXHAUSTIVE_TRAFFIC_DATASET_KEY,
    EXHAUSTIVE_OBSTACLE_DATASET_KEY,
    AIHUB_LANE_DATASET_KEY,
)


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (width, height), color).save(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _raw_box_to_yolo_row(class_name: str, raw_box: list[float], raw_hw: tuple[int, int]) -> str:
    raw_h, raw_w = raw_hw
    x1, y1, x2, y2 = [float(value) for value in raw_box]
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return (
        f"{OD_CLASS_TO_ID[class_name]} "
        f"{center_x / raw_w:.6f} "
        f"{center_y / raw_h:.6f} "
        f"{width / raw_w:.6f} "
        f"{height / raw_h:.6f}"
    )


def prepared_sample_id(dataset_key: str, split: str) -> str:
    return f"{dataset_key}_{split}_001"


def _sample_payload(dataset_key: str) -> dict[str, Any]:
    if dataset_key == EXHAUSTIVE_BDD_DATASET_KEY:
        return {
            "image_hw": (720, 1280),
            "color": "#303840",
            "detections": [
                {"class_name": "vehicle", "bbox": [120.0, 180.0, 420.0, 420.0]},
                {"class_name": "pedestrian", "bbox": [720.0, 210.0, 780.0, 360.0]},
            ],
            "traffic_lights": [],
            "lanes": [],
            "stop_lines": [],
            "crosswalks": [],
        }
    if dataset_key == EXHAUSTIVE_TRAFFIC_DATASET_KEY:
        return {
            "image_hw": (1080, 1920),
            "color": "#202830",
            "detections": [
                {"class_name": "traffic_light", "bbox": [120.0, 80.0, 180.0, 220.0]},
                {"class_name": "sign", "bbox": [360.0, 120.0, 500.0, 260.0]},
            ],
            "traffic_lights": [
                {
                    "detection_id": 0,
                    "tl_bits": {"red": 1, "yellow": 0, "green": 0, "arrow": 1},
                    "tl_attr_valid": True,
                    "collapse_reason": "valid",
                }
            ],
            "lanes": [],
            "stop_lines": [],
            "crosswalks": [],
        }
    if dataset_key == EXHAUSTIVE_OBSTACLE_DATASET_KEY:
        return {
            "image_hw": (720, 1280),
            "color": "#403030",
            "detections": [
                {"class_name": "traffic_cone", "bbox": [80.0, 240.0, 150.0, 410.0]},
                {"class_name": "obstacle", "bbox": [640.0, 250.0, 760.0, 420.0]},
            ],
            "traffic_lights": [],
            "lanes": [],
            "stop_lines": [],
            "crosswalks": [],
        }
    if dataset_key == AIHUB_LANE_DATASET_KEY:
        return {
            "image_hw": (720, 1280),
            "color": "#202020",
            "detections": [],
            "traffic_lights": [],
            "lanes": [
                {
                    "class_name": "white_lane",
                    "source_style": "solid",
                    "points": [[240.0, 700.0], [260.0, 520.0], [280.0, 340.0]],
                    "visibility": [1.0, 1.0, 1.0],
                },
                {
                    "class_name": "yellow_lane",
                    "source_style": "dotted",
                    "points": [[980.0, 700.0], [960.0, 520.0], [940.0, 340.0]],
                    "visibility": [1.0, 1.0, 1.0],
                },
            ],
            "stop_lines": [
                {
                    "points": [[260.0, 620.0], [1000.0, 620.0]],
                }
            ],
            "crosswalks": [
                {
                    "points": [[330.0, 650.0], [470.0, 650.0], [500.0, 710.0], [300.0, 710.0]],
                }
            ],
        }
    raise KeyError(f"unsupported prepared dataset key: {dataset_key}")


def create_prepared_pv26_dataset(
    root: Path,
    *,
    splits: Iterable[str] = ("train", "val"),
    dataset_keys: Iterable[str] = DEFAULT_PREPARED_DATASET_KEYS,
) -> Path:
    dataset_root = Path(root).resolve()
    split_values = tuple(str(value) for value in splits)
    key_values = tuple(str(value) for value in dataset_keys)
    for split in split_values:
        for dataset_key in key_values:
            payload = _sample_payload(dataset_key)
            sample_id = prepared_sample_id(dataset_key, split)
            raw_h, raw_w = payload["image_hw"]
            image_name = f"{sample_id}.png"
            image_path = dataset_root / "images" / split / image_name
            scene_path = dataset_root / "labels_scene" / split / f"{sample_id}.json"
            det_path = dataset_root / "labels_det" / split / f"{sample_id}.txt"
            _make_image(image_path, raw_w, raw_h, str(payload["color"]))
            scene = {
                "image": {
                    "file_name": image_name,
                    "original_file_name": image_name,
                    "width": raw_w,
                    "height": raw_h,
                },
                "source": {
                    "dataset": dataset_key,
                    "split": split,
                    "final_sample_id": sample_id,
                },
                "detections": [
                    {
                        "id": index,
                        "class_name": item["class_name"],
                        "bbox": list(item["bbox"]),
                    }
                    for index, item in enumerate(payload["detections"])
                ],
                "traffic_lights": list(payload["traffic_lights"]),
                "lanes": list(payload["lanes"]),
                "stop_lines": list(payload["stop_lines"]),
                "crosswalks": list(payload["crosswalks"]),
            }
            _write_json(scene_path, scene)
            if payload["detections"]:
                det_rows = [
                    _raw_box_to_yolo_row(
                        str(item["class_name"]),
                        list(item["bbox"]),
                        (int(raw_h), int(raw_w)),
                    )
                    for item in payload["detections"]
                ]
                det_path.parent.mkdir(parents=True, exist_ok=True)
                det_path.write_text("\n".join(det_rows) + "\n", encoding="utf-8")
    return dataset_root


def select_prepared_samples(
    dataset: Any,
    *,
    split: str,
    dataset_keys: Iterable[str],
    limit_per_key: int = 1,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts = {str(dataset_key): 0 for dataset_key in dataset_keys}
    for index, record in enumerate(dataset.records):
        dataset_key = str(record.dataset_key)
        if str(record.split) != str(split) or dataset_key not in counts:
            continue
        if counts[dataset_key] >= int(limit_per_key):
            continue
        selected.append(dataset[index])
        counts[dataset_key] += 1
        if all(count >= int(limit_per_key) for count in counts.values()):
            break
    missing = [dataset_key for dataset_key, count in counts.items() if count < int(limit_per_key)]
    if missing:
        raise ValueError(f"prepared dataset fixture missing samples for split={split}: {missing}")
    return selected


def build_prepared_dataset_e2e_scenario(
    *,
    dataset_root: Path,
    run_root: Path,
) -> MetaTrainScenario:
    phase = PhaseConfig(
        name="prepared_dataset_stage1_e2e",
        stage="stage_1_frozen_trunk_warmup",
        min_epochs=1,
        max_epochs=1,
        patience=1,
    )
    return MetaTrainScenario(
        dataset=DatasetConfig(root=Path(dataset_root).resolve()),
        run=RunConfig(
            run_root=Path(run_root).resolve(),
            run_name_prefix="prepared_dataset_e2e",
            run_dir=(Path(run_root).resolve() / "prepared_dataset_e2e"),
        ),
        train_defaults=TrainDefaultsConfig(
            device="cpu",
            batch_size=4,
            train_batches=1,
            val_batches=1,
            schedule="none",
            amp=False,
            checkpoint_every=1,
            num_workers=0,
            pin_memory=False,
            log_every_n_steps=1,
            profile_window=4,
            profile_device_sync=False,
            persistent_workers=False,
            prefetch_factor=None,
            backbone_variant="n",
            det_conf_threshold=0.0,
            det_iou_threshold=0.70,
            lane_obj_threshold=0.0,
            stop_line_obj_threshold=0.0,
            crosswalk_obj_threshold=0.0,
        ),
        selection=SelectionConfig(metric_path="val.losses.total.mean", mode="min", eps=1.0e-8),
        preview=PreviewConfig(
            enabled=False,
            split="val",
            dataset_keys=(),
            max_samples_per_dataset=1,
            write_overlay=False,
        ),
        phases=(phase,),
    )
