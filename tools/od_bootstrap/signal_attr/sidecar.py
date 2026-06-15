from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from PIL import Image
import torch

from common.pv26_schema import TL_BITS

from .classifier import (
    SignalAttrPrediction,
    SignalAttrThresholdPolicy,
    load_signal_attr_classifier_checkpoint,
    predict_signal_attr_crop_image,
)
from .crop import (
    SIGNAL_ATTR_CROP_REASON_INVALID_ROI,
    SignalAttrCropConfig,
    crop_signal_attr_roi,
)


@dataclass(frozen=True)
class SignalAttrSidecarStats:
    traffic_light_count: int
    valid_count: int
    invalid_count: int
    reason_counts: dict[str, int]


class SignalAttrSidecarTeacher:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        crop_config: SignalAttrCropConfig,
        threshold_policy: SignalAttrThresholdPolicy,
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.crop_config = crop_config
        self.threshold_policy = threshold_policy
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._lock = Lock()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device: str = "cuda:0",
    ) -> "SignalAttrSidecarTeacher":
        loaded = load_signal_attr_classifier_checkpoint(checkpoint_path, device=device)
        return cls(
            model=loaded["model"],
            crop_config=_crop_config_from_payload(loaded["crop_config"]),
            threshold_policy=loaded["threshold_policy"],
            checkpoint_path=Path(checkpoint_path).resolve(),
            device=loaded["device"],
        )

    def apply_to_scene(
        self,
        scene: dict[str, Any],
        image_path: Path,
        *,
        run_id: str,
        created_at: str,
    ) -> SignalAttrSidecarStats:
        detections = scene.get("detections")
        if not isinstance(detections, list):
            raise TypeError("signal attr sidecar requires scene.detections list")
        _assert_detection_row_order(detections)
        rows: list[dict[str, Any]] = []
        reason_counts: Counter[str] = Counter()
        valid_count = 0
        with Image.open(image_path) as image:
            image.load()
            for detection_index, detection in enumerate(detections):
                if not isinstance(detection, Mapping):
                    raise TypeError(f"scene detections[{detection_index}] must be an object")
                class_name = str(detection.get("class_name") or "")
                if class_name != "traffic_light":
                    continue
                bbox = _bbox_from_detection(detection)
                crop = crop_signal_attr_roi(image, bbox, config=self.crop_config)
                if not crop.valid or crop.crop_image is None:
                    prediction = _invalid_prediction(SIGNAL_ATTR_CROP_REASON_INVALID_ROI)
                    crop_payload = {"crop_box": None, "clipped_box": None}
                else:
                    with self._lock:
                        prediction = predict_signal_attr_crop_image(
                            self.model,
                            crop.crop_image,
                            crop_config=_crop_config_payload(self.crop_config),
                            threshold_policy=self.threshold_policy,
                            device=self.device,
                        )
                    crop_payload = {
                        "crop_box": list(crop.crop_box) if crop.crop_box is not None else None,
                        "clipped_box": list(crop.clipped_box) if crop.clipped_box is not None else None,
                    }
                reason_counts[prediction.collapse_reason] += 1
                valid_count += int(prediction.tl_attr_valid)
                rows.append(
                    {
                        "id": len(rows),
                        "detection_id": detection_index,
                        "bbox": bbox,
                        "tl_bits": {bit: int(prediction.tl_bits.get(bit, 0)) for bit in TL_BITS},
                        "tl_attr_valid": int(prediction.tl_attr_valid),
                        "collapse_reason": prediction.collapse_reason,
                        "base_color": prediction.base_color,
                        "arrow": int(prediction.arrow),
                        "base_color_confidence": float(prediction.base_color_confidence),
                        "arrow_probability": float(prediction.arrow_probability),
                        "meta": {
                            "label_origin": "signal_attr_sidecar",
                            "teacher_name": "signal_attr",
                            "checkpoint_path": str(self.checkpoint_path),
                            "threshold_policy": self.threshold_policy.name,
                            "bootstrap_run_id": run_id,
                            "created_at": created_at,
                            **crop_payload,
                        },
                    }
                )
        scene["traffic_lights"] = rows
        scene.setdefault("tasks", {})
        scene["tasks"]["has_tl_attr"] = int(valid_count > 0)
        scene.setdefault("notes", [])
        scene["notes"].append("Traffic-light attributes were materialized by signal_attr sidecar on final detection row order.")
        return SignalAttrSidecarStats(
            traffic_light_count=len(rows),
            valid_count=valid_count,
            invalid_count=len(rows) - valid_count,
            reason_counts=dict(sorted(reason_counts.items())),
        )


def _crop_config_from_payload(payload: Mapping[str, Any]) -> SignalAttrCropConfig:
    allowed = SignalAttrCropConfig.__dataclass_fields__
    values = {key: payload[key] for key in allowed if key in payload}
    return SignalAttrCropConfig(**values)


def _crop_config_payload(config: SignalAttrCropConfig) -> dict[str, Any]:
    return {
        "input_size": int(config.input_size),
        "normalization": str(config.normalization),
    }


def _assert_detection_row_order(detections: Sequence[Any]) -> None:
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            raise TypeError(f"scene detections[{index}] must be an object")
        if int(detection.get("id", -1)) != index:
            raise ValueError(f"scene detections[{index}].id must match detection row order before signal attr sidecar")


def _bbox_from_detection(detection: Mapping[str, Any]) -> list[float]:
    raw_bbox = detection.get("bbox")
    if isinstance(raw_bbox, Mapping):
        return [
            float(raw_bbox.get("x1", 0.0)),
            float(raw_bbox.get("y1", 0.0)),
            float(raw_bbox.get("x2", 0.0)),
            float(raw_bbox.get("y2", 0.0)),
        ]
    if isinstance(raw_bbox, Sequence) and not isinstance(raw_bbox, (str, bytes)):
        values = [float(item) for item in raw_bbox[:4]]
        if len(values) == 4:
            return values
    return [0.0, 0.0, 0.0, 0.0]


def _invalid_prediction(reason: str) -> SignalAttrPrediction:
    return SignalAttrPrediction(
        tl_bits={bit: 0 for bit in TL_BITS},
        tl_attr_valid=0,
        collapse_reason=reason,
        base_color="off",
        arrow=0,
        base_color_confidence=0.0,
        arrow_probability=0.0,
        base_color_scores={"off": 0.0, "red": 0.0, "yellow": 0.0, "green": 0.0},
    )


__all__ = [
    "SignalAttrSidecarStats",
    "SignalAttrSidecarTeacher",
]
