from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import torch

from .classifier import (
    SignalAttrThresholdPolicy,
    load_signal_attr_classifier_checkpoint,
    product_signal_attr_prediction_from_logits,
    signal_attr_crop_image_to_tensor,
    signal_attr_prediction_from_logits,
)
from .crop import SignalAttrCropConfig, crop_signal_attr_roi

SIGNAL_CLASSES = {"vehicle_signal": "car", "pedestrian_signal": "pedestrian"}


class SignalAttrRuntime:
    """Batch the signal crops from one image and bind results by detection ID."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        crop_config: SignalAttrCropConfig,
        threshold_policy: SignalAttrThresholdPolicy,
        state_semantics: str,
        all_off_is_valid: bool = False,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.crop_config = crop_config
        self.threshold_policy = threshold_policy
        self.state_semantics = state_semantics
        self.all_off_is_valid = bool(all_off_is_valid)
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, *, device: str = "cpu") -> "SignalAttrRuntime":
        loaded = load_signal_attr_classifier_checkpoint(checkpoint_path, device=device)
        crop_payload = loaded["crop_config"]
        crop_values = {
            key: crop_payload[key]
            for key in SignalAttrCropConfig.__dataclass_fields__
            if key in crop_payload
        }
        return cls(
            loaded["model"],
            crop_config=SignalAttrCropConfig(**crop_values),
            threshold_policy=loaded["threshold_policy"],
            state_semantics=str(loaded["payload"].get("state_semantics") or "legacy_arrow"),
            all_off_is_valid=bool(loaded["payload"].get("all_off_is_valid", False)),
            device=loaded["device"],
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, detections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        tensors: list[torch.Tensor] = []
        inference_indices: list[int] = []
        for detection in detections:
            class_name = str(detection.get("class_name") or "")
            light_type = SIGNAL_CLASSES.get(class_name)
            if light_type is None:
                continue
            bbox = detection.get("bbox_xyxy")
            detection_id = detection.get("id")
            row = {
                "detection_id": detection_id,
                "class_name": class_name,
                "light_type": light_type,
                "bbox_xyxy": bbox,
                "state_valid": False,
                "reason": "invalid_roi",
                "base_color": None,
                "red_score": 0.0,
                "yellow_score": 0.0,
                "green_score": 0.0,
                "left_arrow_score": None,
                "left_arrow": None,
            }
            rows.append(row)
            crop = crop_signal_attr_roi(image, bbox, config=self.crop_config)
            if not crop.valid or crop.crop_image is None:
                continue
            tensors.append(
                signal_attr_crop_image_to_tensor(
                    crop.crop_image,
                    input_size=self.crop_config.input_size,
                    normalization=self.crop_config.normalization,
                )
            )
            inference_indices.append(len(rows) - 1)

        if not tensors:
            return rows
        batch = torch.stack(tensors, dim=0).to(device=self.device)
        outputs = self.model(batch)
        for batch_index, row_index in enumerate(inference_indices):
            row = rows[row_index]
            base_logits = outputs["base_color_logits"][batch_index]
            arrow_logit = outputs["arrow_logit"][batch_index]
            if self.state_semantics == "left_arrow":
                prediction = product_signal_attr_prediction_from_logits(
                    base_logits,
                    arrow_logit,
                    light_type=row["light_type"],
                    all_off_is_valid=self.all_off_is_valid,
                    policy=self.threshold_policy,
                )
            else:
                prediction = signal_attr_prediction_from_logits(base_logits, arrow_logit, policy=self.threshold_policy)
            row["base_color"] = prediction.base_color
            row["red_score"] = prediction.base_color_scores["red"]
            row["yellow_score"] = prediction.base_color_scores["yellow"]
            row["green_score"] = prediction.base_color_scores["green"]
            if self.state_semantics != "left_arrow":
                row["reason"] = "left_arrow_untrained"
                continue
            if row["light_type"] == "pedestrian":
                row["state_valid"] = bool(prediction.tl_attr_valid)
                row["reason"] = prediction.collapse_reason
                continue
            row["left_arrow_score"] = prediction.arrow_probability
            if prediction.tl_attr_valid:
                row["state_valid"] = True
                row["reason"] = "valid"
                row["left_arrow"] = prediction.arrow
            else:
                row["reason"] = prediction.collapse_reason
        return rows


__all__ = ["SIGNAL_CLASSES", "SignalAttrRuntime"]
