"""Image-to-observation inference using only models inside the PV26 package."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image
import torch

from common.schema import DEFAULT_IMAGE_HW
from model.data.dataset import letterbox_focused_image
from model.engine.postprocess import decode_focused_detections, decode_roadmark_points
from model.net.pv26 import PV26FocusedModel
from model.signal_attr.runtime import SignalAttrRuntime


class FocusedPerception:
    def __init__(self, model: PV26FocusedModel, *, image_hw=DEFAULT_IMAGE_HW,
                 device: str = "cpu", signal: SignalAttrRuntime | None = None,
                 roadmark_decode: dict | None = None) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.image_hw = tuple(image_hw)
        self.signal = signal
        self.roadmark_decode = dict(roadmark_decode or {})

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cpu",
                        signal_checkpoint: Path | None = None) -> "FocusedPerception":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = PV26FocusedModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        metadata = checkpoint.get("run_metadata") or {}
        image_hw = metadata.get("model", {}).get("image_hw", DEFAULT_IMAGE_HW)
        signal = SignalAttrRuntime.from_checkpoint(signal_checkpoint, device=device) if signal_checkpoint else None
        # The run's decoding profile travels with its weights.
        roadmark_decode = metadata.get("train", {}).get("roadmark_decode")
        return cls(model, image_hw=image_hw, device=device, signal=signal,
                   roadmark_decode=roadmark_decode)

    @torch.inference_mode()
    def predict(self, images: Sequence[Image.Image], *, confidence: float = 0.25,
                roadmark_threshold: float | Sequence[float] = 0.5,
                roadmark_localization: str = "grid") -> list[dict]:
        if not images:
            return []
        prepared = [letterbox_focused_image(image, self.image_hw) for image in images]
        batch = torch.stack([tensor for tensor, _ in prepared]).to(self.device)
        meta = [metadata for _, metadata in prepared]
        outputs = self.model(batch)
        detections = decode_focused_detections(outputs["det"], meta, conf_threshold=confidence)
        roadmarks = decode_roadmark_points(outputs["roadmark_logits"], meta,
                                            threshold=roadmark_threshold,
                                            localization=roadmark_localization,
                                            **self.roadmark_decode)
        results = []
        for image, image_meta, boxes, lines in zip(images, meta, detections, roadmarks):
            for detection_id, detection in enumerate(boxes):
                detection["id"] = detection_id
            if self.signal is not None:
                attributes = {row["detection_id"]: row for row in self.signal.predict(image, boxes)}
                for detection in boxes:
                    detection["state"] = attributes[detection["id"]]
            for line_id, line in enumerate(lines):
                line["id"] = line_id
            results.append({"image_hw": list(image_meta["raw_hw"]), "detections": boxes, "roadmarks": lines})
        return results
