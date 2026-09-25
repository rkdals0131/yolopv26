"""Write golden fixtures for the C++ PV26 runtime (pv26_perception) from camera frames.

Each case stores the source JPEG, the PV26 road-mark probabilities (float16,
[3,H/4,W/4]), the raw detection rows ([N,6] float32, network pixels) and the
Python reference decodings. Road marks are decoded from the stored float16
probabilities, so the C++ decoder can be compared on identical input values.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
import site
import sys

site.addsitedir(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from model.data.dataset import letterbox_focused_image
from model.engine.postprocess import decode_focused_detections, decode_roadmark_probabilities
from model.net.pv26 import PV26FocusedModel
from model.signal_attr.classifier import (
    load_signal_attr_classifier_checkpoint, product_signal_attr_prediction_from_logits,
    signal_attr_crop_image_to_tensor)
from model.signal_attr.crop import SignalAttrCropConfig, crop_signal_attr_roi
from model.signal_attr.runtime import SIGNAL_CLASSES

LEGACY_PROFILE = {"threshold": [0.5, 0.5, 0.5], "min_points": 6, "orientation": "fixed",
                  "max_gap": [3, 3, 3], "min_score": [0.0, 0.0, 0.0], "min_length_px": [0.0, 0.0, 0.0]}


def read_frames(bag: Path, topics: list[str], every: int, count: int) -> list[tuple[str, int, bytes]]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    seen = {topic: 0 for topic in topics}
    frames = []
    while reader.has_next() and len(frames) < count:
        topic, data, _ = reader.read_next()
        seen[topic] += 1
        if seen[topic] % every:
            continue
        message = deserialize_message(data, CompressedImage)
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        data = bytes(message.data)
        # usb_cam publishes the whole MJPEG buffer; keep the JPEG, drop its zero padding.
        end = data.rfind(b"\xff\xd9")
        frames.append((topic, stamp, data[:end + 2] if end >= 0 else data))
    return frames


def lines_json(lines: list[dict]) -> list[dict]:
    return [{"class_id": line["class_id"], "score": line["score"], "points_xy": line["points_xy"]}
            for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="pv26.manifest.json of the export")
    parser.add_argument("--signal-checkpoint", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=360)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    profile = {key: manifest["roadmark_decode"][key] for key in LEGACY_PROFILE}
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PV26FocusedModel(**payload["model_config"])
    model.load_state_dict(payload["model"])
    model.eval()
    image_hw = tuple(manifest["input"]["image_hw"])
    signal = load_signal_attr_classifier_checkpoint(args.signal_checkpoint, device="cpu")
    crop_config = SignalAttrCropConfig(**signal["crop_config"])
    signal_payload = signal["payload"]
    topics = ["/perception/camera/left/source/image_raw/compressed",
              "/perception/camera/right/source/image_raw/compressed"]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, (topic, stamp, jpeg) in enumerate(read_frames(args.bag, topics, args.every, args.count)):
        name = f"case_{index:02d}"
        (args.output / f"{name}.jpg").write_bytes(jpeg)

        image = Image.open(BytesIO(jpeg)).convert("RGB")
        tensor, meta = letterbox_focused_image(image, image_hw)
        with torch.no_grad():
            detections, logits = model.forward_export(tensor.unsqueeze(0))
        probability = logits.float().sigmoid()[0].numpy().astype(np.float16)
        probability.tofile(args.output / f"{name}.prob_f16.bin")
        detections[0].numpy().astype(np.float32).tofile(args.output / f"{name}.det_f32.bin")
        decoded_input = probability.astype(np.float32)[None]
        expected = {
            "manifest": lines_json(decode_roadmark_probabilities(decoded_input, [meta], **profile)[0]),
            "legacy": lines_json(decode_roadmark_probabilities(decoded_input, [meta], **LEGACY_PROFILE)[0]),
        }
        expected_detections = decode_focused_detections(
            detections, [meta], conf_threshold=float(manifest["detection_confidence"]))[0]
        signal_rows = []
        for crop_index, detection in enumerate(expected_detections):
            crop = crop_signal_attr_roi(image, detection["bbox_xyxy"], config=crop_config)
            row = {"detection_index": crop_index, "crop_box": list(crop.crop_box) if crop.valid else None}
            if crop.valid:
                np.asarray(crop.crop_image, dtype=np.uint8).tofile(
                    args.output / f"{name}.crop{crop_index}_u8.bin")
                # PIL-decoded source pixels of the crop box, so the resize can be
                # checked exactly, independent of the JPEG decoder.
                np.asarray(image.crop(crop.crop_box), dtype=np.uint8).tofile(
                    args.output / f"{name}.crop{crop_index}_src_u8.bin")
                with torch.no_grad():
                    outputs = signal["model"](signal_attr_crop_image_to_tensor(
                        crop.crop_image, input_size=crop_config.input_size,
                        normalization=crop_config.normalization).unsqueeze(0))
                base_logits = outputs["base_color_logits"][0]
                arrow_logit = outputs["arrow_logit"][0]
                prediction = product_signal_attr_prediction_from_logits(
                    base_logits, arrow_logit, light_type=SIGNAL_CLASSES[detection["class_name"]],
                    all_off_is_valid=bool(signal_payload.get("all_off_is_valid", False)),
                    policy=signal["threshold_policy"])
                row.update(base_color_logits=base_logits.tolist(), arrow_logit=float(arrow_logit),
                           valid=bool(prediction.tl_attr_valid), reason=prediction.collapse_reason,
                           base_color=prediction.base_color, arrow=prediction.arrow)
            signal_rows.append(row)
        cases.append({
            "name": name, "topic": topic, "stamp_ns": stamp,
            "raw_hw": list(meta["raw_hw"]), "transform": meta["transform"],
            "probability_shape": list(probability.shape), "detection_shape": list(detections[0].shape),
            "expected_lines": expected,
            "expected_detections": [{"class_id": d["class_id"], "score": d["score"],
                                     "bbox_xyxy": d["bbox_xyxy"]} for d in expected_detections],
            "expected_signal_attr": signal_rows,
        })
        print(json.dumps({"case": name, "lines": {k: len(v) for k, v in expected.items()},
                          "detections": len(expected_detections)}), flush=True)
    (args.output / "fixtures.json").write_text(json.dumps({
        "format_version": 1, "checkpoint_sha256": manifest["checkpoint_sha256"],
        "profiles": {"manifest": profile, "legacy": LEGACY_PROFILE},
        "detection_confidence": manifest["detection_confidence"],
        "image_hw": list(image_hw), "cases": cases}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
