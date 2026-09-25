"""Run PV26 on recorded camera frames without starting ROS nodes."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import site
import tempfile
import time

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch

from model.engine.inference import FocusedPerception


CAMERA_TOPICS = {
    "/perception/camera/left/source/image_raw/compressed": "left",
    "/perception/camera/right/source/image_raw/compressed": "right",
}


def predict_bag(args: argparse.Namespace) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"prediction output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = FocusedPerception.from_checkpoint(
        args.checkpoint, device=args.device, signal_checkpoint=args.signal_checkpoint,
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    available = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if not any(name in available for name in CAMERA_TOPICS):
        raise ValueError("recording has no supported left/right compressed camera topic")
    for name in CAMERA_TOPICS:
        if name in available and available[name] != "sensor_msgs/msg/CompressedImage":
            raise ValueError(f"camera topic has unexpected type: {name}")
    counts: Counter[str] = Counter()
    pending: list[tuple[Image.Image, dict]] = []
    image_count = 0
    started = time.monotonic()
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as stream:
            staged = Path(stream.name)

            def flush_batch() -> None:
                if not pending:
                    return
                if pipeline.device.type == "cuda":
                    torch.cuda.synchronize(pipeline.device)
                inference_started = time.perf_counter()
                observations = pipeline.predict(
                    [image for image, _ in pending],
                    confidence=args.confidence,
                    roadmark_threshold=args.roadmark_thresholds or args.roadmark_threshold,
                    roadmark_localization=args.roadmark_localization,
                )
                if pipeline.device.type == "cuda":
                    torch.cuda.synchronize(pipeline.device)
                elapsed_ms = (time.perf_counter() - inference_started) * 1000
                for (_, frame), observation in zip(pending, observations):
                    stream.write(json.dumps({
                        **frame,
                        "checkpoint": str(args.checkpoint),
                        "signal_checkpoint": str(args.signal_checkpoint) if args.signal_checkpoint else None,
                        "inference_batch_ms": elapsed_ms,
                        "inference_batch_size": len(pending),
                        "observation": observation,
                    }, ensure_ascii=False) + "\n")
                pending.clear()

            while reader.has_next():
                topic, serialized, bag_time_ns = reader.read_next()
                camera = CAMERA_TOPICS.get(topic)
                if camera is None:
                    continue
                message = deserialize_message(serialized, CompressedImage)
                with Image.open(BytesIO(bytes(message.data))) as opened:
                    image = opened.convert("RGB")
                topic_sequence = counts[camera]
                counts[camera] += 1
                pending.append((image, {
                    "bag": str(args.bag),
                    "camera": camera,
                    "topic": topic,
                    "topic_sequence": topic_sequence,
                    "bag_time_ns": int(bag_time_ns),
                    "header_time_ns": int(message.header.stamp.sec) * 1_000_000_000
                        + int(message.header.stamp.nanosec),
                    "frame_id": message.header.frame_id,
                }))
                image_count += 1
                if len(pending) == args.batch_size:
                    flush_batch()
                if args.max_images is not None and image_count >= args.max_images:
                    break
                if image_count % 1000 == 0:
                    print(f"images={image_count} left={counts['left']} right={counts['right']}", flush=True)
            flush_batch()
            stream.flush()
            os.fsync(stream.fileno())
        # link() publishes without replacing a prior result from another run.
        os.link(staged, output)
        staged.unlink()
        staged = None
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return {"output": str(output), "images": image_count,
            "by_camera": dict(counts), "elapsed_sec": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signal-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--confidence", type=float, default=0.25)
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument("--roadmark-threshold", type=float, default=0.5)
    threshold.add_argument("--roadmark-thresholds", type=float, nargs=3,
                           metavar=("WHITE", "YELLOW", "STOP"))
    parser.add_argument("--roadmark-localization", choices=("grid", "subpixel", "smooth"), default="grid")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_images is not None and args.max_images < 1:
        parser.error("batch size and max images must be positive")
    print(json.dumps(predict_bag(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
