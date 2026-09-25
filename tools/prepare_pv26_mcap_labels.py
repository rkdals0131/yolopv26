"""Extract fixed-time camera frames for manual MCAP annotation."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import site

from PIL import Image

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from tools.predict_pv26_mcap import CAMERA_TOPICS


PERIOD_NS = 500_000_000  # One image per camera every 0.5 seconds.


def prepare(args: argparse.Namespace) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"annotation output already exists: {output}")
    event_windows = []
    if args.event_windows is not None:
        with args.event_windows.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                start_ns, end_ns = int(row["start_ns"]), int(row["end_ns"])
                if end_ns < start_ns:
                    raise ValueError("event window end precedes start")
                event_windows.append((start_ns, end_ns))
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    available = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if not any(topic in available for topic in CAMERA_TOPICS):
        raise ValueError("recording has no supported left/right compressed camera topic")
    for name in CAMERA_TOPICS:
        if name in available and available[name] != "sensor_msgs/msg/CompressedImage":
            raise ValueError(f"camera topic has unexpected type: {name}")

    output.mkdir(parents=True)
    counts: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    last_selected: dict[str, int] = {}
    previous: dict[str, tuple[str, bytes, int, int]] = {}
    next_grid: dict[str, int] = {}
    started_at: int | None = None
    manifest_stage = output / ".frames.jsonl.tmp"
    try:
        with manifest_stage.open("w", encoding="utf-8") as manifest:
            def publish(camera: str, frame: tuple[str, bytes, int, int]) -> None:
                topic, data, timestamp, sequence = frame
                if last_selected.get(camera) == sequence:
                    return
                message = deserialize_message(data, CompressedImage)
                content = bytes(message.data)
                with Image.open(BytesIO(content)) as image:
                    image.load()
                    width, height = image.size
                    suffix = ".jpg" if image.format == "JPEG" else ".png" if image.format == "PNG" else None
                if suffix is None:
                    raise ValueError(f"unsupported compressed camera format: {message.format}")
                destination = output / "images" / camera / f"{timestamp}_{sequence:06d}{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as image_file:
                    image_file.write(content)
                    image_file.flush()
                    os.fsync(image_file.fileno())
                manifest.write(json.dumps({
                    "bag": str(args.bag.expanduser().resolve()),
                    "camera": camera,
                    "topic": topic,
                    "topic_sequence": sequence,
                    "bag_time_ns": timestamp,
                    "header_time_ns": int(message.header.stamp.sec) * 1_000_000_000
                        + int(message.header.stamp.nanosec),
                    "frame_id": message.header.frame_id,
                    "image_hw": [height, width],
                    "image": str(destination.relative_to(output)),
                }, ensure_ascii=False) + "\n")
                selected[camera] += 1
                last_selected[camera] = sequence

            while reader.has_next():
                topic, serialized, timestamp = reader.read_next()
                if started_at is None:
                    started_at = int(timestamp)
                camera = CAMERA_TOPICS.get(topic)
                if camera is None:
                    continue
                if camera not in next_grid:
                    next_grid[camera] = started_at
                sequence = counts[camera]
                counts[camera] += 1
                current = (topic, serialized, int(timestamp), sequence)
                while next_grid[camera] <= timestamp:
                    older = previous.get(camera)
                    choice = (older if older is not None
                              and next_grid[camera] - older[2] <= timestamp - next_grid[camera]
                              else current)
                    publish(camera, choice)
                    next_grid[camera] += PERIOD_NS
                    if args.max_images is not None and sum(selected.values()) >= args.max_images:
                        break
                previous[camera] = current
                if args.max_images is not None and sum(selected.values()) >= args.max_images:
                    break
                if any(start_ns <= timestamp <= end_ns for start_ns, end_ns in event_windows):
                    publish(camera, current)
                if args.max_images is not None and sum(selected.values()) >= args.max_images:
                    break
            if not selected:
                raise ValueError("no camera frame was selected")
            manifest.flush()
            os.fsync(manifest.fileno())
        for directory in (output / "images" / camera for camera in selected):
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        directory_fd = os.open(output / "images", os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(manifest_stage, output / "frames.jsonl")
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(output)
        raise
    return {"output": str(output), "selected": dict(selected),
            "source_messages": dict(counts), "period_ns": PERIOD_NS,
            "event_windows": len(event_windows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-windows", type=Path,
                        help="JSONL time windows whose camera frames are all included.")
    parser.add_argument("--max-images", type=int, help="Bounded reader check; omit for all frames.")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    print(json.dumps(prepare(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
