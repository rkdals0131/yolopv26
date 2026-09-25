from __future__ import annotations

import argparse
import json
from pathlib import Path
import site
import time

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch

from common.io import write_json
from model.engine.inference import FocusedPerception


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict signal states and roadmark polylines from RGB images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signal-checkpoint", type=Path)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument("--roadmark-threshold", type=float, default=0.5)
    threshold.add_argument("--roadmark-thresholds", type=float, nargs=3,
                           metavar=("WHITE", "YELLOW", "STOP"))
    parser.add_argument("--roadmark-localization", choices=("grid", "subpixel", "smooth"), default="grid")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    pipeline = FocusedPerception.from_checkpoint(args.checkpoint, device=args.device,
                                                signal_checkpoint=args.signal_checkpoint)
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if pipeline.device.type == "cuda":
        torch.cuda.synchronize(pipeline.device)
    started = time.perf_counter()
    observations = pipeline.predict(images, confidence=args.confidence,
                                    roadmark_threshold=args.roadmark_thresholds or args.roadmark_threshold,
                                    roadmark_localization=args.roadmark_localization)
    if pipeline.device.type == "cuda":
        torch.cuda.synchronize(pipeline.device)
    result = {"images": [str(path) for path in args.images], "observations": observations,
              "processing_ms": (time.perf_counter() - started) * 1000}
    if args.output:
        write_json(args.output, result, ensure_ascii=False)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.overlay:
        from PIL import ImageDraw
        args.overlay.mkdir(parents=True, exist_ok=True)
        colors = {"white_lane": "cyan", "yellow_lane": "yellow", "stop_line": "red"}
        for index, (source, observation) in enumerate(zip(images, observations)):
            draw = ImageDraw.Draw(source)
            for detection in observation["detections"]:
                draw.rectangle(detection["bbox_xyxy"], outline="lime", width=2)
            for line in observation["roadmarks"]:
                draw.line([tuple(point) for point in line["points_xy"]], fill=colors[line["class_name"]], width=2)
            source.save(args.overlay / f"{index:02d}_{args.images[index].stem}.png")


if __name__ == "__main__":
    main()
