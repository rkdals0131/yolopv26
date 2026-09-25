"""Score recorded PV26 observations against manually checked frame labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import site
import tempfile

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from common.schema import ROADMARK_CLASSES, SIGNAL_CLASSES
from model.engine.evaluation import _scores
from model.engine.geometry_metrics import match_roadmark_lines


BASE_COLORS = {
    "vehicle_signal": ("off", "red", "yellow", "green"),
    "pedestrian_signal": ("off", "red", "green"),
}
STATE_NAMES = {**BASE_COLORS,
               "vehicle_signal": (*BASE_COLORS["vehicle_signal"], "left_arrow")}


def _frame_key(row: dict) -> tuple[str, str, int, int]:
    return (str(Path(row["bag"]).expanduser().resolve()), str(row["camera"]),
            int(row["bag_time_ns"]), int(row["topic_sequence"]))


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return overlap / max(first_area + second_area - overlap, 1e-12)


def _state_labels(light_type: str, row: dict | None) -> set[str]:
    if row is None or not row.get("state_valid", False):
        return set()
    color = row.get("base_color")
    if color not in BASE_COLORS[light_type]:
        raise ValueError(f"invalid state color for {light_type}: {color}")
    labels = {str(color)}
    if light_type == "vehicle_signal" and row.get("left_arrow"):
        labels.add("left_arrow")
    return labels


def evaluate(predictions: Path, labels: Path, *, iou_threshold: float,
             line_tolerance_px: float) -> dict:
    gt_by_key: dict[tuple[str, str, int, int], dict] = {}
    with labels.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = _frame_key(row)
            if key in gt_by_key:
                raise ValueError(f"duplicate annotated frame: {key}")
            for signal in row["signals"]:
                if signal["class_name"] not in SIGNAL_CLASSES:
                    raise ValueError(f"unknown signal class: {signal['class_name']}")
                box = [float(value) for value in signal["bbox_xyxy"]]
                if (len(box) != 4 or not all(math.isfinite(value) for value in box)
                        or box[2] <= box[0] or box[3] <= box[1]):
                    raise ValueError(f"invalid signal box for frame: {key}")
                signal["bbox_xyxy"] = box
                if not isinstance(signal.get("state_valid"), bool):
                    raise ValueError(f"invalid signal state validity for frame: {key}")
                if signal["state_valid"]:
                    if signal.get("base_color") not in BASE_COLORS[signal["class_name"]]:
                        raise ValueError(f"invalid signal color for frame: {key}")
                    if (signal["class_name"] == "vehicle_signal"
                            and not isinstance(signal.get("left_arrow"), bool)):
                        raise ValueError(f"vehicle left_arrow must be labeled for frame: {key}")
            for roadmark in row["roadmarks"]:
                points = [[float(x), float(y)] for x, y in roadmark["points_xy"]]
                if (roadmark["class_name"] not in ROADMARK_CLASSES or len(points) < 2
                        or any(not all(math.isfinite(v) for v in point) for point in points)):
                    raise ValueError(f"invalid roadmark line for frame: {key}")
                roadmark["points_xy"] = points
            gt_by_key[key] = row
    det_counts = {name: [0, 0, 0] for name in SIGNAL_CLASSES}
    state_counts = {kind: {name: [0, 0, 0] for name in names}
                    for kind, names in STATE_NAMES.items()}
    geometry: dict[str, Counter] = {name: Counter() for name in ROADMARK_CLASSES}
    state_context = Counter()
    seen: set[tuple[str, str, int, int]] = set()
    prediction_frames = scored_frames = skipped_ignore_frames = 0

    with predictions.open(encoding="utf-8") as source:
        for line in source:
            prediction = json.loads(line)
            prediction_frames += 1
            key = _frame_key(prediction)
            gt = gt_by_key.get(key)
            if gt is None:
                continue
            if key in seen:
                raise ValueError(f"duplicate prediction for annotated frame: {key}")
            seen.add(key)
            if gt.get("ignore_regions"):
                skipped_ignore_frames += 1
                continue  # Partial region clipping needs an explicit matching contract.
            scored_frames += 1
            observations = prediction["observation"]
            gt_signals = gt["signals"]
            predicted_signals = observations["detections"]
            height, width = observations["image_hw"]
            for target in gt_signals:
                x1, y1, x2, y2 = target["bbox_xyxy"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"signal box lies outside the source image: {key}")
            for target in gt["roadmarks"]:
                if any(not (0 <= x <= width and 0 <= y <= height)
                       for x, y in target["points_xy"]):
                    raise ValueError(f"roadmark line lies outside the source image: {key}")

            for kind in SIGNAL_CLASSES:
                truth = [row for row in gt_signals if row["class_name"] == kind]
                predicted = sorted((row for row in predicted_signals if row["class_name"] == kind),
                                   key=lambda row: row["score"], reverse=True)
                matched_truth: set[int] = set()
                matches: list[tuple[dict, dict | None]] = []
                for candidate in predicted:
                    ranked = sorted(
                        ((index, _iou(candidate["bbox_xyxy"], target["bbox_xyxy"]))
                         for index, target in enumerate(truth) if index not in matched_truth),
                        key=lambda pair: pair[1], reverse=True,
                    )
                    if ranked and ranked[0][1] >= iou_threshold:
                        index = ranked[0][0]
                        matched_truth.add(index)
                        det_counts[kind][0] += 1
                        matches.append((candidate, truth[index]))
                    else:
                        det_counts[kind][1] += 1
                        matches.append((candidate, None))
                det_counts[kind][2] += len(truth) - len(matched_truth)

                for predicted_row, truth_row in matches:
                    prediction_state = predicted_row.get("state")
                    predicted_labels = _state_labels(kind, prediction_state)
                    truth_labels = _state_labels(kind, truth_row)
                    if truth_row is None:
                        for name in predicted_labels:
                            state_counts[kind][name][1] += 1
                        continue
                    if truth_row.get("state_valid", False):
                        state_context["valid_gt"] += 1
                        if prediction_state and prediction_state.get("state_valid", False):
                            state_context["valid_on_matched"] += 1
                            if predicted_labels == truth_labels:
                                state_context["correct_on_matched"] += 1
                            if truth_row.get("base_color") == "red" and prediction_state.get("base_color") == "green":
                                state_context["red_as_green"] += 1
                        for name in truth_labels:
                            state_counts[kind][name][0 if name in predicted_labels else 2] += 1
                        for name in predicted_labels - truth_labels:
                            state_counts[kind][name][1] += 1
                    elif predicted_labels:
                        state_context["valid_prediction_on_unreadable_gt"] += 1
                for index, truth_row in enumerate(truth):
                    if index not in matched_truth and truth_row.get("state_valid", False):
                        state_context["valid_gt"] += 1
                        for name in _state_labels(kind, truth_row):
                            state_counts[kind][name][2] += 1

            line_metrics = match_roadmark_lines(
                observations["roadmarks"], gt["roadmarks"], tolerance_px=line_tolerance_px,
            )
            for name, values in line_metrics.items():
                geometry[name].update(values)

    missing = set(gt_by_key) - seen
    if missing:
        raise ValueError(f"predictions are missing {len(missing)} annotated frames")
    if not scored_frames:
        raise ValueError("no annotated frames without ignored regions were scored")
    roadmarks = {}
    for name, counts in geometry.items():
        result = _scores(*(int(counts[key]) for key in ("tp", "fp", "fn")))
        matched = int(counts["matched_count"])
        result["mean_distance_px"] = counts["matched_distance_sum"] / matched if matched else None
        result["mean_gt_coverage"] = counts["matched_gt_coverage_sum"] / matched if matched else None
        result["mean_pred_coverage"] = counts["matched_pred_coverage_sum"] / matched if matched else None
        if name == "stop_line":
            angled = int(counts["matched_angle_count"])
            result["mean_angle_error_deg"] = (
                counts["matched_angle_error_sum_deg"] / angled if angled else None
            )
        roadmarks[name] = result
    state = {}
    for kind, per_class in state_counts.items():
        state[kind] = {name: _scores(*values) for name, values in per_class.items()}
        for name, values in per_class.items():
            state[kind][name]["support"] = values[0] + values[2]
            if not state[kind][name]["support"] and not values[1]:
                state[kind][name]["f1"] = None
    scored_states = [row["f1"] for per_class in state.values() for row in per_class.values()
                     if row["f1"] is not None]
    return {
        "prediction_frames": prediction_frames,
        "annotated_frames": len(gt_by_key),
        "scored_frames": scored_frames,
        "skipped_ignore_frames": skipped_ignore_frames,
        "box_iou_threshold": iou_threshold,
        "line_tolerance_px": line_tolerance_px,
        "signal_detection": {kind: _scores(*values) for kind, values in det_counts.items()},
        "signal_detection_total": _scores(*(sum(values[i] for values in det_counts.values())
                                             for i in range(3))),
        "state": state,
        "state_macro_f1": sum(scored_states) / len(scored_states) if scored_states else None,
        "state_context": dict(state_context),
        "roadmark_lines": roadmarks,
        "roadmark_lines_total": _scores(*(sum(int(counts[key]) for counts in geometry.values())
                                          for key in ("tp", "fp", "fn"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-iou-threshold", type=float, default=0.5)
    parser.add_argument("--line-tolerance-px", type=float, default=8.0)
    args = parser.parse_args()
    if not math.isfinite(args.box_iou_threshold) or not 0 < args.box_iou_threshold <= 1:
        parser.error("box IoU threshold must be in (0, 1]")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    result = evaluate(args.predictions, args.labels,
                      iou_threshold=args.box_iou_threshold,
                      line_tolerance_px=args.line_tolerance_px)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as stream:
            staged = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
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
    print(json.dumps({"output": str(output), "scored_frames": result["scored_frames"],
                      "signal_f1": result["signal_detection_total"]["f1"],
                      "roadmark_f1": result["roadmark_lines_total"]["f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
