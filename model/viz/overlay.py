from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


LANE_COLOR_MAP = {
    "white_lane": "#ffffff",
    "yellow_lane": "#ffd400",
    "blue_lane": "#3aa5ff",
}
DETECTION_COLOR_MAP = {
    "vehicle": "#ffb000",
    "bike": "#ff7a00",
    "pedestrian": "#ff4040",
    "traffic_cone": "#ff66b3",
    "obstacle": "#c84dff",
}
DEBUG_RECTANGLE_DEFAULT_COLOR = "#00e5ff"
TEXT_STROKE_COLOR = "#000000"
TEXT_POINTSIZE = "18"


def _format_points(points: list[list[float]]) -> str:
    return " ".join(f"{point[0]},{point[1]}" for point in points)


def _append_polyline(command: list[str], points: list[list[float]], color: str) -> None:
    if len(points) < 2:
        return
    command.extend(
        [
            "-fill",
            "none",
            "-stroke",
            color,
            "-strokewidth",
            "3",
            "-draw",
            f"polyline {_format_points(points)}",
        ]
    )


def _append_polygon(command: list[str], points: list[list[float]], color: str) -> None:
    if len(points) < 3:
        return
    command.extend(
        [
            "-fill",
            "none",
            "-stroke",
            color,
            "-strokewidth",
            "2",
            "-draw",
            f"polygon {_format_points(points)}",
        ]
    )


def _append_rectangle(command: list[str], bbox: list[float], color: str) -> None:
    if len(bbox) != 4:
        return
    x1, y1, x2, y2 = bbox
    command.extend(
        [
            "-fill",
            "none",
            "-stroke",
            color,
            "-strokewidth",
            "3",
            "-draw",
            f"rectangle {x1},{y1} {x2},{y2}",
        ]
    )


def _safe_label(label: str) -> str:
    return str(label).replace("'", "").replace('"', "").strip()


def _append_text(command: list[str], x_value: float, y_value: float, label: str, color: str) -> None:
    safe_label = _safe_label(label)
    if not safe_label:
        return
    x_coord = max(0.0, float(x_value))
    y_coord = max(18.0, float(y_value))
    command.extend(
        [
            "-stroke",
            TEXT_STROKE_COLOR,
            "-strokewidth",
            "2",
            "-fill",
            color,
            "-pointsize",
            TEXT_POINTSIZE,
            "-draw",
            f"text {x_coord},{y_coord} '{safe_label}'",
        ]
    )


def _label_anchor_from_points(points: list[list[float]]) -> tuple[float, float] | None:
    if not points:
        return None
    first_point = points[0]
    if len(first_point) < 2:
        return None
    return float(first_point[0]), float(first_point[1]) - 6.0


def _label_anchor_from_bbox(bbox: list[float]) -> tuple[float, float] | None:
    if len(bbox) != 4:
        return None
    return float(bbox[0]) + 4.0, float(bbox[1]) - 6.0


def render_overlay(scene: dict[str, Any], output_path: Path) -> None:
    """Render a human-readable overlay on top of the original image."""
    image_source_path = scene.get("image", {}).get("source_path") or scene.get("source", {}).get("image_path")
    if not image_source_path:
        raise KeyError("scene does not contain an image source path")
    image_path = Path(image_source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["convert", str(image_path)]

    for lane in scene.get("lanes", []):
        color = LANE_COLOR_MAP.get(lane.get("class_name"), "#aaaaaa")
        points = lane.get("points", [])
        _append_polyline(command, points, color)
        anchor = _label_anchor_from_points(points)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], str(lane.get("class_name") or "lane"), color)

    for stop_line in scene.get("stop_lines", []):
        points = stop_line.get("points") or [stop_line.get("p1", []), stop_line.get("p2", [])]
        _append_polyline(command, points, "#ff4040")
        anchor = _label_anchor_from_points(points)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], "stop_line", "#ff4040")

    for crosswalk in scene.get("crosswalks", []):
        points = crosswalk.get("points", [])
        _append_polygon(command, points, "#ff66ff")
        anchor = _label_anchor_from_points(points)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], "crosswalk", "#ff66ff")

    for ignored in scene.get("ignored_regions", []):
        points = ignored.get("points", [])
        _append_polygon(command, points, "#ff66ff")
        anchor = _label_anchor_from_points(points)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], "ignored_region", "#ff66ff")

    for detection in scene.get("detections", []):
        class_name = str(detection.get("class_name") or "").strip().lower()
        if class_name in {"traffic_light", "sign"}:
            continue
        color = DETECTION_COLOR_MAP.get(class_name, "#ffaa00")
        bbox = detection.get("bbox", [])
        _append_rectangle(command, bbox, color)
        anchor = _label_anchor_from_bbox(bbox)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], class_name or "detection", color)

    for light in scene.get("traffic_lights", []):
        bbox = light.get("bbox", [])
        _append_rectangle(command, bbox, "#00ff99")
        anchor = _label_anchor_from_bbox(bbox)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], "traffic_light", "#00ff99")

    for sign in scene.get("traffic_signs", []):
        bbox = sign.get("bbox", [])
        _append_rectangle(command, bbox, "#00b7ff")
        anchor = _label_anchor_from_bbox(bbox)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], "sign", "#00b7ff")

    for debug_rectangle in scene.get("debug_rectangles", []):
        color = str(debug_rectangle.get("color") or DEBUG_RECTANGLE_DEFAULT_COLOR)
        bbox = debug_rectangle.get("bbox", [])
        _append_rectangle(command, bbox, color)
        anchor = _label_anchor_from_bbox(bbox)
        if anchor is not None:
            _append_text(command, anchor[0], anchor[1], str(debug_rectangle.get("label") or "debug"), color)

    command.append(str(output_path))
    subprocess.run(command, check=True, capture_output=True, text=True)
