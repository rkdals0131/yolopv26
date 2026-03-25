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
        _append_polyline(command, lane.get("points", []), color)

    for stop_line in scene.get("stop_lines", []):
        points = stop_line.get("points") or [stop_line.get("p1", []), stop_line.get("p2", [])]
        _append_polyline(command, points, "#ff4040")

    for crosswalk in scene.get("crosswalks", []):
        _append_polygon(command, crosswalk.get("points", []), "#ff66ff")

    for ignored in scene.get("ignored_regions", []):
        _append_polygon(command, ignored.get("points", []), "#ff66ff")

    for detection in scene.get("detections", []):
        class_name = str(detection.get("class_name") or "").strip().lower()
        if class_name in {"traffic_light", "sign"}:
            continue
        _append_rectangle(command, detection.get("bbox", []), DETECTION_COLOR_MAP.get(class_name, "#ffaa00"))

    for light in scene.get("traffic_lights", []):
        _append_rectangle(command, light.get("bbox", []), "#00ff99")

    for sign in scene.get("traffic_signs", []):
        _append_rectangle(command, sign.get("bbox", []), "#00b7ff")

    command.append(str(output_path))
    subprocess.run(command, check=True, capture_output=True, text=True)
