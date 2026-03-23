from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on external environment.
    Image = None

LANE_DATASET_KEY = "aihub_lane_source"
TRAFFIC_DATASET_KEY = "aihub_traffic_source"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
VALID_TL_STATES = {"red", "yellow", "green", "off", "unknown"}


@dataclass(frozen=True)
class PairRecord:
    dataset_key: str
    dataset_root: Path
    split: str
    image_path: Path | None
    label_path: Path
    image_file_name: str
    relative_id: str


@dataclass
class DiscoveryReport:
    dataset_key: str
    dataset_root: Path
    image_counts: Counter[str] = field(default_factory=Counter)
    label_counts: Counter[str] = field(default_factory=Counter)
    pairs: list[PairRecord] = field(default_factory=list)
    missing_images: list[dict[str, Any]] = field(default_factory=list)
    missing_labels: list[dict[str, Any]] = field(default_factory=list)
    label_parse_failures: list[dict[str, Any]] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._") or "sample"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _repo_root() -> Path:
    env_value = os.environ.get("PV26_REPO_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _seg_dataset_root(repo_root: Path | None = None) -> Path:
    env_value = os.environ.get("PV26_SEG_DATASET_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root or _repo_root()) / "seg_dataset"


def _env_path(name: str, default: Path) -> Path:
    env_value = os.environ.get(name)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return default.resolve()


def _probe_image_size(path: Path) -> tuple[int, int]:
    if Image is not None:
        try:
            with Image.open(path) as image:
                return int(image.width), int(image.height)
        except Exception:
            pass
    result = subprocess.run(
        ["identify", "-format", "%w %h", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    width_text, height_text = result.stdout.strip().split()
    return int(width_text), int(height_text)


def _parse_imsize(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
        if width is not None and height is not None:
            return int(width), int(height)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
    return None


def _primary_image_path(dataset_key: str, path: Path) -> bool:
    raw_parts = [str(part).lower() for part in path.parts]
    if dataset_key == TRAFFIC_DATASET_KEY:
        if any("crop" in part for part in raw_parts):
            return False
        if any(part.startswith("result_") for part in raw_parts):
            return False
        return any("원천" in part or "source" in part or "images" == part for part in raw_parts)
    return True


def _extract_filename(raw: dict[str, Any], fallback_name: str) -> str:
    image_section = raw.get("image")
    if isinstance(image_section, dict):
        for key in ("filename", "file_name", "name"):
            if image_section.get(key):
                return Path(str(image_section[key])).name
    for key in ("filename", "file_name", "name", "image_file"):
        if raw.get(key):
            return Path(str(raw[key])).name
    return fallback_name


def _extract_image_size(raw: dict[str, Any], image_path: Path) -> tuple[int, int]:
    image_section = raw.get("image")
    parsed_size = None
    if isinstance(image_section, dict):
        for key in ("imsize", "image_size"):
            parsed_size = _parse_imsize(image_section.get(key))
            if parsed_size is not None:
                break
        if parsed_size is None:
            parsed_size = _parse_imsize(image_section)
    if parsed_size is None:
        parsed_size = _parse_imsize(raw.get("imsize"))
    if parsed_size is None:
        parsed_size = _parse_imsize(raw.get("image_size"))
    try:
        actual_size = _probe_image_size(image_path)
    except Exception:
        if parsed_size is not None:
            return parsed_size
        raise
    if parsed_size is None:
        return actual_size
    if parsed_size == actual_size or parsed_size[::-1] == actual_size:
        return actual_size
    return actual_size


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_split(path: Path) -> str:
    split_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    for part in path.parts:
        normalized = _normalize_text(part)
        if normalized in split_map:
            return split_map[normalized]
    return "unspecified"


def _path_similarity(label_path: Path, image_path: Path) -> int:
    label_parts = [_normalize_text(part) for part in label_path.parts[:-1]]
    image_parts = [_normalize_text(part) for part in image_path.parts[:-1]]
    score = 0
    for left, right in zip(reversed(label_parts), reversed(image_parts)):
        if left != right:
            break
        score += 1
    if label_path.stem == image_path.stem:
        score += 100
    return score


def _extract_minimal_label_metadata(label_path: Path) -> tuple[str, str, dict[str, Any]]:
    raw = _load_json(label_path)
    filename = _extract_filename(raw, f"{label_path.stem}.jpg")
    split = _infer_split(label_path)
    return filename, split, raw


def _discover_pairs(dataset_key: str, dataset_root: Path) -> DiscoveryReport:
    report = DiscoveryReport(dataset_key=dataset_key, dataset_root=dataset_root)
    image_candidates: dict[str, list[Path]] = defaultdict(list)
    matched_images: set[Path] = set()

    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and _primary_image_path(dataset_key, path):
            split = _infer_split(path)
            report.image_counts[split] += 1
            image_candidates[path.name].append(path)

    label_paths = sorted(path for path in dataset_root.rglob("*.json") if path.is_file())
    for label_path in label_paths:
        split = _infer_split(label_path)
        report.label_counts[split] += 1
        try:
            filename, label_split, _raw = _extract_minimal_label_metadata(label_path)
        except Exception as exc:
            report.label_parse_failures.append(
                {
                    "label_path": str(label_path),
                    "split": split,
                    "error": str(exc),
                }
            )
            continue

        split = label_split or split
        candidates = image_candidates.get(filename, [])
        image_path = None
        if candidates:
            image_path = max(candidates, key=lambda candidate: _path_similarity(label_path, candidate))
            matched_images.add(image_path)

        relative_id = _safe_slug(str(label_path.relative_to(dataset_root).with_suffix("")))
        if image_path is None:
            report.missing_images.append(
                {
                    "split": split,
                    "image_file_name": filename,
                    "label_path": str(label_path),
                    "relative_id": relative_id,
                }
            )
            continue

        report.pairs.append(
            PairRecord(
                dataset_key=dataset_key,
                dataset_root=dataset_root,
                split=split,
                image_path=image_path,
                label_path=label_path,
                image_file_name=filename,
                relative_id=relative_id,
            )
        )

    for image_name, candidates in image_candidates.items():
        for candidate in candidates:
            if candidate in matched_images:
                continue
            report.missing_labels.append(
                {
                    "split": _infer_split(candidate),
                    "image_file_name": image_name,
                    "image_path": str(candidate),
                    "relative_id": _safe_slug(str(candidate.relative_to(dataset_root).with_suffix(""))),
                }
            )

    return report


def _clean_points(points: Iterable[tuple[float, float]]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for x_value, y_value in points:
        point = [round(float(x_value), 3), round(float(y_value), 3)]
        if cleaned and cleaned[-1] == point:
            continue
        cleaned.append(point)
    if len(cleaned) >= 2 and cleaned[0][1] < cleaned[-1][1]:
        cleaned.reverse()
    return cleaned


def _extract_points(annotation: dict[str, Any]) -> list[list[float]]:
    for geometry_key in ("polyline", "polygon"):
        geometry = annotation.get(geometry_key)
        if isinstance(geometry, dict):
            x_values = geometry.get("x")
            y_values = geometry.get("y")
            if isinstance(x_values, list) and isinstance(y_values, list):
                return _clean_points(zip(x_values, y_values))
            points = geometry.get("points")
            if isinstance(points, list):
                return _extract_points({"points": points})

    x_values = annotation.get("x")
    y_values = annotation.get("y")
    if isinstance(x_values, list) and isinstance(y_values, list):
        return _clean_points(zip(x_values, y_values))

    points = annotation.get("points")
    if isinstance(points, list):
        extracted: list[tuple[float, float]] = []
        for point in points:
            if isinstance(point, dict):
                x_value = point.get("x")
                y_value = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x_value, y_value = point[0], point[1]
            else:
                continue
            extracted.append((float(x_value), float(y_value)))
        return _clean_points(extracted)

    data = annotation.get("data")
    if isinstance(data, list):
        return _extract_points({"points": data})

    return []


def _extract_attribute_map(annotation: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    raw_attributes = annotation.get("attributes")
    if isinstance(raw_attributes, list):
        for item in raw_attributes:
            if not isinstance(item, dict):
                continue
            code = _normalize_text(item.get("code") or item.get("key") or item.get("name"))
            if code:
                attributes[code] = item.get("value")
    elif isinstance(raw_attributes, dict):
        for key, value in raw_attributes.items():
            normalized = _normalize_text(key)
            if normalized:
                attributes[normalized] = value

    if annotation.get("color") is not None:
        attributes.setdefault("lane_color", annotation.get("color"))
    if annotation.get("type") is not None:
        attributes.setdefault("lane_type", annotation.get("type"))
    return attributes


def _extract_bbox(annotation: dict[str, Any], width: int, height: int) -> list[float] | None:
    box = annotation.get("box") or annotation.get("bbox")
    if isinstance(box, dict):
        if all(key in box for key in ("x1", "y1", "x2", "y2")):
            coords = [box["x1"], box["y1"], box["x2"], box["y2"]]
        elif all(key in box for key in ("left", "top", "right", "bottom")):
            coords = [box["left"], box["top"], box["right"], box["bottom"]]
        elif all(key in box for key in ("x", "y", "w", "h")):
            coords = [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]]
        else:
            return None
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        coords = [float(value) for value in box]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            coords = [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]]
    elif isinstance(box, (list, tuple)) and len(box) == 2:
        first, second = box
        if (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) >= 2
            and len(second) >= 2
        ):
            coords = [float(first[0]), float(first[1]), float(second[0]), float(second[1])]
        else:
            return None
    else:
        return None

    x1 = max(0.0, min(float(coords[0]), float(width)))
    y1 = max(0.0, min(float(coords[1]), float(height)))
    x2 = max(0.0, min(float(coords[2]), float(width)))
    y2 = max(0.0, min(float(coords[3]), float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _extract_annotations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("annotation", "annotations", "labels", "objects"):
        container = raw.get(key)
        if isinstance(container, list):
            return [item for item in container if isinstance(item, dict)]
        if isinstance(container, dict):
            flattened: list[dict[str, Any]] = []
            for inferred_class, value in container.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            copied = dict(item)
                            copied.setdefault("class", inferred_class)
                            flattened.append(copied)
                elif isinstance(value, dict):
                    copied = dict(value)
                    copied.setdefault("class", inferred_class)
                    flattened.append(copied)
            if flattened:
                return flattened
    if any(key in raw for key in ("class", "color", "box", "bbox", "polyline", "polygon", "x", "y")):
        return [raw]
    return []


def _extract_tl_state(annotation: dict[str, Any]) -> str:
    for key in ("state", "signal_state", "light_state", "color"):
        normalized = _normalize_text(annotation.get(key))
        if normalized in VALID_TL_STATES:
            return normalized

    attribute = annotation.get("attribute")
    if isinstance(attribute, dict):
        for key, value in attribute.items():
            normalized_key = _normalize_text(key)
            normalized_value = _normalize_text(value)
            if normalized_key in VALID_TL_STATES and str(value).lower() not in {"0", "false", ""}:
                return normalized_key
            if normalized_value in VALID_TL_STATES:
                return normalized_value
    if isinstance(attribute, list):
        for item in attribute:
            normalized = _normalize_text(item)
            if normalized in VALID_TL_STATES:
                return normalized
            if isinstance(item, dict):
                for key, value in item.items():
                    normalized_key = _normalize_text(key)
                    normalized_value = _normalize_text(value)
                    if normalized_key in VALID_TL_STATES and str(value).lower() not in {"0", "false", ""}:
                        return normalized_key
                    if normalized_value in VALID_TL_STATES:
                        return normalized_value
    normalized = _normalize_text(attribute)
    if normalized in VALID_TL_STATES:
        return normalized
    return "unknown"


__all__ = [
    "DiscoveryReport",
    "LANE_DATASET_KEY",
    "PairRecord",
    "TRAFFIC_DATASET_KEY",
    "_env_path",
    "_discover_pairs",
    "_extract_annotations",
    "_extract_attribute_map",
    "_extract_bbox",
    "_extract_filename",
    "_extract_image_size",
    "_extract_points",
    "_extract_tl_state",
    "_load_json",
    "_normalize_text",
    "_now_iso",
    "_repo_root",
    "_safe_slug",
    "_seg_dataset_root",
]
