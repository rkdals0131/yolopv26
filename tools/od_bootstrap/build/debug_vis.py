from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Callable, Mapping, Sequence, TypedDict

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is expected in the repo test env.
    Image = None

from common.io import now_iso as _now_iso
from common.io import write_json as _write_json
from common.overlay import render_overlay
from .review import canonical_scene_to_overlay_scene
from .final_dataset import FINAL_DATASET_MANIFEST_NAME
from .image_list import ImageListEntry, build_sample_uid, load_image_list


DEFAULT_DEBUG_VIS_COUNT = 20
DEFAULT_DEBUG_VIS_SEED = 26
DEFAULT_FINAL_LANE_AUDIT_OVERVIEW_COUNT = 120
DEFAULT_FINAL_LANE_AUDIT_BIN_COUNT = 32
DEFAULT_FINAL_LANE_AUDIT_SAMPLES_PER_BIN = 4
DEFAULT_FINAL_LANE_AUDIT_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_FINAL_LANE_AUDIT_DIRNAME = "debug_vis_lane_audit"
DEFAULT_FINAL_LANE_AUDIT_OVERVIEW_QUOTAS: tuple[tuple[str, str, int], ...] = (
    ("aihub_lane_seoul", "train", 24),
    ("aihub_lane_seoul", "val", 24),
    ("pv26_exhaustive_aihub_traffic_seoul", "train", 16),
    ("pv26_exhaustive_aihub_traffic_seoul", "val", 8),
    ("pv26_exhaustive_aihub_obstacle_seoul", "train", 16),
    ("pv26_exhaustive_aihub_obstacle_seoul", "val", 8),
    ("pv26_exhaustive_bdd100k_det_100k", "train", 8),
    ("pv26_exhaustive_bdd100k_det_100k", "val", 8),
    ("pv26_exhaustive_bdd100k_det_100k", "test", 8),
)
_TRAILING_DIGITS_RE = re.compile(r"(\d+)(?!.*\d)")


class DebugSelectionRow(TypedDict, total=False):
    dataset_key: str
    dataset_root: str
    det_path: str | None
    image_path: str
    output_label_path: str
    sample_id: str
    sample_uid: str
    scene_path: str
    source_dataset_key: str
    source_image_path: str
    split: str

class OverlayImage(TypedDict):
    source_path: str
class OverlayItem(TypedDict, total=False):
    bbox: list[float]
    class_name: str


class OverlayPolylineItem(TypedDict, total=False):
    class_name: str
    points: list[list[float]]
    p1: list[float]
    p2: list[float]


class OverlayPolygonItem(TypedDict):
    points: list[list[float]]


class DebugRectangleItem(TypedDict, total=False):
    bbox: list[float]
    color: str
    label: str


class OverlayScene(TypedDict):
    image: OverlayImage
    detections: list[OverlayItem]
    traffic_lights: list[OverlayItem]
    traffic_signs: list[OverlayItem]
    lanes: list[OverlayPolylineItem]
    stop_lines: list[OverlayPolylineItem]
    crosswalks: list[OverlayPolygonItem]
    ignored_regions: list[OverlayPolygonItem]
    debug_rectangles: list[DebugRectangleItem]


class DebugVisItem(TypedDict, total=False):
    dataset_key: str
    overlay_path: str
    sample_id: str
    sample_uid: str
    source_det_path: str | None
    source_image_path: str
    source_label_path: str
    source_scene_path: str
    split: str
    teacher_name: str


class DebugVisPayload(TypedDict, total=False):
    class_names: list[str]
    dataset_root: str
    generated_at: str
    image_list_manifest_path: str
    items: list[DebugVisItem]
    seed: int
    selection_count: int
    teacher_name: str
    version: str


class DebugVisResult(TypedDict):
    debug_vis_dir: Path
    debug_vis_manifest: Path
    selection_count: int


class FinalLaneAuditItem(TypedDict, total=False):
    bucket_dir: str
    category: str
    crosswalk_count: int
    dataset_key: str
    frame_id: int
    group_key: str
    group_slug: str
    lane_count: int
    overlay_path: str
    pick_index: int
    sample_id: str
    sample_uid: str
    source_image_path: str
    source_raw_id: str
    source_scene_path: str
    split: str
    stop_line_count: int
    bin_index: int
    bin_count: int


class FinalLaneAuditResult(TypedDict):
    output_root: Path
    index_path: Path
    summary_path: Path
    selection_count: int


def _selection_str(row: Mapping[str, object] | DebugSelectionRow, key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _selection_optional_str(row: Mapping[str, object] | DebugSelectionRow, key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _debug_selection_row_from_mapping(row: Mapping[str, object] | DebugSelectionRow) -> DebugSelectionRow:
    normalized: DebugSelectionRow = {}
    for key in (
        "dataset_key",
        "dataset_root",
        "image_path",
        "output_label_path",
        "sample_id",
        "sample_uid",
        "scene_path",
        "source_dataset_key",
        "source_image_path",
        "split",
    ):
        text = _selection_str(row, key)
        if text:
            normalized[key] = text
    det_path = _selection_optional_str(row, "det_path")
    if det_path is not None:
        normalized["det_path"] = det_path
    return normalized


def _debug_selection_row_from_image_list_entry(entry: ImageListEntry) -> DebugSelectionRow:
    row: DebugSelectionRow = {
        "dataset_key": entry.dataset_key,
        "dataset_root": str(entry.dataset_root),
        "image_path": str(entry.image_path),
        "sample_id": entry.sample_id,
        "sample_uid": entry.sample_uid,
        "scene_path": str(entry.scene_path),
        "split": entry.split,
    }
    if entry.det_path is not None:
        row["det_path"] = str(entry.det_path)
    return row


def _reset_debug_vis_dir(debug_vis_dir: Path) -> None:
    if debug_vis_dir.exists():
        shutil.rmtree(debug_vis_dir)
    debug_vis_dir.mkdir(parents=True, exist_ok=True)


def _stable_selection_key(seed: int, row: Mapping[str, object] | DebugSelectionRow) -> tuple[str, str, str]:
    dataset_key = _selection_str(row, "dataset_key") or _selection_str(row, "source_dataset_key")
    split = _selection_str(row, "split")
    sample_id = _selection_str(row, "sample_uid") or _selection_str(row, "sample_id")
    digest = hashlib.sha256(f"{seed}:{dataset_key}:{split}:{sample_id}".encode("utf-8")).hexdigest()
    return digest, split, sample_id


def _copy_debug_row(row: Mapping[str, object] | DebugSelectionRow) -> DebugSelectionRow:
    return _debug_selection_row_from_mapping(row)


def _select_debug_rows(rows: Sequence[DebugSelectionRow], *, count: int, seed: int) -> list[DebugSelectionRow]:
    if count <= 0 or not rows:
        return []
    grouped: dict[str, list[DebugSelectionRow]] = defaultdict(list)
    for row in rows:
        dataset_key = _selection_str(row, "dataset_key") or _selection_str(row, "source_dataset_key") or "default"
        grouped[dataset_key].append(_copy_debug_row(row))
    for dataset_key, items in grouped.items():
        items.sort(key=lambda item: _stable_selection_key(seed, {"dataset_key": dataset_key, **item}))
    selected: list[DebugSelectionRow] = []
    remaining = min(int(count), len(rows))
    active_keys = sorted(key for key, items in grouped.items() if items)
    while remaining > 0 and active_keys:
        next_active: list[str] = []
        for dataset_key in active_keys:
            bucket = grouped[dataset_key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            remaining -= 1
            if bucket:
                next_active.append(dataset_key)
            if remaining == 0:
                break
        active_keys = next_active
    return selected


def _empty_overlay_scene(*, image_path: Path) -> OverlayScene:
    return {
        "image": {"source_path": str(image_path.resolve())},
        "detections": [],
        "traffic_lights": [],
        "traffic_signs": [],
        "lanes": [],
        "stop_lines": [],
        "crosswalks": [],
        "ignored_regions": [],
        "debug_rectangles": [],
    }


def _yolo_row_to_xyxy(values: list[float], *, width: int, height: int) -> list[float]:
    center_x = float(values[0]) * float(width)
    center_y = float(values[1]) * float(height)
    box_w = float(values[2]) * float(width)
    box_h = float(values[3]) * float(height)
    half_w = box_w * 0.5
    half_h = box_h * 0.5
    return [
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    ]


def _teacher_overlay_scene(*, image_path: Path, label_path: Path, class_names: Sequence[str]) -> OverlayScene:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required to render teacher debug overlays")
    with Image.open(image_path) as image:
        width, height = image.size
    overlay_scene = _empty_overlay_scene(image_path=image_path)
    if not label_path.is_file():
        return overlay_scene
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        class_index = int(parts[0])
        if not 0 <= class_index < len(class_names):
            continue
        class_name = str(class_names[class_index])
        bbox = _yolo_row_to_xyxy([float(value) for value in parts[1:]], width=width, height=height)
        if class_name == "traffic_light":
            overlay_scene["traffic_lights"].append({"bbox": bbox})
            continue
        if class_name == "sign":
            overlay_scene["traffic_signs"].append({"bbox": bbox})
            continue
        overlay_scene["detections"].append({"class_name": class_name, "bbox": bbox})
    return overlay_scene


def _resolve_sample_uid(row: Mapping[str, object] | DebugSelectionRow) -> str:
    sample_uid = _selection_str(row, "sample_uid")
    if sample_uid:
        return sample_uid
    return build_sample_uid(
        dataset_key=_selection_str(row, "dataset_key") or _selection_str(row, "source_dataset_key") or "default",
        split=_selection_str(row, "split") or "unknown",
        sample_id=_selection_str(row, "sample_id") or "sample",
    )


def _overlay_path_for_row(debug_vis_dir: Path, row: Mapping[str, object] | DebugSelectionRow) -> Path:
    return debug_vis_dir / f"{_resolve_sample_uid(row)}.png"


def _resolve_canonical_dataset_root(row: Mapping[str, object] | DebugSelectionRow, *, canonical_root: Path) -> Path:
    dataset_root = _selection_str(row, "dataset_root")
    if dataset_root:
        return Path(dataset_root).resolve()
    dataset_key = _selection_str(row, "dataset_key") or _selection_str(row, "source_dataset_key")
    if dataset_key == "bdd100k_det_100k":
        return (canonical_root / "bdd100k_det_100k").resolve()
    return (canonical_root / "aihub_standardized").resolve()


def _render_canonical_debug_item(row: DebugSelectionRow, *, output_root: Path) -> DebugVisItem:
    image_path = Path(str(row["image_path"])).resolve()
    scene_path = Path(str(row["scene_path"])).resolve()
    overlay_path = _overlay_path_for_row(output_root, row)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    render_overlay(canonical_scene_to_overlay_scene(scene, image_path=image_path), overlay_path)
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "sample_uid": _resolve_sample_uid(row),
        "dataset_key": _selection_str(row, "dataset_key") or _selection_str(row, "source_dataset_key"),
        "split": _selection_str(row, "split"),
        "source_image_path": str(image_path),
        "source_scene_path": str(scene_path),
        "source_det_path": _selection_optional_str(row, "det_path"),
        "overlay_path": str(overlay_path),
    }


def _render_teacher_debug_item(
    row: DebugSelectionRow,
    *,
    teacher_name: str,
    class_names: Sequence[str],
    output_root: Path,
) -> DebugVisItem:
    image_path = Path(str(row["source_image_path"])).resolve()
    label_path = Path(str(row["output_label_path"])).resolve()
    overlay_path = _overlay_path_for_row(output_root, row)
    render_overlay(_teacher_overlay_scene(image_path=image_path, label_path=label_path, class_names=class_names), overlay_path)
    return {
        "teacher_name": teacher_name,
        "sample_id": str(row.get("sample_id") or ""),
        "sample_uid": _resolve_sample_uid(row),
        "dataset_key": _selection_str(row, "source_dataset_key") or _selection_str(row, "dataset_key"),
        "split": _selection_str(row, "split"),
        "source_image_path": str(image_path),
        "source_label_path": str(label_path),
        "overlay_path": str(overlay_path),
    }


def _render_selected_rows(
    rows: Sequence[DebugSelectionRow],
    *,
    stage_name: str,
    log_fn: Callable[[str], None] | None,
    render_fn: Callable[[DebugSelectionRow], DebugVisItem],
    max_workers: int | None = None,
) -> list[DebugVisItem]:
    if not rows:
        return []
    requested_workers = DEFAULT_FINAL_LANE_AUDIT_WORKERS if max_workers is None else int(max_workers)
    workers = max(1, min(requested_workers, len(rows)))
    if log_fn is not None:
        log_fn(f"{stage_name} debug_vis start samples={len(rows)} workers={workers}")
    completed = 0
    start_time = time.monotonic()
    items: list[DebugVisItem] = []
    log_every = max(10, workers * 4)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{stage_name}_debug_vis") as executor:
        futures = [executor.submit(render_fn, row) for row in rows]
        for future in as_completed(futures):
            items.append(future.result())
            completed += 1
            if log_fn is not None and (completed == len(rows) or completed == 1 or completed % log_every == 0):
                elapsed = max(time.monotonic() - start_time, 1e-6)
                rate = completed / elapsed
                log_fn(f"{stage_name} debug_vis progress {completed}/{len(rows)} samples ({rate:.1f} samples/s)")
    items.sort(key=lambda item: (str(item.get("dataset_key") or ""), str(item.get("split") or ""), str(item.get("sample_uid") or "")))
    return items


def _build_debug_vis_result(
    *,
    manifest_path: Path,
    debug_vis_dir: Path,
    items: Sequence[DebugVisItem],
    payload: DebugVisPayload,
) -> DebugVisResult:
    _write_json(manifest_path, payload)
    return {
        "debug_vis_dir": debug_vis_dir,
        "debug_vis_manifest": manifest_path,
        "selection_count": len(items),
    }


def generate_canonical_debug_vis(
    *,
    image_list_manifest_path: Path,
    canonical_root: Path,
    debug_vis_count: int,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, DebugVisResult]:
    canonical_root = canonical_root.resolve()
    rows = [_debug_selection_row_from_image_list_entry(entry) for entry in load_image_list(image_list_manifest_path)]
    grouped_rows: dict[Path, list[DebugSelectionRow]] = {
        dataset_root.resolve(): []
        for dataset_root in (
            canonical_root / "bdd100k_det_100k",
            canonical_root / "aihub_standardized",
        )
        if dataset_root.exists()
    }
    for row in rows:
        dataset_root = _resolve_canonical_dataset_root(row, canonical_root=canonical_root)
        grouped_rows.setdefault(dataset_root, []).append(_copy_debug_row(row))
    if not grouped_rows:
        grouped_rows = {
            (canonical_root / "bdd100k_det_100k").resolve(): [],
            (canonical_root / "aihub_standardized").resolve(): [],
        }

    outputs: dict[str, DebugVisResult] = {}
    for dataset_root, dataset_rows in sorted(grouped_rows.items(), key=lambda item: str(item[0])):
        debug_vis_dir = dataset_root / "meta" / "debug_vis"
        manifest_path = dataset_root / "meta" / "debug_vis_manifest.json"
        _reset_debug_vis_dir(debug_vis_dir)
        selected = _select_debug_rows(dataset_rows, count=debug_vis_count, seed=debug_vis_seed)
        items = _render_selected_rows(
            selected,
            stage_name=f"canonical:{dataset_root.name}",
            log_fn=log_fn,
            render_fn=lambda row, output_root=debug_vis_dir: _render_canonical_debug_item(row, output_root=output_root),
        )
        outputs[dataset_root.name] = _build_debug_vis_result(
            manifest_path=manifest_path,
            debug_vis_dir=debug_vis_dir,
            items=items,
            payload={
                "version": "od-bootstrap-canonical-debug-vis-v2",
                "generated_at": _now_iso(),
                "image_list_manifest_path": str(Path(image_list_manifest_path).resolve()),
                "dataset_root": str(dataset_root),
                "selection_count": len(items),
                "seed": int(debug_vis_seed),
                "items": items,
            },
        )
    return outputs


def generate_teacher_dataset_debug_vis(
    *,
    dataset_root: Path,
    teacher_name: str,
    class_names: Sequence[str],
    manifest_rows: Sequence[DebugSelectionRow],
    debug_vis_count: int,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
) -> DebugVisResult:
    dataset_root = dataset_root.resolve()
    debug_vis_dir = dataset_root / "meta" / "debug_vis"
    manifest_path = dataset_root / "meta" / "debug_vis_manifest.json"
    _reset_debug_vis_dir(debug_vis_dir)
    selected = _select_debug_rows(list(manifest_rows), count=debug_vis_count, seed=debug_vis_seed)
    items = _render_selected_rows(
        selected,
        stage_name=f"teacher:{teacher_name}",
        log_fn=log_fn,
        render_fn=lambda row, output_root=debug_vis_dir: _render_teacher_debug_item(
            row,
            teacher_name=teacher_name,
            class_names=class_names,
            output_root=output_root,
        ),
    )
    return _build_debug_vis_result(
        manifest_path=manifest_path,
        debug_vis_dir=debug_vis_dir,
        items=items,
        payload={
            "version": "od-bootstrap-teacher-debug-vis-v2",
            "generated_at": _now_iso(),
            "teacher_name": teacher_name,
            "dataset_root": str(dataset_root),
            "class_names": list(class_names),
            "selection_count": len(items),
            "seed": int(debug_vis_seed),
            "items": items,
        },
    )


def generate_exhaustive_debug_vis(
    *,
    dataset_root: Path,
    manifest_rows: Sequence[DebugSelectionRow],
    debug_vis_count: int,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
) -> DebugVisResult:
    dataset_root = dataset_root.resolve()
    debug_vis_dir = dataset_root / "meta" / "debug_vis"
    manifest_path = dataset_root / "meta" / "debug_vis_manifest.json"
    _reset_debug_vis_dir(debug_vis_dir)
    selected = _select_debug_rows(list(manifest_rows), count=debug_vis_count, seed=debug_vis_seed)
    items = _render_selected_rows(
        selected,
        stage_name=f"exhaustive:{dataset_root.name}",
        log_fn=log_fn,
        render_fn=lambda row, output_root=debug_vis_dir: _render_canonical_debug_item(row, output_root=output_root),
    )
    return _build_debug_vis_result(
        manifest_path=manifest_path,
        debug_vis_dir=debug_vis_dir,
        items=items,
        payload={
            "version": "od-bootstrap-exhaustive-debug-vis-v1",
            "generated_at": _now_iso(),
            "dataset_root": str(dataset_root),
            "selection_count": len(items),
            "seed": int(debug_vis_seed),
            "items": items,
        },
    )


def _debug_selection_row_from_final_dataset_row(row: Mapping[str, object]) -> DebugSelectionRow:
    dataset_key = _selection_str(row, "source_dataset_key")
    split = _selection_str(row, "split")
    sample_id = _selection_str(row, "final_sample_id")
    return {
        "dataset_key": dataset_key,
        "image_path": _selection_str(row, "image_path"),
        "scene_path": _selection_str(row, "scene_path"),
        "sample_id": sample_id,
        "sample_uid": build_sample_uid(
            dataset_key=dataset_key or "default",
            split=split or "unknown",
            sample_id=sample_id or "sample",
        ),
        "split": split,
    }


def _resolve_final_dataset_scene_path(*, dataset_root: Path, row: Mapping[str, object]) -> Path:
    scene_path = Path(_selection_str(row, "scene_path")).resolve()
    if scene_path.is_file():
        return scene_path
    split = _selection_str(row, "split")
    sample_id = _selection_str(row, "final_sample_id")
    fallback_path = (dataset_root / "labels_scene" / split / f"{sample_id}.json").resolve()
    if fallback_path.is_file():
        return fallback_path
    raise FileNotFoundError(f"final dataset scene not found: {scene_path}")


def _resolve_final_dataset_image_path(*, dataset_root: Path, row: Mapping[str, object]) -> Path:
    image_path = Path(_selection_str(row, "image_path")).resolve()
    if image_path.is_file():
        return image_path
    split = _selection_str(row, "split")
    image_name = image_path.name
    fallback_path = (dataset_root / "images" / split / image_name).resolve()
    if fallback_path.is_file():
        return fallback_path
    raise FileNotFoundError(f"final dataset image not found: {image_path}")


def generate_final_dataset_debug_vis(
    *,
    dataset_root: Path,
    manifest_rows: Sequence[Mapping[str, object] | DebugSelectionRow],
    debug_vis_count: int,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
) -> DebugVisResult:
    dataset_root = dataset_root.resolve()
    debug_vis_dir = dataset_root / "meta" / "debug_vis"
    manifest_path = dataset_root / "meta" / "debug_vis_manifest.json"
    _reset_debug_vis_dir(debug_vis_dir)
    selection_rows: list[DebugSelectionRow] = []
    for row in manifest_rows:
        selection_row = _debug_selection_row_from_final_dataset_row(row)
        selection_row["scene_path"] = str(_resolve_final_dataset_scene_path(dataset_root=dataset_root, row=row))
        selection_row["image_path"] = str(_resolve_final_dataset_image_path(dataset_root=dataset_root, row=row))
        selection_rows.append(selection_row)
    selected = _select_debug_rows(selection_rows, count=debug_vis_count, seed=debug_vis_seed)
    items = _render_selected_rows(
        selected,
        stage_name=f"final:{dataset_root.name}",
        log_fn=log_fn,
        render_fn=lambda row, output_root=debug_vis_dir: _render_canonical_debug_item(row, output_root=output_root),
    )
    return _build_debug_vis_result(
        manifest_path=manifest_path,
        debug_vis_dir=debug_vis_dir,
        items=items,
        payload={
            "version": "od-bootstrap-final-debug-vis-v1",
            "generated_at": _now_iso(),
            "dataset_root": str(dataset_root),
            "final_dataset_manifest_path": str((dataset_root / "meta" / FINAL_DATASET_MANIFEST_NAME).resolve()),
            "selection_count": len(items),
            "seed": int(debug_vis_seed),
            "items": items,
        },
    )


def _stable_final_dataset_order_key(seed: int, row: Mapping[str, object]) -> tuple[str, str, str]:
    dataset_key = _selection_str(row, "source_dataset_key") or _selection_str(row, "dataset_key")
    split = _selection_str(row, "split")
    sample_id = _selection_str(row, "final_sample_id") or _selection_str(row, "sample_id")
    sample_uid = build_sample_uid(
        dataset_key=dataset_key or "default",
        split=split or "unknown",
        sample_id=sample_id or "sample",
    )
    digest = hashlib.sha256(f"{seed}:{sample_uid}".encode("utf-8")).hexdigest()
    return digest, split, sample_uid


def _slugify_token(value: str, *, default: str, max_length: int = 64) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_").lower()
    if not normalized:
        normalized = default
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
    prefix = normalized[: max_length - 9].rstrip("_")
    return f"{prefix}_{digest}" if prefix else digest


def _sample_hash_suffix(sample_uid: str) -> str:
    return hashlib.sha256(str(sample_uid).encode("utf-8")).hexdigest()[:10]


def _extract_frame_id(*values: str) -> int | None:
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        match = _TRAILING_DIGITS_RE.search(text)
        if match is None:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _resolve_source_group_name(*, dataset_key: str, split: str, source_image_path: str, sample_id: str) -> str:
    if dataset_key == "pv26_exhaustive_bdd100k_det_100k":
        return split or "unknown"
    if source_image_path:
        parent_name = Path(source_image_path).parent.name.strip()
        if parent_name:
            return parent_name
    sample_text = sample_id or "unknown"
    if dataset_key == "aihub_lane_seoul":
        match = re.match(r"aihub_lane_seoul_(?:train|val)_(.+)_\d+$", sample_text)
        if match is not None:
            return match.group(1)
    return sample_text


def _load_scene_payload(scene_path: Path) -> dict[str, object]:
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"scene JSON root must be a mapping: {scene_path}")
    return payload


def _build_final_lane_audit_row(
    *,
    dataset_root: Path,
    row: Mapping[str, object],
    frame_index_fallback: int | None = None,
) -> DebugSelectionRow:
    scene_path = _resolve_final_dataset_scene_path(dataset_root=dataset_root, row=row)
    image_path = _resolve_final_dataset_image_path(dataset_root=dataset_root, row=row)
    scene = _load_scene_payload(scene_path)
    source = scene.get("source") if isinstance(scene.get("source"), dict) else {}
    source = source if isinstance(source, dict) else {}
    dataset_key = _selection_str(row, "source_dataset_key") or str(source.get("dataset") or "").strip()
    split = _selection_str(row, "split") or str(source.get("split") or "").strip()
    sample_id = _selection_str(row, "final_sample_id") or str(source.get("final_sample_id") or scene_path.stem)
    sample_uid = build_sample_uid(
        dataset_key=dataset_key or "default",
        split=split or "unknown",
        sample_id=sample_id or "sample",
    )
    source_image_path = str(source.get("image_path") or image_path)
    source_raw_id = str(source.get("raw_id") or "").strip()
    frame_id = _extract_frame_id(sample_id, source_raw_id, Path(source_image_path).stem)
    if frame_id is None and frame_index_fallback is not None:
        frame_id = int(frame_index_fallback)
    group_key = _resolve_source_group_name(
        dataset_key=dataset_key,
        split=split,
        source_image_path=source_image_path,
        sample_id=sample_id,
    )
    group_slug = _slugify_token(group_key, default="group", max_length=32)
    return {
        "dataset_key": dataset_key,
        "split": split,
        "sample_id": sample_id,
        "sample_uid": sample_uid,
        "scene_path": str(scene_path),
        "image_path": str(image_path),
        "source_image_path": source_image_path,
        "source_raw_id": source_raw_id,
        "group_key": group_key,
        "group_slug": group_slug,
        "frame_id": -1 if frame_id is None else int(frame_id),
        "lane_count": int(len(scene.get("lanes") or [])),
        "stop_line_count": int(len(scene.get("stop_lines") or [])),
        "crosswalk_count": int(len(scene.get("crosswalks") or [])),
    }


def _scaled_overview_quotas(total_count: int) -> list[tuple[str, str, int]]:
    if total_count <= 0:
        return []
    base_rows = list(DEFAULT_FINAL_LANE_AUDIT_OVERVIEW_QUOTAS)
    base_total = sum(count for _, _, count in base_rows)
    if total_count == base_total:
        return base_rows
    scaled: list[list[object]] = []
    used = 0
    for dataset_key, split, count in base_rows:
        exact = (float(count) * float(total_count)) / float(base_total)
        floor_count = int(exact)
        scaled.append([dataset_key, split, floor_count, exact - float(floor_count)])
        used += floor_count
    remaining = max(0, total_count - used)
    scaled.sort(key=lambda item: (float(item[3]), str(item[0]), str(item[1])), reverse=True)
    for index in range(min(remaining, len(scaled))):
        scaled[index][2] = int(scaled[index][2]) + 1
    normalized = [
        (str(dataset_key), str(split), int(count))
        for dataset_key, split, count, _ in scaled
        if int(count) > 0
    ]
    normalized.sort(key=lambda item: (item[0], item[1]))
    return normalized


def _select_overview_manifest_rows(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    total_count: int,
    seed: int,
) -> list[Mapping[str, object]]:
    quotas = _scaled_overview_quotas(total_count)
    rows_by_bucket: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        dataset_key = _selection_str(row, "source_dataset_key")
        split = _selection_str(row, "split")
        rows_by_bucket[(dataset_key, split)].append(row)
    for bucket_key, rows in rows_by_bucket.items():
        rows.sort(key=lambda item: _stable_final_dataset_order_key(seed, item))
        rows_by_bucket[bucket_key] = rows
    selected: list[Mapping[str, object]] = []
    for dataset_key, split, quota in quotas:
        selected.extend(rows_by_bucket.get((dataset_key, split), [])[:quota])
    return selected


def _pick_even_indices(*, start: int, end: int, count: int) -> list[int]:
    span = max(0, int(end) - int(start))
    if span <= 0 or count <= 0:
        return []
    if count == 1:
        return [start + (span // 2)]
    fractions = [((2 * index) + 1) / (2 * count) for index in range(count)]
    selected = [min(end - 1, start + int(fraction * span)) for fraction in fractions]
    return selected


def _select_lane_audit_rows(
    lane_rows: Sequence[DebugSelectionRow],
    *,
    bin_count: int,
    samples_per_bin: int,
) -> list[DebugSelectionRow]:
    grouped_rows: dict[tuple[str, str], list[DebugSelectionRow]] = defaultdict(list)
    for row in lane_rows:
        grouped_rows[(_selection_str(row, "split"), _selection_str(row, "group_key"))].append(dict(row))
    selected: list[DebugSelectionRow] = []
    for (split, group_key), rows in sorted(grouped_rows.items(), key=lambda item: item[0]):
        ordered_rows = sorted(
            rows,
            key=lambda item: (
                int(item.get("frame_id") or -1),
                _selection_str(item, "sample_uid"),
            ),
        )
        group_slug = _selection_str(ordered_rows[0], "group_slug") or _slugify_token(group_key, default="group", max_length=32)
        total_rows = len(ordered_rows)
        for bin_index in range(bin_count):
            start = (bin_index * total_rows) // bin_count
            end = ((bin_index + 1) * total_rows) // bin_count
            for pick_index, row_index in enumerate(
                _pick_even_indices(start=start, end=end, count=samples_per_bin),
                start=1,
            ):
                row = dict(ordered_rows[row_index])
                row["category"] = "lane"
                row["bucket_dir"] = f"lane_{split}"
                row["group_key"] = group_key
                row["group_slug"] = group_slug
                row["bin_index"] = int(bin_index) + 1
                row["bin_count"] = int(bin_count)
                row["pick_index"] = int(pick_index)
                frame_id = int(row.get("frame_id") or -1)
                row["overlay_file_name"] = (
                    f"lane__{split}__{group_slug}__bin{bin_index + 1:02d}-of-{bin_count}"
                    f"__pick{pick_index:02d}__frame{frame_id}"
                    f"__{_sample_hash_suffix(_selection_str(row, 'sample_uid'))}.png"
                )
                selected.append(row)
    return selected


def _enrich_overview_audit_rows(
    *,
    dataset_root: Path,
    rows: Sequence[Mapping[str, object]],
    seed: int,
) -> list[DebugSelectionRow]:
    enriched: list[DebugSelectionRow] = []
    for ordinal, row in enumerate(
        sorted(rows, key=lambda item: _stable_final_dataset_order_key(seed, item)),
        start=1,
    ):
        audit_row = _build_final_lane_audit_row(dataset_root=dataset_root, row=row, frame_index_fallback=ordinal)
        dataset_key = _selection_str(audit_row, "dataset_key")
        split = _selection_str(audit_row, "split")
        audit_row["category"] = "overview"
        audit_row["bucket_dir"] = "overview"
        audit_row["overlay_file_name"] = (
            f"overview__{dataset_key}__{split}__{ordinal:03d}"
            f"__{_sample_hash_suffix(_selection_str(audit_row, 'sample_uid'))}.png"
        )
        enriched.append(audit_row)
    return enriched


def _render_final_lane_audit_item(row: DebugSelectionRow) -> FinalLaneAuditItem:
    image_path = Path(_selection_str(row, "image_path")).resolve()
    scene_path = Path(_selection_str(row, "scene_path")).resolve()
    overlay_path = Path(_selection_str(row, "overlay_path")).resolve()
    scene = _load_scene_payload(scene_path)
    render_overlay(canonical_scene_to_overlay_scene(scene, image_path=image_path), overlay_path)
    item: FinalLaneAuditItem = {
        "category": _selection_str(row, "category"),
        "bucket_dir": _selection_str(row, "bucket_dir"),
        "dataset_key": _selection_str(row, "dataset_key"),
        "split": _selection_str(row, "split"),
        "sample_id": _selection_str(row, "sample_id"),
        "sample_uid": _selection_str(row, "sample_uid"),
        "overlay_path": str(overlay_path),
        "source_image_path": str(image_path),
        "source_scene_path": str(scene_path),
        "group_key": _selection_str(row, "group_key"),
        "group_slug": _selection_str(row, "group_slug"),
        "source_raw_id": _selection_str(row, "source_raw_id"),
        "lane_count": int(row.get("lane_count") or 0),
        "stop_line_count": int(row.get("stop_line_count") or 0),
        "crosswalk_count": int(row.get("crosswalk_count") or 0),
        "frame_id": int(row.get("frame_id") or -1),
    }
    if row.get("bin_index") is not None:
        item["bin_index"] = int(row.get("bin_index") or 0)
    if row.get("bin_count") is not None:
        item["bin_count"] = int(row.get("bin_count") or 0)
    if row.get("pick_index") is not None:
        item["pick_index"] = int(row.get("pick_index") or 0)
    return item


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def generate_final_lane_label_audit(
    *,
    dataset_root: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    output_root: Path | None = None,
    overview_count: int = DEFAULT_FINAL_LANE_AUDIT_OVERVIEW_COUNT,
    lane_bin_count: int = DEFAULT_FINAL_LANE_AUDIT_BIN_COUNT,
    lane_samples_per_bin: int = DEFAULT_FINAL_LANE_AUDIT_SAMPLES_PER_BIN,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    workers: int = DEFAULT_FINAL_LANE_AUDIT_WORKERS,
    log_fn: Callable[[str], None] | None = None,
) -> FinalLaneAuditResult:
    if overview_count < 0:
        raise ValueError("overview_count must be >= 0")
    if lane_bin_count <= 0:
        raise ValueError("lane_bin_count must be > 0")
    if lane_samples_per_bin <= 0:
        raise ValueError("lane_samples_per_bin must be > 0")
    if workers <= 0:
        raise ValueError("workers must be > 0")
    dataset_root = dataset_root.resolve()
    audit_root = (output_root or (dataset_root / DEFAULT_FINAL_LANE_AUDIT_DIRNAME)).resolve()
    _reset_debug_vis_dir(audit_root)

    lane_manifest_rows = sorted(
        (
            row for row in manifest_rows
            if _selection_str(row, "source_dataset_key") == "aihub_lane_seoul"
        ),
        key=lambda item: _stable_final_dataset_order_key(debug_vis_seed, item),
    )
    lane_scan_rows: list[DebugSelectionRow] = []
    lane_zero_counter: Counter[str] = Counter()
    for index, row in enumerate(lane_manifest_rows, start=1):
        audit_row = _build_final_lane_audit_row(dataset_root=dataset_root, row=row, frame_index_fallback=index)
        lane_scan_rows.append(audit_row)
        if int(audit_row.get("lane_count") or 0) == 0:
            lane_zero_counter["total"] += 1
            lane_zero_counter[_selection_str(audit_row, "split") or "unknown"] += 1
        if log_fn is not None and (index == len(lane_manifest_rows) or index == 1 or index % 5000 == 0):
            log_fn(f"lane_audit scan progress {index}/{len(lane_manifest_rows)} lane scenes")

    overview_rows = _enrich_overview_audit_rows(
        dataset_root=dataset_root,
        rows=_select_overview_manifest_rows(
            manifest_rows,
            total_count=overview_count,
            seed=debug_vis_seed,
        ),
        seed=debug_vis_seed,
    )
    lane_rows = _select_lane_audit_rows(
        lane_scan_rows,
        bin_count=lane_bin_count,
        samples_per_bin=lane_samples_per_bin,
    )

    selection_rows = [*overview_rows, *lane_rows]
    for row in selection_rows:
        bucket_dir = _selection_str(row, "bucket_dir")
        overlay_file_name = _selection_str(row, "overlay_file_name")
        row["overlay_path"] = str((audit_root / bucket_dir / overlay_file_name).resolve())

    rendered_items_raw = _render_selected_rows(
        selection_rows,
        stage_name=f"final-audit:{dataset_root.name}",
        log_fn=log_fn,
        render_fn=lambda row: _render_final_lane_audit_item(row),
        max_workers=workers,
    )
    rendered_items: list[FinalLaneAuditItem] = []
    for item in rendered_items_raw:
        rendered_items.append(dict(item))
    rendered_items.sort(
        key=lambda item: (
            str(item.get("bucket_dir") or ""),
            str(item.get("dataset_key") or ""),
            str(item.get("split") or ""),
            str(item.get("overlay_path") or ""),
        )
    )

    overview_counter = Counter(
        f"{item['dataset_key']}::{item['split']}"
        for item in rendered_items
        if item.get("category") == "overview"
    )
    lane_group_counter = Counter(
        f"{item['split']}::{item['group_key']}"
        for item in rendered_items
        if item.get("category") == "lane"
    )
    lane_scanned_counter = Counter(
        _selection_str(row, "split")
        for row in lane_scan_rows
    )
    index_path = audit_root / "index.json"
    summary_path = audit_root / "summary.json"
    _write_json(
        index_path,
        {
            "version": "od-bootstrap-final-lane-audit-v1",
            "generated_at": _now_iso(),
            "dataset_root": str(dataset_root),
            "output_root": str(audit_root),
            "seed": int(debug_vis_seed),
            "workers": int(workers),
            "selection_count": len(rendered_items),
            "items": rendered_items,
        },
    )
    _write_json(
        summary_path,
        {
            "version": "od-bootstrap-final-lane-audit-v1",
            "generated_at": _now_iso(),
            "dataset_root": str(dataset_root),
            "output_root": str(audit_root),
            "seed": int(debug_vis_seed),
            "workers": int(workers),
            "overview_count_requested": int(overview_count),
            "overview_count_rendered": int(sum(1 for item in rendered_items if item.get("category") == "overview")),
            "lane_bin_count": int(lane_bin_count),
            "lane_samples_per_bin": int(lane_samples_per_bin),
            "lane_selection_count_rendered": int(sum(1 for item in rendered_items if item.get("category") == "lane")),
            "selection_count": int(len(rendered_items)),
            "overview_dataset_split_counts": _counter_dict(overview_counter),
            "lane_rendered_group_counts": _counter_dict(lane_group_counter),
            "lane_scanned_split_counts": _counter_dict(lane_scanned_counter),
            "lane_zero_counts": _counter_dict(lane_zero_counter),
        },
    )
    return {
        "output_root": audit_root,
        "index_path": index_path,
        "summary_path": summary_path,
        "selection_count": len(rendered_items),
    }
