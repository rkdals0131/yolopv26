from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is expected in the repo test env.
    Image = None

from model.viz import render_overlay
from tools.od_bootstrap.smoke.review import canonical_scene_to_overlay_scene
from tools.od_bootstrap.sweep.image_list import build_sample_uid, load_image_list


DEFAULT_DEBUG_VIS_COUNT = 20
DEFAULT_DEBUG_VIS_SEED = 26


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _reset_debug_vis_dir(debug_vis_dir: Path) -> None:
    if debug_vis_dir.exists():
        shutil.rmtree(debug_vis_dir)
    debug_vis_dir.mkdir(parents=True, exist_ok=True)


def _stable_selection_key(seed: int, row: Mapping[str, Any]) -> tuple[str, str, str]:
    dataset_key = str(row.get("dataset_key") or row.get("source_dataset_key") or "").strip()
    split = str(row.get("split") or "").strip()
    sample_id = str(row.get("sample_uid") or row.get("sample_id") or "").strip()
    digest = hashlib.sha256(f"{seed}:{dataset_key}:{split}:{sample_id}".encode("utf-8")).hexdigest()
    return digest, split, sample_id


def _select_debug_rows(rows: Sequence[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_key = str(row.get("dataset_key") or row.get("source_dataset_key") or "").strip() or "default"
        grouped[dataset_key].append(dict(row))
    for dataset_key, items in grouped.items():
        items.sort(key=lambda item: _stable_selection_key(seed, {"dataset_key": dataset_key, **item}))
    selected: list[dict[str, Any]] = []
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


def _empty_overlay_scene(*, image_path: Path) -> dict[str, Any]:
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


def _teacher_overlay_scene(*, image_path: Path, label_path: Path, class_names: Sequence[str]) -> dict[str, Any]:
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


def _resolve_sample_uid(row: Mapping[str, Any]) -> str:
    sample_uid = str(row.get("sample_uid") or "").strip()
    if sample_uid:
        return sample_uid
    return build_sample_uid(
        dataset_key=str(row.get("dataset_key") or row.get("source_dataset_key") or "default"),
        split=str(row.get("split") or "unknown"),
        sample_id=str(row.get("sample_id") or "sample"),
    )


def _overlay_path_for_row(debug_vis_dir: Path, row: Mapping[str, Any]) -> Path:
    return debug_vis_dir / f"{_resolve_sample_uid(row)}.png"


def _resolve_canonical_dataset_root(row: Mapping[str, Any], *, canonical_root: Path) -> Path:
    dataset_root = str(row.get("dataset_root") or "").strip()
    if dataset_root:
        return Path(dataset_root).resolve()
    dataset_key = str(row.get("dataset_key") or row.get("source_dataset_key") or "").strip()
    if dataset_key == "bdd100k_det_100k":
        return (canonical_root / "bdd100k_det_100k").resolve()
    return (canonical_root / "aihub_standardized").resolve()


def _render_canonical_debug_item(row: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    image_path = Path(str(row["image_path"])).resolve()
    scene_path = Path(str(row["scene_path"])).resolve()
    overlay_path = _overlay_path_for_row(output_root, row)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    render_overlay(canonical_scene_to_overlay_scene(scene, image_path=image_path), overlay_path)
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "sample_uid": _resolve_sample_uid(row),
        "dataset_key": str(row.get("dataset_key") or row.get("source_dataset_key") or ""),
        "split": str(row.get("split") or ""),
        "source_image_path": str(image_path),
        "source_scene_path": str(scene_path),
        "source_det_path": str(row.get("det_path") or "") or None,
        "overlay_path": str(overlay_path),
    }


def _render_teacher_debug_item(
    row: Mapping[str, Any],
    *,
    teacher_name: str,
    class_names: Sequence[str],
    output_root: Path,
) -> dict[str, Any]:
    image_path = Path(str(row["source_image_path"])).resolve()
    label_path = Path(str(row["output_label_path"])).resolve()
    overlay_path = _overlay_path_for_row(output_root, row)
    render_overlay(_teacher_overlay_scene(image_path=image_path, label_path=label_path, class_names=class_names), overlay_path)
    return {
        "teacher_name": teacher_name,
        "sample_id": str(row.get("sample_id") or ""),
        "sample_uid": _resolve_sample_uid(row),
        "dataset_key": str(row.get("source_dataset_key") or row.get("dataset_key") or ""),
        "split": str(row.get("split") or ""),
        "source_image_path": str(image_path),
        "source_label_path": str(label_path),
        "overlay_path": str(overlay_path),
    }


def _render_selected_rows(
    rows: Sequence[dict[str, Any]],
    *,
    stage_name: str,
    log_fn: Callable[[str], None] | None,
    render_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    workers = max(1, min(8, len(rows)))
    if log_fn is not None:
        log_fn(f"{stage_name} debug_vis start samples={len(rows)} workers={workers}")
    completed = 0
    start_time = time.monotonic()
    items: list[dict[str, Any]] = []
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
    items: Sequence[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
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
) -> dict[str, dict[str, Any]]:
    canonical_root = canonical_root.resolve()
    rows = [entry.to_dict() for entry in load_image_list(image_list_manifest_path)]
    grouped_rows: dict[Path, list[dict[str, Any]]] = {
        dataset_root.resolve(): []
        for dataset_root in (
            canonical_root / "bdd100k_det_100k",
            canonical_root / "aihub_standardized",
        )
        if dataset_root.exists()
    }
    for row in rows:
        dataset_root = _resolve_canonical_dataset_root(row, canonical_root=canonical_root)
        grouped_rows.setdefault(dataset_root, []).append(dict(row))
    if not grouped_rows:
        grouped_rows = {
            (canonical_root / "bdd100k_det_100k").resolve(): [],
            (canonical_root / "aihub_standardized").resolve(): [],
        }

    outputs: dict[str, dict[str, Any]] = {}
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
    manifest_rows: Sequence[dict[str, Any]],
    debug_vis_count: int,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
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
    manifest_rows: Sequence[dict[str, Any]],
    debug_vis_count: int,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
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
