from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..io import utc_now_iso, write_json


WOD_BULK_STATE_VERSION = "wod-bulk-v1"

WOD_STATUS_PENDING = "pending"
WOD_STATUS_BLOCKED = "blocked_missing_components"
WOD_STATUS_IN_PROGRESS = "in_progress"
WOD_STATUS_COMPLETED = "completed"
WOD_STATUS_FAILED = "failed"

WOD_COMPONENTS = ("camera_image", "camera_segmentation", "camera_box")


@dataclass(frozen=True)
class WodContextScan:
    context_name: str
    image_relpath: str
    segmentation_relpath: str
    box_relpath: str
    image_bytes: int
    segmentation_bytes: int
    box_bytes: int
    has_image: bool
    has_segmentation: bool
    has_box: bool

    @property
    def processable_now(self) -> bool:
        return self.has_image and self.has_segmentation

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "context_name": self.context_name,
            "image_relpath": self.image_relpath,
            "segmentation_relpath": self.segmentation_relpath,
            "box_relpath": self.box_relpath,
            "image_bytes": int(self.image_bytes),
            "segmentation_bytes": int(self.segmentation_bytes),
            "box_bytes": int(self.box_bytes),
            "has_image": bool(self.has_image),
            "has_segmentation": bool(self.has_segmentation),
            "has_box": bool(self.has_box),
            "processable_now": bool(self.processable_now),
        }


def _context_name_from_relpath(path: Path) -> str:
    return path.stem


def _relpath_if_exists(path: Path, root: Path) -> str:
    if not path.exists():
        return ""
    return path.relative_to(root).as_posix()


def _bytes_if_exists(path: Path) -> int:
    if not path.exists():
        return 0
    return int(path.stat().st_size)


def _safe_context_dirname(context_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", context_name.strip())
    return s or "unknown_context"


def shard_root_for_context(*, shards_root: Path, context_name: str) -> Path:
    return shards_root / f"pv26_wod_{_safe_context_dirname(context_name)}"


def scan_wod_training_root(training_root: Path) -> List[WodContextScan]:
    training_root = Path(training_root).expanduser().resolve()
    image_dir = training_root / "camera_image"
    seg_dir = training_root / "camera_segmentation"
    box_dir = training_root / "camera_box"

    names = set()
    for component_dir in (image_dir, seg_dir, box_dir):
        if not component_dir.exists():
            continue
        for p in sorted(component_dir.glob("*.parquet")):
            names.add(_context_name_from_relpath(p))

    out: List[WodContextScan] = []
    for context_name in sorted(names):
        image_p = image_dir / f"{context_name}.parquet"
        seg_p = seg_dir / f"{context_name}.parquet"
        box_p = box_dir / f"{context_name}.parquet"
        out.append(
            WodContextScan(
                context_name=context_name,
                image_relpath=_relpath_if_exists(image_p, training_root),
                segmentation_relpath=_relpath_if_exists(seg_p, training_root),
                box_relpath=_relpath_if_exists(box_p, training_root),
                image_bytes=_bytes_if_exists(image_p),
                segmentation_bytes=_bytes_if_exists(seg_p),
                box_bytes=_bytes_if_exists(box_p),
                has_image=image_p.exists(),
                has_segmentation=seg_p.exists(),
                has_box=box_p.exists(),
            )
        )
    return out


def load_wod_bulk_state(state_path: Path) -> Dict[str, Any]:
    state_path = Path(state_path).expanduser()
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid state json root: {state_path}")
    return data


def summarize_wod_bulk_state(state: Mapping[str, Any]) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "total_contexts": 0,
        "processable_now": 0,
        "with_box": 0,
        "status_pending": 0,
        "status_blocked_missing_components": 0,
        "status_in_progress": 0,
        "status_completed": 0,
        "status_failed": 0,
    }
    for ctx in state.get("contexts", []):
        summary["total_contexts"] += 1
        if ctx.get("processable_now"):
            summary["processable_now"] += 1
        if ctx.get("has_box"):
            summary["with_box"] += 1
        status = str(ctx.get("status", "")).strip()
        key = f"status_{status}"
        if key in summary:
            summary[key] += 1
    return summary


def reconcile_wod_bulk_state(
    *,
    training_root: Path,
    shards_root: Path,
    prior_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    training_root = Path(training_root).expanduser().resolve()
    shards_root = Path(shards_root).expanduser().resolve()

    previous_by_context: Dict[str, Mapping[str, Any]] = {}
    if prior_state:
        for entry in prior_state.get("contexts", []):
            name = str(entry.get("context_name", "")).strip()
            if name:
                previous_by_context[name] = entry

    scanned = {item.context_name: item for item in scan_wod_training_root(training_root)}
    all_context_names = sorted(set(scanned.keys()) | set(previous_by_context.keys()))

    contexts: List[Dict[str, Any]] = []
    for context_name in all_context_names:
        scan = scanned.get(context_name)
        prev = previous_by_context.get(context_name, {})

        if scan is None:
            base = {
                "context_name": context_name,
                "image_relpath": str(prev.get("image_relpath", "")),
                "segmentation_relpath": str(prev.get("segmentation_relpath", "")),
                "box_relpath": str(prev.get("box_relpath", "")),
                "image_bytes": int(prev.get("image_bytes", 0) or 0),
                "segmentation_bytes": int(prev.get("segmentation_bytes", 0) or 0),
                "box_bytes": int(prev.get("box_bytes", 0) or 0),
                "has_image": False,
                "has_segmentation": False,
                "has_box": False,
                "processable_now": False,
            }
        else:
            base = scan.to_state_dict()

        shard_root = str(shard_root_for_context(shards_root=shards_root, context_name=context_name))
        prev_status = str(prev.get("status", "")).strip()
        if prev_status == WOD_STATUS_IN_PROGRESS:
            status = WOD_STATUS_PENDING if base["processable_now"] else WOD_STATUS_BLOCKED
        elif prev_status in {WOD_STATUS_COMPLETED, WOD_STATUS_FAILED}:
            status = prev_status
        else:
            status = WOD_STATUS_PENDING if base["processable_now"] else WOD_STATUS_BLOCKED

        contexts.append(
            {
                **base,
                "status": status,
                "attempt_count": int(prev.get("attempt_count", 0) or 0),
                "last_error": str(prev.get("last_error", "")),
                "last_started_at": str(prev.get("last_started_at", "")),
                "last_completed_at": str(prev.get("last_completed_at", "")),
                "raw_deleted_at": str(prev.get("raw_deleted_at", "")),
                "shard_root": str(prev.get("shard_root", shard_root) or shard_root),
                "output_num_rows": int(prev.get("output_num_rows", 0) or 0),
                "output_rows_by_split": dict(prev.get("output_rows_by_split", {}) or {}),
                "output_has_det_rows": int(prev.get("output_has_det_rows", 0) or 0),
            }
        )

    state: Dict[str, Any] = {
        "version": WOD_BULK_STATE_VERSION,
        "training_root": str(training_root),
        "shards_root": str(shards_root),
        "created_at": str((prior_state or {}).get("created_at", "") or utc_now_iso()),
        "updated_at": utc_now_iso(),
        "contexts": contexts,
    }
    state["summary"] = summarize_wod_bulk_state(state)
    return state


def write_wod_bulk_state(state_path: Path, state: Mapping[str, Any]) -> None:
    write_json(Path(state_path).expanduser(), dict(state))


def write_wod_bulk_state_csv(csv_path: Path, state: Mapping[str, Any]) -> None:
    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context_name",
        "status",
        "processable_now",
        "has_image",
        "has_segmentation",
        "has_box",
        "image_bytes",
        "segmentation_bytes",
        "box_bytes",
        "image_relpath",
        "segmentation_relpath",
        "box_relpath",
        "attempt_count",
        "output_num_rows",
        "output_has_det_rows",
        "last_error",
        "last_started_at",
        "last_completed_at",
        "raw_deleted_at",
        "shard_root",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ctx in state.get("contexts", []):
            writer.writerow({k: ctx.get(k, "") for k in fieldnames})


def find_context_entry(state: Mapping[str, Any], context_name: str) -> Dict[str, Any]:
    for ctx in state.get("contexts", []):
        if str(ctx.get("context_name", "")).strip() == context_name:
            return ctx
    raise KeyError(f"context not found in state: {context_name}")


def iter_contexts_for_processing(
    state: Mapping[str, Any],
    *,
    include_failed: bool = False,
    selected_contexts: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    allow = {s.strip() for s in selected_contexts or [] if s.strip()}
    out: List[Dict[str, Any]] = []
    for ctx in state.get("contexts", []):
        name = str(ctx.get("context_name", "")).strip()
        if allow and name not in allow:
            continue
        if not bool(ctx.get("processable_now", False)):
            continue
        status = str(ctx.get("status", "")).strip()
        if status == WOD_STATUS_PENDING or (include_failed and status == WOD_STATUS_FAILED):
            out.append(ctx)
    return out


def completed_shard_roots_from_state(state: Mapping[str, Any]) -> List[Path]:
    out: List[Path] = []
    for ctx in state.get("contexts", []):
        if str(ctx.get("status", "")).strip() != WOD_STATUS_COMPLETED:
            continue
        shard_root = str(ctx.get("shard_root", "")).strip()
        if shard_root:
            out.append(Path(shard_root).expanduser())
    return out
