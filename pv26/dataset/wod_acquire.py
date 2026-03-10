from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..io import utc_now_iso, write_json


WOD_ACQUIRE_STATE_VERSION = "wod-acquire-v1"
WOD_REMOTE_COMPONENTS = ("camera_image", "camera_segmentation", "camera_box")

WOD_DOWNLOAD_STATUS_REMOTE_ONLY = "remote_only"
WOD_DOWNLOAD_STATUS_DOWNLOADING = "downloading"
WOD_DOWNLOAD_STATUS_DOWNLOADED = "downloaded"
WOD_DOWNLOAD_STATUS_RAW_DELETED = "raw_deleted"
WOD_DOWNLOAD_STATUS_FAILED = "failed"

_GCLOUD_LS_LONG_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(gs://\S+)\s*$")


@dataclass(frozen=True)
class WodRemoteObject:
    component: str
    context_name: str
    url: str
    size_bytes: int
    updated_at: str

    def to_state_patch(self) -> Dict[str, Any]:
        return {
            f"{self.component}_url": self.url,
            f"{self.component}_bytes": int(self.size_bytes),
            f"{self.component}_updated_at": self.updated_at,
            f"remote_has_{_component_suffix(self.component)}": True,
        }


def _component_suffix(component: str) -> str:
    mapping = {
        "camera_image": "image",
        "camera_segmentation": "segmentation",
        "camera_box": "box",
    }
    if component not in mapping:
        raise KeyError(f"unsupported component: {component}")
    return mapping[component]


def _safe_context_dirname(context_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", context_name.strip()) or "unknown_context"


def _context_name_from_url(url: str) -> str:
    name = str(url).rstrip("/").split("/")[-1]
    if not name.endswith(".parquet"):
        raise ValueError(f"not a parquet object url: {url}")
    return name[: -len(".parquet")]


def _local_component_relpath(component: str, context_name: str) -> str:
    return f"{component}/{_safe_context_dirname(context_name)}.parquet"


def _local_component_path(*, training_root: Path, component: str, context_name: str) -> Path:
    return training_root / _local_component_relpath(component, context_name)


def parse_gcloud_storage_ls_long(text: str, *, component: str) -> List[WodRemoteObject]:
    component = str(component).strip()
    if component not in WOD_REMOTE_COMPONENTS:
        raise ValueError(f"unsupported component: {component}")
    out: List[WodRemoteObject] = []
    for raw_line in str(text).splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("TOTAL:"):
            continue
        match = _GCLOUD_LS_LONG_RE.match(line)
        if match is None:
            continue
        size_text, updated_at, url = match.groups()
        if url.endswith("/") or not url.endswith(".parquet"):
            continue
        needle = f"/training/{component}/"
        if needle not in url:
            continue
        out.append(
            WodRemoteObject(
                component=component,
                context_name=_context_name_from_url(url),
                url=url,
                size_bytes=int(size_text),
                updated_at=updated_at,
            )
        )
    out.sort(key=lambda item: item.context_name)
    return out


def _build_remote_inventory_from_prior(prior_state: Mapping[str, Any]) -> Dict[str, Dict[str, WodRemoteObject]]:
    inventory: Dict[str, Dict[str, WodRemoteObject]] = {component: {} for component in WOD_REMOTE_COMPONENTS}
    for ctx in prior_state.get("contexts", []):
        context_name = str(ctx.get("context_name", "")).strip()
        if not context_name:
            continue
        for component in WOD_REMOTE_COMPONENTS:
            url = str(ctx.get(f"{component}_url", "")).strip()
            if not url:
                continue
            inventory[component][context_name] = WodRemoteObject(
                component=component,
                context_name=context_name,
                url=url,
                size_bytes=int(ctx.get(f"{component}_bytes", 0) or 0),
                updated_at=str(ctx.get(f"{component}_updated_at", "")),
            )
    return inventory


def _bulk_info_by_context(bulk_state: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not bulk_state:
        return out
    for ctx in bulk_state.get("contexts", []):
        name = str(ctx.get("context_name", "")).strip()
        if name:
            out[name] = dict(ctx)
    return out


def _derive_download_status(
    *,
    base: Mapping[str, Any],
    prev: Mapping[str, Any],
    bulk_info: Mapping[str, Any],
) -> str:
    local_has_required = bool(base.get("local_has_image")) and bool(base.get("local_has_segmentation"))
    any_local = bool(base.get("local_has_image")) or bool(base.get("local_has_segmentation")) or bool(base.get("local_has_box"))
    prev_status = str(prev.get("download_status", "")).strip()
    bulk_status = str(bulk_info.get("status", "")).strip()
    if local_has_required:
        return WOD_DOWNLOAD_STATUS_DOWNLOADED
    if prev_status == WOD_DOWNLOAD_STATUS_DOWNLOADING:
        return WOD_DOWNLOAD_STATUS_REMOTE_ONLY
    if prev_status == WOD_DOWNLOAD_STATUS_RAW_DELETED and not any_local:
        return WOD_DOWNLOAD_STATUS_RAW_DELETED
    if prev_status == WOD_DOWNLOAD_STATUS_FAILED and not any_local:
        return WOD_DOWNLOAD_STATUS_FAILED
    if bulk_status == "completed" and not any_local:
        return WOD_DOWNLOAD_STATUS_RAW_DELETED
    return WOD_DOWNLOAD_STATUS_REMOTE_ONLY


def reconcile_wod_acquire_state(
    *,
    training_root: Path,
    prior_state: Optional[Mapping[str, Any]] = None,
    remote_objects_by_component: Optional[Mapping[str, Iterable[WodRemoteObject]]] = None,
    bulk_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    training_root = Path(training_root).expanduser().resolve()
    prior_state = dict(prior_state or {})
    if remote_objects_by_component is None:
        inventory = _build_remote_inventory_from_prior(prior_state)
    else:
        inventory = {
            component: {item.context_name: item for item in items}
            for component, items in remote_objects_by_component.items()
        }
        for component in WOD_REMOTE_COMPONENTS:
            inventory.setdefault(component, {})

    previous_by_context: Dict[str, Mapping[str, Any]] = {}
    for ctx in prior_state.get("contexts", []):
        name = str(ctx.get("context_name", "")).strip()
        if name:
            previous_by_context[name] = ctx

    bulk_by_context = _bulk_info_by_context(bulk_state)
    all_context_names = sorted(
        set(previous_by_context.keys())
        | set(bulk_by_context.keys())
        | {name for by_component in inventory.values() for name in by_component.keys()}
    )

    contexts: List[Dict[str, Any]] = []
    for context_name in all_context_names:
        prev = previous_by_context.get(context_name, {})
        bulk_info = bulk_by_context.get(context_name, {})
        base: Dict[str, Any] = {
            "context_name": context_name,
        }
        for component in WOD_REMOTE_COMPONENTS:
            suffix = _component_suffix(component)
            item = inventory.get(component, {}).get(context_name)
            if item is None:
                base[f"{component}_url"] = str(prev.get(f"{component}_url", ""))
                base[f"{component}_bytes"] = int(prev.get(f"{component}_bytes", 0) or 0)
                base[f"{component}_updated_at"] = str(prev.get(f"{component}_updated_at", ""))
                base[f"remote_has_{suffix}"] = bool(prev.get(f"remote_has_{suffix}", False))
            else:
                base.update(item.to_state_patch())

            local_path = _local_component_path(training_root=training_root, component=component, context_name=context_name)
            local_relpath = _local_component_relpath(component, context_name)
            base[f"local_{suffix}_relpath"] = local_relpath
            base[f"local_has_{suffix}"] = bool(local_path.exists())
            base[f"local_{suffix}_bytes"] = int(local_path.stat().st_size) if local_path.exists() else 0

        base["remote_processable"] = bool(base.get("remote_has_image")) and bool(base.get("remote_has_segmentation"))
        download_status = _derive_download_status(base=base, prev=prev, bulk_info=bulk_info)

        contexts.append(
            {
                **base,
                "download_status": download_status,
                "attempt_count": int(prev.get("attempt_count", 0) or 0),
                "last_error": str(prev.get("last_error", "")),
                "last_started_at": str(prev.get("last_started_at", "")),
                "last_completed_at": str(prev.get("last_completed_at", "")),
                "raw_deleted_at": str(prev.get("raw_deleted_at", "")),
                "bulk_status": str(bulk_info.get("status", "")),
                "bulk_output_num_rows": int(bulk_info.get("output_num_rows", 0) or 0),
                "bulk_output_has_det_rows": int(bulk_info.get("output_has_det_rows", 0) or 0),
            }
        )

    state: Dict[str, Any] = {
        "version": WOD_ACQUIRE_STATE_VERSION,
        "training_root": str(training_root),
        "created_at": str(prior_state.get("created_at", "") or utc_now_iso()),
        "updated_at": utc_now_iso(),
        "remote_synced_at": str(prior_state.get("remote_synced_at", "")),
        "contexts": contexts,
    }
    state["summary"] = summarize_wod_acquire_state(state)
    return state


def load_wod_acquire_state(state_path: Path) -> Dict[str, Any]:
    state_path = Path(state_path).expanduser()
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid state json root: {state_path}")
    return data


def write_wod_acquire_state(state_path: Path, state: Mapping[str, Any]) -> None:
    write_json(Path(state_path).expanduser(), dict(state))


def summarize_wod_acquire_state(state: Mapping[str, Any]) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "total_contexts": 0,
        "remote_processable": 0,
        "remote_with_box": 0,
        "download_status_remote_only": 0,
        "download_status_downloading": 0,
        "download_status_downloaded": 0,
        "download_status_raw_deleted": 0,
        "download_status_failed": 0,
        "bulk_status_pending": 0,
        "bulk_status_blocked_missing_components": 0,
        "bulk_status_in_progress": 0,
        "bulk_status_completed": 0,
        "bulk_status_failed": 0,
    }
    for ctx in state.get("contexts", []):
        summary["total_contexts"] += 1
        if ctx.get("remote_processable"):
            summary["remote_processable"] += 1
        if ctx.get("remote_has_box"):
            summary["remote_with_box"] += 1
        status = str(ctx.get("download_status", "")).strip()
        key = f"download_status_{status}"
        if key in summary:
            summary[key] += 1
        bulk_status = str(ctx.get("bulk_status", "")).strip()
        bulk_key = f"bulk_status_{bulk_status}"
        if bulk_key in summary:
            summary[bulk_key] += 1
    return summary


def write_wod_acquire_state_csv(csv_path: Path, state: Mapping[str, Any]) -> None:
    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context_name",
        "download_status",
        "remote_processable",
        "remote_has_image",
        "remote_has_segmentation",
        "remote_has_box",
        "camera_image_bytes",
        "camera_segmentation_bytes",
        "camera_box_bytes",
        "local_has_image",
        "local_has_segmentation",
        "local_has_box",
        "local_image_bytes",
        "local_segmentation_bytes",
        "local_box_bytes",
        "camera_image_url",
        "camera_segmentation_url",
        "camera_box_url",
        "attempt_count",
        "last_error",
        "last_started_at",
        "last_completed_at",
        "raw_deleted_at",
        "bulk_status",
        "bulk_output_num_rows",
        "bulk_output_has_det_rows",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ctx in state.get("contexts", []):
            writer.writerow({k: ctx.get(k, "") for k in fieldnames})


def find_wod_acquire_context_entry(state: Mapping[str, Any], context_name: str) -> Dict[str, Any]:
    for ctx in state.get("contexts", []):
        if str(ctx.get("context_name", "")).strip() == context_name:
            return dict(ctx)
    raise KeyError(f"context not found in acquire state: {context_name}")


def iter_contexts_for_download(
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
        if not bool(ctx.get("remote_processable", False)):
            continue
        status = str(ctx.get("download_status", "")).strip()
        if status == WOD_DOWNLOAD_STATUS_REMOTE_ONLY or (include_failed and status == WOD_DOWNLOAD_STATUS_FAILED):
            out.append(dict(ctx))
    return out
