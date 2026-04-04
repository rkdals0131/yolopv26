from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from common.io import now_iso, read_json, read_jsonl, timestamp_token, write_json


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def safe_name(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value).strip()
    )
    normalized = normalized.strip("_")
    return normalized or "meta_train"


def _normalize_scenario_snapshot(snapshot: dict[str, Any] | None, *, run_dir: Path) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    normalized = json_ready(snapshot)
    if not isinstance(normalized, dict):
        return None
    run_mapping = normalized.get("run")
    if not isinstance(run_mapping, dict):
        run_mapping = {}
    run_mapping["run_dir"] = str(run_dir)
    normalized["run"] = run_mapping
    return normalized


def resolve_meta_run_dir(scenario: Any, *, scenario_path: Path) -> Path:
    if scenario.run.run_dir is not None:
        return Path(scenario.run.run_dir)
    run_root = Path(scenario.run.run_root)
    timestamp = timestamp_token()
    prefix = safe_name(f"{scenario.run.run_name_prefix}_{scenario_path.stem}")
    candidate = run_root / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def build_meta_manifest_template(
    *,
    scenario: Any,
    scenario_path: Path,
    run_dir: Path,
    meta_manifest_version: str,
    scenario_snapshot: dict[str, Any] | None = None,
    selected_phase_window: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_snapshot = _normalize_scenario_snapshot(scenario_snapshot, run_dir=run_dir)
    manifest = {
        "version": meta_manifest_version,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "scenario_path": str(scenario_path),
        "run_dir": str(run_dir),
        "dataset": json_ready(asdict(scenario.dataset)),
        "run": json_ready(asdict(scenario.run)),
        "train_defaults": json_ready(asdict(scenario.train_defaults)),
        "selection": json_ready(asdict(scenario.selection)),
        "preview": json_ready(asdict(scenario.preview)),
        "active_phase_index": None,
        "active_phase_name": None,
        "phases": [
            {
                "index": index + 1,
                "name": phase.name,
                "stage": phase.stage,
                "status": "pending",
                "run_dir": str(run_dir / f"phase_{index + 1}"),
                "summary_path": str(run_dir / f"phase_{index + 1}" / "summary.json"),
                "best_checkpoint_path": None,
                "last_checkpoint_path": None,
                "completed_epochs": 0,
                "best_metric_value": None,
                "best_epoch": None,
                "promotion_reason": None,
                "preview": {},
            }
            for index, phase in enumerate(scenario.phases)
        ],
    }
    if selected_phase_window is not None:
        manifest["selected_phase_window"] = json_ready(selected_phase_window)
    if lineage is not None:
        manifest["lineage"] = json_ready(lineage)
    if normalized_snapshot is not None:
        manifest["scenario_snapshot"] = normalized_snapshot
    return manifest


def load_or_init_meta_manifest(
    *,
    scenario: Any,
    scenario_path: Path,
    run_dir: Path,
    meta_manifest_version: str,
    scenario_snapshot: dict[str, Any] | None = None,
    selected_phase_window: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    manifest_path = run_dir / "meta_manifest.json"
    normalized_snapshot = _normalize_scenario_snapshot(scenario_snapshot, run_dir=run_dir)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if normalized_snapshot is not None and not isinstance(manifest.get("scenario_snapshot"), dict):
            manifest["scenario_snapshot"] = normalized_snapshot
            manifest["updated_at"] = now_iso()
            write_json(manifest_path, manifest)
        if selected_phase_window is not None and not isinstance(manifest.get("selected_phase_window"), dict):
            manifest["selected_phase_window"] = json_ready(selected_phase_window)
            manifest["updated_at"] = now_iso()
            write_json(manifest_path, manifest)
        if lineage is not None and not isinstance(manifest.get("lineage"), dict):
            manifest["lineage"] = json_ready(lineage)
            manifest["updated_at"] = now_iso()
            write_json(manifest_path, manifest)
        return manifest, manifest_path
    manifest = build_meta_manifest_template(
        scenario=scenario,
        scenario_path=scenario_path,
        run_dir=run_dir,
        meta_manifest_version=meta_manifest_version,
        scenario_snapshot=normalized_snapshot,
        selected_phase_window=selected_phase_window,
        lineage=lineage,
    )
    write_json(manifest_path, manifest)
    return manifest, manifest_path


def write_meta_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    write_json(manifest_path, manifest)


def write_meta_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    phases = manifest.get("phases", [])
    completed = [phase for phase in phases if phase.get("status") == "completed"]
    skipped = [phase for phase in phases if phase.get("status") == "skipped"]
    executable = [phase for phase in phases if phase.get("status") != "skipped"]
    latest_phase = None
    if completed:
        latest_phase = completed[-1]
    elif phases:
        active_phase_index = int(manifest.get("active_phase_index") or 0)
        if 1 <= active_phase_index <= len(phases):
            latest_phase = phases[active_phase_index - 1]
    summary = {
        "version": manifest["version"],
        "status": manifest["status"],
        "scenario_path": manifest["scenario_path"],
        "run_dir": manifest["run_dir"],
        "train_defaults": manifest.get("train_defaults", {}),
        "completed_phases": len(completed),
        "skipped_phases": len(skipped),
        "completed_selected_phases": len(completed),
        "total_selected_phases": len(executable),
        "total_phases": len(phases),
        "active_phase_index": manifest.get("active_phase_index"),
        "active_phase_name": manifest.get("active_phase_name"),
        "final_checkpoint_path": completed[-1]["best_checkpoint_path"] if completed else None,
        "phases": phases,
        "updated_at": manifest["updated_at"],
    }
    if isinstance(manifest.get("selected_phase_window"), dict):
        summary["selected_phase_window"] = json_ready(manifest["selected_phase_window"])
    if isinstance(manifest.get("lineage"), dict):
        summary["lineage"] = json_ready(manifest["lineage"])
    if isinstance(latest_phase, dict):
        backbone = latest_phase.get("backbone", {})
        postprocess = latest_phase.get("postprocess", {})
        selection = latest_phase.get("selection", {})
        summary["latest_phase_name"] = latest_phase.get("name")
        summary["latest_phase_stage"] = latest_phase.get("stage")
        if isinstance(backbone, dict):
            summary["latest_backbone_variant"] = backbone.get("variant")
            summary["latest_backbone_weights"] = backbone.get("weights")
        if isinstance(postprocess, dict):
            summary["latest_postprocess"] = postprocess
        if isinstance(selection, dict):
            summary["latest_selection_metric_path"] = selection.get("metric_path")
            summary["latest_selection_mode"] = selection.get("mode")
    write_json(run_dir / "summary.json", summary)


def phase_summary_indicates_complete(phase_run_dir: Path, phase: Any) -> bool:
    summary_path = phase_run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    if "early_exit" in summary:
        return True
    return int(summary.get("completed_epochs", 0)) >= int(phase.max_epochs)


def recover_phase_entry_from_run_dir(entry: dict[str, Any], phase: Any) -> dict[str, Any] | None:
    if str(entry.get("status") or "") == "skipped":
        return None
    phase_run_dir = Path(entry["run_dir"])
    if not phase_summary_indicates_complete(phase_run_dir, phase):
        return None
    summary_path = phase_run_dir / "summary.json"
    summary = read_json(summary_path)
    checkpoint_paths = summary.get("checkpoint_paths", {}) if isinstance(summary.get("checkpoint_paths"), dict) else {}
    best_checkpoint = checkpoint_paths.get("best")
    last_checkpoint = checkpoint_paths.get("last")
    return {
        "status": "completed",
        "run_dir": str(phase_run_dir),
        "summary_path": str(summary_path),
        "run_manifest_path": str(phase_run_dir / "run_manifest.json"),
        "best_checkpoint_path": str(best_checkpoint) if best_checkpoint else None,
        "last_checkpoint_path": str(last_checkpoint) if last_checkpoint else None,
        "completed_epochs": int(summary.get("completed_epochs", 0)),
        "best_metric_value": summary.get("best_metric_value"),
        "best_epoch": summary.get("best_epoch"),
        "promotion_reason": (
            summary.get("early_exit", {}).get("reason", "completed")
            if isinstance(summary.get("early_exit"), dict)
            else "completed"
        ),
        "phase_state": summary.get("early_exit", {}).get("phase_state")
        if isinstance(summary.get("early_exit"), dict)
        else None,
    }


def phase_entry_is_completed(entry: dict[str, Any], phase: Any) -> bool:
    if entry.get("status") == "completed" and entry.get("best_checkpoint_path"):
        return True
    phase_run_dir = Path(entry["run_dir"])
    return phase_summary_indicates_complete(phase_run_dir, phase)


def phase_entry_is_skipped(entry: dict[str, Any]) -> bool:
    return str(entry.get("status") or "") == "skipped"


def phase_entry_is_terminal(entry: dict[str, Any], phase: Any) -> bool:
    return phase_entry_is_skipped(entry) or phase_entry_is_completed(entry, phase)
