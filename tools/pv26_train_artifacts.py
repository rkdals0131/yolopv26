from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        return []
    return [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_name(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value).strip()
    )
    normalized = normalized.strip("_")
    return normalized or "meta_train"


def resolve_meta_run_dir(scenario: Any, *, scenario_path: Path) -> Path:
    if scenario.run.run_dir is not None:
        return Path(scenario.run.run_dir)
    run_root = Path(scenario.run.run_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
) -> dict[str, Any]:
    return {
        "version": meta_manifest_version,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "scenario_path": str(scenario_path),
        "run_dir": str(run_dir),
        "dataset": json_ready(asdict(scenario.dataset)),
        "run": json_ready(asdict(scenario.run)),
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


def load_or_init_meta_manifest(
    *,
    scenario: Any,
    scenario_path: Path,
    run_dir: Path,
    meta_manifest_version: str,
) -> tuple[dict[str, Any], Path]:
    manifest_path = run_dir / "meta_manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path), manifest_path
    manifest = build_meta_manifest_template(
        scenario=scenario,
        scenario_path=scenario_path,
        run_dir=run_dir,
        meta_manifest_version=meta_manifest_version,
    )
    write_json(manifest_path, manifest)
    return manifest, manifest_path


def write_meta_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    write_json(manifest_path, manifest)


def write_meta_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    phases = manifest.get("phases", [])
    completed = [phase for phase in phases if phase.get("status") == "completed"]
    summary = {
        "version": manifest["version"],
        "status": manifest["status"],
        "scenario_path": manifest["scenario_path"],
        "run_dir": manifest["run_dir"],
        "completed_phases": len(completed),
        "total_phases": len(phases),
        "active_phase_index": manifest.get("active_phase_index"),
        "active_phase_name": manifest.get("active_phase_name"),
        "final_checkpoint_path": completed[-1]["best_checkpoint_path"] if completed else None,
        "phases": phases,
        "updated_at": manifest["updated_at"],
    }
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
