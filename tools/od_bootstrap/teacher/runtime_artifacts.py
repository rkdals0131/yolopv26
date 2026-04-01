from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from common.io import now_iso, write_json


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _remove_path(destination)
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        try:
            os.symlink(source, destination)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def _refresh_latest_teacher_artifacts(
    *,
    teacher_root: Path,
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    latest_summary_path = teacher_root / "train_summary.json"
    write_json(latest_summary_path, summary)

    alias_weights_root = teacher_root / "weights"
    alias_actions: dict[str, str] = {}
    for checkpoint_name in ("best.pt", "last.pt", "last_resume.pt"):
        source_path = run_dir / "weights" / checkpoint_name
        if not source_path.is_file():
            continue
        alias_actions[checkpoint_name] = _link_or_copy_file(source_path, alias_weights_root / checkpoint_name)

    latest_run_payload = {
        "teacher_root": str(teacher_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "updated_at": now_iso(),
        "alias_actions": alias_actions,
    }
    latest_run_path = teacher_root / "latest_run.json"
    write_json(latest_run_path, latest_run_payload)
    return {
        "train_summary_path": str(latest_summary_path),
        "latest_run_path": str(latest_run_path),
        "weights_root": str(alias_weights_root),
        "alias_actions": alias_actions,
    }
