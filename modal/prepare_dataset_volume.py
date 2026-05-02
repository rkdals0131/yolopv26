from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from constants import (
    ARCHIVE_EXCLUDE_PATHS,
    DATASET_ARCHIVE_NAME,
    DATASET_ARCHIVE_REMOTE_PATH,
    DATASET_ROOT_DIRNAME,
    DATA_VOLUME_NAME,
    LOCAL_REPO_DATASET_ARCHIVE,
    LOCAL_REPO_DATASET_ROOT,
    REQUIRED_DATASET_DIRS,
    RUNS_VOLUME_NAME,
    TRAIN_PRESET,
    validate_modal_constants,
)
from dataset_archive import verify_archive_contract


def _log(message: str) -> None:
    print(f"[modal-prepare] {message}", flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    _log(f"RUN cwd={cwd} command={' '.join(command)}")
    try:
        return subprocess.run(command, cwd=str(cwd), check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "command failed "
            f"exit_code={exc.returncode} command={' '.join(command)}"
        ) from exc


def _capture(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    _log(f"RUN cwd={cwd} command={' '.join(command)}")
    try:
        return subprocess.run(command, cwd=str(cwd), check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "command failed "
            f"exit_code={exc.returncode} command={' '.join(command)} "
            f"stdout={stdout!r} stderr={stderr!r}"
        ) from exc


def _modal_cli(repo_root: Path) -> str:
    local_modal = repo_root / ".venv" / "bin" / "modal"
    if local_modal.is_file():
        return str(local_modal)
    resolved = shutil.which("modal")
    if resolved is not None:
        return resolved
    raise FileNotFoundError("Modal CLI not found. Expected .venv/bin/modal or modal on PATH.")


def _check_dataset(dataset_root: Path) -> dict[str, Any]:
    missing = [rel for rel in REQUIRED_DATASET_DIRS if not (dataset_root / rel).exists()]
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if missing:
        raise FileNotFoundError(f"dataset layout missing under {dataset_root}: {missing}")
    stats_path = dataset_root / "meta" / "final_dataset_stats.json"
    sample_count = None
    if stats_path.is_file():
        sample_count = json.loads(stats_path.read_text(encoding="utf-8")).get("sample_count")
    payload = {
        "dataset_root": str(dataset_root),
        "stats_path": str(stats_path),
        "sample_count": sample_count,
        "required_dirs": len(REQUIRED_DATASET_DIRS),
    }
    _log(f"OK dataset layout {payload}")
    return payload


def _create_archive(repo_root: Path, archive_path: Path, *, force: bool) -> None:
    if archive_path.exists() and not force:
        _log(f"OK archive exists path={archive_path} size_bytes={archive_path.stat().st_size}")
        verify_archive_contract(archive_path)
        return
    if archive_path.exists() and force:
        archive_path.unlink()
    command = [
        "tar",
        "--zstd",
        "-cf",
        DATASET_ARCHIVE_NAME,
        "-C",
        "seg_dataset",
    ]
    for excluded in ARCHIVE_EXCLUDE_PATHS:
        command.append(f"--exclude={excluded}")
    command.append(DATASET_ROOT_DIRNAME)
    _run(command, cwd=repo_root)
    verify_archive_contract(archive_path)
    _log(f"OK archive ready path={archive_path} size_bytes={archive_path.stat().st_size}")


def _json_contains_name(payload: Any, expected_name: str) -> bool:
    if isinstance(payload, dict):
        return any(_json_contains_name(value, expected_name) for value in payload.values())
    if isinstance(payload, list):
        return any(_json_contains_name(item, expected_name) for item in payload)
    value = str(payload).strip()
    return value == expected_name or Path(value).name == expected_name


def _volume_exists(modal_cli: str, repo_root: Path, volume_name: str) -> bool:
    try:
        result = _capture([modal_cli, "volume", "list", "--json"], cwd=repo_root)
        payload = json.loads(result.stdout)
        return _json_contains_name(payload, volume_name)
    except (json.JSONDecodeError, RuntimeError):
        result = _capture([modal_cli, "volume", "list"], cwd=repo_root)
        return volume_name in result.stdout


def _ensure_volume(modal_cli: str, repo_root: Path, volume_name: str) -> None:
    if _volume_exists(modal_cli, repo_root, volume_name):
        _log(f"OK volume exists name={volume_name}")
        return
    _run([modal_cli, "volume", "create", volume_name], cwd=repo_root)
    if not _volume_exists(modal_cli, repo_root, volume_name):
        raise RuntimeError(f"Modal volume creation was not confirmed: {volume_name}")
    _log(f"OK volume created name={volume_name}")


def _remote_archive_exists(modal_cli: str, repo_root: Path) -> bool:
    try:
        result = _capture([modal_cli, "volume", "ls", DATA_VOLUME_NAME, "/", "--json"], cwd=repo_root)
        payload = json.loads(result.stdout)
        return _json_contains_name(payload, DATASET_ARCHIVE_NAME)
    except (json.JSONDecodeError, RuntimeError):
        result = _capture([modal_cli, "volume", "ls", DATA_VOLUME_NAME, "/"], cwd=repo_root)
        return DATASET_ARCHIVE_NAME in result.stdout


def _upload_archive(modal_cli: str, repo_root: Path, archive_path: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive missing before upload: {archive_path}")
    verify_archive_contract(archive_path)
    _run(
        [
            modal_cli,
            "volume",
            "put",
            "-f",
            DATA_VOLUME_NAME,
            DATASET_ARCHIVE_NAME,
            str(DATASET_ARCHIVE_REMOTE_PATH),
        ],
        cwd=repo_root,
    )
    if not _remote_archive_exists(modal_cli, repo_root):
        raise RuntimeError(
            f"uploaded archive was not confirmed in volume root: {DATA_VOLUME_NAME}:{DATASET_ARCHIVE_REMOTE_PATH}"
        )
    _log(f"OK uploaded archive volume={DATA_VOLUME_NAME} remote_path={DATASET_ARCHIVE_REMOTE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally upload the PV26 exhaustive dataset archive to Modal Volume."
    )
    parser.add_argument("--create-archive", action="store_true", help="Create the local .tar.zst archive if missing.")
    parser.add_argument("--force-archive", action="store_true", help="Rebuild the local archive if it already exists.")
    parser.add_argument("--ensure-volumes", action="store_true", help="Create Modal Volumes if they are missing.")
    parser.add_argument("--upload", action="store_true", help="Upload the archive to the dataset Modal Volume.")
    args = parser.parse_args()

    repo_root = _repo_root()
    constants_summary = validate_modal_constants()
    _log(f"OK constants {constants_summary}")

    dataset_root = repo_root / LOCAL_REPO_DATASET_ROOT
    archive_path = repo_root / LOCAL_REPO_DATASET_ARCHIVE
    _check_dataset(dataset_root)

    if args.create_archive or args.force_archive:
        _create_archive(repo_root, archive_path, force=bool(args.force_archive))
    elif archive_path.is_file():
        verify_archive_contract(archive_path)
        _log(f"OK archive exists path={archive_path} size_bytes={archive_path.stat().st_size}")
    else:
        _log(f"ARCHIVE_MISSING path={archive_path}")
        _log("NEXT rerun with --create-archive before --upload")

    modal_cli = _modal_cli(repo_root)
    version = _capture([modal_cli, "--version"], cwd=repo_root).stdout.strip()
    _log(f"OK modal cli path={modal_cli} version={version}")

    if args.ensure_volumes or args.upload:
        _ensure_volume(modal_cli, repo_root, DATA_VOLUME_NAME)
        _ensure_volume(modal_cli, repo_root, RUNS_VOLUME_NAME)

    if args.upload:
        _upload_archive(modal_cli, repo_root, archive_path)

    _log(
        "READY "
        f"preset={TRAIN_PRESET} dataset_volume={DATA_VOLUME_NAME} runs_volume={RUNS_VOLUME_NAME} "
        f"archive_remote_path={DATASET_ARCHIVE_REMOTE_PATH}"
    )


if __name__ == "__main__":
    main()
