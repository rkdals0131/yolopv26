from __future__ import annotations

import json
from pathlib import Path

from constants import (
    ARCHIVE_EXCLUDE_PATHS,
    DATASET_ARCHIVE_IN_VOLUME,
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


def _log(message: str) -> None:
    print(f"[modal-local-preflight] {message}", flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_status(dataset_root: Path) -> dict[str, object]:
    missing = [rel for rel in REQUIRED_DATASET_DIRS if not (dataset_root / rel).exists()]
    stats_path = dataset_root / "meta" / "final_dataset_stats.json"
    sample_count = None
    if stats_path.is_file():
        sample_count = json.loads(stats_path.read_text(encoding="utf-8")).get("sample_count")
    return {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.is_dir(),
        "missing_required_dirs": missing,
        "stats_path": str(stats_path),
        "sample_count": sample_count,
    }


def main() -> None:
    repo_root = _repo_root()
    _log("step 1/4 validating script constants")
    constants_summary = validate_modal_constants()
    _log(f"OK constants {constants_summary}")

    dataset_root = repo_root / LOCAL_REPO_DATASET_ROOT
    archive_path = repo_root / LOCAL_REPO_DATASET_ARCHIVE
    _log(f"step 2/4 checking local dataset root: {dataset_root}")
    status = _dataset_status(dataset_root)
    if not status["exists"]:
        raise FileNotFoundError(f"local dataset root not found: {dataset_root}")
    if status["missing_required_dirs"]:
        raise FileNotFoundError(
            f"local dataset layout missing under {dataset_root}: {status['missing_required_dirs']}"
        )
    _log(f"OK dataset status {status}")

    _log(f"step 3/4 checking local archive path: {archive_path}")
    if archive_path.is_file():
        _log(f"OK archive exists path={archive_path} size_bytes={archive_path.stat().st_size}")
    else:
        _log(f"ARCHIVE_MISSING path={archive_path}")
        _log("NEXT create it with the tar command printed below")

    _log("step 4/4 exact next commands")
    exclude_flags = "".join(f"  --exclude='{path}' \\\n" for path in ARCHIVE_EXCLUDE_PATHS)
    print(
        "\n# preferred: one checked local prepare/upload script\n"
        ".venv/bin/python modal/prepare_dataset_volume.py --create-archive --ensure-volumes --upload\n\n"
        "# manual equivalent, if you want to run each command yourself\n"
        "# 1. create dataset archive if missing\n"
        f"tar --zstd -cf {DATASET_ARCHIVE_NAME} \\\n"
        "  -C seg_dataset \\\n"
        f"{exclude_flags}"
        f"  {DATASET_ROOT_DIRNAME}\n\n"
        "# 2. create Modal volumes if needed\n"
        f".venv/bin/modal volume create {DATA_VOLUME_NAME}\n"
        f".venv/bin/modal volume create {RUNS_VOLUME_NAME}\n\n"
        "# 3. upload archive to the exact in-volume path expected by scripts\n"
        f".venv/bin/modal volume put -f {DATA_VOLUME_NAME} {DATASET_ARCHIVE_NAME} {DATASET_ARCHIVE_REMOTE_PATH}\n\n"
        "# 4. verify upload and Modal extraction/CUDA/layout\n"
        f".venv/bin/modal volume ls {DATA_VOLUME_NAME} /\n"
        ".venv/bin/modal run modal/check.py\n\n"
        "# 5. launch detached training\n"
        ".venv/bin/modal run --detach modal/train.py\n\n"
        "# expected Modal archive path:\n"
        f"# {DATASET_ARCHIVE_IN_VOLUME}\n"
        "# preset:\n"
        f"# {TRAIN_PRESET}\n"
    )


if __name__ == "__main__":
    main()
