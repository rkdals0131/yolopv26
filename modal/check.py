from __future__ import annotations

from pathlib import Path

from sdk_import import modal

from constants import (
    APP_NAME,
    DATASET_ARCHIVE_IN_VOLUME,
    DATA_VOLUME_MOUNT,
    DATA_VOLUME_NAME,
    GPU_TYPE,
    LOCAL_DATASET_ROOT,
    MODAL_CPU_CORES,
    MODAL_EPHEMERAL_DISK_MB,
    MODAL_MEMORY_MB,
    REPO_UPLOAD_IGNORE,
    REMOTE_REPO_ROOT,
    REQUIRED_DATASET_DIRS,
    RUNS_VOLUME_MOUNT,
    RUNS_VOLUME_NAME,
    TIMEOUT_SECONDS,
    validate_modal_constants,
)


def _image() -> modal.Image:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "zstd", "libgl1", "libglib2.0-0")
        .pip_install_from_requirements(str(repo_root / "requirements.txt"))
        .add_local_dir(str(repo_root), remote_path=str(REMOTE_REPO_ROOT), ignore=REPO_UPLOAD_IGNORE)
    )


app = modal.App(f"{APP_NAME}-check")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)


from dataset_archive import extract_archive, verify_archive_contract, verify_layout


def _log(message: str) -> None:
    print(f"[modal-check] {message}", flush=True)


@app.function(
    image=_image(),
    gpu=GPU_TYPE,
    cpu=MODAL_CPU_CORES,
    memory=MODAL_MEMORY_MB,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MB,
    volumes={
        str(DATA_VOLUME_MOUNT): data_volume,
        str(RUNS_VOLUME_MOUNT): runs_volume,
    },
    timeout=TIMEOUT_SECONDS,
)
def check() -> dict[str, object]:
    import torch

    _log("step 1/5 validating hardcoded constants")
    constants_summary = validate_modal_constants()
    _log(f"OK constants {constants_summary}")

    _log(f"step 2/5 checking archive in mounted volume: {DATASET_ARCHIVE_IN_VOLUME}")
    verify_archive_contract(DATASET_ARCHIVE_IN_VOLUME)

    _log(f"step 3/5 extracting archive to local SSD: {LOCAL_DATASET_ROOT}")
    extract_archive(DATASET_ARCHIVE_IN_VOLUME, LOCAL_DATASET_ROOT)

    _log("step 4/5 verifying dataset layout")
    layout = verify_layout(LOCAL_DATASET_ROOT)

    _log("step 5/5 checking CUDA device")
    result = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_root": str(LOCAL_DATASET_ROOT),
        "layout": layout,
    }
    _log(f"OK check complete {result}")
    return result


@app.local_entrypoint()
def main() -> None:
    print(check.remote())
