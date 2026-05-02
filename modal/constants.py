from __future__ import annotations

from pathlib import Path

APP_NAME = "pv26-unified-roadmark-segfirst-a100"
GPU_TYPE = "A100-40GB"
TIMEOUT_SECONDS = 60 * 60 * 24
MODAL_CPU_CORES = 24
MODAL_MEMORY_MB = 98_304  # 96 GiB
MODAL_EPHEMERAL_DISK_MB = 786_432  # 768 GiB local SSD quota for archive extraction + run cache

DATASET_ROOT_DIRNAME = "pv26_exhaustive_od_lane_dataset"
DATASET_ARCHIVE_NAME = f"{DATASET_ROOT_DIRNAME}.tar.zst"
LOCAL_REPO_DATASET_ROOT = Path("seg_dataset") / DATASET_ROOT_DIRNAME
LOCAL_REPO_DATASET_ARCHIVE = Path(DATASET_ARCHIVE_NAME)
DATASET_ARCHIVE_REMOTE_PATH = Path("/") / DATASET_ARCHIVE_NAME
ARCHIVE_EXCLUDE_PATHS = (
    f"{DATASET_ROOT_DIRNAME}/debug_vis_lane_audit",
)

# Modal Volumes and in-volume dataset archive location. Edit these constants in source;
# the Modal entrypoints intentionally accept no CLI arguments.
DATA_VOLUME_NAME = "pv26-dataset-archives"
RUNS_VOLUME_NAME = "pv26-training-runs"
DATA_VOLUME_MOUNT = Path("/volumes/pv26_dataset_archives")
RUNS_VOLUME_MOUNT = Path("/volumes/pv26_training_runs")
DATASET_ARCHIVE_IN_VOLUME = DATA_VOLUME_MOUNT / DATASET_ARCHIVE_NAME

REMOTE_REPO_ROOT = Path("/root/yolopv26")
LOCAL_SSD_ROOT = Path("/local")
LOCAL_DATASET_ROOT = LOCAL_SSD_ROOT / DATASET_ROOT_DIRNAME

TRAIN_PRESET = "pv26_unified_roadmark_segfirst_a100"
RUN_GROUP_NAME = TRAIN_PRESET
REMOTE_RUN_ROOT = REMOTE_REPO_ROOT / "runs" / RUN_GROUP_NAME
PERSISTED_RUN_ROOT = RUNS_VOLUME_MOUNT / RUN_GROUP_NAME
REQUIRED_DATASET_DIRS = (
    "images/train",
    "images/val",
    "images/test",
    "labels_scene/train",
    "labels_scene/val",
    "labels_scene/test",
    "labels_det/train",
    "labels_det/val",
    "labels_det/test",
    "meta",
)

# Code is uploaded to Modal separately from the dataset archive. Keep local
# datasets/runs out of the app upload path; the train/check entrypoints read
# the dataset only from DATA_VOLUME_NAME -> DATASET_ARCHIVE_IN_VOLUME.
REPO_UPLOAD_IGNORE = (
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "seg_dataset/**",
    "runs/**",
    "analysis_exports/**",
    "debug_vis_lane_audit/**",
    "yolo26m.pt",
)


def validate_modal_constants() -> dict[str, str]:
    """Validate path/name constants that must stay in lockstep across scripts/docs."""

    errors: list[str] = []
    if DATASET_ARCHIVE_IN_VOLUME.name != DATASET_ARCHIVE_NAME:
        errors.append(
            "DATASET_ARCHIVE_IN_VOLUME filename does not match DATASET_ARCHIVE_NAME: "
            f"{DATASET_ARCHIVE_IN_VOLUME.name} != {DATASET_ARCHIVE_NAME}"
        )
    expected_archive_in_volume = DATA_VOLUME_MOUNT / DATASET_ARCHIVE_REMOTE_PATH.relative_to("/")
    if DATASET_ARCHIVE_IN_VOLUME != expected_archive_in_volume:
        errors.append(
            "DATASET_ARCHIVE_IN_VOLUME must match DATA_VOLUME_MOUNT + DATASET_ARCHIVE_REMOTE_PATH: "
            f"{DATASET_ARCHIVE_IN_VOLUME} != {expected_archive_in_volume}"
        )
    if DATASET_ARCHIVE_REMOTE_PATH.name != DATASET_ARCHIVE_NAME:
        errors.append(
            "DATASET_ARCHIVE_REMOTE_PATH filename does not match DATASET_ARCHIVE_NAME: "
            f"{DATASET_ARCHIVE_REMOTE_PATH.name} != {DATASET_ARCHIVE_NAME}"
        )
    if LOCAL_DATASET_ROOT.name != DATASET_ROOT_DIRNAME:
        errors.append(
            "LOCAL_DATASET_ROOT dirname does not match DATASET_ROOT_DIRNAME: "
            f"{LOCAL_DATASET_ROOT.name} != {DATASET_ROOT_DIRNAME}"
        )
    if LOCAL_REPO_DATASET_ROOT.name != DATASET_ROOT_DIRNAME:
        errors.append(
            "LOCAL_REPO_DATASET_ROOT dirname does not match DATASET_ROOT_DIRNAME: "
            f"{LOCAL_REPO_DATASET_ROOT.name} != {DATASET_ROOT_DIRNAME}"
        )
    if LOCAL_REPO_DATASET_ARCHIVE.name != DATASET_ARCHIVE_NAME:
        errors.append(
            "LOCAL_REPO_DATASET_ARCHIVE filename does not match DATASET_ARCHIVE_NAME: "
            f"{LOCAL_REPO_DATASET_ARCHIVE.name} != {DATASET_ARCHIVE_NAME}"
        )
    if TRAIN_PRESET != "pv26_unified_roadmark_segfirst_a100":
        errors.append(f"unexpected TRAIN_PRESET: {TRAIN_PRESET}")
    if RUN_GROUP_NAME != TRAIN_PRESET:
        errors.append(f"RUN_GROUP_NAME must equal TRAIN_PRESET: {RUN_GROUP_NAME} vs {TRAIN_PRESET}")
    if REMOTE_RUN_ROOT.name != RUN_GROUP_NAME:
        errors.append(f"REMOTE_RUN_ROOT dirname must equal RUN_GROUP_NAME: {REMOTE_RUN_ROOT} vs {RUN_GROUP_NAME}")
    if PERSISTED_RUN_ROOT.name != RUN_GROUP_NAME:
        errors.append(f"PERSISTED_RUN_ROOT dirname must equal RUN_GROUP_NAME: {PERSISTED_RUN_ROOT} vs {RUN_GROUP_NAME}")
    if errors:
        raise ValueError("Modal constant mismatch:\n- " + "\n- ".join(errors))
    return {
        "dataset_root_dirname": DATASET_ROOT_DIRNAME,
        "dataset_archive_name": DATASET_ARCHIVE_NAME,
        "local_repo_dataset_root": str(LOCAL_REPO_DATASET_ROOT),
        "local_repo_dataset_archive": str(LOCAL_REPO_DATASET_ARCHIVE),
        "modal_archive_remote_path": str(DATASET_ARCHIVE_REMOTE_PATH),
        "modal_archive_path": str(DATASET_ARCHIVE_IN_VOLUME),
        "modal_local_dataset_root": str(LOCAL_DATASET_ROOT),
        "train_preset": TRAIN_PRESET,
        "run_group_name": RUN_GROUP_NAME,
        "remote_run_root": str(REMOTE_RUN_ROOT),
        "persisted_run_root": str(PERSISTED_RUN_ROOT),
    }
