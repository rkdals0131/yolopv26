from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    PERSISTED_RUN_ROOT,
    REPO_UPLOAD_IGNORE,
    REMOTE_REPO_ROOT,
    REMOTE_RUN_ROOT,
    REQUIRED_DATASET_DIRS,
    RUN_GROUP_NAME,
    RUNS_VOLUME_MOUNT,
    RUNS_VOLUME_NAME,
    TIMEOUT_SECONDS,
    TRAIN_PRESET,
    validate_modal_constants,
)
from dataset_archive import extract_archive, verify_archive_contract, verify_layout


def _log(message: str) -> None:
    print(f"[modal-train] {message}", flush=True)


def _image() -> modal.Image:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "zstd", "libgl1", "libglib2.0-0")
        .pip_install_from_requirements(str(repo_root / "requirements.txt"))
        .add_local_dir(str(repo_root), remote_path=str(REMOTE_REPO_ROOT), ignore=REPO_UPLOAD_IGNORE)
    )


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)


def _write_modal_path_config() -> None:
    config_path = REMOTE_REPO_ROOT / "config" / "user_paths.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        "pv26_train:\n"
        f"  dataset_root: {LOCAL_DATASET_ROOT}\n"
        f"  run_root: {REMOTE_RUN_ROOT}\n"
    )
    config_path.write_text(payload, encoding="utf-8")
    written = config_path.read_text(encoding="utf-8")
    if written != payload:
        raise RuntimeError(f"failed to write expected Modal path config: {config_path}")
    _log(f"OK wrote path config path={config_path} dataset_root={LOCAL_DATASET_ROOT} run_root={REMOTE_RUN_ROOT}")


def _verify_training_plan() -> None:
    if str(REMOTE_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REMOTE_REPO_ROOT))
    from model.engine.loss import STAGE_LOSS_WEIGHTS
    from tools.pv26_train.cli import load_meta_train_scenario

    scenario = load_meta_train_scenario(TRAIN_PRESET)
    errors: list[str] = []
    if Path(scenario.dataset.root) != LOCAL_DATASET_ROOT:
        errors.append(f"dataset.root {scenario.dataset.root} != {LOCAL_DATASET_ROOT}")
    if Path(scenario.run.run_root) != REMOTE_RUN_ROOT:
        errors.append(f"run.run_root {scenario.run.run_root} != {REMOTE_RUN_ROOT}")
    if str(scenario.run.run_name_prefix) != TRAIN_PRESET:
        errors.append(f"run.run_name_prefix {scenario.run.run_name_prefix} != {TRAIN_PRESET}")
    if str(scenario.train_defaults.task_mode) != "roadmark_joint":
        errors.append(f"train_defaults.task_mode {scenario.train_defaults.task_mode} != roadmark_joint")
    if bool(scenario.train_defaults.amp):
        errors.append("seg-first Modal preset must run amp=False")
    if int(scenario.train_defaults.train_batches) != 512:
        errors.append(f"train_defaults.train_batches {scenario.train_defaults.train_batches} != 512")
    if int(scenario.train_defaults.val_batches) != 256:
        errors.append(f"train_defaults.val_batches {scenario.train_defaults.val_batches} != 256")
    if str(scenario.train_defaults.task_positive_task) != "multi:lane,stopline,crosswalk":
        errors.append(
            "train_defaults.task_positive_task "
            f"{scenario.train_defaults.task_positive_task} != multi:lane,stopline,crosswalk"
        )
    if float(scenario.train_defaults.task_positive_fraction or 0.0) != 0.5:
        errors.append(f"train_defaults.task_positive_fraction {scenario.train_defaults.task_positive_fraction} != 0.5")
    if not bool(getattr(scenario.train_defaults, "train_augmentation", False)):
        errors.append("seg-first Modal preset must run train_augmentation=True")
    if int(getattr(scenario.train_defaults, "train_augmentation_seed", -1) or -1) != 26:
        errors.append(
            "train_defaults.train_augmentation_seed "
            f"{getattr(scenario.train_defaults, 'train_augmentation_seed', None)} != 26"
        )

    phase_weights: dict[str, dict[str, float]] = {}
    for phase in scenario.phases:
        weights = dict(STAGE_LOSS_WEIGHTS[str(phase.stage)])
        weights.update({str(name): float(value) for name, value in dict(phase.loss_weights).items()})
        phase_weights[str(phase.name)] = weights

    for phase_name in ("head_warmup", "partial_unfreeze", "end_to_end_finetune"):
        weights = phase_weights.get(phase_name, {})
        for task_name in ("det", "tl_attr", "lane", "stop_line"):
            if float(weights.get(task_name, 0.0)) <= 0.0:
                errors.append(f"{phase_name} must train {task_name}, got weight={weights.get(task_name)}")
    for phase_name in ("partial_unfreeze", "end_to_end_finetune", "lane_family_finetune"):
        weights = phase_weights.get(phase_name, {})
        if float(weights.get("crosswalk", 0.0)) <= 0.0:
            errors.append(f"{phase_name} must train crosswalk, got weight={weights.get('crosswalk')}")
    lane_family = phase_weights.get("lane_family_finetune", {})
    if float(lane_family.get("det", 0.0)) != 0.0 or float(lane_family.get("tl_attr", 0.0)) != 0.0:
        errors.append(f"lane_family_finetune must freeze detector/TL losses, got {lane_family}")

    if errors:
        raise RuntimeError("Modal training plan mismatch:\n- " + "\n- ".join(errors))

    payload: dict[str, Any] = {
        "preset": TRAIN_PRESET,
        "dataset_root": str(scenario.dataset.root),
        "run_root": str(scenario.run.run_root),
        "run_name_prefix": scenario.run.run_name_prefix,
        "task_mode": scenario.train_defaults.task_mode,
        "train_augmentation": scenario.train_defaults.train_augmentation,
        "train_augmentation_seed": scenario.train_defaults.train_augmentation_seed,
        "phase_loss_weights": phase_weights,
    }
    _log(f"OK training plan {payload}")


def _sync_runs_to_volume() -> None:
    _log(f"syncing runs to volume source={REMOTE_RUN_ROOT} dest={PERSISTED_RUN_ROOT}")
    PERSISTED_RUN_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not REMOTE_RUN_ROOT.exists():
        _log(f"WARN no remote run root to sync yet: {REMOTE_RUN_ROOT}")
        runs_volume.commit()
        return
    shutil.copytree(REMOTE_RUN_ROOT, PERSISTED_RUN_ROOT, dirs_exist_ok=True)
    runs_volume.commit()
    manifest_count = len(list(PERSISTED_RUN_ROOT.glob("*/meta_manifest.json")))
    _log(f"OK committed runs volume group={RUN_GROUP_NAME} dest={PERSISTED_RUN_ROOT} meta_runs={manifest_count}")


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
def train() -> dict[str, str]:
    _log("step 1/8 validating hardcoded constants")
    constants_summary = validate_modal_constants()
    _log(f"OK constants {constants_summary}")

    _log(f"step 2/8 checking archive in mounted volume: {DATASET_ARCHIVE_IN_VOLUME}")
    verify_archive_contract(DATASET_ARCHIVE_IN_VOLUME)

    _log(f"step 3/8 extracting archive to local SSD: {LOCAL_DATASET_ROOT}")
    extract_archive(DATASET_ARCHIVE_IN_VOLUME, LOCAL_DATASET_ROOT)

    _log("step 4/8 verifying dataset layout")
    verify_layout(LOCAL_DATASET_ROOT)

    _log("step 5/8 writing Modal path config")
    _write_modal_path_config()

    _log("step 6/8 verifying training plan")
    _verify_training_plan()

    command = ["python3", "-m", "tools.pv26_train.cli", "--preset", TRAIN_PRESET]
    _log(f"step 7/8 launching training cwd={REMOTE_REPO_ROOT} command={' '.join(command)}")
    try:
        subprocess.run(command, cwd=str(REMOTE_REPO_ROOT), check=True)
    finally:
        _log("step 8/8 preserving runs in Modal Volume")
        _sync_runs_to_volume()
    result = {
        "preset": TRAIN_PRESET,
        "dataset_root": str(LOCAL_DATASET_ROOT),
        "remote_run_root": str(REMOTE_RUN_ROOT),
        "persisted_run_root": str(PERSISTED_RUN_ROOT),
    }
    _log(f"OK train complete {result}")
    return result


@app.local_entrypoint()
def main() -> None:
    print(train.remote())
