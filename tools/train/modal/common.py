from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def default_torch_specs_for_gpu(gpu_name: str) -> tuple[str, str]:
    normalized = gpu_name.strip().upper()
    if "B200" in normalized:
        return "torch>=2.7.0", "torchvision>=0.22.0"
    return "torch==2.6.0", "torchvision==0.21.0"


@dataclass(frozen=True)
class ModalRuntimeDefaults:
    app_name: str
    dataset_volume_name: str
    artifact_volume_name: str
    gpu_name: str
    torch_spec: str
    torchvision_spec: str
    timeout_sec: int
    cpu: float
    memory_mb: int


@dataclass(frozen=True)
class ModalTrainDefaults:
    epochs: int
    batch_size: int
    workers: int
    prefetch_factor: int
    persistent_workers: bool
    augment: bool
    lr: str
    optimizer: str
    weight_decay: str
    momentum: str
    scheduler: str
    min_lr_ratio: str
    warmup_epochs: int
    warmup_start_factor: str
    compile: bool
    compile_mode: str
    compile_fullgraph: bool
    compile_seg_loss: bool
    seg_output_stride: int
    det_pretrained: str
    log_every: int
    progress: bool
    tensorboard: bool
    profile_every: int
    profile_sync_cuda: bool
    profile_system: bool
    eval_map_every: int
    train_drop_last: bool


@dataclass(frozen=True)
class ModalDatasetDefaults:
    dataset_dir_in_volume: str
    dataset_tar_in_volume: str
    artifact_root_in_volume: str


@dataclass(frozen=True)
class ModalSyncDefaults:
    auto_download_artifacts: bool
    local_artifact_root: str
    sync_every_n_epochs: int
    sync_poll_sec: int


def build_train_command(
    *,
    train_script: Path,
    dataset_root: Path,
    out_root: Path,
    run_name: str,
    train_defaults: ModalTrainDefaults,
    augment: bool | None = None,
) -> list[str]:
    effective_augment = train_defaults.augment if augment is None else bool(augment)
    cmd = [
        str(train_script),
        "--dataset-root",
        str(dataset_root),
        "--out-dir",
        str(out_root),
        "--epochs",
        str(train_defaults.epochs),
        "--batch-size",
        str(train_defaults.batch_size),
        "--workers",
        str(train_defaults.workers),
        "--prefetch-factor",
        str(train_defaults.prefetch_factor),
        "--lr",
        str(train_defaults.lr),
        "--optimizer",
        str(train_defaults.optimizer),
        "--weight-decay",
        str(train_defaults.weight_decay),
        "--momentum",
        str(train_defaults.momentum),
        "--scheduler",
        str(train_defaults.scheduler),
        "--min-lr-ratio",
        str(train_defaults.min_lr_ratio),
        "--warmup-epochs",
        str(train_defaults.warmup_epochs),
        "--warmup-start-factor",
        str(train_defaults.warmup_start_factor),
        "--eval-map-every",
        str(train_defaults.eval_map_every),
        "--compile-mode",
        str(train_defaults.compile_mode),
        "--log-every",
        str(train_defaults.log_every),
        "--profile-every",
        str(train_defaults.profile_every),
        "--seg-output-stride",
        str(train_defaults.seg_output_stride),
        "--run-name",
        run_name,
    ]

    cmd.append("--compile" if train_defaults.compile else "--no-compile")
    cmd.append("--compile-fullgraph" if train_defaults.compile_fullgraph else "--no-compile-fullgraph")
    cmd.append("--compile-seg-loss" if train_defaults.compile_seg_loss else "--no-compile-seg-loss")
    cmd.append("--persistent-workers" if train_defaults.persistent_workers else "--no-persistent-workers")
    cmd.append("--progress" if train_defaults.progress else "--no-progress")
    cmd.append("--tensorboard" if train_defaults.tensorboard else "--no-tensorboard")
    cmd.append("--augment" if effective_augment else "--no-augment")

    if train_defaults.profile_sync_cuda:
        cmd.append("--profile-sync-cuda")
    if train_defaults.profile_system:
        cmd.append("--profile-system")
    if train_defaults.train_drop_last:
        cmd.append("--train-drop-last")
    if train_defaults.det_pretrained.strip():
        cmd.extend(["--det-pretrained", train_defaults.det_pretrained.strip()])

    return cmd


def format_modal_profile(
    *,
    runtime_defaults: ModalRuntimeDefaults,
    train_defaults: ModalTrainDefaults,
    augment: bool,
) -> str:
    return (
        "[modal] profile: "
        f"gpu={runtime_defaults.gpu_name} "
        f"cpu={runtime_defaults.cpu} "
        f"memory_mb={runtime_defaults.memory_mb} "
        f"batch={train_defaults.batch_size} "
        f"augment={bool(augment)} "
        f"optimizer={train_defaults.optimizer} "
        f"lr={train_defaults.lr} "
        f"scheduler={train_defaults.scheduler} "
        f"warmup_epochs={train_defaults.warmup_epochs} "
        f"compile={train_defaults.compile} "
        f"compile_mode={train_defaults.compile_mode} "
        f"compile_fullgraph={train_defaults.compile_fullgraph} "
        f"compile_seg_loss={train_defaults.compile_seg_loss} "
        f"seg_output_stride={train_defaults.seg_output_stride} "
        f"workers={train_defaults.workers} "
        f"prefetch={train_defaults.prefetch_factor} "
        f"persistent_workers={train_defaults.persistent_workers} "
        f"progress={train_defaults.progress} "
        f"tensorboard={train_defaults.tensorboard} "
        f"log_every={train_defaults.log_every} "
        f"profile_every={train_defaults.profile_every} "
        f"profile_sync_cuda={train_defaults.profile_sync_cuda} "
        f"profile_system={train_defaults.profile_system} "
        f"eval_map_every={train_defaults.eval_map_every} "
        f"torch_spec={runtime_defaults.torch_spec} "
        f"torchvision_spec={runtime_defaults.torchvision_spec}"
    )
