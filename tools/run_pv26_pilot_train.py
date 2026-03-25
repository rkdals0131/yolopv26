from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.heads import PV26Heads
from model.loading import PV26CanonicalDataset, build_pv26_eval_dataloader, build_pv26_train_dataloader
from model.training import PV26Trainer, build_pv26_optimizer, build_pv26_scheduler
from model.trunk import build_yolo26n_trunk


DEFAULT_AIHUB_ROOT = REPO_ROOT / "seg_dataset" / "pv26_aihub_standardized"
DEFAULT_BDD_ROOT = REPO_ROOT / "seg_dataset" / "pv26_bdd100k_standardized"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "pv26_train" / "pilot"
HEAD_CHANNELS = (64, 128, 256)


@dataclass(frozen=True)
class DatasetConfig:
    # AIHUB canonical root. Must exist before training.
    aihub_root: Path = DEFAULT_AIHUB_ROOT
    # Turn on when BDD canonical root should be mixed into training.
    include_bdd: bool = True
    # BDD canonical root. Used only when include_bdd is True.
    bdd_root: Path = DEFAULT_BDD_ROOT


@dataclass(frozen=True)
class TrainConfig:
    # Stage name from the documented training schedule.
    stage: str = "stage_1_frozen_trunk_warmup"
    # Device string passed to torch.
    device: str = "cpu"
    # Epoch count for this run.
    epochs: int = 3
    # Per-batch sample count.
    batch_size: int = 4
    # Number of train batches consumed each epoch.
    train_batches: int = 50
    # Number of validation batches. Set 0 to disable validation.
    val_batches: int = 10
    # Optimizer hyperparameters.
    trunk_lr: float = 1e-4
    head_lr: float = 5e-3
    weight_decay: float = 1e-4
    # Scheduler name: "none" or "cosine".
    schedule: str = "cosine"
    # Runtime knobs.
    amp: bool = False
    accumulate_steps: int = 1
    grad_clip_norm: float = 5.0
    auto_resume: bool = True
    # Validation/checkpoint cadence in epochs.
    val_every: int = 1
    checkpoint_every: int = 1
    # DataLoader worker settings.
    num_workers: int = 0
    pin_memory: bool = False
    # Console/TensorBoard timing log cadence and rolling profile window size.
    log_every_n_steps: int = 1
    profile_window: int = 20
    # When True, synchronize CUDA before/after timed regions for more accurate profiles.
    profile_device_sync: bool = False
    # Run artifact root. `summary.json`, `run_manifest.json`, `history/`, `checkpoints/`, `tensorboard/` land here.
    run_dir: Path = DEFAULT_RUN_DIR
    # TensorBoard scalar logging under `<run_dir>/tensorboard`.
    enable_tensorboard: bool = True


# Edit this block directly before running `python3 tools/run_pv26_pilot_train.py`.
DATASET_CONFIG = DatasetConfig()
TRAIN_CONFIG = TrainConfig()


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    dataset_roots = [DATASET_CONFIG.aihub_root]
    if DATASET_CONFIG.include_bdd:
        dataset_roots.append(DATASET_CONFIG.bdd_root)
    missing_roots = [str(path) for path in dataset_roots if not path.is_dir()]
    if missing_roots:
        raise SystemExit(f"canonical dataset roots not found: {missing_roots}")

    dataset = PV26CanonicalDataset(dataset_roots)
    train_loader = build_pv26_train_dataloader(
        dataset,
        batch_size=TRAIN_CONFIG.batch_size,
        num_batches=TRAIN_CONFIG.train_batches,
        split="train",
        seed=26,
        num_workers=TRAIN_CONFIG.num_workers,
        pin_memory=TRAIN_CONFIG.pin_memory,
    )
    val_loader = None
    if TRAIN_CONFIG.val_batches > 0:
        try:
            val_loader = build_pv26_eval_dataloader(
                dataset,
                batch_size=TRAIN_CONFIG.batch_size,
                num_batches=TRAIN_CONFIG.val_batches,
                split="val",
                num_workers=TRAIN_CONFIG.num_workers,
                pin_memory=TRAIN_CONFIG.pin_memory,
            )
        except ValueError:
            val_loader = None

    adapter = build_yolo26n_trunk()
    heads = PV26Heads(in_channels=HEAD_CHANNELS)
    optimizer = build_pv26_optimizer(
        adapter,
        heads,
        trunk_lr=TRAIN_CONFIG.trunk_lr,
        head_lr=TRAIN_CONFIG.head_lr,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )
    scheduler = build_pv26_scheduler(
        optimizer,
        epochs=TRAIN_CONFIG.epochs,
        schedule=TRAIN_CONFIG.schedule,
    )
    trainer = PV26Trainer(
        adapter,
        heads,
        stage=TRAIN_CONFIG.stage,
        device=TRAIN_CONFIG.device,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=TRAIN_CONFIG.amp,
        accumulate_steps=TRAIN_CONFIG.accumulate_steps,
        grad_clip_norm=TRAIN_CONFIG.grad_clip_norm,
    )
    summary = trainer.fit(
        train_loader,
        epochs=TRAIN_CONFIG.epochs,
        val_loader=val_loader,
        run_dir=TRAIN_CONFIG.run_dir,
        val_every=TRAIN_CONFIG.val_every,
        checkpoint_every=TRAIN_CONFIG.checkpoint_every,
        max_train_batches=TRAIN_CONFIG.train_batches,
        max_val_batches=TRAIN_CONFIG.val_batches,
        auto_resume=TRAIN_CONFIG.auto_resume,
        enable_tensorboard=TRAIN_CONFIG.enable_tensorboard,
        log_every_n_steps=TRAIN_CONFIG.log_every_n_steps,
        profile_window=TRAIN_CONFIG.profile_window,
        profile_device_sync=TRAIN_CONFIG.profile_device_sync,
        run_manifest_extra={
            "entry_script": "tools/run_pv26_pilot_train.py",
            "dataset_config": _json_ready(asdict(DATASET_CONFIG)),
            "train_config": _json_ready(asdict(TRAIN_CONFIG)),
            "head_channels": list(HEAD_CHANNELS),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
