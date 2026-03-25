from __future__ import annotations

import json
from datetime import datetime
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
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "pv26_train"
HEAD_CHANNELS = (64, 128, 256)


@dataclass(frozen=True)
class DatasetConfig:
    aihub_root: Path   # 학습에 사용할 AIHUB canonical root
    include_bdd: bool  # True면 BDD canonical root도 함께 섞어서 학습
    bdd_root: Path     # include_bdd=True일 때 사용할 BDD canonical root


@dataclass(frozen=True)
class TrainConfig:
    stage: str               # 학습 단계 프리셋: "stage_0_smoke", "stage_1_frozen_trunk_warmup", "stage_2_partial_unfreeze", "stage_3_end_to_end_finetune"
    device: str              # torch device
    epochs: int              # 전체 epoch 수

    batch_size: int          # step당 배치 크기
    train_batches: int       # epoch당 train batch 수, 0 또는 -1이면 train split 전체 사용
    val_batches: int         # epoch당 val batch 수, 0이면 validation 비활성화, -1이면 val split 전체 사용

    trunk_lr: float          # trunk learning rate
    head_lr: float           # head learning rate
    weight_decay: float      # optimizer weight decay
    schedule: str            # lr scheduler, "none" 또는 "cosine"

    amp: bool                # mixed precision 사용 여부
    accumulate_steps: int    # gradient accumulation step 수
    grad_clip_norm: float    # grad clip norm
    auto_resume: bool        # last checkpoint 자동 resume 여부

    val_every: int           # 몇 epoch마다 validation 할지
    checkpoint_every: int    # 몇 epoch마다 epoch checkpoint 저장할지

    num_workers: int         # DataLoader worker 수
    pin_memory: bool         # DataLoader pin_memory 사용 여부

    log_every_n_steps: int   # 몇 step마다 콘솔 로그를 찍을지
    profile_window: int      # rolling timing profile window 크기
    profile_device_sync: bool  # CUDA timing 정확도를 위해 sync할지

    run_name_prefix: str     # 자동 생성 run 폴더 앞에 붙일 문자열
    run_root: Path           # 자동 생성 run 폴더들이 쌓일 상위 디렉터리
    run_dir: Path | None     # 직접 경로를 고정하고 싶을 때만 사용, None이면 prefix+timestamp 자동 생성
    enable_tensorboard: bool # TensorBoard 기록 여부


# `python3 tools/run_pv26_train.py` 실행 전 이 블록만 수정한다.
DATASET_CONFIG = DatasetConfig(
    aihub_root  = DEFAULT_AIHUB_ROOT,
    include_bdd = True,
    bdd_root    = DEFAULT_BDD_ROOT,
)

TRAIN_CONFIG = TrainConfig(
    stage               = "stage_1_frozen_trunk_warmup",  # "stage_0_smoke", "stage_1_frozen_trunk_warmup", "stage_2_partial_unfreeze", "stage_3_end_to_end_finetune"
    device              = "cuda:0",
    epochs              = 5,

    batch_size          = 80,
    train_batches       = -1,
    val_batches         = -1,

    trunk_lr            = 1e-4,
    head_lr             = 5e-3,
    weight_decay        = 1e-4,
    schedule            = "cosine",

    amp                 = True,
    accumulate_steps    = 1,
    grad_clip_norm      = 5.0,
    auto_resume         = True,

    val_every           = 1,
    checkpoint_every    = 10,

    num_workers         = 4,
    pin_memory          = True,

    log_every_n_steps   = 20,
    profile_window      = 20,
    profile_device_sync = True,

    run_name_prefix     = "train",
    run_root            = DEFAULT_RUN_ROOT,
    run_dir             = None,
    enable_tensorboard  = True,
)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _resolve_train_batch_limit(value: int) -> int | None:
    if int(value) <= 0:
        return None
    return int(value)


def _resolve_val_batch_limit(value: int) -> int | None:
    if int(value) < 0:
        return None
    return int(value)


def _safe_run_name_component(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(value).strip())
    normalized = normalized.strip("_")
    return normalized or "train"


def _resolve_run_dir(config: TrainConfig) -> Path:
    if config.run_dir is not None:
        return Path(config.run_dir)

    run_root = Path(config.run_root)
    prefix = _safe_run_name_component(config.run_name_prefix)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def main() -> None:
    train_batches = _resolve_train_batch_limit(TRAIN_CONFIG.train_batches)
    val_batches = _resolve_val_batch_limit(TRAIN_CONFIG.val_batches)
    run_dir = _resolve_run_dir(TRAIN_CONFIG)
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
        num_batches=train_batches,
        split="train",
        seed=26,
        num_workers=TRAIN_CONFIG.num_workers,
        pin_memory=TRAIN_CONFIG.pin_memory,
    )
    val_loader = None
    if val_batches != 0:
        try:
            val_loader = build_pv26_eval_dataloader(
                dataset,
                batch_size=TRAIN_CONFIG.batch_size,
                num_batches=val_batches,
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
        run_dir=run_dir,
        val_every=TRAIN_CONFIG.val_every,
        checkpoint_every=TRAIN_CONFIG.checkpoint_every,
        max_train_batches=train_batches,
        max_val_batches=val_batches,
        auto_resume=TRAIN_CONFIG.auto_resume,
        enable_tensorboard=TRAIN_CONFIG.enable_tensorboard,
        log_every_n_steps=TRAIN_CONFIG.log_every_n_steps,
        profile_window=TRAIN_CONFIG.profile_window,
        profile_device_sync=TRAIN_CONFIG.profile_device_sync,
        run_manifest_extra={
            "entry_script": "tools/run_pv26_train.py",
            "dataset_config": _json_ready(asdict(DATASET_CONFIG)),
            "train_config": _json_ready(asdict(TRAIN_CONFIG)),
            "head_channels": list(HEAD_CHANNELS),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
