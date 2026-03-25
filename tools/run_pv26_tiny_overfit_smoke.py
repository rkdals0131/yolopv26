from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.heads import PV26Heads
from model.loading import PV26CanonicalDataset, collate_pv26_samples
from model.training import PV26Trainer, run_pv26_tiny_overfit
from model.trunk import build_yolo26n_trunk


DEFAULT_AIHUB_ROOT = REPO_ROOT / "seg_dataset" / "pv26_aihub_standardized"
DEFAULT_BDD_ROOT = REPO_ROOT / "seg_dataset" / "pv26_bdd100k_standardized"
HEAD_CHANNELS = (64, 128, 256)


@dataclass(frozen=True)
class DatasetConfig:
    # AIHUB canonical root. Must exist before training.
    aihub_root: Path = DEFAULT_AIHUB_ROOT
    # Turn on when BDD canonical root should be mixed into the smoke batch.
    include_bdd: bool = True
    # BDD canonical root. Used only when include_bdd is True.
    bdd_root: Path = DEFAULT_BDD_ROOT


@dataclass(frozen=True)
class TrainConfig:
    # Number of repeated updates on the same tiny batch.
    steps: int = 8
    # Stage name from the documented training schedule.
    stage: str = "stage_1_frozen_trunk_warmup"
    # Device string passed to torch.
    device: str = "cpu"
    # Optimizer hyperparameters for the tiny overfit check.
    trunk_lr: float = 1e-4
    head_lr: float = 5e-3


# Edit this block directly before running `python3 tools/run_pv26_tiny_overfit_smoke.py`.
DATASET_CONFIG = DatasetConfig()
TRAIN_CONFIG = TrainConfig()


def _select_samples(dataset: PV26CanonicalDataset, include_bdd: bool) -> list[dict]:
    wanted = ["aihub_traffic_seoul", "aihub_obstacle_seoul", "aihub_lane_seoul"]
    if include_bdd:
        wanted.append("bdd100k_det_100k")

    selected: list[dict] = []
    seen: set[str] = set()
    for sample in dataset:
        dataset_key = str(sample["meta"]["dataset_key"])
        split = str(sample["meta"]["split"])
        if dataset_key not in wanted or dataset_key in seen or split != "train":
            continue
        selected.append(sample)
        seen.add(dataset_key)
        if len(seen) == len(wanted):
            break
    missing = [dataset_key for dataset_key in wanted if dataset_key not in seen]
    if missing:
        raise RuntimeError(f"missing required tiny-overfit samples for datasets: {missing}")
    return selected


def main() -> None:
    dataset_roots = [DATASET_CONFIG.aihub_root]
    if DATASET_CONFIG.include_bdd:
        dataset_roots.append(DATASET_CONFIG.bdd_root)
    missing_roots = [str(path) for path in dataset_roots if not path.is_dir()]
    if missing_roots:
        raise SystemExit(f"canonical dataset roots not found: {missing_roots}")

    dataset = PV26CanonicalDataset(dataset_roots)
    samples = _select_samples(dataset, include_bdd=DATASET_CONFIG.include_bdd)
    batch = collate_pv26_samples(samples)

    adapter = build_yolo26n_trunk()
    heads = PV26Heads(in_channels=HEAD_CHANNELS)
    trainer = PV26Trainer(
        adapter,
        heads,
        stage=TRAIN_CONFIG.stage,
        device=TRAIN_CONFIG.device,
        trunk_lr=TRAIN_CONFIG.trunk_lr,
        head_lr=TRAIN_CONFIG.head_lr,
    )
    summary = run_pv26_tiny_overfit(trainer, batch, steps=TRAIN_CONFIG.steps)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
