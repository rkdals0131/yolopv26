from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.heads import PV26Heads
from model.loading import PV26CanonicalDataset, build_pv26_train_dataloader
from model.training import PV26Trainer
from model.trunk import build_yolo26n_trunk

DEFAULT_AIHUB_ROOT = REPO_ROOT / "seg_dataset" / "pv26_aihub_standardized"
DEFAULT_BDD_ROOT = REPO_ROOT / "seg_dataset" / "pv26_bdd100k_standardized"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "pv26_train" / "fit_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PV26 epoch-level training smoke loop.")
    parser.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    parser.add_argument("--bdd-root", type=Path, default=DEFAULT_BDD_ROOT)
    parser.add_argument("--include-bdd", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--stage", type=str, default="stage_1_frozen_trunk_warmup")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--trunk-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-3)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = [args.aihub_root]
    if args.include_bdd:
        dataset_roots.append(args.bdd_root)
    missing_roots = [str(path) for path in dataset_roots if not path.is_dir()]
    if missing_roots:
        raise SystemExit(f"canonical dataset roots not found: {missing_roots}")

    dataset = PV26CanonicalDataset(dataset_roots)
    train_loader = build_pv26_train_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_batches=args.train_batches,
        split="train",
        seed=26,
    )
    val_loader = None
    if args.val_batches > 0:
        try:
            val_loader = build_pv26_train_dataloader(
                dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                split="val",
                seed=52,
            )
        except ValueError:
            val_loader = None

    adapter = build_yolo26n_trunk()
    heads = PV26Heads(in_channels=(64, 128, 256))
    trainer = PV26Trainer(
        adapter,
        heads,
        stage=args.stage,
        device=args.device,
        trunk_lr=args.trunk_lr,
        head_lr=args.head_lr,
    )
    summary = trainer.fit(
        train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        run_dir=args.run_dir,
        max_train_batches=args.train_batches,
        max_val_batches=args.val_batches,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
