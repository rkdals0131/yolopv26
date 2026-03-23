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
from model.training import PV26Trainer, build_pv26_optimizer, build_pv26_scheduler
from model.trunk import build_yolo26n_trunk

DEFAULT_AIHUB_ROOT = REPO_ROOT / "seg_dataset" / "pv26_aihub_standardized"
DEFAULT_BDD_ROOT = REPO_ROOT / "seg_dataset" / "pv26_bdd100k_standardized"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "pv26_train" / "pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PV26 pilot training loop on canonical datasets.")
    parser.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    parser.add_argument("--bdd-root", type=Path, default=DEFAULT_BDD_ROOT)
    parser.add_argument("--include-bdd", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--stage", type=str, default="stage_1_frozen_trunk_warmup")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--trunk-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-3)
    parser.add_argument("--schedule", type=str, default="cosine", choices=("none", "cosine"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
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
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
        except ValueError:
            val_loader = None

    adapter = build_yolo26n_trunk()
    heads = PV26Heads(in_channels=(64, 128, 256))
    optimizer = build_pv26_optimizer(
        adapter,
        heads,
        trunk_lr=args.trunk_lr,
        head_lr=args.head_lr,
    )
    scheduler = build_pv26_scheduler(optimizer, epochs=args.epochs, schedule=args.schedule)
    trainer = PV26Trainer(
        adapter,
        heads,
        stage=args.stage,
        device=args.device,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=args.amp,
        accumulate_steps=args.accumulate_steps,
        grad_clip_norm=args.grad_clip_norm,
    )
    summary = trainer.fit(
        train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        run_dir=args.run_dir,
        max_train_batches=args.train_batches,
        max_val_batches=args.val_batches,
        auto_resume=args.auto_resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
