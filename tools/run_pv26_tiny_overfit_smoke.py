from __future__ import annotations

import argparse
import json
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


def _select_samples(dataset: PV26CanonicalDataset, include_bdd: bool) -> list[dict]:
    wanted = ["aihub_traffic_seoul", "aihub_lane_seoul"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PV26 tiny overfit smoke loop.")
    parser.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    parser.add_argument("--bdd-root", type=Path, default=DEFAULT_BDD_ROOT)
    parser.add_argument("--include-bdd", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--stage", type=str, default="stage_1_frozen_trunk_warmup")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--trunk-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-3)
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
    samples = _select_samples(dataset, include_bdd=args.include_bdd)
    batch = collate_pv26_samples(samples)

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
    summary = run_pv26_tiny_overfit(trainer, batch, steps=args.steps)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
