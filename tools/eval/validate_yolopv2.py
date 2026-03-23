#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.loading.manifest_dataset import Pv26ManifestDataset
from pv26.eval.yolopv2_validation import (
    collate_yolopv2_eval,
    load_yolopv2_torchscript,
    validate_yolopv2,
    validation_summary_to_dict,
)
from pv26.training.train_config import SCRIPT_DEFAULTS


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate legacy yolopv2.pt on the PV26 val split.")
    p.add_argument("--weights", type=Path, default=Path("datasets/weights/yolopv2.pt"))
    p.add_argument("--dataset-root", type=Path, default=SCRIPT_DEFAULTS.dataset_root)
    p.add_argument("--batch-size", type=int, default=SCRIPT_DEFAULTS.batch_size)
    p.add_argument("--workers", type=int, default=SCRIPT_DEFAULTS.workers)
    p.add_argument("--prefetch-factor", type=int, default=SCRIPT_DEFAULTS.prefetch_factor)
    p.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across the validation loop.",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    p.add_argument("--device", type=str, default=SCRIPT_DEFAULTS.device, help="auto|cpu|cuda|cuda:N")
    p.add_argument("--conf-thres", type=float, default=0.01)
    p.add_argument("--iou-thres", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=200)
    p.add_argument("--max-batches", type=int, default=0, help="Only validate the first N batches (0=all).")
    p.add_argument("--progress-every", type=int, default=10, help="Print progress every N batches (0=off).")
    p.add_argument("--validate-masks", action="store_true", help="Enable strict dataset mask checks.")
    p.add_argument("--out-json", type=Path, default=None)
    p.set_defaults(persistent_workers=SCRIPT_DEFAULTS.persistent_workers)
    return p


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _format_metric(name: str, metric) -> list[str]:
    prefix = f"[pv2-val] {name}"
    if metric.iou is None:
        return [
            f"{prefix}: skipped(no_supervision)",
            f"{prefix}_supervised_samples={metric.supervised_samples}",
        ]
    return [
        f"{prefix}: iou={metric.iou:.4f} precision={metric.precision:.4f} recall={metric.recall:.4f} f1={metric.f1:.4f}",
        f"{prefix}_supervised_samples={metric.supervised_samples} valid_pixels={metric.valid_pixels} "
        f"tp={metric.true_positive} fp={metric.false_positive} fn={metric.false_negative} tn={metric.true_negative}",
    ]


def main() -> int:
    args = build_argparser().parse_args()
    device = _resolve_device(args.device)

    dataset = Pv26ManifestDataset(
        dataset_root=args.dataset_root,
        splits=("val",),
        validate_masks=bool(args.validate_masks),
    )
    if len(dataset) == 0:
        raise RuntimeError("val split is empty")

    num_workers = max(0, int(args.workers))
    loader_perf_kwargs = {}
    if num_workers > 0:
        loader_perf_kwargs = {
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": max(2, int(args.prefetch_factor)),
        }
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_yolopv2_eval,
        pin_memory=(device.type == "cuda"),
        **loader_perf_kwargs,
    )

    model = load_yolopv2_torchscript(args.weights, device=device)

    progress_every = max(0, int(args.progress_every))

    def _progress(done: int, total: int) -> None:
        if progress_every <= 0:
            return
        if done == 1 or done == total or (done % progress_every) == 0:
            print(f"[pv2-val] progress {done}/{total}", flush=True)

    summary = validate_yolopv2(
        model=model,
        loader=loader,
        device=device,
        weights_path=args.weights,
        dataset_root=args.dataset_root,
        conf_thres=float(args.conf_thres),
        iou_thres=float(args.iou_thres),
        max_det=max(1, int(args.max_det)),
        max_batches=max(0, int(args.max_batches)),
        progress_hook=_progress if progress_every > 0 else None,
    )

    print(f"[pv2-val] weights={summary.weights_path}")
    print(f"[pv2-val] dataset_root={summary.dataset_root}")
    print(f"[pv2-val] val_samples={summary.num_samples} batches={summary.num_batches} input={summary.input_width}x{summary.input_height}")
    if summary.det_map50 is None:
        print("[pv2-val] det_map50=skipped(no_det_gt)")
    else:
        print(
            f"[pv2-val] det_map50={summary.det_map50:.4f} "
            f"eval_images={summary.det_eval_images} gt_boxes={summary.det_gt_boxes} preds={summary.det_predictions}"
        )
    for line in _format_metric("da", summary.da):
        print(line)
    for line in _format_metric("lane", summary.lane):
        print(line)
    print(f"[pv2-val] unsupported_metrics={','.join(summary.unsupported_metrics)}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(validation_summary_to_dict(summary), indent=2), encoding="utf-8")
        print(f"[pv2-val] wrote_json={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
