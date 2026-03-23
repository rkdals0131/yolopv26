#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.dataset.loading.manifest_dataset import Pv26ManifestDataset
from pv26.eval.bdd_yolopv2_validation import (
    BddYoloPv2ValDataset,
    collate_bdd_yolopv2_eval,
    validate_bdd_yolopv2,
    validation_summary_to_dict as pv2_bdd_summary_to_dict,
)
from pv26.eval.common_validation import BinaryMetricSummary
from pv26.eval.pv26_validation import (
    collate_pv26_eval,
    load_pv26_checkpoint,
    validate_pv26,
    validation_summary_to_dict as pv26_summary_to_dict,
)
from pv26.eval.yolopv2_validation import load_yolopv2_torchscript
from pv26.training.train_config import SCRIPT_DEFAULTS

EVAL_DIR = Path(__file__).resolve().parent
PV2_WEIGHTS = EVAL_DIR / "yolopv2.pt"
PV26_WEIGHTS = EVAL_DIR / "yolopv26_epoch148.pt"
BDD_ROOT = REPO_ROOT / "datasets" / "BDD100K"
COMMON_COMPARE_KEYS = ("da.iou", "da.f1", "lane.iou", "lane.f1")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate YOLOPv2 on raw BDD100K val and PV26 on merged PV26 val, then print both side by side."
    )
    p.add_argument("--pv26-dataset-root", type=Path, default=SCRIPT_DEFAULTS.dataset_root)
    p.add_argument("--bdd-root", type=Path, default=BDD_ROOT)
    p.add_argument("--batch-size", type=int, default=SCRIPT_DEFAULTS.batch_size)
    p.add_argument("--workers", type=int, default=SCRIPT_DEFAULTS.workers)
    p.add_argument("--prefetch-factor", type=int, default=SCRIPT_DEFAULTS.prefetch_factor)
    p.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across validation.",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    p.add_argument("--device", type=str, default=SCRIPT_DEFAULTS.device, help="auto|cpu|cuda|cuda:N")
    p.add_argument("--max-batches", type=int, default=0, help="Only validate the first N batches (0=all).")
    p.add_argument("--progress-every", type=int, default=10, help="Print progress every N batches per model (0=off).")
    p.add_argument("--validate-masks", action="store_true", help="Enable strict dataset mask checks.")
    p.add_argument("--out-json", type=Path, default=None)
    p.set_defaults(persistent_workers=SCRIPT_DEFAULTS.persistent_workers)
    return p


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _format_metric(prefix: str, metric: BinaryMetricSummary) -> list[str]:
    if metric.iou is None:
        return [f"{prefix}: skipped(no_supervision)"]
    return [
        f"{prefix}: iou={metric.iou:.4f} precision={metric.precision:.4f} recall={metric.recall:.4f} f1={metric.f1:.4f}",
        f"{prefix}_supervised_samples={metric.supervised_samples} valid_pixels={metric.valid_pixels} "
        f"tp={metric.true_positive} fp={metric.false_positive} fn={metric.false_negative} tn={metric.true_negative}",
    ]


def _format_ap_by_class(prefix: str, ap_by_class: dict[str, Optional[float]]) -> list[str]:
    parts = [f"{name}={_fmt(ap)}" for name, ap in ap_by_class.items()]
    return [f"{prefix}: " + " ".join(parts)]


def _metric_lookup(report: dict[str, Any], path: str) -> Optional[float]:
    cur: Any = report
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    return float(cur)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _build_compare_block(pv2_report: dict[str, Any], pv26_report: dict[str, Any]) -> dict[str, Any]:
    common: dict[str, dict[str, Optional[float]]] = {}
    for key in COMMON_COMPARE_KEYS:
        pv2_val = _metric_lookup(pv2_report, key)
        pv26_val = _metric_lookup(pv26_report, key)
        common[key] = {"pv2_bdd_raw": pv2_val, "pv26_merged": pv26_val}
    return {
        "note": "cross-dataset directional comparison only; not apples-to-apples",
        "common_metrics": common,
        "pv26_extra": {
            "rm_road_marker_non_lane.iou": _metric_lookup(pv26_report, "rm_road_marker_non_lane.iou"),
            "rm_stop_line.iou": _metric_lookup(pv26_report, "rm_stop_line.iou"),
            "lane_subclass_miou4": _metric_lookup(pv26_report, "lane_subclass_miou4"),
            "lane_subclass_miou4_present": _metric_lookup(pv26_report, "lane_subclass_miou4_present"),
        },
    }


def main() -> int:
    args = build_argparser().parse_args()
    device = _resolve_device(args.device)
    pv26_dataset = Pv26ManifestDataset(
        dataset_root=args.pv26_dataset_root,
        splits=("val",),
        validate_masks=bool(args.validate_masks),
    )
    if len(pv26_dataset) == 0:
        raise RuntimeError("pv26 val split is empty")
    bdd_dataset = BddYoloPv2ValDataset(bdd_root=args.bdd_root)
    if not PV2_WEIGHTS.exists():
        raise FileNotFoundError(f"missing hardcoded PV2 weights: {PV2_WEIGHTS}")
    if not PV26_WEIGHTS.exists():
        raise FileNotFoundError(f"missing hardcoded PV26 weights: {PV26_WEIGHTS}")

    num_workers = max(0, int(args.workers))
    loader_perf_kwargs = {}
    if num_workers > 0:
        loader_perf_kwargs = {
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": max(2, int(args.prefetch_factor)),
        }

    pv2_loader = DataLoader(
        bdd_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bdd_yolopv2_eval,
        pin_memory=(device.type == "cuda"),
        **loader_perf_kwargs,
    )
    pv26_loader = DataLoader(
        pv26_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pv26_eval,
        pin_memory=(device.type == "cuda"),
        **loader_perf_kwargs,
    )

    progress_every = max(0, int(args.progress_every))

    def _progress(prefix: str):
        def inner(done: int, total: int) -> None:
            if progress_every <= 0:
                return
            if done == 1 or done == total or (done % progress_every) == 0:
                print(f"[{prefix}] progress {done}/{total}", flush=True)

        return inner

    pv2_model = load_yolopv2_torchscript(PV2_WEIGHTS, device=device)
    pv2_summary = validate_bdd_yolopv2(
        model=pv2_model,
        loader=pv2_loader,
        device=device,
        weights_path=PV2_WEIGHTS,
        bdd_root=args.bdd_root,
        max_batches=max(0, int(args.max_batches)),
        progress_hook=_progress("pv2-bdd") if progress_every > 0 else None,
    )

    pv26_model, checkpoint_layout = load_pv26_checkpoint(PV26_WEIGHTS, device=device)
    pv26_summary = validate_pv26(
        model=pv26_model,
        loader=pv26_loader,
        device=device,
        weights_path=PV26_WEIGHTS,
        dataset_root=args.pv26_dataset_root,
        checkpoint_layout=checkpoint_layout,
        conf_thres=0.01,
        max_batches=max(0, int(args.max_batches)),
        progress_hook=_progress("pv26-val") if progress_every > 0 else None,
    )

    pv2_report = pv2_bdd_summary_to_dict(pv2_summary)
    pv26_report = pv26_summary_to_dict(pv26_summary)
    compare_report = _build_compare_block(pv2_report, pv26_report)

    print(f"[validate-yolopv] bdd_root={args.bdd_root}")
    print(f"[validate-yolopv] pv26_dataset_root={args.pv26_dataset_root}")
    print(f"[validate-yolopv] pv2_weights={PV2_WEIGHTS}")
    print(f"[validate-yolopv] pv26_weights={PV26_WEIGHTS}")

    print("[pv2-bdd] summary")
    print(f"[pv2-bdd] detection={pv2_summary.detection_status}")
    if pv2_summary.det_map50 is None:
        print("[pv2-bdd] det_map50=skipped")
    else:
        print(
            f"[pv2-bdd] det_map50={pv2_summary.det_map50:.4f} "
            f"eval_images={pv2_summary.det_eval_images} gt_boxes={pv2_summary.det_gt_boxes} preds={pv2_summary.det_predictions}"
        )
        for line in _format_ap_by_class("[pv2-bdd] det_ap_by_class", pv2_summary.det_ap_by_class):
            print(line)
    for line in _format_metric("[pv2-bdd] da", pv2_summary.da):
        print(line)
    for line in _format_metric("[pv2-bdd] lane", pv2_summary.lane):
        print(line)

    print("[pv26-val] summary")
    print(f"[pv26-val] checkpoint_layout={pv26_summary.checkpoint_layout}")
    if pv26_summary.det_map50 is None:
        print("[pv26-val] det_map50=skipped(no_det_gt)")
    else:
        print(
            f"[pv26-val] det_map50={pv26_summary.det_map50:.4f} "
            f"eval_images={pv26_summary.det_eval_images} gt_boxes={pv26_summary.det_gt_boxes} preds={pv26_summary.det_predictions}"
        )
    for metric_name, metric in [
        ("[pv26-val] da", pv26_summary.da),
        ("[pv26-val] lane", pv26_summary.lane),
        ("[pv26-val] rm_road_marker_non_lane", pv26_summary.rm_road_marker_non_lane),
        ("[pv26-val] rm_stop_line", pv26_summary.rm_stop_line),
    ]:
        for line in _format_metric(metric_name, metric):
            print(line)
    if pv26_summary.lane_subclass_miou4 is None:
        print("[pv26-val] lane_subclass_miou4=skipped(no_supervision)")
    else:
        print(
            f"[pv26-val] lane_subclass_miou4={pv26_summary.lane_subclass_miou4:.4f} "
            f"lane_subclass_miou4_present={pv26_summary.lane_subclass_miou4_present:.4f}"
        )
    for name, metric in pv26_summary.lane_subclass_groups.items():
        for line in _format_metric(f"[pv26-val] lane_subclass_group_{name}", metric):
            print(line)

    print("[compare] common_metrics")
    print(f"[compare] note={compare_report['note']}")
    for key, vals in compare_report["common_metrics"].items():
        print(
            f"[compare] {key}: pv2_bdd_raw={_fmt(vals['pv2_bdd_raw'])} pv26_merged={_fmt(vals['pv26_merged'])}"
        )
    print("[compare] pv26_extra")
    for key, value in compare_report["pv26_extra"].items():
        print(f"[compare] {key}: pv26={_fmt(value)}")

    if args.out_json is not None:
        payload = {
            "bdd_root": str(args.bdd_root),
            "pv26_dataset_root": str(args.pv26_dataset_root),
            "pv2": pv2_report,
            "pv26": pv26_report,
            "comparison": compare_report,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[validate-yolopv] wrote_json={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
