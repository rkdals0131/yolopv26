from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from model.data import PV26CanonicalDataset, encode_pv26_batch
from model.engine._trainer_epochs import _merge_raw_batches
from model.engine.batch import augment_lane_family_metrics, raw_batch_for_metrics
from model.engine.loss import build_loss_spec
from model.engine.metrics import summarize_pv26_metrics
from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api


SPEC = build_loss_spec()
LANE_QUERY_COUNT = int(SPEC["heads"]["lane"]["query_count"])
LANE_VECTOR_DIM = int(SPEC["heads"]["lane"]["shape"].split(" x ")[-1])
STOP_LINE_QUERY_COUNT = int(SPEC["heads"]["stop_line"]["query_count"])
STOP_LINE_VECTOR_DIM = int(SPEC["heads"]["stop_line"]["shape"].split(" x ")[-1])
CROSSWALK_QUERY_COUNT = int(SPEC["heads"]["crosswalk"]["query_count"])
CROSSWALK_VECTOR_DIM = int(SPEC["heads"]["crosswalk"]["shape"].split(" x ")[-1])
DET_FEATURE_SHAPES = [(76, 100), (38, 50), (19, 25)]
DET_FEATURE_STRIDES = [8, 16, 32]
DET_QUERY_COUNT = sum(rows * cols for rows, cols in DET_FEATURE_SHAPES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode stop-line HAF targets through the runtime postprocess path as a target/decoder sanity gate."
    )
    parser.add_argument(
        "--dataset-root",
        default="seg_dataset/pv26_exhaustive_od_lane_dataset",
        help="Canonical PV26 dataset root.",
    )
    parser.add_argument("--preset", default="default")
    parser.add_argument("--phase-index", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--validation-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--haf-valid-threshold", type=float, default=0.90)
    parser.add_argument("--haf-min-votes", type=int, default=2)
    parser.add_argument("--haf-cluster-endpoint-tolerance", type=float, default=0.25)
    parser.add_argument("--haf-max-endpoint-covariance", type=float, default=0.01)
    parser.add_argument("--haf-max-segments", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default="runs/pv26_exhaustive_od_lane_train/stopline_haf_target_decode_probe",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist() if value.numel() <= 1024 else {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist() if value.size <= 1024 else {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _advance_validation_sampler(val_loader: Any, *, validation_epoch: int) -> None:
    for _ in range(max(0, int(validation_epoch) - 1)):
        batch_sampler = getattr(val_loader, "batch_sampler", None)
        if batch_sampler is None:
            raise ValueError("validation loader does not expose a batch_sampler to advance")
        for _batch_indices in batch_sampler:
            pass


def _ensure_encoded(batch: dict[str, Any]) -> dict[str, Any]:
    if "roadmark_v2" in batch:
        return batch
    return encode_pv26_batch(batch)


def _haf_target_predictions(encoded: dict[str, Any]) -> dict[str, torch.Tensor | list[Any]]:
    batch_size = int(encoded["image"].shape[0])
    roadmark_v2 = encoded["roadmark_v2"]
    haf_valid = roadmark_v2["stop_line_haf_valid"].to(dtype=torch.float32)
    haf_ignore = roadmark_v2.get("stop_line_haf_ignore")
    if isinstance(haf_ignore, torch.Tensor):
        haf_valid = torch.where(haf_ignore.to(dtype=torch.bool), torch.zeros_like(haf_valid), haf_valid)
    haf_logits = torch.where(
        haf_valid > 0.5,
        torch.full_like(haf_valid, 8.0),
        torch.full_like(haf_valid, -8.0),
    )
    return {
        "det": torch.zeros((batch_size, DET_QUERY_COUNT, 12), dtype=torch.float32),
        "tl_attr": torch.zeros((batch_size, DET_QUERY_COUNT, 4), dtype=torch.float32),
        "lane": torch.zeros((batch_size, LANE_QUERY_COUNT, LANE_VECTOR_DIM), dtype=torch.float32),
        "stop_line": torch.zeros((batch_size, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32),
        "crosswalk": torch.zeros((batch_size, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM), dtype=torch.float32),
        "stop_line_haf_endpoint": roadmark_v2["stop_line_haf_endpoint"].to(dtype=torch.float32),
        "stop_line_haf_valid_logits": haf_logits,
        "det_feature_shapes": DET_FEATURE_SHAPES,
        "det_feature_strides": DET_FEATURE_STRIDES,
    }


def _metric_row(metrics: dict[str, Any], *, prediction_count: int, processed_batches: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": "haf_target_decode",
        "processed_batches": int(processed_batches),
        "prediction_count": int(prediction_count),
    }
    for task in ("lane", "stop_line", "crosswalk"):
        payload = metrics.get(task, {}) if isinstance(metrics.get(task), dict) else {}
        for key in ("f1", "precision", "recall", "tp", "fp", "fn", "support"):
            if key == "support" and key not in payload:
                row[f"{task}_{key}"] = int(payload.get("tp", 0)) + int(payload.get("fn", 0))
            else:
                row[f"{task}_{key}"] = payload.get(key, 0)
    lane_family = metrics.get("lane_family", {}) if isinstance(metrics.get("lane_family"), dict) else {}
    row["lane_family_mean_f1"] = lane_family.get("mean_f1", 0.0)
    row["lane_family_min_f1"] = lane_family.get("min_f1", 0.0)
    return row


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    scenario = train_cli.load_meta_train_scenario(args.preset)
    phase = scenario.phases[int(args.phase_index) - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    train_config = replace(
        train_config,
        batch_size=int(args.batch_size),
        val_batches=int(args.max_val_batches),
        num_workers=int(args.num_workers),
        encode_val_batches_in_loader=False,
    )

    dataset = PV26CanonicalDataset(
        [dataset_root],
        train_augmentation=False,
        progress_callback=lambda message: print(f"[stopline_haf_target_decode] {message}", flush=True),
    )
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=train_config, phase=phase)
    if val_loader is None:
        raise ValueError("HAF target decode probe requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    postprocess_config = PV26PostprocessConfig(
        det_conf_threshold=0.999,
        lane_obj_threshold=0.999,
        crosswalk_obj_threshold=0.999,
        stop_line_haf_enabled=True,
        stop_line_haf_valid_threshold=float(args.haf_valid_threshold),
        stop_line_haf_min_votes=int(args.haf_min_votes),
        stop_line_haf_cluster_endpoint_tolerance=float(args.haf_cluster_endpoint_tolerance),
        stop_line_haf_max_endpoint_covariance=float(args.haf_max_endpoint_covariance),
        stop_line_haf_max_segments=int(args.haf_max_segments),
    )
    predictions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    processed_batches = 0
    for batch_index, batch in enumerate(val_loader, start=1):
        if batch_index > int(args.max_val_batches):
            break
        raw_batch = raw_batch_for_metrics(batch)
        if raw_batch is None:
            raise ValueError("validation loader must provide raw batches for metrics")
        encoded = _ensure_encoded(batch)
        predictions.extend(
            postprocess_pv26_batch(
                _haf_target_predictions(encoded),
                list(encoded["meta"]),
                config=postprocess_config,
            )
        )
        raw_batches.append(raw_batch)
        processed_batches += 1
        print(f"[stopline_haf_target_decode] batch {batch_index}/{args.max_val_batches}", flush=True)

    if not raw_batches:
        raise ValueError("no validation batches were processed")
    merged_raw = _merge_raw_batches(raw_batches)
    metrics = augment_lane_family_metrics(summarize_pv26_metrics(predictions, merged_raw))
    prediction_count = sum(len(sample.get("stop_lines", [])) for sample in predictions)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _metric_row(metrics, prediction_count=prediction_count, processed_batches=processed_batches)
    _write_csv(output_dir / "metrics.csv", [row])
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "postprocess_config": _json_ready(postprocess_config),
                "metrics": _json_ready(metrics),
                "row": row,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row, sort_keys=True), flush=True)
    print(f"[stopline_haf_target_decode] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
