"""Tune line-level road-marking decoding on a development split, then report a held-out split.

The network is run once per split. Each candidate stop-line gap is decoded once
with no line filters; per-class score and length filters are then applied to the
decoded lines, since both filters only remove whole lines. Filters are chosen on
the development split by per-class line F1 and reported on the held-out split
beside the run's configured decoding and the legacy fixed-direction decoding.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import atomic_write_json  # noqa: E402
from common.schema import ROADMARK_CLASSES  # noqa: E402
from model.data.dataset import FocusedDataset, FocusedSource, collate_focused  # noqa: E402
from model.engine.geometry_metrics import match_roadmark_lines  # noqa: E402
from model.engine.postprocess import decode_roadmark_points  # noqa: E402
from model.net.pv26 import PV26FocusedModel  # noqa: E402

SCORES = (0.0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
LENGTHS_PX = (0.0, 40.0, 80.0, 120.0)
STOP_GAPS = (3, 6, 10)
LANE_GAP = 3


def _length(points: list[list[float]]) -> float:
    array = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum()) if len(array) > 1 else 0.0


def _decode_variants(logits: torch.Tensor, meta: list[dict]) -> dict[str, list[list[dict]]]:
    """Unfiltered auto-orientation lines per stop gap, plus the legacy decoding."""
    variants = {}
    for gap in STOP_GAPS:
        variants[f"auto_gap{gap}"] = decode_roadmark_points(
            logits, meta, orientation="auto", max_gap=(LANE_GAP, LANE_GAP, gap))
    variants["legacy"] = decode_roadmark_points(logits, meta)
    for images in variants.values():
        for lines in images:
            for line in lines:
                line["length_px"] = _length(line["points_xy"])
    return variants


@torch.inference_mode()
def collect(model: PV26FocusedModel, dataset: FocusedDataset, device: torch.device,
            batch_size: int, num_workers: int, configured: dict) -> list[dict]:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, collate_fn=collate_focused)
    rows = []
    for batch_index, batch in enumerate(loader):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.forward_for_loss(batch["image"].to(device, non_blocking=True))["roadmark_logits"]
        logits = logits.float().cpu()
        variants = _decode_variants(logits, batch["meta"])
        variants["configured"] = decode_roadmark_points(logits, batch["meta"], **configured)
        for lines in variants["configured"]:
            for line in lines:
                line["length_px"] = _length(line["points_xy"])
        for offset, meta in enumerate(batch["meta"]):
            valid = batch["roadmark_valid"][offset].flatten(1).any(dim=1).tolist()
            rows.append({
                "supervised": valid,
                "gt": [line for line in meta["roadmark_gt"] if valid[line["class_id"]]],
                "pred": {name: [line for line in images[offset] if valid[line["class_id"]]]
                         for name, images in variants.items()},
            })
        if batch_index % 100 == 0:
            print(json.dumps({"collect": {"images": len(rows), "total": len(dataset)}}), flush=True)
    return rows


def _count(rows: list[dict], variant: str, class_id: int, min_score: float, min_length: float) -> tuple[int, int, int]:
    name = ROADMARK_CLASSES[class_id]
    tp = fp = fn = 0
    for row in rows:
        if not row["supervised"][class_id]:
            continue
        predicted = [line for line in row["pred"][variant] if line["class_id"] == class_id
                     and line["score"] >= min_score and line["length_px"] >= min_length]
        truth = [line for line in row["gt"] if line["class_id"] == class_id]
        result = match_roadmark_lines(predicted, truth)[name]
        tp, fp, fn = tp + result["tp"], fp + result["fp"], fn + result["fn"]
    return tp, fp, fn


def _scores(counts: tuple[int, int, int]) -> dict:
    tp, fp, fn = counts
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


_ROWS: list[dict] = []


def _init(path: str) -> None:
    global _ROWS
    _ROWS = torch.load(path, weights_only=False)


def _task(arguments: tuple[str, int, float, float]) -> tuple[tuple[str, int, float, float], tuple[int, int, int]]:
    return arguments, _count(_ROWS, *arguments)


def tune(dev_path: Path, workers: int) -> dict:
    tasks = []
    for class_id in range(len(ROADMARK_CLASSES)):
        gaps = STOP_GAPS if ROADMARK_CLASSES[class_id] == "stop_line" else (STOP_GAPS[0],)
        for gap, score, length in itertools.product(gaps, SCORES, LENGTHS_PX):
            tasks.append((f"auto_gap{gap}", class_id, score, length))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(dev_path),)) as pool:
        results = dict(pool.map(_task, tasks, chunksize=1))
    chosen = {}
    for class_id, name in enumerate(ROADMARK_CLASSES):
        candidates = {key: value for key, value in results.items() if key[1] == class_id}
        key = max(candidates, key=lambda item: (_scores(candidates[item])["f1"], -item[2], -item[3]))
        chosen[name] = {"variant": key[0], "min_score": key[2], "min_length_px": key[3],
                        "dev": _scores(candidates[key])}
    return chosen


def decode_profile(chosen: dict) -> dict:
    return {
        "orientation": "auto",
        "max_gap": [int(chosen[name]["variant"].removeprefix("auto_gap")) if name == "stop_line" else LANE_GAP
                    for name in ROADMARK_CLASSES],
        "min_score": [chosen[name]["min_score"] for name in ROADMARK_CLASSES],
        "min_length_px": [chosen[name]["min_length_px"] for name in ROADMARK_CLASSES],
    }


def report(rows: list[dict], chosen: dict) -> dict:
    result = {}
    for label in ("legacy", "configured", "tuned"):
        classes = {}
        for class_id, name in enumerate(ROADMARK_CLASSES):
            if label == "tuned":
                pick = chosen[name]
                counts = _count(rows, pick["variant"], class_id, pick["min_score"], pick["min_length_px"])
            else:
                counts = _count(rows, label, class_id, 0.0, 0.0)
            classes[name] = _scores(counts)
        total = tuple(sum(classes[name][key] for name in ROADMARK_CLASSES) for key in ("tp", "fp", "fn"))
        classes["total"] = _scores(total)
        result[label] = classes
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", default="best_stop_line")
    parser.add_argument("--dev-index-run", type=Path, required=True)
    parser.add_argument("--test-index-run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tune-workers", type=int, default=10)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    checkpoint_path = run / "checkpoints" / f"{args.role}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = PV26FocusedModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    configured = dict(cfg["train"].get("roadmark_decode") or {})
    sources = [FocusedSource(**{**source, "root": Path(source["root"])}) for source in cfg["data"]["sources"]]
    work = run / "postprocess_tuning" / args.role
    work.mkdir(parents=True, exist_ok=True)
    cached = {}
    for split, index_run in (("dev", args.dev_index_run), ("test", args.test_index_run)):
        dataset = FocusedDataset(sources, split="val", image_hw=tuple(cfg["model"]["image_hw"]),
                                 index_path=index_run / "val_samples.jsonl", selected_kind="roadmark")
        cached[split] = work / f"{split}_lines.pt"
        if not cached[split].is_file():
            torch.save(collect(model, dataset, device, args.batch_size, args.num_workers, configured),
                       cached[split])
    del model
    chosen = tune(cached["dev"], args.tune_workers)
    profile = decode_profile(chosen)
    test_rows = torch.load(cached["test"], weights_only=False)
    dev_rows = torch.load(cached["dev"], weights_only=False)
    summary = {"checkpoint": str(checkpoint_path), "global_step": int(checkpoint["global_step"]),
               "configured_decode": configured, "tuned_decode": profile, "chosen": chosen,
               "dev": report(dev_rows, chosen), "test": report(test_rows, chosen),
               "note": "filters chosen on dev only; test is the held-out report"}
    atomic_write_json(work / "summary.json", summary, ensure_ascii=False)
    tuned = copy.deepcopy(checkpoint)
    metadata = copy.deepcopy(tuned.get("run_metadata") or cfg)
    metadata.setdefault("train", {})["roadmark_decode"] = profile
    tuned["run_metadata"] = metadata
    tuned["roadmark_decode_tuning"] = str(work / "summary.json")
    torch.save(tuned, run / "checkpoints" / f"{args.role}_tuned.pt")
    print(json.dumps({"tuning": {k: summary[k] for k in ("global_step", "tuned_decode", "test")}},
                     ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
