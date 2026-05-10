from __future__ import annotations

import argparse
import json
from pathlib import Path
import site
import sys
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from common.overlay import render_overlay
from model.engine.batch import raw_batch_for_metrics
from model.engine.postprocess import postprocess_pv26_batch
from tools.evaluate_pv26_lane60_checkpoint import _advance_validation_sampler
from tools.pv26_train import cli as train_cli
from tools.pv26_train import config as train_config_api
from tools.pv26_train.epoch_visualization import _gt_scene_from_sample, _prediction_to_overlay_scene
from tools.run_pv26_lane60_probe import _lane60_scenario


SOURCE_RUN = (
    REPO_ROOT
    / "runs"
    / "pv26_exhaustive_od_lane_train"
    / "lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412"
)
DEFAULT_CHECKPOINT = SOURCE_RUN / "phase_4" / "checkpoints" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GT/component/row-scan comparison grids for a lane60 checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument("--preset", default="default")
    parser.add_argument("--baseline-experiment", default="core_centerline_refine_cross_retain")
    parser.add_argument("--row-scan-experiment", default="core_centerline_refine_row_scan_vectorizer")
    parser.add_argument("--max-val-batches", type=int, default=128)
    parser.add_argument("--validation-epoch", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--include-unchanged", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(SOURCE_RUN / "analysis_exports" / "row_scan_visual_compare_epoch2"),
    )
    return parser.parse_args()


def _scenario_for(args: argparse.Namespace, *, experiment: str) -> tuple[Any, Any, int]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run = Path(args.source_run).expanduser().resolve()
    scenario_args = argparse.Namespace(
        preset=str(args.preset),
        source_run=str(source_run),
        seed_checkpoint=str(checkpoint),
        experiment=str(experiment),
        epochs=1,
        train_batches=int(args.train_batches),
        val_batches=int(args.max_val_batches),
        batch_size=int(args.batch_size),
        device=str(args.device),
        run_root="",
        preview=False,
    )
    scenario, _scenario_path, options = _lane60_scenario(
        scenario_args,
        source_run=source_run,
        seed_checkpoint=checkpoint,
    )
    phase_index = int(tuple(options["selected_phase_indices"])[0])
    phase = scenario.phases[phase_index - 1]
    train_config = train_config_api.scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    return scenario, train_config, phase_index


def _detach_to_cpu(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu()
    if isinstance(item, dict):
        return {key: _detach_to_cpu(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_detach_to_cpu(value) for value in item]
    if isinstance(item, tuple):
        return tuple(_detach_to_cpu(value) for value in item)
    return item


def _sample_from_raw(raw_batch: dict[str, Any], index: int) -> dict[str, Any]:
    sample = {
        "det_targets": raw_batch["det_targets"][index],
        "tl_attr_targets": raw_batch["tl_attr_targets"][index],
        "lane_targets": raw_batch["lane_targets"][index],
        "source_mask": raw_batch["source_mask"][index],
        "valid_mask": raw_batch["valid_mask"][index],
        "meta": raw_batch["meta"][index],
    }
    if "image" in raw_batch and isinstance(raw_batch["image"], torch.Tensor):
        sample["image"] = raw_batch["image"][index]
    return sample


def _lane_signature(prediction: dict[str, Any]) -> tuple[Any, ...]:
    signature = []
    for lane in prediction.get("lanes", []):
        points = lane.get("points_xy", [])
        compact = tuple((round(float(x), 1), round(float(y), 1)) for x, y in points[:: max(1, len(points) // 6 or 1)])
        signature.append((str(lane.get("class_name")), str(lane.get("lane_type")), compact))
    return tuple(signature)


def _safe_id(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)[:120]


def _compose_triplet(
    *,
    sample_id: str,
    gt_path: Path,
    component_path: Path,
    row_scan_path: Path,
    output_path: Path,
) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for visual comparison grids")
    images = [Image.open(path).convert("RGB") for path in (gt_path, component_path, row_scan_path)]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    header_height = 48
    canvas = Image.new("RGB", (width * 3, height + header_height), color=(12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ("ground_truth", "component", "row_scan")
    for column, (label, image) in enumerate(zip(labels, images)):
        canvas.paste(image, (column * width, header_height))
        draw.text((column * width + 8, 24), label, fill=(255, 255, 255), font=font)
    draw.text((8, 6), sample_id, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _compose_grid(tile_paths: list[Path], *, columns: int, output_path: Path) -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for visual comparison grids")
    if not tile_paths:
        raise ValueError("cannot compose visual grid with zero tiles")
    tiles = [Image.open(path).convert("RGB") for path in tile_paths]
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + max(1, int(columns)) - 1) // max(1, int(columns))
    canvas = Image.new("RGB", (tile_width * max(1, int(columns)), tile_height * rows), color=(8, 8, 8))
    for index, tile in enumerate(tiles):
        row = index // max(1, int(columns))
        col = index % max(1, int(columns))
        canvas.paste(tile, (col * tile_width, row * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    _baseline_scenario, baseline_train_config, _baseline_phase_index = _scenario_for(
        args, experiment=str(args.baseline_experiment)
    )
    row_scenario, row_train_config, row_phase_index = _scenario_for(args, experiment=str(args.row_scan_experiment))
    baseline_postprocess = train_cli._build_postprocess_config(baseline_train_config)
    row_postprocess = train_cli._build_postprocess_config(row_train_config)

    train_cli._configure_torch_multiprocessing()
    dataset = train_cli.PV26CanonicalDataset(
        train_cli._existing_dataset_roots(row_scenario),
        train_augmentation=False,
        progress_callback=lambda message: print(f"[row_scan_visual] {message}", flush=True),
    )
    phase = row_scenario.phases[row_phase_index - 1]
    _, val_loader = train_cli._build_phase_train_loaders(dataset, train_config=row_train_config, phase=phase)
    if val_loader is None:
        raise ValueError("row-scan visual compare requires validation batches")
    _advance_validation_sampler(val_loader, validation_epoch=int(args.validation_epoch))

    trainer = train_cli._build_phase_trainer(phase, row_train_config)
    trainer.load_model_weights(checkpoint, map_location=row_train_config.device)
    evaluator = trainer.build_evaluator()
    evaluator.adapter.raw_model.eval()
    evaluator.heads.eval()

    tile_paths: list[Path] = []
    entries: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader, start=1):
            if batch_index > int(args.max_val_batches) or len(entries) >= int(args.sample_count):
                break
            if batch_index == 1 or batch_index % 20 == 0:
                print(f"[row_scan_visual] eval batch {batch_index}/{args.max_val_batches}", flush=True)
            raw_batch = raw_batch_for_metrics(batch)
            if raw_batch is None:
                raise ValueError("validation loader must provide raw batches")
            encoded = evaluator.prepare_batch(batch)
            predictions = _detach_to_cpu(evaluator.forward_encoded_batch(encoded))
            meta = _detach_to_cpu(encoded["meta"])
            component_predictions = postprocess_pv26_batch(predictions, meta, config=baseline_postprocess)
            row_scan_predictions = postprocess_pv26_batch(predictions, meta, config=row_postprocess)
            for sample_index, (component_pred, row_scan_pred) in enumerate(
                zip(component_predictions, row_scan_predictions)
            ):
                if len(entries) >= int(args.sample_count):
                    break
                changed = _lane_signature(component_pred) != _lane_signature(row_scan_pred)
                if not changed and not bool(args.include_unchanged):
                    continue
                sample = _sample_from_raw(raw_batch, sample_index)
                sample_id = str(sample["meta"].get("sample_id", f"batch{batch_index}_sample{sample_index}"))
                sample_dir = output_dir / f"{len(entries) + 1:02d}__{_safe_id(sample_id)}"
                gt_path = sample_dir / "ground_truth.png"
                component_path = sample_dir / "component.png"
                row_scan_path = sample_dir / "row_scan.png"
                tile_path = sample_dir / "comparison.png"
                render_overlay(_gt_scene_from_sample(sample), gt_path)
                render_overlay(_prediction_to_overlay_scene(component_pred, sample), component_path)
                render_overlay(_prediction_to_overlay_scene(row_scan_pred, sample), row_scan_path)
                _compose_triplet(
                    sample_id=sample_id,
                    gt_path=gt_path,
                    component_path=component_path,
                    row_scan_path=row_scan_path,
                    output_path=tile_path,
                )
                tile_paths.append(tile_path)
                entries.append(
                    {
                        "sample_id": sample_id,
                        "dataset_key": str(sample["meta"].get("dataset_key")),
                        "image_path": str(sample["meta"].get("image_path")),
                        "component_lane_count": len(component_pred.get("lanes", [])),
                        "row_scan_lane_count": len(row_scan_pred.get("lanes", [])),
                        "comparison_path": str(tile_path),
                    }
                )

    grid_path = output_dir / "row_scan_component_comparison_grid.png"
    _compose_grid(tile_paths, columns=int(args.columns), output_path=grid_path)
    manifest = {
        "checkpoint": str(checkpoint),
        "baseline_experiment": str(args.baseline_experiment),
        "row_scan_experiment": str(args.row_scan_experiment),
        "validation_epoch": int(args.validation_epoch),
        "max_val_batches": int(args.max_val_batches),
        "sample_count": len(entries),
        "grid_path": str(grid_path),
        "samples": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grid_path": str(grid_path), "sample_count": len(entries)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
