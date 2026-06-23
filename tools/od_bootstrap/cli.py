from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from common.io import read_json
from tools.od_bootstrap.build.lane_val_odpseudo import (
    DEFAULT_EXPECTED_BASE_VAL_COUNT,
    DEFAULT_MAX_REJECTED_CANDIDATES_PER_SAMPLE,
    VARIANTS_BY_NAME as LANE_VAL_ODPSEUDO_VARIANTS_BY_NAME,
    build_lane_val_odpseudo_eval_root,
    resolve_lane_val_odpseudo_variant,
    run_lane_val_odpseudo_teacher_sample_results,
)
from tools.od_bootstrap.build.debug_vis import (
    DEFAULT_DEBUG_VIS_COUNT,
    DEFAULT_DEBUG_VIS_SEED,
    DEFAULT_FINAL_LANE_AUDIT_BIN_COUNT,
    DEFAULT_FINAL_LANE_AUDIT_DIRNAME,
    DEFAULT_FINAL_LANE_AUDIT_OVERVIEW_COUNT,
    DEFAULT_FINAL_LANE_AUDIT_SAMPLES_PER_BIN,
    DEFAULT_FINAL_LANE_AUDIT_WORKERS,
    generate_canonical_debug_vis,
    generate_exhaustive_debug_vis,
    generate_final_dataset_debug_vis,
    generate_final_lane_label_audit,
    generate_teacher_dataset_debug_vis,
)
from tools.od_bootstrap.build.exhaustive_od import EXHAUSTIVE_MATERIALIZATION_MANIFEST_NAME
from tools.od_bootstrap.build.final_dataset import FINAL_DATASET_MANIFEST_NAME, build_pv26_exhaustive_od_lane_dataset
from tools.od_bootstrap.build.final_dataset_stats import analyze_final_dataset
from tools.od_bootstrap.build.review import render_final_dataset_review_bundle
from tools.od_bootstrap.build.sweep import run_model_centric_sweep_scenario
from tools.od_bootstrap.build.teacher_dataset import build_teacher_datasets
from tools.od_bootstrap.source.prepare import prepare_od_bootstrap_sources
from tools.od_bootstrap.source.types import CanonicalSourceBundle
from tools.od_bootstrap.signal_attr import (
    SignalAttrClassifierConfig,
    SignalAttrSidecarTeacher,
    SignalAttrTrainConfig,
    evaluate_signal_attr_checkpoint,
    materialize_aihub_signal_attr_crop_dataset_from_canonical_root,
    train_signal_attr_classifier,
)
from tools.od_bootstrap.presets import (
    build_calibration_preset,
    build_default_source_preset,
    build_final_dataset_preset,
    build_sweep_preset,
    build_teacher_dataset_preset,
    build_teacher_eval_preset,
    build_teacher_train_preset,
)
from tools.od_bootstrap.teacher.calibrate import calibrate_class_policy_scenario
from tools.od_bootstrap.teacher.eval import eval_teacher_checkpoint
from tools.od_bootstrap.teacher.registry import teacher_checkpoint_path, teacher_choices, teacher_definition
from tools.od_bootstrap.teacher.train import run_teacher_train_scenario


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


class _InlineProgressPrinter:
    def __init__(self) -> None:
        self._active = False
        self._rendered_lines = 1
        self._enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def __call__(self, message: str) -> None:
        text = str(message)
        if self._enabled and " progress " in text:
            self._clear_active()
            sys.stdout.write(text)
            sys.stdout.flush()
            self._active = True
            self._rendered_lines = max(1, text.count("\n") + 1)
            return
        self.finish()
        print(text, flush=True)

    def finish(self) -> None:
        if self._active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._active = False
            self._rendered_lines = 1

    def _clear_active(self) -> None:
        if not self._active:
            sys.stdout.write("\r\033[K")
            return
        for line_index in range(max(1, int(self._rendered_lines))):
            if line_index:
                sys.stdout.write("\r\033[1A")
            sys.stdout.write("\r\033[K")


def _add_common_path_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=None, help="Override the preset output root.")


def _resolve_output_root(args: argparse.Namespace, default: Path) -> Path:
    if args.output_root is None:
        return default
    return Path(args.output_root).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return payload


def _resolve_teacher_names(teacher_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted({item.strip() for item in requested if item.strip()})
    names: list[str] = []
    for child in sorted(teacher_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta" / "teacher_dataset_manifest.json").is_file():
            names.append(child.name)
    return names


def _resolve_exhaustive_run_root(exhaustive_root: Path, run_name: str | None) -> Path:
    exhaustive_root = exhaustive_root.resolve()
    if run_name:
        run_root = exhaustive_root / run_name
        if not run_root.is_dir():
            raise FileNotFoundError(f"exhaustive run not found: {run_root}")
        return run_root
    run_roots = sorted((child for child in exhaustive_root.iterdir() if child.is_dir()), key=lambda item: item.name)
    if not run_roots:
        raise FileNotFoundError(f"no exhaustive runs found under: {exhaustive_root}")
    return run_roots[-1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.od_bootstrap", description="PV26 OD bootstrap tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-sources", help="Prepare canonical sources.")
    _add_common_path_overrides(prepare)
    prepare.set_defaults(handler=_run_prepare_sources)

    teacher_datasets = subparsers.add_parser("build-teacher-datasets", help="Build teacher datasets.")
    _add_common_path_overrides(teacher_datasets)
    teacher_datasets.set_defaults(handler=_run_teacher_datasets)

    signal_attr_dataset = subparsers.add_parser(
        "build-signal-attr-dataset",
        help="Build the TL attribute crop dataset from canonical AIHUB traffic scenes.",
    )
    signal_attr_dataset.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help="Override canonical AIHUB standardized root.",
    )
    _add_common_path_overrides(signal_attr_dataset)
    signal_attr_dataset.set_defaults(handler=_run_signal_attr_dataset)

    signal_attr_train = subparsers.add_parser("train-signal-attr", help="Train the TL attribute crop classifier.")
    signal_attr_train.add_argument("--dataset-root", type=Path, default=None, help="Override signal_attr crop dataset root.")
    signal_attr_train.add_argument("--epochs", type=int, default=SignalAttrTrainConfig.epochs)
    signal_attr_train.add_argument("--batch", type=int, default=SignalAttrTrainConfig.batch_size)
    signal_attr_train.add_argument("--device", type=str, default=SignalAttrTrainConfig.device)
    signal_attr_train.add_argument("--num-workers", type=int, default=SignalAttrTrainConfig.num_workers)
    signal_attr_train.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=SignalAttrTrainConfig.pin_memory)
    signal_attr_train.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=SignalAttrTrainConfig.persistent_workers,
    )
    signal_attr_train.add_argument("--prefetch-factor", type=int, default=SignalAttrTrainConfig.prefetch_factor)
    signal_attr_train.add_argument("--learning-rate", type=float, default=SignalAttrTrainConfig.learning_rate)
    signal_attr_train.add_argument("--width", type=int, default=SignalAttrClassifierConfig.width)
    signal_attr_train.add_argument("--dropout", type=float, default=SignalAttrClassifierConfig.dropout)
    _add_common_path_overrides(signal_attr_train)
    signal_attr_train.set_defaults(handler=_run_signal_attr_train, teacher="signal_attr")

    signal_attr_eval = subparsers.add_parser("eval-signal-attr", help="Evaluate the TL attribute crop classifier.")
    signal_attr_eval.add_argument("--dataset-root", type=Path, default=None, help="Override signal_attr crop dataset root.")
    signal_attr_eval.add_argument("--checkpoint", type=Path, default=None, help="Override best_signal_attr.pt path.")
    signal_attr_eval.add_argument("--split", default="val", help="Dataset split to evaluate.")
    signal_attr_eval.add_argument("--batch", type=int, default=SignalAttrTrainConfig.batch_size)
    signal_attr_eval.add_argument("--device", type=str, default=SignalAttrTrainConfig.device)
    signal_attr_eval.add_argument("--num-workers", type=int, default=SignalAttrTrainConfig.num_workers)
    _add_common_path_overrides(signal_attr_eval)
    signal_attr_eval.set_defaults(handler=_run_signal_attr_eval, teacher="signal_attr")

    train = subparsers.add_parser("train", help="Train a teacher preset.")
    train.add_argument("--teacher", choices=teacher_choices(), default="mobility")
    train.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from the latest resumable checkpoint, or provide an exact checkpoint path.",
    )
    train.add_argument("--dataset-root", type=Path, default=None, help="signal_attr only: override crop dataset root.")
    train.add_argument("--epochs", type=int, default=None, help="signal_attr only: training epochs.")
    train.add_argument("--batch", type=int, default=None, help="signal_attr only: training batch size.")
    train.add_argument("--device", type=str, default=None, help="signal_attr only: torch device.")
    train.add_argument("--num-workers", type=int, default=None, help="signal_attr only: dataloader workers.")
    train.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="signal_attr only: enable or disable pinned host memory.",
    )
    train.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="signal_attr only: keep dataloader workers alive across epochs.",
    )
    train.add_argument("--prefetch-factor", type=int, default=None, help="signal_attr only: dataloader prefetch factor.")
    train.add_argument("--learning-rate", type=float, default=None, help="signal_attr only: optimizer learning rate.")
    train.add_argument("--width", type=int, default=None, help="signal_attr only: classifier width.")
    train.add_argument("--dropout", type=float, default=None, help="signal_attr only: classifier dropout.")
    _add_common_path_overrides(train)
    train.set_defaults(handler=_run_teacher_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a teacher checkpoint preset.")
    eval_parser.add_argument("--teacher", choices=teacher_choices(), default="mobility")
    eval_parser.add_argument("--dataset-root", type=Path, default=None, help="signal_attr only: override crop dataset root.")
    eval_parser.add_argument("--checkpoint", type=Path, default=None, help="signal_attr only: override best_signal_attr.pt path.")
    eval_parser.add_argument("--split", default=None, help="signal_attr only: dataset split to evaluate.")
    eval_parser.add_argument("--batch", type=int, default=None, help="signal_attr only: evaluation batch size.")
    eval_parser.add_argument("--device", type=str, default=None, help="signal_attr only: torch device.")
    eval_parser.add_argument("--num-workers", type=int, default=None, help="signal_attr only: dataloader workers.")
    _add_common_path_overrides(eval_parser)
    eval_parser.set_defaults(handler=_run_teacher_eval)

    calibrate = subparsers.add_parser("calibrate", help="Calibrate class policies.")
    _add_common_path_overrides(calibrate)
    calibrate.set_defaults(handler=_run_calibration)

    exhaustive_od = subparsers.add_parser("build-exhaustive-od", help="Build the exhaustive OD dataset preset.")
    exhaustive_od.add_argument(
        "--allow-default-class-policy",
        action="store_true",
        help="Allow exhaustive OD build to run without calibration/class_policy.yaml by using config defaults.",
    )
    exhaustive_od.add_argument(
        "--signal-attr-checkpoint",
        type=Path,
        default=None,
        help="Override the best_signal_attr.pt checkpoint used for attrpseudo materialization.",
    )
    exhaustive_od.add_argument(
        "--no-signal-attr-sidecar",
        action="store_true",
        help="Disable automatic signal_attr sidecar discovery for legacy OD-only materialization.",
    )
    _add_common_path_overrides(exhaustive_od)
    exhaustive_od.set_defaults(handler=_run_exhaustive_od)

    lane_val_odpseudo = subparsers.add_parser(
        "build-lane-val-odpseudo",
        help="Materialize the lane validation OD pseudo eval root from teacher/audit sample results.",
    )
    lane_val_odpseudo.add_argument(
        "--variant",
        choices=tuple(LANE_VAL_ODPSEUDO_VARIANTS_BY_NAME),
        default="v1",
        help="Eval-root contract variant to materialize.",
    )
    lane_val_odpseudo.add_argument(
        "--base-lane-root",
        type=Path,
        default=None,
        help="Override canonical AIHUB standardized lane root.",
    )
    lane_val_odpseudo.add_argument(
        "--sample-results",
        type=Path,
        default=None,
        help="JSON/JSONL sample results from the OD teacher/audit pass. Defaults to output_root/meta/sample_results.jsonl.",
    )
    lane_val_odpseudo.add_argument(
        "--generate-sample-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate missing sample_results.jsonl by running OD teachers over the lane-val set.",
    )
    lane_val_odpseudo.add_argument(
        "--overwrite-sample-results",
        action="store_true",
        help="Regenerate sample_results.jsonl even when it already exists.",
    )
    lane_val_odpseudo.add_argument(
        "--allow-default-class-policy",
        action="store_true",
        help="Allow OD teacher sweep to use default class policy when calibration class_policy.yaml is absent.",
    )
    lane_val_odpseudo.add_argument(
        "--expected-base-count",
        type=int,
        default=None,
        help=f"Expected base val sample count. Default: {DEFAULT_EXPECTED_BASE_VAL_COUNT}.",
    )
    lane_val_odpseudo.add_argument(
        "--signal-attr-checkpoint",
        type=Path,
        default=None,
        help="attr_v2 only: override the best_signal_attr.pt sidecar checkpoint.",
    )
    lane_val_odpseudo.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for OD teacher sweep and attr_v2 signal_attr sidecar. Defaults to the sweep preset.",
    )
    lane_val_odpseudo.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override OD teacher sweep batch size.",
    )
    lane_val_odpseudo.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override OD teacher sweep image size.",
    )
    lane_val_odpseudo.add_argument(
        "--predict-conf",
        type=float,
        default=None,
        help="Override OD teacher sweep raw prediction confidence floor.",
    )
    lane_val_odpseudo.add_argument(
        "--predict-iou",
        type=float,
        default=None,
        help="Override OD teacher sweep raw prediction IoU setting.",
    )
    lane_val_odpseudo.add_argument(
        "--max-rejected-candidates-per-sample",
        type=int,
        default=DEFAULT_MAX_REJECTED_CANDIDATES_PER_SAMPLE,
        help=f"Maximum rejected candidate audit rows to emit per sample. Default: {DEFAULT_MAX_REJECTED_CANDIDATES_PER_SAMPLE}.",
    )
    lane_val_odpseudo.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy images instead of the default hardlink-with-copy-fallback behavior.",
    )
    _add_common_path_overrides(lane_val_odpseudo)
    lane_val_odpseudo.set_defaults(handler=_run_lane_val_odpseudo)

    finalize = subparsers.add_parser("build-final-dataset", help="Build the final exhaustive OD lane dataset.")
    _add_common_path_overrides(finalize)
    finalize.set_defaults(handler=_run_final_dataset)

    analyze_final = subparsers.add_parser("analyze-final-dataset", help="Scan final dataset class/task stats and audit manifest integrity.")
    analyze_final.add_argument("--final-root", type=Path, default=None, help="Override the final dataset root.")
    analyze_final.set_defaults(handler=_run_analyze_final_dataset)

    audit_final = subparsers.add_parser("audit-final-lane-labels", help="Render shallow final-dataset audit overlays with lane-focused stratified sampling.")
    audit_final.add_argument("--final-root", type=Path, default=None, help="Override the final dataset root.")
    audit_final.add_argument("--output-root", type=Path, default=None, help="Override the audit output root.")
    audit_final.add_argument("--overview-count", type=int, default=DEFAULT_FINAL_LANE_AUDIT_OVERVIEW_COUNT, help="Approximate overview overlay budget across the final dataset.")
    audit_final.add_argument("--lane-bin-count", type=int, default=DEFAULT_FINAL_LANE_AUDIT_BIN_COUNT, help="Number of contiguous bins per lane clip.")
    audit_final.add_argument("--lane-samples-per-bin", type=int, default=DEFAULT_FINAL_LANE_AUDIT_SAMPLES_PER_BIN, help="How many overlays to sample from each lane bin.")
    audit_final.add_argument("--workers", type=int, default=DEFAULT_FINAL_LANE_AUDIT_WORKERS, help="Maximum parallel overlay render workers.")
    audit_final.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED, help="Sampling seed.")
    audit_final.set_defaults(handler=_run_audit_final_lane_labels)

    review_final = subparsers.add_parser("review-final-dataset", help="Render focused final-dataset overlay review samples.")
    review_final.add_argument("--final-root", type=Path, default=None, help="Override the final dataset root.")
    review_final.add_argument("--focus", required=True, help="Focus target such as traffic_light, lane, stop_line, crosswalk, tl_attr, or a detector class.")
    review_final.add_argument("--split", default="val", help="Dataset split to review.")
    review_final.add_argument("--count", type=int, default=50, help="Maximum number of review overlays to render.")
    review_final.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED, help="Sampling seed.")
    review_final.add_argument("--output-root", type=Path, default=None, help="Override the review output root.")
    review_final.set_defaults(handler=_run_review_final_dataset)

    debug_vis = subparsers.add_parser("generate-debug-vis", help="Render bootstrap debug visualizations.")
    debug_vis.add_argument(
        "--mode",
        choices=("canonical", "teacher", "exhaustive", "final", "all"),
        default="canonical",
        help="Which debug-vis target to render.",
    )
    debug_vis.add_argument("--bootstrap-root", type=Path, default=None, help="Override the bootstrap root.")
    debug_vis.add_argument("--teacher-root", type=Path, default=None, help="Override the teacher dataset root.")
    debug_vis.add_argument("--teacher", action="append", default=None, help="Teacher name to render. Repeatable.")
    debug_vis.add_argument("--exhaustive-root", type=Path, default=None, help="Override the exhaustive OD root.")
    debug_vis.add_argument("--final-root", type=Path, default=None, help="Override the final dataset root.")
    debug_vis.add_argument("--run", type=str, default=None, help="Sweep run directory name. Default: latest.")
    debug_vis.add_argument("--count", type=int, default=DEFAULT_DEBUG_VIS_COUNT)
    debug_vis.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED)
    debug_vis.set_defaults(handler=_run_debug_vis)

    return parser


def _run_prepare_sources(args: argparse.Namespace) -> int:
    preset = replace(build_default_source_preset(), output_root=_resolve_output_root(args, build_default_source_preset().output_root))
    result = prepare_od_bootstrap_sources(preset)
    _print_json(
        {
            "bundle": {
                "bdd_root": str(result.bundle.bdd_root),
                "aihub_root": str(result.bundle.aihub_root),
                "output_root": str(result.bundle.output_root),
                "bootstrap_source_keys": list(result.bundle.bootstrap_source_keys),
                "excluded_source_keys": list(result.bundle.excluded_source_keys),
            },
            "manifest_path": str(result.manifest_path),
            "image_list_manifest_path": str(result.image_list_manifest_path),
            "canonical_debug_vis_manifest_paths": {
                dataset_name: str(path)
                for dataset_name, path in result.canonical_debug_vis_manifest_paths.items()
            },
            "bdd_output_root": str(result.bdd_outputs["output_root"]),
            "aihub_output_root": str(result.aihub_outputs["output_root"]),
        }
    )
    return 0


def _run_teacher_datasets(args: argparse.Namespace) -> int:
    log_printer = _InlineProgressPrinter()
    preset = replace(
        build_teacher_dataset_preset(),
        output_root=_resolve_output_root(args, build_teacher_dataset_preset().output_root),
    )
    canonical_bundle = CanonicalSourceBundle(
        bdd_root=preset.canonical_root / "canonical" / "bdd100k_det_100k",
        aihub_root=preset.canonical_root / "canonical" / "aihub_standardized",
        output_root=preset.canonical_root,
        bootstrap_source_keys=("bdd100k_det_100k", "aihub_traffic_seoul", "aihub_obstacle_seoul"),
        excluded_source_keys=("aihub_lane_seoul",),
    )
    results = build_teacher_datasets(
        canonical_bundle,
        preset.output_root,
        copy_images=preset.copy_images,
        workers=preset.workers,
        log_every=preset.log_every,
        debug_vis_count=preset.debug_vis_count,
        debug_vis_seed=preset.debug_vis_seed,
        log_fn=log_printer,
    )
    signal_attr_manifest = materialize_aihub_signal_attr_crop_dataset_from_canonical_root(
        preset.canonical_root / "canonical" / "aihub_standardized",
        (preset.output_root / "signal_attr").resolve(),
        workers=preset.workers,
        log_every=preset.log_every,
        log_fn=log_printer,
    )
    log_printer.finish()
    _print_json(
        {
            "teachers": {
                teacher_name: {
                    "dataset_root": str(result.dataset_root),
                    "manifest_path": str(result.manifest_path),
                    "debug_vis_manifest_path": str(result.debug_vis_manifest_path),
                    "sample_count": result.sample_count,
                    "detection_count": result.detection_count,
                    "class_counts": result.class_counts,
                }
                for teacher_name, result in results.items()
            },
            "signal_attr": signal_attr_manifest,
        }
    )
    return 0


def _run_signal_attr_dataset(args: argparse.Namespace) -> int:
    log_printer = _InlineProgressPrinter()
    preset = build_teacher_dataset_preset()
    canonical_root = (
        Path(args.canonical_root).resolve()
        if args.canonical_root is not None
        else preset.canonical_root / "canonical" / "aihub_standardized"
    )
    output_root = (
        _resolve_output_root(args, preset.output_root / "signal_attr")
        if args.output_root is not None
        else (preset.output_root / "signal_attr").resolve()
    )
    manifest = materialize_aihub_signal_attr_crop_dataset_from_canonical_root(
        canonical_root,
        output_root,
        workers=preset.workers,
        log_every=preset.log_every,
        log_fn=log_printer,
    )
    log_printer.finish()
    _print_json(manifest)
    return 0


def _signal_attr_dataset_root_override(path: Path | None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return (build_teacher_dataset_preset().output_root / "signal_attr").resolve()


def _value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _signal_attr_train_output_root(args: argparse.Namespace) -> Path:
    default_root = build_teacher_train_preset("signal").run.output_root / "signal_attr"
    if args.output_root is not None:
        return _resolve_output_root(args, default_root)
    return default_root.resolve()


def _signal_attr_eval_output_root(args: argparse.Namespace) -> Path:
    default_root = build_teacher_eval_preset("signal").run.output_root / "signal_attr"
    if args.output_root is not None:
        return _resolve_output_root(args, default_root)
    return default_root.resolve()


def _run_signal_attr_train(args: argparse.Namespace) -> int:
    log_printer = _InlineProgressPrinter()
    dataset_root = _signal_attr_dataset_root_override(args.dataset_root)
    train_output_root = _signal_attr_train_output_root(args)
    summary = train_signal_attr_classifier(
        dataset_root,
        train_output_root,
        train_config=SignalAttrTrainConfig(
            epochs=int(_value_or_default(args.epochs, SignalAttrTrainConfig.epochs)),
            batch_size=int(_value_or_default(args.batch, SignalAttrTrainConfig.batch_size)),
            learning_rate=float(_value_or_default(args.learning_rate, SignalAttrTrainConfig.learning_rate)),
            device=str(_value_or_default(args.device, SignalAttrTrainConfig.device)),
            num_workers=int(_value_or_default(args.num_workers, SignalAttrTrainConfig.num_workers)),
            pin_memory=bool(_value_or_default(args.pin_memory, SignalAttrTrainConfig.pin_memory)),
            persistent_workers=bool(
                _value_or_default(args.persistent_workers, SignalAttrTrainConfig.persistent_workers)
            ),
            prefetch_factor=int(_value_or_default(args.prefetch_factor, SignalAttrTrainConfig.prefetch_factor)),
        ),
        model_config=SignalAttrClassifierConfig(
            width=int(_value_or_default(args.width, SignalAttrClassifierConfig.width)),
            dropout=float(_value_or_default(args.dropout, SignalAttrClassifierConfig.dropout)),
        ),
        log_fn=log_printer,
    )
    log_printer.finish()
    _print_json(summary)
    return 0


def _run_signal_attr_eval(args: argparse.Namespace) -> int:
    log_printer = _InlineProgressPrinter()
    dataset_root = _signal_attr_dataset_root_override(args.dataset_root)
    train_root = build_teacher_train_preset("signal").run.output_root / "signal_attr"
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint is not None else (train_root / "best_signal_attr.pt").resolve()
    output_root = _signal_attr_eval_output_root(args)
    report = evaluate_signal_attr_checkpoint(
        dataset_root,
        checkpoint_path,
        output_root,
        split=str(_value_or_default(args.split, "val")),
        batch_size=int(_value_or_default(args.batch, SignalAttrTrainConfig.batch_size)),
        device=str(_value_or_default(args.device, SignalAttrTrainConfig.device)),
        num_workers=int(_value_or_default(args.num_workers, SignalAttrTrainConfig.num_workers)),
        log_fn=log_printer,
    )
    log_printer.finish()
    _print_json(report)
    return 0


def _reject_signal_attr_only_train_overrides(args: argparse.Namespace) -> None:
    signal_attr_only = {
        "dataset_root": args.dataset_root,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "learning_rate": args.learning_rate,
        "width": args.width,
        "dropout": args.dropout,
    }
    used = sorted(name for name, value in signal_attr_only.items() if value is not None)
    if used:
        raise ValueError(f"{', '.join(used)} are only supported with --teacher signal_attr")


def _reject_signal_attr_only_eval_overrides(args: argparse.Namespace) -> None:
    signal_attr_only = {
        "dataset_root": args.dataset_root,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "batch": args.batch,
        "device": args.device,
        "num_workers": args.num_workers,
    }
    used = sorted(name for name, value in signal_attr_only.items() if value is not None)
    if used:
        raise ValueError(f"{', '.join(used)} are only supported with --teacher signal_attr")


def _run_teacher_train(args: argparse.Namespace) -> int:
    if teacher_definition(args.teacher).kind == "signal_attr":
        return _run_signal_attr_train(args)
    _reject_signal_attr_only_train_overrides(args)
    scenario = build_teacher_train_preset(args.teacher)
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    if args.resume is not None:
        scenario = replace(scenario, train=replace(scenario.train, resume=args.resume))
    run_teacher_train_scenario(scenario, scenario_path=Path(f"preset_{scenario.teacher_name}"))
    return 0


def _run_teacher_eval(args: argparse.Namespace) -> int:
    if teacher_definition(args.teacher).kind == "signal_attr":
        return _run_signal_attr_eval(args)
    _reject_signal_attr_only_eval_overrides(args)
    scenario = build_teacher_eval_preset(args.teacher)
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    eval_teacher_checkpoint(scenario=scenario, scenario_path=Path(f"preset_{scenario.teacher_name}"))
    return 0


def _run_calibration(args: argparse.Namespace) -> int:
    scenario = build_calibration_preset()
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    calibrate_class_policy_scenario(scenario, scenario_path=Path("preset_calibration"))
    return 0


def _run_exhaustive_od(args: argparse.Namespace) -> int:
    scenario = build_sweep_preset(allow_default_class_policy=bool(args.allow_default_class_policy))
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    signal_attr_checkpoint_path = _resolve_signal_attr_sidecar_checkpoint(args)
    run_model_centric_sweep_scenario(
        scenario,
        scenario_path=Path("preset_model_centric"),
        signal_attr_checkpoint_path=signal_attr_checkpoint_path,
    )
    return 0


def _resolve_signal_attr_sidecar_checkpoint(args: argparse.Namespace) -> Path | None:
    if bool(getattr(args, "no_signal_attr_sidecar", False)):
        return None
    if args.signal_attr_checkpoint is not None:
        return Path(args.signal_attr_checkpoint).resolve()
    default_checkpoint = (build_teacher_train_preset("signal").run.output_root / "signal_attr" / "best_signal_attr.pt").resolve()
    default_eval_report = (build_teacher_eval_preset("signal").run.output_root / "signal_attr" / "signal_attr_eval_report.json").resolve()
    if default_checkpoint.is_file() and default_eval_report.is_file():
        return default_checkpoint
    return None


def _load_lane_val_sample_results(path: Path) -> list[dict[str, Any]] | dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"lane-val OD pseudo sample results not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"sample results row must be an object: {path}:{line_number}")
            rows.append(payload)
        return rows
    payload = read_json(path)
    if isinstance(payload.get("sample_results"), list):
        return [dict(item) for item in payload["sample_results"]]
    if isinstance(payload.get("sample_results"), dict):
        return {str(key): dict(value) for key, value in payload["sample_results"].items()}
    if isinstance(payload.get("samples"), list):
        return [dict(item) for item in payload["samples"]]
    return {str(key): dict(value) for key, value in payload.items()}


def _lane_val_teacher_checkpoints() -> dict[str, Path]:
    train_root = build_teacher_train_preset("signal").run.output_root
    return {
        teacher_name: teacher_checkpoint_path(train_root, teacher_name)
        for teacher_name in ("mobility", "signal", "obstacle")
    }


def _run_lane_val_odpseudo(args: argparse.Namespace) -> int:
    variant = resolve_lane_val_odpseudo_variant(args.variant)
    teacher_dataset_preset = build_teacher_dataset_preset()
    base_lane_root = (
        Path(args.base_lane_root).resolve()
        if args.base_lane_root is not None
        else teacher_dataset_preset.canonical_root / "canonical" / "aihub_standardized"
    )
    default_output_root = Path(__file__).resolve().parents[2] / "seg_dataset" / variant.dataset_key
    output_root = _resolve_output_root(args, default_output_root)
    sample_results_path = (
        Path(args.sample_results).resolve()
        if args.sample_results is not None
        else output_root / "meta" / "sample_results.jsonl"
    )
    expected_base_count = (
        DEFAULT_EXPECTED_BASE_VAL_COUNT
        if args.expected_base_count is None
        else int(args.expected_base_count)
    )
    sweep_scenario = None
    if bool(args.generate_sample_results) and (bool(args.overwrite_sample_results) or not sample_results_path.is_file()):
        sweep_scenario = build_sweep_preset(allow_default_class_policy=bool(args.allow_default_class_policy))
        run_config = sweep_scenario.run
        if args.device is not None:
            run_config = replace(run_config, device=str(args.device))
        if args.batch is not None:
            run_config = replace(run_config, batch_size=int(args.batch))
        if args.imgsz is not None:
            run_config = replace(run_config, imgsz=int(args.imgsz))
        if args.predict_conf is not None:
            run_config = replace(run_config, predict_conf=float(args.predict_conf))
        if args.predict_iou is not None:
            run_config = replace(run_config, predict_iou=float(args.predict_iou))
        sample_summary = run_lane_val_odpseudo_teacher_sample_results(
            base_lane_root=base_lane_root,
            output_root=output_root,
            teachers=sweep_scenario.teachers,
            class_policy=sweep_scenario.class_policy,
            run_config=run_config,
            expected_base_count=expected_base_count,
            sample_results_path=sample_results_path,
            max_rejected_candidates_per_sample=args.max_rejected_candidates_per_sample,
            overwrite=bool(args.overwrite_sample_results),
            log_fn=lambda message: print(message, flush=True),
        )
        print(json.dumps({"sample_results": sample_summary}, indent=2, ensure_ascii=True, default=str), flush=True)

    signal_attr_checkpoint = None
    signal_attr_sidecar = None
    if variant.tl_attr_enabled:
        train_root = build_teacher_train_preset("signal").run.output_root
        signal_attr_checkpoint = (
            Path(args.signal_attr_checkpoint).resolve()
            if args.signal_attr_checkpoint is not None
            else teacher_checkpoint_path(train_root, "signal_attr")
        )
        sidecar_device = str(args.device or (sweep_scenario.run.device if sweep_scenario is not None else "cuda:0"))
        signal_attr_sidecar = SignalAttrSidecarTeacher.from_checkpoint(
            signal_attr_checkpoint,
            device=sidecar_device,
        )
    teacher_checkpoints = (
        {teacher.name: teacher.checkpoint_path for teacher in sweep_scenario.teachers}
        if sweep_scenario is not None
        else _lane_val_teacher_checkpoints()
    )

    summary = build_lane_val_odpseudo_eval_root(
        base_lane_root=base_lane_root,
        output_root=output_root,
        sample_results=_load_lane_val_sample_results(sample_results_path),
        teacher_checkpoints=teacher_checkpoints,
        signal_attr_checkpoint=signal_attr_checkpoint,
        signal_attr_sidecar=signal_attr_sidecar,
        copy_images=bool(args.copy_images),
        variant=variant,
        expected_base_count=expected_base_count,
    )
    _print_json(summary)
    return 0


def _run_final_dataset(args: argparse.Namespace) -> int:
    preset = replace(build_final_dataset_preset(), output_root=_resolve_output_root(args, build_final_dataset_preset().output_root))
    result = build_pv26_exhaustive_od_lane_dataset(
        exhaustive_od_root=preset.exhaustive_od_root,
        aihub_canonical_root=preset.aihub_canonical_root,
        output_root=preset.output_root,
        copy_images=preset.copy_images,
        log_fn=lambda message: print(message, flush=True),
    )
    _print_json(result)
    return 0


def _resolve_final_root_override(path: Path | None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return build_final_dataset_preset().output_root.resolve()


def _run_analyze_final_dataset(args: argparse.Namespace) -> int:
    final_root = _resolve_final_root_override(args.final_root)
    result = analyze_final_dataset(dataset_root=final_root, write_artifacts=True)
    _print_json(result)
    return 0


def _run_audit_final_lane_labels(args: argparse.Namespace) -> int:
    final_root = _resolve_final_root_override(args.final_root)
    manifest_path = final_root / "meta" / FINAL_DATASET_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final dataset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    sample_rows = [dict(item) for item in manifest.get("samples") or []]
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else final_root / DEFAULT_FINAL_LANE_AUDIT_DIRNAME
    )
    result = generate_final_lane_label_audit(
        dataset_root=final_root,
        manifest_rows=sample_rows,
        output_root=output_root,
        overview_count=int(args.overview_count),
        lane_bin_count=int(args.lane_bin_count),
        lane_samples_per_bin=int(args.lane_samples_per_bin),
        debug_vis_seed=int(args.seed),
        workers=int(args.workers),
        log_fn=lambda message: print(message, flush=True),
    )
    _print_json(
        {
            "dataset_root": str(final_root),
            "output_root": str(result["output_root"]),
            "index_path": str(result["index_path"]),
            "summary_path": str(result["summary_path"]),
            "selection_count": int(result["selection_count"]),
        }
    )
    return 0


def _run_review_final_dataset(args: argparse.Namespace) -> int:
    final_root = _resolve_final_root_override(args.final_root)
    focus = str(args.focus).strip()
    split = str(args.split).strip()
    count = int(args.count)
    seed = int(args.seed)
    if count <= 0:
        raise SystemExit("--count must be > 0")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else final_root / "meta" / "review" / focus / split
    )
    result = render_final_dataset_review_bundle(
        dataset_root=final_root,
        output_root=output_root,
        focus=focus,
        split=split,
        count=count,
        seed=seed,
    )
    _print_json(result)
    return 0


def _run_canonical_debug_vis(*, bootstrap_root: Path, count: int, seed: int) -> dict[str, Any]:
    bootstrap_root = bootstrap_root.resolve()
    image_list_manifest_path = bootstrap_root / "meta" / "bootstrap_image_list.jsonl"
    canonical_root = bootstrap_root / "canonical"
    if not image_list_manifest_path.is_file():
        raise FileNotFoundError(f"bootstrap image list manifest not found: {image_list_manifest_path}")
    outputs = generate_canonical_debug_vis(
        image_list_manifest_path=image_list_manifest_path,
        canonical_root=canonical_root,
        debug_vis_count=count,
        debug_vis_seed=seed,
        log_fn=lambda message: print(message, flush=True),
    )
    return {
        dataset_name: {
            "debug_vis_dir": str(payload["debug_vis_dir"]),
            "debug_vis_manifest_path": str(payload["debug_vis_manifest"]),
            "selection_count": int(payload["selection_count"]),
        }
        for dataset_name, payload in outputs.items()
    }


def _run_teacher_debug_vis(*, teacher_root: Path, teacher_names: list[str], count: int, seed: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for teacher_name in teacher_names:
        dataset_root = teacher_root / teacher_name
        manifest_path = dataset_root / "meta" / "teacher_dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"teacher dataset manifest not found: {manifest_path}")
        manifest = _load_json(manifest_path)
        class_names = [str(item) for item in manifest.get("class_names") or []]
        sample_rows = [dict(item) for item in manifest.get("samples") or []]
        result = generate_teacher_dataset_debug_vis(
            dataset_root=dataset_root,
            teacher_name=teacher_name,
            class_names=class_names,
            manifest_rows=sample_rows,
            debug_vis_count=count,
            debug_vis_seed=seed,
            log_fn=lambda message: print(message, flush=True),
        )
        outputs[teacher_name] = {
            "dataset_root": str(dataset_root),
            "debug_vis_dir": str(result["debug_vis_dir"]),
            "debug_vis_manifest_path": str(result["debug_vis_manifest"]),
            "selection_count": int(result["selection_count"]),
        }
    return outputs


def _run_exhaustive_debug_vis(*, exhaustive_root: Path, run_name: str | None, count: int, seed: int) -> dict[str, Any]:
    run_root = _resolve_exhaustive_run_root(exhaustive_root, run_name)
    manifest_path = run_root / "meta" / EXHAUSTIVE_MATERIALIZATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"materialization manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    sample_rows = [dict(item) for item in manifest.get("samples") or []]
    result = generate_exhaustive_debug_vis(
        dataset_root=run_root,
        manifest_rows=sample_rows,
        debug_vis_count=count,
        debug_vis_seed=seed,
        log_fn=lambda message: print(message, flush=True),
    )
    return {
        "run_root": str(run_root),
        "run_id": run_root.name,
        "debug_vis_dir": str(result["debug_vis_dir"]),
        "debug_vis_manifest_path": str(result["debug_vis_manifest"]),
        "selection_count": int(result["selection_count"]),
    }


def _run_final_debug_vis(*, final_root: Path, count: int, seed: int) -> dict[str, Any]:
    dataset_root = final_root.resolve()
    manifest_path = dataset_root / "meta" / FINAL_DATASET_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final dataset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    sample_rows = [dict(item) for item in manifest.get("samples") or []]
    result = generate_final_dataset_debug_vis(
        dataset_root=dataset_root,
        manifest_rows=sample_rows,
        debug_vis_count=count,
        debug_vis_seed=seed,
        log_fn=lambda message: print(message, flush=True),
    )
    return {
        "dataset_root": str(dataset_root),
        "debug_vis_dir": str(result["debug_vis_dir"]),
        "debug_vis_manifest_path": str(result["debug_vis_manifest"]),
        "selection_count": int(result["selection_count"]),
    }


def _run_debug_vis(args: argparse.Namespace) -> int:
    bootstrap_root = Path(args.bootstrap_root).resolve() if args.bootstrap_root is not None else build_default_source_preset().output_root
    teacher_root = Path(args.teacher_root).resolve() if args.teacher_root is not None else bootstrap_root / "teacher_datasets"
    exhaustive_root = Path(args.exhaustive_root).resolve() if args.exhaustive_root is not None else bootstrap_root / "exhaustive_od"
    final_root = (
        Path(args.final_root).resolve()
        if args.final_root is not None
        else build_final_dataset_preset().output_root
    )
    count = int(args.count)
    seed = int(args.seed)

    if count < 0:
        raise SystemExit("--count must be >= 0")

    if args.mode == "canonical":
        result = {"canonical": _run_canonical_debug_vis(bootstrap_root=bootstrap_root, count=count, seed=seed)}
    elif args.mode == "teacher":
        result = {
            "teachers": _run_teacher_debug_vis(
                teacher_root=teacher_root,
                teacher_names=_resolve_teacher_names(teacher_root, args.teacher),
                count=count,
                seed=seed,
            )
        }
    elif args.mode == "exhaustive":
        result = {
            "exhaustive": _run_exhaustive_debug_vis(
                exhaustive_root=exhaustive_root,
                run_name=args.run,
                count=count,
                seed=seed,
            )
        }
    elif args.mode == "final":
        result = {"final": _run_final_debug_vis(final_root=final_root, count=count, seed=seed)}
    else:
        result = {
            "canonical": _run_canonical_debug_vis(bootstrap_root=bootstrap_root, count=count, seed=seed),
            "teachers": _run_teacher_debug_vis(
                teacher_root=teacher_root,
                teacher_names=_resolve_teacher_names(teacher_root, args.teacher),
                count=count,
                seed=seed,
            ),
        }
        if exhaustive_root.is_dir():
            try:
                result["exhaustive"] = _run_exhaustive_debug_vis(
                    exhaustive_root=exhaustive_root,
                    run_name=args.run,
                    count=count,
                    seed=seed,
                )
            except FileNotFoundError:
                pass
        if final_root.is_dir():
            try:
                result["final"] = _run_final_debug_vis(final_root=final_root, count=count, seed=seed)
            except FileNotFoundError:
                pass

    _print_json(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
