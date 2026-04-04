from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from tools.od_bootstrap.build.debug_vis import (
    DEFAULT_DEBUG_VIS_COUNT,
    DEFAULT_DEBUG_VIS_SEED,
    generate_canonical_debug_vis,
    generate_exhaustive_debug_vis,
    generate_final_dataset_debug_vis,
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
from tools.od_bootstrap.teacher.train import run_teacher_train_scenario


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


def _add_common_path_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=None, help="Override the preset output root.")


def _resolve_output_root(args: argparse.Namespace, default: Path) -> Path:
    if args.output_root is None:
        return default
    return Path(args.output_root).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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

    train = subparsers.add_parser("train", help="Train a teacher preset.")
    train.add_argument("--teacher", choices=("mobility", "signal", "obstacle"), default="mobility")
    _add_common_path_overrides(train)
    train.set_defaults(handler=_run_teacher_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a teacher checkpoint preset.")
    eval_parser.add_argument("--teacher", choices=("mobility", "signal", "obstacle"), default="mobility")
    _add_common_path_overrides(eval_parser)
    eval_parser.set_defaults(handler=_run_teacher_eval)

    calibrate = subparsers.add_parser("calibrate", help="Calibrate class policies.")
    _add_common_path_overrides(calibrate)
    calibrate.set_defaults(handler=_run_calibration)

    exhaustive_od = subparsers.add_parser("build-exhaustive-od", help="Build the exhaustive OD dataset preset.")
    _add_common_path_overrides(exhaustive_od)
    exhaustive_od.set_defaults(handler=_run_exhaustive_od)

    finalize = subparsers.add_parser("build-final-dataset", help="Build the final exhaustive OD lane dataset.")
    _add_common_path_overrides(finalize)
    finalize.set_defaults(handler=_run_final_dataset)

    analyze_final = subparsers.add_parser("analyze-final-dataset", help="Scan final dataset class/task stats and audit manifest integrity.")
    analyze_final.add_argument("--final-root", type=Path, default=None, help="Override the final dataset root.")
    analyze_final.set_defaults(handler=_run_analyze_final_dataset)

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
        log_fn=lambda message: print(message, flush=True),
    )
    _print_json(
        {
            teacher_name: {
                "dataset_root": str(result.dataset_root),
                "manifest_path": str(result.manifest_path),
                "debug_vis_manifest_path": str(result.debug_vis_manifest_path),
                "sample_count": result.sample_count,
                "detection_count": result.detection_count,
                "class_counts": result.class_counts,
            }
            for teacher_name, result in results.items()
        }
    )
    return 0


def _run_teacher_train(args: argparse.Namespace) -> int:
    scenario = build_teacher_train_preset(args.teacher)
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    run_teacher_train_scenario(scenario, scenario_path=Path(f"preset_{scenario.teacher_name}"))
    return 0


def _run_teacher_eval(args: argparse.Namespace) -> int:
    scenario = build_teacher_eval_preset(args.teacher)
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    eval_teacher_checkpoint(scenario=scenario, scenario_path=Path(f"preset_{scenario.teacher_name}"))
    return 0


def _run_calibration(args: argparse.Namespace) -> int:
    scenario = build_calibration_preset()
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    calibrate_class_policy_scenario(scenario)
    return 0


def _run_exhaustive_od(args: argparse.Namespace) -> int:
    scenario = build_sweep_preset()
    if args.output_root is not None:
        scenario = replace(scenario, run=replace(scenario.run, output_root=_resolve_output_root(args, scenario.run.output_root)))
    run_model_centric_sweep_scenario(scenario, scenario_path=Path("preset_model_centric"))
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
