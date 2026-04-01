from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.preprocess.debug_vis import (  # noqa: E402
    DEFAULT_DEBUG_VIS_COUNT,
    DEFAULT_DEBUG_VIS_SEED,
    generate_canonical_debug_vis,
    generate_exhaustive_debug_vis,
    generate_teacher_dataset_debug_vis,
)


DEFAULT_BOOTSTRAP_ROOT = REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap"
DEFAULT_TEACHER_ROOT = DEFAULT_BOOTSTRAP_ROOT / "teacher_datasets"
DEFAULT_EXHAUSTIVE_ROOT = DEFAULT_BOOTSTRAP_ROOT / "exhaustive_od"


def _log(message: str) -> None:
    print(f"[debug-vis] {message}", file=sys.stderr, flush=True)


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


def _run_canonical(*, bootstrap_root: Path, count: int, seed: int) -> dict[str, Any]:
    image_list_manifest_path = bootstrap_root / "meta" / "bootstrap_image_list.jsonl"
    canonical_root = bootstrap_root / "canonical"
    _log(f"canonical bootstrap_root={bootstrap_root} count={count} seed={seed}")
    outputs = generate_canonical_debug_vis(
        image_list_manifest_path=image_list_manifest_path,
        canonical_root=canonical_root,
        debug_vis_count=count,
        debug_vis_seed=seed,
        log_fn=_log,
    )
    return {
        dataset_name: {
            "debug_vis_dir": str(payload["debug_vis_dir"]),
            "debug_vis_manifest_path": str(payload["debug_vis_manifest"]),
            "selection_count": int(payload["selection_count"]),
        }
        for dataset_name, payload in outputs.items()
    }


def _run_teacher(*, teacher_root: Path, teacher_names: list[str], count: int, seed: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for teacher_name in teacher_names:
        dataset_root = teacher_root / teacher_name
        manifest_path = dataset_root / "meta" / "teacher_dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"teacher dataset manifest not found: {manifest_path}")
        manifest = _load_json(manifest_path)
        class_names = [str(item) for item in manifest.get("class_names") or []]
        sample_rows = [dict(item) for item in manifest.get("samples") or []]
        _log(f"teacher={teacher_name} dataset_root={dataset_root} count={count} seed={seed}")
        result = generate_teacher_dataset_debug_vis(
            dataset_root=dataset_root,
            teacher_name=teacher_name,
            class_names=class_names,
            manifest_rows=sample_rows,
            debug_vis_count=count,
            debug_vis_seed=seed,
            log_fn=_log,
        )
        outputs[teacher_name] = {
            "dataset_root": str(dataset_root),
            "debug_vis_dir": str(result["debug_vis_dir"]),
            "debug_vis_manifest_path": str(result["debug_vis_manifest"]),
            "selection_count": int(result["selection_count"]),
        }
    return outputs


def _run_exhaustive(*, exhaustive_root: Path, run_name: str | None, count: int, seed: int) -> dict[str, Any]:
    run_root = _resolve_exhaustive_run_root(exhaustive_root, run_name)
    manifest_path = run_root / "meta" / "materialization_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"materialization manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    sample_rows = [dict(item) for item in manifest.get("samples") or []]
    _log(f"exhaustive run_root={run_root} count={count} seed={seed}")
    result = generate_exhaustive_debug_vis(
        dataset_root=run_root,
        manifest_rows=sample_rows,
        debug_vis_count=count,
        debug_vis_seed=seed,
        log_fn=_log,
    )
    return {
        "run_root": str(run_root),
        "run_id": run_root.name,
        "debug_vis_dir": str(result["debug_vis_dir"]),
        "debug_vis_manifest_path": str(result["debug_vis_manifest"]),
        "selection_count": int(result["selection_count"]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate flat debug overlay images from existing OD bootstrap artifacts.")
    subparsers = parser.add_subparsers(dest="target", required=True)

    canonical_parser = subparsers.add_parser("canonical", help="Generate canonical debug-vis only.")
    canonical_parser.add_argument("--bootstrap-root", type=Path, default=DEFAULT_BOOTSTRAP_ROOT)
    canonical_parser.add_argument("--count", type=int, default=DEFAULT_DEBUG_VIS_COUNT)
    canonical_parser.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED)

    teacher_parser = subparsers.add_parser("teacher", help="Generate teacher-dataset debug-vis only.")
    teacher_parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    teacher_parser.add_argument("--teacher", action="append", default=None, help="Teacher name to render. Repeatable. Default: all.")
    teacher_parser.add_argument("--count", type=int, default=DEFAULT_DEBUG_VIS_COUNT)
    teacher_parser.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED)

    exhaustive_parser = subparsers.add_parser("exhaustive", help="Generate exhaustive OD debug-vis for one sweep run.")
    exhaustive_parser.add_argument("--exhaustive-root", type=Path, default=DEFAULT_EXHAUSTIVE_ROOT)
    exhaustive_parser.add_argument("--run", type=str, default=None, help="Sweep run directory name. Default: latest.")
    exhaustive_parser.add_argument("--count", type=int, default=DEFAULT_DEBUG_VIS_COUNT)
    exhaustive_parser.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED)

    all_parser = subparsers.add_parser("all", help="Generate canonical, teacher-dataset, and latest exhaustive debug-vis.")
    all_parser.add_argument("--bootstrap-root", type=Path, default=DEFAULT_BOOTSTRAP_ROOT)
    all_parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    all_parser.add_argument("--teacher", action="append", default=None, help="Teacher name to render. Repeatable. Default: all.")
    all_parser.add_argument("--exhaustive-root", type=Path, default=DEFAULT_EXHAUSTIVE_ROOT)
    all_parser.add_argument("--run", type=str, default=None, help="Sweep run directory name. Default: latest.")
    all_parser.add_argument("--count", type=int, default=DEFAULT_DEBUG_VIS_COUNT)
    all_parser.add_argument("--seed", type=int, default=DEFAULT_DEBUG_VIS_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if int(args.count) < 0:
        raise SystemExit("--count must be >= 0")

    if args.target == "canonical":
        result = {
            "canonical": _run_canonical(
                bootstrap_root=Path(args.bootstrap_root).resolve(),
                count=int(args.count),
                seed=int(args.seed),
            )
        }
    elif args.target == "teacher":
        teacher_root = Path(args.teacher_root).resolve()
        result = {
            "teachers": _run_teacher(
                teacher_root=teacher_root,
                teacher_names=_resolve_teacher_names(teacher_root, args.teacher),
                count=int(args.count),
                seed=int(args.seed),
            )
        }
    elif args.target == "exhaustive":
        result = {
            "exhaustive": _run_exhaustive(
                exhaustive_root=Path(args.exhaustive_root).resolve(),
                run_name=args.run,
                count=int(args.count),
                seed=int(args.seed),
            )
        }
    else:
        bootstrap_root = Path(args.bootstrap_root).resolve()
        teacher_root = Path(args.teacher_root).resolve()
        exhaustive_root = Path(args.exhaustive_root).resolve()
        result = {
            "canonical": _run_canonical(
                bootstrap_root=bootstrap_root,
                count=int(args.count),
                seed=int(args.seed),
            ),
            "teachers": _run_teacher(
                teacher_root=teacher_root,
                teacher_names=_resolve_teacher_names(teacher_root, args.teacher),
                count=int(args.count),
                seed=int(args.seed),
            ),
        }
        if exhaustive_root.is_dir():
            try:
                result["exhaustive"] = _run_exhaustive(
                    exhaustive_root=exhaustive_root,
                    run_name=args.run,
                    count=int(args.count),
                    seed=int(args.seed),
                )
            except FileNotFoundError:
                _log(f"skip exhaustive debug-vis: no run under {exhaustive_root}")

    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
