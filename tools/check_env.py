from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import site
import subprocess
import sys
from typing import Any, Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(REPO_ROOT))

from common.user_config import (
    USER_OD_BOOTSTRAP_HYPERPARAMETERS_CONFIG_PATH,
    USER_PATHS_CONFIG_PATH,
    USER_PV26_TRAIN_HYPERPARAMETERS_CONFIG_PATH,
    load_user_paths_config,
    nested_get,
    resolve_repo_path,
)
from tools.od_bootstrap.presets import (
    build_calibration_preset,
    build_default_source_preset,
    build_final_dataset_preset,
    build_sweep_preset,
    build_teacher_dataset_preset,
    build_teacher_eval_preset,
)

TEACHER_NAMES = ("mobility", "signal", "obstacle")
STAGE_ICON = {
    "OK": "✅",
    "WARN": "⚠️",
    "TODO": "🟡",
    "FAIL": "❌",
}


@dataclass(frozen=True)
class PipelinePaths:
    repo_root: Path
    raw_bdd_root: Path
    raw_aihub_root: Path
    bootstrap_root: Path
    teacher_dataset_root: Path
    teacher_train_root: Path
    teacher_eval_root: Path
    calibration_root: Path
    exhaustive_run_root: Path
    exhaustive_dataset_root: Path
    final_dataset_root: Path
    pv26_run_root: Path
    user_paths_config_path: Path
    od_hyperparameters_config_path: Path
    pv26_hyperparameters_config_path: Path


@dataclass(frozen=True)
class StageRow:
    stage: str
    success_condition: str
    current_state: str
    verdict: str


@dataclass(frozen=True)
class WorkspaceSnapshot:
    paths: PipelinePaths
    rows: tuple[StageRow, ...]
    flags: dict[str, bool]
    recommendation: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    command_display: str
    argv: tuple[str, ...]
    output_hint: str


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _check_torchvision_nms() -> dict[str, Any]:
    result = {
        "importable": False,
        "callable": False,
        "error": None,
        "fallback_available": True,
    }
    try:
        import torch
        from torchvision.ops import nms

        result["importable"] = True
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        keep = nms(boxes, scores, 0.5)
        result["callable"] = bool(keep.numel() >= 1)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _check_yolo26(check_runtime: bool) -> dict[str, Any]:
    result = {
        "importable": False,
        "supported": False,
        "version": None,
        "runtime_load_ok": None,
        "error": None,
    }
    try:
        from model.net.trunk import (
            ULTRALYTICS_VERSION,
            build_yolo26n_trunk,
            ensure_yolo26_support,
        )

        result["importable"] = True
        result["version"] = ULTRALYTICS_VERSION
        ensure_yolo26_support()
        result["supported"] = True
        if check_runtime:
            adapter = build_yolo26n_trunk()
            result["runtime_load_ok"] = bool(adapter.detect_head is not None)
    except Exception as exc:
        result["error"] = str(exc)
        if result["runtime_load_ok"] is None:
            result["runtime_load_ok"] = False
    return result


def check_env(*, check_yolo_runtime: bool = False) -> dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "versions": {
            "torch": _module_version("torch"),
            "torchvision": _module_version("torchvision"),
            "ultralytics": _module_version("ultralytics"),
            "numpy": _module_version("numpy"),
            "scipy": _module_version("scipy"),
            "PIL": _module_version("PIL"),
        },
        "checks": {
            "torchvision_nms": _check_torchvision_nms(),
            "yolo26": _check_yolo26(check_yolo_runtime),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PV26 runtime environment portability prerequisites.")
    parser.add_argument("--check-yolo-runtime", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", help="Always print JSON instead of launching the interactive TUI.")
    return parser


def _should_run_interactive(
    args: argparse.Namespace,
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> bool:
    in_tty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
    out_tty = sys.stdout.isatty() if stdout_isatty is None else stdout_isatty
    return bool(in_tty and out_tty and not args.strict and not args.json)


def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _json_load_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _json_load(path)


def _manifest_header(path: Path, *, array_key: str = "samples", max_bytes: int = 4_000_000) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    marker = f'"{array_key}": ['.encode("ascii")
    buffer = bytearray()
    with path.open("rb") as handle:
        while len(buffer) < max_bytes:
            chunk = handle.read(min(65_536, max_bytes - len(buffer)))
            if not chunk:
                break
            buffer.extend(chunk)
            marker_index = bytes(buffer).find(marker)
            if marker_index == -1:
                continue
            prefix = bytes(buffer[:marker_index]).decode("ascii")
            trimmed = prefix.rstrip()
            if trimmed.endswith(","):
                trimmed = trimmed[:-1]
            return json.loads(trimmed + "\n}\n")
    return None


def _compact_or_manifest(
    *,
    summary_path: Path,
    manifest_path: Path,
    array_key: str = "samples",
) -> dict[str, Any] | None:
    summary = _json_load_if_exists(summary_path)
    if summary is not None:
        return summary
    header = _manifest_header(manifest_path, array_key=array_key)
    if header is not None:
        return header
    return _json_load_if_exists(manifest_path)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sum_inventory_splits(payload: dict[str, Any]) -> int:
    total = 0
    for split_payload in payload.values():
        if not isinstance(split_payload, dict):
            continue
        if "json_files" in split_payload:
            total += _safe_int(split_payload.get("json_files"))
            continue
        if "raw_images" in split_payload:
            total += _safe_int(split_payload.get("raw_images"))
            continue
        if "images" in split_payload:
            total += _safe_int(split_payload.get("images"))
    return total


def _resolve_pipeline_paths() -> PipelinePaths:
    source_preset = build_default_source_preset()
    teacher_dataset_preset = build_teacher_dataset_preset()
    calibration_preset = build_calibration_preset()
    sweep_preset = build_sweep_preset()
    final_preset = build_final_dataset_preset()
    teacher_eval_preset = build_teacher_eval_preset("mobility")
    user_paths = load_user_paths_config()
    pv26_run_root = resolve_repo_path(
        nested_get(user_paths, "pv26_train", "run_root"),
        repo_root=REPO_ROOT,
    ) or (REPO_ROOT / "runs" / "pv26_exhaustive_od_lane_train").resolve()
    return PipelinePaths(
        repo_root=REPO_ROOT,
        raw_bdd_root=source_preset.roots.bdd_root.resolve(),
        raw_aihub_root=source_preset.roots.aihub_root.resolve(),
        bootstrap_root=source_preset.output_root.resolve(),
        teacher_dataset_root=teacher_dataset_preset.output_root.resolve(),
        teacher_train_root=Path(calibration_preset.teachers[0].checkpoint_path).resolve().parents[2],
        teacher_eval_root=Path(teacher_eval_preset.run.output_root).resolve(),
        calibration_root=Path(calibration_preset.run.output_root).resolve(),
        exhaustive_run_root=Path(sweep_preset.run.output_root).resolve(),
        exhaustive_dataset_root=Path(sweep_preset.materialization.output_root).resolve(),
        final_dataset_root=Path(final_preset.output_root).resolve(),
        pv26_run_root=pv26_run_root.resolve(),
        user_paths_config_path=USER_PATHS_CONFIG_PATH.resolve(),
        od_hyperparameters_config_path=USER_OD_BOOTSTRAP_HYPERPARAMETERS_CONFIG_PATH.resolve(),
        pv26_hyperparameters_config_path=USER_PV26_TRAIN_HYPERPARAMETERS_CONFIG_PATH.resolve(),
    )


def _latest_child_with_meta(root: Path, relative_paths: tuple[str, ...]) -> tuple[Path | None, Path | None]:
    if not root.is_dir():
        return None, None
    candidates = sorted((child for child in root.iterdir() if child.is_dir()), key=lambda item: item.name)
    for child in reversed(candidates):
        for relative in relative_paths:
            candidate = child / relative
            if candidate.is_file():
                return child, candidate
    return None, None


def _teacher_dataset_summary(dataset_root: Path) -> dict[str, Any] | None:
    meta_root = dataset_root / "meta"
    return _compact_or_manifest(
        summary_path=meta_root / "teacher_dataset_summary.json",
        manifest_path=meta_root / "teacher_dataset_manifest.json",
    )


def _exhaustive_summary(dataset_root: Path) -> dict[str, Any] | None:
    meta_root = dataset_root / "meta"
    return _compact_or_manifest(
        summary_path=meta_root / "materialization_summary.json",
        manifest_path=meta_root / "materialization_manifest.json",
    )


def _final_dataset_summary(dataset_root: Path) -> dict[str, Any] | None:
    meta_root = dataset_root / "meta"
    return _compact_or_manifest(
        summary_path=meta_root / "final_dataset_summary.json",
        manifest_path=meta_root / "final_dataset_manifest.json",
    )


def _build_runtime_row(report: dict[str, Any]) -> tuple[StageRow, dict[str, bool]]:
    versions = report["versions"]
    checks = report["checks"]
    nms_ok = checks["torchvision_nms"]["callable"] is True
    core_ready = versions["torch"] is not None and versions["ultralytics"] is not None and nms_ok
    yolo_ok = checks["yolo26"]["importable"] is True and checks["yolo26"]["runtime_load_ok"] is True
    state = (
        f"torch={versions['torch'] or 'missing'} | "
        f"torchvision={versions['torchvision'] or 'missing'} | "
        f"ultralytics={versions['ultralytics'] or 'missing'} | "
        f"NMS={'OK' if nms_ok else 'X'} | "
        f"YOLO26 runtime={'OK' if yolo_ok else 'X'}"
    )
    verdict = "OK" if core_ready and yolo_ok else ("FAIL" if not core_ready else "WARN")
    return (
        StageRow(
            stage="환경 런타임",
            success_condition="torch/ultralytics, torchvision NMS, YOLO26 runtime이 현재 환경에서 모두 동작",
            current_state=state,
            verdict=verdict,
        ),
        {
            "runtime_core": core_ready,
            "pv26_runtime": core_ready and yolo_ok,
        },
    )


def _build_raw_roots_row(paths: PipelinePaths, counts: dict[str, int]) -> tuple[StageRow, dict[str, bool]]:
    bdd_exists = paths.raw_bdd_root.exists()
    aihub_exists = paths.raw_aihub_root.exists()
    verdict = "OK" if bdd_exists and aihub_exists else ("WARN" if bdd_exists or aihub_exists else "FAIL")
    state = (
        f"BDD root={'O' if bdd_exists else 'X'} | "
        f"AIHUB root={'O' if aihub_exists else 'X'} | "
        f"raw BDD={counts.get('bdd_raw', 0)} | "
        f"lane={counts.get('lane_raw', 0)} | "
        f"traffic={counts.get('traffic_raw', 0)} | "
        f"obstacle={counts.get('obstacle_raw', 0)}"
    )
    return (
        StageRow(
            stage="원본 데이터",
            success_condition="현재 config 기준 BDD100K / AIHUB raw root가 존재",
            current_state=state,
            verdict=verdict,
        ),
        {
            "raw_roots": bdd_exists and aihub_exists,
        },
    )


def _collect_source_counts(paths: PipelinePaths) -> dict[str, int]:
    counts = {
        "bdd_raw": 0,
        "bdd_processed": 0,
        "lane_raw": 0,
        "lane_processed": 0,
        "traffic_raw": 0,
        "traffic_processed": 0,
        "obstacle_raw": 0,
        "obstacle_processed": 0,
    }
    bdd_report = _json_load_if_exists(paths.bootstrap_root / "canonical" / "bdd100k_det_100k" / "meta" / "conversion_report.json")
    if bdd_report is not None:
        dataset = bdd_report.get("dataset", {})
        counts["bdd_processed"] = _safe_int(dataset.get("processed_samples"))
        source_snapshot = bdd_report.get("source_inventory_snapshot", {}).get("dataset", {})
        local_inventory = source_snapshot.get("local_inventory", {})
        counts["bdd_raw"] = _sum_inventory_splits(local_inventory.get("splits", {}))

    aihub_report = _json_load_if_exists(paths.bootstrap_root / "canonical" / "aihub_standardized" / "meta" / "conversion_report.json")
    if aihub_report is not None:
        datasets = aihub_report.get("datasets", [])
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            dataset_key = str(dataset.get("dataset_key"))
            if dataset_key == "aihub_lane_seoul":
                counts["lane_processed"] = _safe_int(dataset.get("processed_samples"))
            elif dataset_key == "aihub_traffic_seoul":
                counts["traffic_processed"] = _safe_int(dataset.get("processed_samples"))
            elif dataset_key == "aihub_obstacle_seoul":
                counts["obstacle_processed"] = _safe_int(dataset.get("processed_samples"))

    aihub_inventory = _json_load_if_exists(paths.bootstrap_root / "canonical" / "aihub_standardized" / "meta" / "source_inventory.json")
    if aihub_inventory is not None:
        for dataset in aihub_inventory.get("datasets", []):
            if not isinstance(dataset, dict):
                continue
            dataset_key = str(dataset.get("dataset_key"))
            local_inventory = dataset.get("local_inventory", {})
            total = _sum_inventory_splits(local_inventory.get("splits", {}))
            if dataset_key == "aihub_lane_seoul":
                counts["lane_raw"] = total
            elif dataset_key == "aihub_traffic_seoul":
                counts["traffic_raw"] = total
            elif dataset_key == "aihub_obstacle_seoul":
                counts["obstacle_raw"] = total

    if counts["bdd_raw"] == 0:
        bdd_inventory = _json_load_if_exists(paths.bootstrap_root / "canonical" / "bdd100k_det_100k" / "meta" / "source_inventory.json")
        if bdd_inventory is not None:
            local_inventory = bdd_inventory.get("dataset", {}).get("local_inventory", {})
            counts["bdd_raw"] = _sum_inventory_splits(local_inventory.get("splits", {}))

    return counts


def _build_source_prep_row(paths: PipelinePaths, counts: dict[str, int]) -> tuple[StageRow, dict[str, bool]]:
    manifest_path = paths.bootstrap_root / "meta" / "source_prep_manifest.json"
    image_list_path = paths.bootstrap_root / "meta" / "bootstrap_image_list.jsonl"
    bdd_report_ok = (paths.bootstrap_root / "canonical" / "bdd100k_det_100k" / "meta" / "conversion_report.json").is_file()
    aihub_report_ok = (paths.bootstrap_root / "canonical" / "aihub_standardized" / "meta" / "conversion_report.json").is_file()
    bootstrap_total = counts["bdd_processed"] + counts["traffic_processed"] + counts["obstacle_processed"]
    bootstrap_raw = counts["bdd_raw"] + counts["traffic_raw"] + counts["obstacle_raw"]
    lane_text = f"lane {counts['lane_processed']}/{counts['lane_raw']}" if counts["lane_raw"] else f"lane {counts['lane_processed']}"
    current_state = (
        f"bootstrap {bootstrap_total}/{bootstrap_raw or bootstrap_total} | "
        f"{lane_text} | "
        f"image_list={'O' if image_list_path.is_file() else 'X'}"
    )
    has_any = manifest_path.is_file() or image_list_path.is_file() or bdd_report_ok or aihub_report_ok
    ready = manifest_path.is_file() and image_list_path.is_file() and bdd_report_ok and aihub_report_ok and bootstrap_total > 0
    verdict = "OK" if ready else ("WARN" if has_any else "TODO")
    return (
        StageRow(
            stage="소스 준비 / canonical",
            success_condition="source_prep manifest, canonical report, bootstrap image list가 모두 준비",
            current_state=current_state,
            verdict=verdict,
        ),
        {
            "source_prep": ready,
            "lane_canonical": counts["lane_processed"] > 0,
            "bootstrap_image_count": bootstrap_total,
        },
    )


def _teacher_stage_state(prefix: str, root: Path, *, summary_name: str | None = None) -> tuple[str, bool, dict[str, bool]]:
    ready_map: dict[str, bool] = {}
    parts: list[str] = []
    done = 0
    for teacher_name in TEACHER_NAMES:
        ready = False
        if summary_name == "dataset":
            summary = _teacher_dataset_summary(root / teacher_name)
            if summary is not None and _safe_int(summary.get("sample_count")) > 0:
                ready = True
                parts.append(f"{teacher_name} {_safe_int(summary.get('sample_count'))}")
        elif summary_name == "train":
            checkpoint_path = root / teacher_name / "weights" / "best.pt"
            run_summary_path = root / teacher_name / "run_summary.json"
            ready = checkpoint_path.is_file() and run_summary_path.is_file()
            if ready:
                parts.append(teacher_name)
        elif summary_name == "eval":
            eval_summary = _json_load_if_exists(root / teacher_name / "checkpoint_eval_summary.json")
            ready = eval_summary is not None
            if ready:
                prediction_summary = eval_summary.get("prediction_summary", {})
                parts.append(f"{teacher_name} {_safe_int(prediction_summary.get('sample_count'))}")
        ready_map[f"{prefix}.{teacher_name}"] = ready
        done += int(ready)
    verdict = "OK" if done == len(TEACHER_NAMES) else ("WARN" if done else "TODO")
    detail = " | ".join(parts) if parts else "없음"
    return f"{done}/{len(TEACHER_NAMES)} 완료 | {detail}", verdict == "OK", ready_map


def _build_teacher_rows(paths: PipelinePaths) -> tuple[tuple[StageRow, StageRow, StageRow], dict[str, bool]]:
    dataset_state, dataset_ready, dataset_map = _teacher_stage_state(
        "teacher_dataset",
        paths.teacher_dataset_root,
        summary_name="dataset",
    )
    train_state, train_ready, train_map = _teacher_stage_state(
        "teacher_train",
        paths.teacher_train_root,
        summary_name="train",
    )
    eval_state, eval_ready, eval_map = _teacher_stage_state(
        "teacher_eval",
        paths.teacher_eval_root,
        summary_name="eval",
    )
    rows = (
        StageRow(
            stage="Teacher dataset",
            success_condition="mobility / signal / obstacle dataset 3종이 모두 생성",
            current_state=dataset_state,
            verdict="OK" if dataset_ready else ("WARN" if any(dataset_map.values()) else "TODO"),
        ),
        StageRow(
            stage="Teacher 학습",
            success_condition="teacher별 `weights/best.pt`와 `run_summary.json`이 모두 존재",
            current_state=train_state,
            verdict="OK" if train_ready else ("WARN" if any(train_map.values()) else "TODO"),
        ),
        StageRow(
            stage="Teacher 평가",
            success_condition="teacher별 `checkpoint_eval_summary.json`이 모두 존재",
            current_state=eval_state,
            verdict="OK" if eval_ready else ("WARN" if any(eval_map.values()) else "TODO"),
        ),
    )
    flags = {"teacher_datasets": dataset_ready, "teacher_trains": train_ready, "teacher_evals": eval_ready}
    flags.update(dataset_map)
    flags.update(train_map)
    flags.update(eval_map)
    return rows, flags


def _build_calibration_row(paths: PipelinePaths) -> tuple[StageRow, dict[str, bool]]:
    report_path = paths.calibration_root / "calibration_report.json"
    policy_path = paths.calibration_root / "class_policy.yaml"
    hard_negative_path = paths.calibration_root / "hard_negative_manifest.json"
    report = _json_load_if_exists(report_path)
    class_ok = 0
    class_total = 0
    teacher_parts: list[str] = []
    if report is not None:
        class_payload = report.get("classes", {})
        if isinstance(class_payload, dict):
            class_total = len(class_payload)
            class_ok = sum(1 for item in class_payload.values() if isinstance(item, dict) and item.get("meets_precision_floor") is True)
        for teacher in report.get("teachers", []):
            if not isinstance(teacher, dict):
                continue
            teacher_parts.append(f"{teacher.get('teacher_name')} {_safe_int(teacher.get('sample_count'))}")
    current_state = (
        f"class {class_ok}/{class_total or 7} precision OK | "
        f"policy={'O' if policy_path.is_file() else 'X'} | "
        f"hard-negative={'O' if hard_negative_path.is_file() else 'X'}"
    )
    if teacher_parts:
        current_state += f" | {' / '.join(teacher_parts)}"
    ready = report is not None and policy_path.is_file() and hard_negative_path.is_file()
    has_any = report is not None or policy_path.is_file() or hard_negative_path.is_file()
    verdict = "OK" if ready else ("WARN" if has_any else "TODO")
    return (
        StageRow(
            stage="Calibration",
            success_condition="class policy, calibration report, hard-negative manifest가 모두 준비",
            current_state=current_state,
            verdict=verdict,
        ),
        {
            "calibration": ready,
        },
    )


def _build_exhaustive_row(paths: PipelinePaths) -> tuple[StageRow, dict[str, bool]]:
    dataset_dir, _ = _latest_child_with_meta(
        paths.exhaustive_dataset_root,
        ("meta/materialization_summary.json", "meta/materialization_manifest.json"),
    )
    summary = _exhaustive_summary(dataset_dir) if dataset_dir is not None else None
    if dataset_dir is not None and summary is not None:
        run_id = str(summary.get("run_id") or dataset_dir.name)
        sample_count = _safe_int(summary.get("sample_count"))
        current_state = f"latest={run_id} | samples={sample_count}"
        verdict = "OK"
        ready = True
    else:
        has_any = paths.exhaustive_dataset_root.is_dir() and any(paths.exhaustive_dataset_root.iterdir())
        current_state = "없음" if not has_any else "run 디렉터리는 있지만 manifest를 못 찾음"
        verdict = "WARN" if has_any else "TODO"
        ready = False
    return (
        StageRow(
            stage="Exhaustive OD",
            success_condition="최신 run의 `materialization_manifest` 또는 summary가 존재",
            current_state=current_state,
            verdict=verdict,
        ),
        {
            "exhaustive": ready,
        },
    )


def _build_final_dataset_row(paths: PipelinePaths) -> tuple[StageRow, dict[str, bool]]:
    summary = _final_dataset_summary(paths.final_dataset_root)
    manifest_path = paths.final_dataset_root / "meta" / "final_dataset_manifest.json"
    if summary is not None:
        sample_count = _safe_int(summary.get("sample_count"))
        dataset_counts = summary.get("dataset_counts", {})
        state = f"samples={sample_count}"
        if isinstance(dataset_counts, dict) and dataset_counts:
            state += " | " + ", ".join(f"{key}={_safe_int(value)}" for key, value in sorted(dataset_counts.items()))
        verdict = "OK"
        ready = True
    else:
        has_any = manifest_path.exists()
        if not has_any and paths.final_dataset_root.is_dir():
            has_any = any(paths.final_dataset_root.iterdir())
        state = "없음" if not has_any else "meta는 일부 있지만 summary를 못 찾음"
        verdict = "WARN" if has_any else "TODO"
        ready = False
    return (
        StageRow(
            stage="최종 병합 데이터셋",
            success_condition="`meta/final_dataset_manifest.json` 또는 summary가 존재",
            current_state=state,
            verdict=verdict,
        ),
        {
            "final_dataset": ready,
        },
    )


def _build_pv26_row(paths: PipelinePaths) -> tuple[StageRow, dict[str, bool]]:
    run_dir, summary_path = _latest_child_with_meta(paths.pv26_run_root, ("summary.json", "meta_manifest.json"))
    summary = _json_load_if_exists(summary_path) if summary_path is not None else None
    if run_dir is not None and summary is not None:
        status = str(summary.get("status") or "unknown")
        completed_phases = _safe_int(summary.get("completed_phases"))
        total_phases = _safe_int(summary.get("total_phases") or len(summary.get("phases", [])))
        train_defaults = summary.get("train_defaults", {})
        backbone_variant = None
        if isinstance(train_defaults, dict):
            backbone_variant = train_defaults.get("backbone_variant")
        latest_phase_stage = summary.get("latest_phase_stage")
        latest_selection = summary.get("latest_selection_metric_path")
        latest_backbone_variant = summary.get("latest_backbone_variant") or backbone_variant
        current_state_parts = [
            f"latest={run_dir.name}",
            f"status={status}",
            f"phases={completed_phases}/{total_phases}",
        ]
        if latest_phase_stage:
            current_state_parts.append(f"stage={latest_phase_stage}")
        if latest_backbone_variant:
            current_state_parts.append(f"backbone={latest_backbone_variant}")
        if latest_selection:
            current_state_parts.append(f"selection={latest_selection}")
        current_state = " | ".join(current_state_parts)
        ready = status == "completed"
        verdict = "OK" if ready else "WARN"
    else:
        ready = False
        verdict = "TODO"
        current_state = "없음"
    return (
        StageRow(
            stage="PV26 학습 run",
            success_condition="최신 meta train run의 `summary.json` 또는 `meta_manifest.json`이 존재",
            current_state=current_state,
            verdict=verdict,
        ),
        {
            "pv26_train": ready,
        },
    )


def _recommendation(flags: dict[str, bool]) -> str:
    if not flags.get("runtime_core", False):
        return "환경 런타임부터 정리하세요. 학습/평가 계열 메뉴는 잠깁니다."
    if not flags.get("raw_roots", False):
        return "raw dataset root가 안 맞습니다. H를 눌러 config 위치를 확인하세요."
    if not flags.get("source_prep", False):
        return "1번 source prep부터 시작하는 편이 안전합니다."
    if not flags.get("teacher_dataset.mobility", False):
        return "2번으로 teacher dataset을 먼저 맞추세요."
    if not flags.get("teacher_train.mobility", False):
        return "3번 mobility teacher 학습이 다음 순서입니다."
    if not flags.get("teacher_train.signal", False):
        return "4번 signal teacher 학습이 다음 순서입니다."
    if not flags.get("teacher_train.obstacle", False):
        return "5번 obstacle teacher 학습이 다음 순서입니다."
    if not flags.get("teacher_eval.mobility", False):
        return "6번 mobility teacher 평가를 돌려 상태를 확인하세요."
    if not flags.get("teacher_eval.signal", False):
        return "7번 signal teacher 평가를 돌려 상태를 확인하세요."
    if not flags.get("teacher_eval.obstacle", False):
        return "8번 obstacle teacher 평가를 돌려 상태를 확인하세요."
    if not flags.get("calibration", False):
        return "9번 calibration으로 class policy를 먼저 고정하세요."
    if not flags.get("exhaustive", False):
        return "A로 exhaustive OD를 만들 차례입니다."
    if not flags.get("final_dataset", False):
        return "B로 최종 병합 데이터셋을 만드세요."
    if not flags.get("pv26_train", False):
        return "C로 PV26 기본 학습을 돌리면 됩니다."
    return "상태는 좋아 보입니다. 필요한 메뉴만 골라 실행하면 됩니다."


def scan_workspace_status(report: dict[str, Any], *, paths: PipelinePaths | None = None) -> WorkspaceSnapshot:
    resolved_paths = paths or _resolve_pipeline_paths()
    rows: list[StageRow] = []
    flags: dict[str, bool] = {}
    notes: list[str] = []

    runtime_row, runtime_flags = _build_runtime_row(report)
    rows.append(runtime_row)
    flags.update(runtime_flags)

    source_counts = _collect_source_counts(resolved_paths)
    raw_row, raw_flags = _build_raw_roots_row(resolved_paths, source_counts)
    rows.append(raw_row)
    flags.update(raw_flags)

    source_prep_row, source_prep_flags = _build_source_prep_row(resolved_paths, source_counts)
    rows.append(source_prep_row)
    flags.update(source_prep_flags)

    teacher_rows, teacher_flags = _build_teacher_rows(resolved_paths)
    rows.extend(teacher_rows)
    flags.update(teacher_flags)

    calibration_row, calibration_flags = _build_calibration_row(resolved_paths)
    rows.append(calibration_row)
    flags.update(calibration_flags)

    exhaustive_row, exhaustive_flags = _build_exhaustive_row(resolved_paths)
    rows.append(exhaustive_row)
    flags.update(exhaustive_flags)

    final_dataset_row, final_dataset_flags = _build_final_dataset_row(resolved_paths)
    rows.append(final_dataset_row)
    flags.update(final_dataset_flags)

    pv26_row, pv26_flags = _build_pv26_row(resolved_paths)
    rows.append(pv26_row)
    flags.update(pv26_flags)

    if flags.get("source_prep", False):
        notes.append("manifest 안 절대경로가 다른 머신 경로여도, 현재 config 기준 실제 파일을 우선 판정합니다.")
    if not flags.get("calibration", False):
        notes.append("calibration이 없어도 exhaustive OD는 fallback class policy로 실행할 수 있습니다.")

    return WorkspaceSnapshot(
        paths=resolved_paths,
        rows=tuple(rows),
        flags=flags,
        recommendation=_recommendation(flags),
        notes=tuple(notes),
    )


def _action_catalog(paths: PipelinePaths) -> tuple[ActionSpec, ...]:
    python_exe = sys.executable
    return (
        ActionSpec("1", "OD bootstrap 소스 준비", "python -m tools.od_bootstrap prepare-sources", (python_exe, "-m", "tools.od_bootstrap", "prepare-sources"), str(paths.bootstrap_root)),
        ActionSpec("2", "Teacher dataset 생성", "python -m tools.od_bootstrap build-teacher-datasets", (python_exe, "-m", "tools.od_bootstrap", "build-teacher-datasets"), str(paths.teacher_dataset_root)),
        ActionSpec("3", "Mobility teacher 학습", "python -m tools.od_bootstrap train --teacher mobility", (python_exe, "-m", "tools.od_bootstrap", "train", "--teacher", "mobility"), str(paths.teacher_train_root / "mobility")),
        ActionSpec("4", "Signal teacher 학습", "python -m tools.od_bootstrap train --teacher signal", (python_exe, "-m", "tools.od_bootstrap", "train", "--teacher", "signal"), str(paths.teacher_train_root / "signal")),
        ActionSpec("5", "Obstacle teacher 학습", "python -m tools.od_bootstrap train --teacher obstacle", (python_exe, "-m", "tools.od_bootstrap", "train", "--teacher", "obstacle"), str(paths.teacher_train_root / "obstacle")),
        ActionSpec("6", "Mobility teacher 평가", "python -m tools.od_bootstrap eval --teacher mobility", (python_exe, "-m", "tools.od_bootstrap", "eval", "--teacher", "mobility"), str(paths.teacher_eval_root / "mobility")),
        ActionSpec("7", "Signal teacher 평가", "python -m tools.od_bootstrap eval --teacher signal", (python_exe, "-m", "tools.od_bootstrap", "eval", "--teacher", "signal"), str(paths.teacher_eval_root / "signal")),
        ActionSpec("8", "Obstacle teacher 평가", "python -m tools.od_bootstrap eval --teacher obstacle", (python_exe, "-m", "tools.od_bootstrap", "eval", "--teacher", "obstacle"), str(paths.teacher_eval_root / "obstacle")),
        ActionSpec("9", "Calibration", "python -m tools.od_bootstrap calibrate", (python_exe, "-m", "tools.od_bootstrap", "calibrate"), str(paths.calibration_root)),
        ActionSpec("A", "Exhaustive OD 생성", "python -m tools.od_bootstrap build-exhaustive-od", (python_exe, "-m", "tools.od_bootstrap", "build-exhaustive-od"), str(paths.exhaustive_dataset_root)),
        ActionSpec("B", "최종 병합 데이터셋 생성", "python -m tools.od_bootstrap build-final-dataset", (python_exe, "-m", "tools.od_bootstrap", "build-final-dataset"), str(paths.final_dataset_root)),
        ActionSpec("C", "PV26 기본 학습", "python3 tools/run_pv26_train.py --preset default", (python_exe, "tools/run_pv26_train.py", "--preset", "default"), str(paths.pv26_run_root)),
    )


def _action_blockers(action: ActionSpec, snapshot: WorkspaceSnapshot) -> list[str]:
    flags = snapshot.flags
    blockers: list[str] = []
    if action.key == "1":
        if not flags.get("raw_roots", False):
            blockers.append("raw dataset root가 config 기준으로 안 맞습니다.")
    elif action.key == "2":
        if not flags.get("source_prep", False):
            blockers.append("source prep이 아직 준비되지 않았습니다.")
    elif action.key in {"3", "4", "5"}:
        teacher_name = {"3": "mobility", "4": "signal", "5": "obstacle"}[action.key]
        if not flags.get("runtime_core", False):
            blockers.append("학습에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get(f"teacher_dataset.{teacher_name}", False):
            blockers.append(f"{teacher_name} teacher dataset이 아직 없습니다.")
    elif action.key in {"6", "7", "8"}:
        teacher_name = {"6": "mobility", "7": "signal", "8": "obstacle"}[action.key]
        if not flags.get("runtime_core", False):
            blockers.append("평가에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get(f"teacher_train.{teacher_name}", False):
            blockers.append(f"{teacher_name} teacher checkpoint가 아직 없습니다.")
    elif action.key == "9":
        if not flags.get("runtime_core", False):
            blockers.append("calibration에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        missing = [name for name in TEACHER_NAMES if not flags.get(f"teacher_train.{name}", False)]
        if missing:
            blockers.append(f"teacher checkpoint가 부족합니다: {', '.join(missing)}")
    elif action.key == "A":
        if not flags.get("runtime_core", False):
            blockers.append("exhaustive OD에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get("source_prep", False):
            blockers.append("source prep이 먼저 준비되어야 합니다.")
        missing = [name for name in TEACHER_NAMES if not flags.get(f"teacher_train.{name}", False)]
        if missing:
            blockers.append(f"teacher checkpoint가 부족합니다: {', '.join(missing)}")
    elif action.key == "B":
        if not flags.get("source_prep", False):
            blockers.append("lane canonical/source prep이 먼저 필요합니다.")
        if not flags.get("exhaustive", False):
            blockers.append("최신 exhaustive OD run이 아직 없습니다.")
    elif action.key == "C":
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 학습에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
        if not flags.get("final_dataset", False):
            blockers.append("최종 병합 데이터셋이 아직 없습니다.")
    return blockers


def _action_advisory(action: ActionSpec, snapshot: WorkspaceSnapshot) -> str | None:
    if action.key == "A" and not snapshot.flags.get("calibration", False):
        return "calibration이 없어서 fallback class policy로 진행될 수 있습니다."
    return None


def _render_dashboard(console: Console, snapshot: WorkspaceSnapshot, actions: tuple[ActionSpec, ...]) -> None:
    console.clear(home=True)
    console.print(
        Panel.fit(
            f"🧭 PV26 체크 허브\nrepo: {snapshot.paths.repo_root}\n📌 추천: {snapshot.recommendation}",
            border_style="cyan",
        )
    )

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("단계", style="bold")
    table.add_column("성공 조건")
    table.add_column("현재 상태")
    table.add_column("판정", justify="center")
    for row in snapshot.rows:
        table.add_row(row.stage, row.success_condition, row.current_state, f"{STAGE_ICON[row.verdict]} {row.verdict}")
    console.print(table)

    path_table = Table(box=box.SIMPLE, title="📂 현재 기준 경로")
    path_table.add_column("항목", style="bold")
    path_table.add_column("경로")
    path_table.add_row("user_paths", str(snapshot.paths.user_paths_config_path))
    path_table.add_row("bootstrap_root", str(snapshot.paths.bootstrap_root))
    path_table.add_row("final_dataset_root", str(snapshot.paths.final_dataset_root))
    path_table.add_row("pv26_run_root", str(snapshot.paths.pv26_run_root))
    console.print(path_table)

    if snapshot.notes:
        console.print(
            Panel(
                "\n".join(f"- {note}" for note in snapshot.notes),
                title="📝 메모",
                border_style="yellow",
            )
        )

    action_table = Table(box=box.SIMPLE, title="🎛️ 메뉴")
    action_table.add_column("키", style="bold cyan", justify="center")
    action_table.add_column("동작")
    action_table.add_column("명령")
    for action in actions:
        action_table.add_row(action.key, action.label, action.command_display)
    action_table.add_row("H", "Help", "간단 README / config 위치 안내")
    action_table.add_row("R", "Refresh", "상태 다시 스캔")
    action_table.add_row("Q", "Quit", "종료")
    console.print(action_table)


def _render_help(console: Console, snapshot: WorkspaceSnapshot) -> None:
    console.clear(home=True)
    console.print(
        Panel(
            "\n".join(
                [
                    "❓ 이 TUI는 설정 편집기가 아닙니다.",
                    f"- 경로를 바꾸려면: {snapshot.paths.user_paths_config_path}",
                    f"- bootstrap/calibration/exhaustive 숫자 파라미터를 바꾸려면: {snapshot.paths.od_hyperparameters_config_path}",
                    f"- PV26 학습 숫자 파라미터를 바꾸려면: {snapshot.paths.pv26_hyperparameters_config_path}",
                    f"- 전체 흐름 설명: {snapshot.paths.repo_root / 'README.md'}",
                    f"- bootstrap 전용 설명: {snapshot.paths.repo_root / 'tools' / 'od_bootstrap' / 'README.md'}",
                    "- 코드에서 빠른 조절 지점을 찾고 싶으면 `USER CONFIG`, `HYPERPARAMETERS`, `PHASE HYPERPARAMETERS`를 검색하세요.",
                    "- 입력은 숫자/영문만 받습니다. yes/no 또는 y/n만 사용하세요.",
                ]
            ),
            title="H 도움말",
            border_style="green",
        )
    )
    _pause(console, prompt="Enter=메인으로 돌아가기 > ")


def _ascii_input(console: Console, prompt: str) -> str | None:
    try:
        raw = input(prompt).strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        console.print("\n[red]입력이 중단되었습니다.[/red]")
        return None
    if raw and not raw.isascii():
        console.print("[red]한글 입력은 받지 않습니다. 숫자/영문만 입력하세요.[/red]")
        return ""
    return raw


def _confirm(console: Console, prompt: str = "yes/no > ") -> bool:
    while True:
        raw = _ascii_input(console, prompt)
        if raw is None:
            return False
        lowered = raw.lower()
        if lowered in {"y", "yes"}:
            return True
        if lowered in {"n", "no", ""}:
            return False
        console.print("[yellow]`yes` 또는 `no`만 입력하세요.[/yellow]")


def _pause(console: Console, *, prompt: str = "Enter=계속 > ") -> str:
    raw = _ascii_input(console, prompt)
    return "" if raw is None else raw.upper()


def _run_action(console: Console, action: ActionSpec, snapshot: WorkspaceSnapshot) -> bool:
    blockers = _action_blockers(action, snapshot)
    if blockers:
        console.print(
            Panel(
                "\n".join(f"- {item}" for item in blockers),
                title=f"{action.key} {action.label} 실행 차단",
                border_style="red",
            )
        )
        return _pause(console, prompt="Enter=메인으로 복귀, Q=종료 > ") != "Q"

    console.clear(home=True)
    lines = [
        f"🎯 작업: {action.label}",
        f"- 명령: {action.command_display}",
        f"- 출력 위치: {action.output_hint}",
    ]
    advisory = _action_advisory(action, snapshot)
    if advisory is not None:
        lines.append(f"- 주의: {advisory}")
    console.print(Panel("\n".join(lines), title=f"{action.key} 실행 확인", border_style="cyan"))
    if not _confirm(console):
        return True

    console.print("[bold green]실행을 시작합니다. 로그는 아래에 그대로 이어집니다.[/bold green]")
    try:
        completed = subprocess.run(action.argv, cwd=str(snapshot.paths.repo_root), check=False)
    except KeyboardInterrupt:
        console.print("\n[red]실행 중단됨[/red]")
        return _pause(console, prompt="Enter=메인으로 복귀, Q=종료 > ") != "Q"
    console.print(
        Panel(
            f"return code: {completed.returncode}\n다시 스캔해서 상태를 갱신합니다.",
            title="실행 종료",
            border_style="green" if completed.returncode == 0 else "yellow",
        )
    )
    return _pause(console, prompt="Enter=상태 새로고침, Q=종료 > ") != "Q"


def _interactive_loop(console: Console) -> int:
    actions = _action_catalog(_resolve_pipeline_paths())
    while True:
        report = check_env(check_yolo_runtime=True)
        snapshot = scan_workspace_status(report)
        actions = _action_catalog(snapshot.paths)
        _render_dashboard(console, snapshot, actions)

        raw = _ascii_input(console, "선택 (1-9/A-C/H/R/Q) > ")
        if raw is None:
            return 0
        if raw == "":
            continue
        choice = raw.upper()
        if choice == "Q":
            return 0
        if choice == "R":
            continue
        if choice == "H":
            _render_help(console, snapshot)
            continue
        action = next((item for item in actions if item.key == choice), None)
        if action is None:
            console.print("[yellow]알 수 없는 입력입니다. 숫자/영문 키만 사용하세요.[/yellow]")
            _pause(console)
            continue
        if not _run_action(console, action, snapshot):
            return 0


def _strict_failures(report: dict[str, Any], *, require_runtime: bool) -> list[str]:
    failures: list[str] = []
    if report["versions"]["torch"] is None:
        failures.append("torch missing")
    if report["checks"]["yolo26"]["importable"] is not True:
        failures.append("ultralytics missing")
    if require_runtime and report["checks"]["yolo26"]["runtime_load_ok"] is not True:
        failures.append("yolo26 runtime load failed")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if _should_run_interactive(args):
        return _interactive_loop(Console())

    report = check_env(check_yolo_runtime=args.check_yolo_runtime)
    print(json.dumps(report, indent=2, ensure_ascii=True))

    if not args.strict:
        return 0

    failures = _strict_failures(report, require_runtime=args.check_yolo_runtime)
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
