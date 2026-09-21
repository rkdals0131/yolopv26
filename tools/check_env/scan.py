"""Read-only snapshot of focused PV26 inputs, runs, and host resources."""

from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_PRUNED_DIRS = {
    "images", "labels", "checkpoints", "data_snapshot", "meta", "overlays",
    "inference", "export", "logs", "cache", "__pycache__",
}
_VERSION_PACKAGES = (
    "torch", "torchvision", "ultralytics", "numpy", "scipy", "Pillow", "PyYAML", "rich",
)


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _read_json(path: Path, errors: list[str], *, strict: bool = False,
               required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            if strict:
                raise FileNotFoundError(path)
            errors.append(f"{path}: 설정 파일을 찾을 수 없음")
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("metadata JSON must be an object")
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        if strict:
            raise ValueError(f"{path}: {exc}") from exc
        errors.append(f"{path}: {exc}")
        return {}


def _read_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError("YAML root must be an object")
        return value
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"{path}: {exc}")
        return {}


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in _VERSION_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    result: dict[str, Any] = {
        "python": f"{sys.executable} (Python {platform.python_version()})",
        "versions": versions,
        "gpus": [],
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=3,
        )
        for row in csv.reader(completed.stdout.splitlines()):
            if len(row) != 4:
                continue
            name, total, free, utilization = (part.strip() for part in row)
            result["gpus"].append({
                "name": name,
                "total_mb": int(total),
                "free_mb": int(free),
                "utilization": int(utilization),
            })
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["error"] = f"GPU 상태 조회 실패: {exc}"
    return result


def _storage(root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "root": str(root),
        "mount": None,
        "free_bytes": None,
        "error": None,
    }
    if not root.is_dir():
        item["error"] = "출력 디렉터리가 없음"
        return item
    try:
        item["free_bytes"] = shutil.disk_usage(root).free
        mount = subprocess.run(
            ["findmnt", "-n", "-o", "TARGET", "-T", str(root)],
            capture_output=True, text=True, check=True, timeout=3,
        ).stdout.strip()
        item["mount"] = mount or None
        if not mount or mount == "/":
            item["error"] = "별도 외장 마운트를 확인하지 못함"
    except (OSError, subprocess.SubprocessError) as exc:
        item["error"] = f"저장소 상태 조회 실패: {exc}"
    return item


def _lock_held(path: Path, errors: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                return False
    except OSError as exc:
        errors.append(f"{path}: 잠금 상태 조회 실패: {exc}")
        return False


def _existing(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def _exports(path: Path, errors: list[str]) -> list[str]:
    found: list[str] = []
    for parent in (path, path / "checkpoints", path / "export"):
        if not parent.is_dir():
            continue
        try:
            found.extend(str(candidate) for candidate in parent.iterdir()
                         if candidate.is_file()
                         and candidate.name.endswith((".torchscript.pt", ".onnx", ".engine")))
        except OSError as exc:
            errors.append(f"{parent}: 내보낸 모델 목록 조회 실패: {exc}")
    return sorted(found)


def _updated_at(paths: list[Path]) -> float | None:
    times = []
    for path in paths:
        try:
            times.append(path.stat().st_mtime)
        except (FileNotFoundError, OSError):
            continue
    return max(times) if times else None


def _run_state(running: bool, step: int | None, max_steps: int | None,
               latest: str | None, summary: dict[str, Any]) -> str:
    progress = f"마지막 기록 {step}" if step is not None else "기록 단계 미상"
    if max_steps is not None:
        progress += f"/{max_steps}"
    if running:
        return f"실행 중 · {progress}"
    if step is not None and max_steps is not None and step >= max_steps:
        return f"설정 단계 완료 · {progress}"
    if summary.get("stopped_by_signal"):
        return f"종료 요청 후 정지 · {progress}"
    if step is not None:
        return f"중단 또는 대기 · {progress}"
    if latest is not None:
        return "체크포인트 있음 · 기록 단계 미상"
    return "설정만 있음"


def _run(path: Path, kind: str, errors: list[str], *, strict: bool = False) -> dict[str, Any]:
    config_name = "run_config.json" if kind == "pv26" else "signal_config.json"
    saved_config = _read_json(path / config_name, errors, strict=strict, required=True)
    summary = _read_json(path / "summary.json", errors, strict=strict)
    validation = _read_json(path / "validation.json", errors, strict=strict)
    stage = (
        saved_config.get("train", {}).get("stage")
        if kind == "pv26" and isinstance(saved_config.get("train"), dict)
        else "signal_attr"
    )
    max_steps = _integer(
        saved_config.get("train", {}).get("max_steps")
        if kind == "pv26" and isinstance(saved_config.get("train"), dict)
        else saved_config.get("max_steps")
    )
    recorded_steps = (_integer(summary.get("global_step")),
                      _integer(validation.get("global_step")))
    step = max((value for value in recorded_steps if value is not None), default=None)
    checkpoints = {
        name: _existing(path / "checkpoints" / f"{name}.pt")
        for name in ("latest", "previous", "best")
    }
    checkpoints["published"] = _existing(path / "best_signal_attr.pt") if kind == "signal_attr" else None
    exports = _exports(path, errors)
    running = _lock_held(path / ".train.lock", errors)
    relevant = [path / config_name, path / "summary.json", path / "validation.json"]
    relevant.extend(path / "checkpoints" / f"{name}.pt" for name in ("latest", "previous", "best"))
    relevant.extend(Path(value) for value in exports)
    if checkpoints["published"] is not None:
        relevant.append(Path(checkpoints["published"]))
    return {
        "path": str(path),
        "kind": kind,
        "stage": stage,
        "running": running,
        "step": step,
        "max_steps": max_steps,
        "state": _run_state(running, step, max_steps, checkpoints["latest"], summary),
        "updated_at": _updated_at(relevant),
        "config": saved_config,
        "summary": summary,
        "validation": validation,
        "checkpoints": checkpoints,
        "exports": exports,
    }


def _crop_dataset(path: Path, errors: list[str]) -> dict[str, Any]:
    manifest_path = path / "meta" / "signal_attr_dataset_manifest.json"
    crop_config_path = path / "meta" / "crop_config.json"
    manifest = _read_json(manifest_path, errors)
    crop_config = _read_json(crop_config_path, errors)
    counts = manifest.get("accepted_count_by_split")
    counts = counts if isinstance(counts, dict) else {}
    train_labels = path / "labels" / "train.jsonl"
    val_labels = path / "labels" / "val.jsonl"
    try:
        usable = all(file.is_file() and file.stat().st_size > 0 for file in (train_labels, val_labels))
    except OSError as exc:
        errors.append(f"{path}: crop 라벨 상태 조회 실패: {exc}")
        usable = False
    usable = usable and bool(manifest) and bool(crop_config)
    return {
        "path": str(path),
        "train_count": _integer(counts.get("train")),
        "val_count": _integer(counts.get("val")),
        "all_off_is_valid": manifest.get("all_off_is_valid"),
        "usable": usable,
        "manifest": manifest,
    }


def _artifacts(roots: list[Path], errors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    crops: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for directory, children, _ in os.walk(root, topdown=True,
                                               onerror=lambda exc: errors.append(str(exc))):
            path = Path(directory)
            if (path / "run_config.json").is_file():
                runs.append(_run(path, "pv26", errors))
                children[:] = []
                continue
            if (path / "signal_config.json").is_file():
                runs.append(_run(path, "signal_attr", errors))
                children[:] = []
                continue
            if (path / "labels").is_dir() and (path / "meta").is_dir():
                crops.append(_crop_dataset(path, errors))
                children[:] = []
                continue
            children[:] = [name for name in children if name not in _PRUNED_DIRS]
    runs.sort(key=lambda item: (item["updated_at"] or 0.0, item["path"]), reverse=True)
    crops.sort(key=lambda item: item["path"])
    return runs, crops


def scan_run(path: Path) -> dict[str, Any]:
    """Inspect one selected run without opening model or optimizer checkpoints."""
    path = _path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    has_pv26 = (path / "run_config.json").is_file()
    has_signal = (path / "signal_config.json").is_file()
    if has_pv26 == has_signal:
        raise ValueError(f"학습 설정 파일을 정확히 하나 찾을 수 없습니다: {path}")
    return _run(path, "pv26" if has_pv26 else "signal_attr", [], strict=True)


def scan_workspace(config_path: Path, signal_config_path: Path) -> dict[str, Any]:
    """Return a JSON-serializable, read-only current-state snapshot for the HMI."""
    config_path = _path(config_path)
    signal_config_path = _path(signal_config_path)
    errors: list[str] = []
    config = _read_yaml(config_path, errors)
    signal_config = _read_yaml(signal_config_path, errors)
    roots: list[Path] = []
    for document in (config, signal_config):
        train = document.get("train")
        output_root = train.get("output_root") if isinstance(train, dict) else None
        if output_root is not None:
            root = _path(output_root)
            if root not in roots:
                roots.append(root)
    sources = []
    data = config.get("data")
    for source in data.get("sources", []) if isinstance(data, dict) else []:
        if not isinstance(source, dict):
            continue
        root_value = source.get("root")
        if not isinstance(root_value, str) or not root_value.strip():
            errors.append(f"{config_path}: 데이터 source root가 비어 있음")
            continue
        root = _path(root_value)
        sources.append({
            "name": source.get("name"),
            "kind": source.get("kind"),
            "root": str(root),
            "weight": source.get("weight"),
            "train_exists": (root / "Training").is_dir(),
            "val_exists": (root / "Validation").is_dir(),
        })
    weights = []
    model = config.get("model")
    signal_train = signal_config.get("train")
    for role, value in (
        ("PV26 초기 가중치", model.get("weights") if isinstance(model, dict) else None),
        ("SignalAttr 초기 가중치", signal_train.get("initial_checkpoint")
         if isinstance(signal_train, dict) else None),
    ):
        if isinstance(value, str) and value.strip():
            path = _path(value)
            weights.append({"role": role, "path": str(path), "exists": path.is_file()})
    runs, crops = _artifacts(roots, errors)
    return {
        "config_path": str(config_path),
        "signal_config_path": str(signal_config_path),
        "config": config,
        "signal_config": signal_config,
        "environment": _environment(),
        "storage": [_storage(root) for root in roots],
        "sources": sources,
        "weights": weights,
        "runs": runs,
        "crop_datasets": crops,
        "errors": errors,
    }
