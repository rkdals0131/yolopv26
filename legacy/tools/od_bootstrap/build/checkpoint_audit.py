from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from common.io import write_json as _write_json

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TeacherCheckpointSpec:
    teacher_name: str
    checkpoint_path: Path
    alias_checkpoint_path: Path
    expected_scale: str = "s"


DEFAULT_TEACHER_CHECKPOINT_SPECS = (
    TeacherCheckpointSpec(
        teacher_name="mobility",
        checkpoint_path=REPO_ROOT / "runs" / "od_bootstrap" / "train" / "mobility" / "20260328_145441" / "weights" / "best.pt",
        alias_checkpoint_path=REPO_ROOT / "runs" / "od_bootstrap" / "train" / "mobility" / "weights" / "best.pt",
    ),
    TeacherCheckpointSpec(
        teacher_name="obstacle",
        checkpoint_path=REPO_ROOT / "runs" / "od_bootstrap" / "train" / "obstacle" / "20260328_143910" / "weights" / "best.pt",
        alias_checkpoint_path=REPO_ROOT / "runs" / "od_bootstrap" / "train" / "obstacle" / "weights" / "best.pt",
    ),
    TeacherCheckpointSpec(
        teacher_name="signal",
        checkpoint_path=REPO_ROOT / "runs" / "od_bootstrap" / "train" / "signal" / "20260329_021050" / "weights" / "best.pt",
        alias_checkpoint_path=REPO_ROOT / "runs" / "od_bootstrap" / "train" / "signal" / "weights" / "best.pt",
    ),
)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value)
    if value is None:
        return None
    return str(value)


def _safe_param_count(model: Any) -> int | None:
    if model is None:
        return None
    try:
        return sum(int(parameter.numel()) for parameter in model.parameters())
    except Exception:
        return None


def _load_checkpoint_payload(path: Path) -> dict[str, Any] | None:
    if torch is None or not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else None


def _checkpoint_info(path: Path) -> dict[str, Any]:
    payload = _load_checkpoint_payload(path)
    if payload is None:
        return {
            "path": str(path),
            "exists": path.is_file(),
            "file_size_bytes": path.stat().st_size if path.is_file() else None,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.is_file() else None,
            "epoch": None,
            "best_fitness": None,
            "train_args_model": None,
            "train_args_name": None,
            "train_args_project": None,
            "scale": None,
            "nc": None,
            "param_count": None,
            "has_optimizer": None,
        }
    train_args = payload.get("train_args") if isinstance(payload.get("train_args"), dict) else {}
    model = payload.get("ema") or payload.get("model")
    yaml_attr = getattr(model, "yaml", None)
    scale = yaml_attr.get("scale") if isinstance(yaml_attr, dict) else None
    nc = yaml_attr.get("nc") if isinstance(yaml_attr, dict) else None
    return {
        "path": str(path),
        "exists": True,
        "file_size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "epoch": _json_scalar(payload.get("epoch")),
        "best_fitness": _json_scalar(payload.get("best_fitness")),
        "train_args_model": _json_scalar(train_args.get("model")),
        "train_args_name": _json_scalar(train_args.get("name")),
        "train_args_project": _json_scalar(train_args.get("project")),
        "scale": _json_scalar(scale),
        "nc": _json_scalar(nc),
        "param_count": _safe_param_count(model),
        "has_optimizer": isinstance(payload.get("optimizer"), dict),
    }


def _same_checkpoint(target_path: Path, alias_path: Path) -> bool | None:
    if not target_path.is_file() or not alias_path.is_file():
        return None
    try:
        return target_path.samefile(alias_path)
    except Exception:
        target_stat = target_path.stat()
        alias_stat = alias_path.stat()
        return (target_stat.st_ino, target_stat.st_dev) == (alias_stat.st_ino, alias_stat.st_dev)


def audit_teacher_checkpoints(
    specs: tuple[TeacherCheckpointSpec, ...] = DEFAULT_TEACHER_CHECKPOINT_SPECS,
) -> dict[str, Any]:
    teachers: list[dict[str, Any]] = []
    expected_ok = True
    for spec in specs:
        checkpoint_info = _checkpoint_info(spec.checkpoint_path)
        alias_info = _checkpoint_info(spec.alias_checkpoint_path)
        scale = checkpoint_info.get("scale")
        alias_scale = alias_info.get("scale")
        is_expected_scale = scale == spec.expected_scale
        expected_ok = expected_ok and bool(is_expected_scale)
        teachers.append(
            {
                "teacher_name": spec.teacher_name,
                "expected_scale": spec.expected_scale,
                "checkpoint": {
                    **checkpoint_info,
                    "is_expected_scale": is_expected_scale,
                },
                "alias_checkpoint": {
                    **alias_info,
                    "same_as_target": _same_checkpoint(spec.checkpoint_path, spec.alias_checkpoint_path),
                    "matches_expected_scale": alias_scale == spec.expected_scale if alias_scale is not None else None,
                },
            }
        )
    return {
        "version": "od-bootstrap-checkpoint-audit-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "all_targets_match_expected_scale": expected_ok,
        "teachers": teachers,
    }


def write_checkpoint_audit(
    output_path: Path,
    specs: tuple[TeacherCheckpointSpec, ...] = DEFAULT_TEACHER_CHECKPOINT_SPECS,
) -> dict[str, Any]:
    payload = audit_teacher_checkpoints(specs)
    resolved_output = _write_json(output_path, payload)
    payload["output_path"] = str(resolved_output)
    return payload
