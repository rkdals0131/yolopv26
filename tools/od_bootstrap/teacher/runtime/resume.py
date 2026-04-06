from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    torch = None

__all__ = [
    "checkpoint_resume_metadata",
    "coerce_weights_name",
    "extract_run_dir",
    "resolve_latest_resumable_checkpoint",
    "resolve_resume_argument",
    "resolve_resume_checkpoint_path",
]


def coerce_weights_name(model_size: str, weights: str | None) -> str:
    if weights:
        return weights
    size = model_size.strip().lower() or "n"
    return f"yolo26{size}.pt"


def _load_checkpoint_payload(checkpoint: Path) -> dict[str, Any] | None:
    if torch is None:
        raise RuntimeError("torch is not available")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"resume checkpoint payload must be a mapping: {checkpoint}")
    return payload


def checkpoint_resume_metadata(checkpoint: Path) -> dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint)
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    optimizer = payload.get("optimizer") if isinstance(payload, dict) else None
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    return {
        "epoch": int(epoch) if isinstance(epoch, int) else -1,
        "train_args": dict(train_args) if isinstance(train_args, dict) else {},
        "resumable": isinstance(optimizer, dict) and isinstance(epoch, int) and int(epoch) >= 0,
    }


def _resume_candidate_sort_key(checkpoint: Path) -> tuple[int, int, int, int, str]:
    metadata = checkpoint_resume_metadata(checkpoint)
    name = checkpoint.name
    kind_priority = 2 if name.startswith("last_resume") else 1 if name.startswith("last") else 0
    return (
        int(metadata["resumable"]),
        kind_priority,
        int(checkpoint.stat().st_mtime_ns),
        int(metadata["epoch"]),
        str(checkpoint),
    )


def resolve_latest_resumable_checkpoint(teacher_root: Path, *, teacher_name: str | None = None) -> Path:
    candidates: set[Path] = set()
    for pattern in ("**/last_resume*.pt", "**/last*.pt", "**/epoch*.pt"):
        candidates.update(path for path in teacher_root.glob(pattern) if path.is_file())
    resumable = [path for path in candidates if checkpoint_resume_metadata(path)["resumable"]]
    if not resumable:
        label = f" for teacher '{teacher_name}'" if teacher_name else ""
        raise FileNotFoundError(f"resume requested{label} but no resumable checkpoint exists under {teacher_root}")
    return max(resumable, key=_resume_candidate_sort_key).resolve()


def resolve_resume_checkpoint_path(checkpoint: Path, *, teacher_root: Path | None = None) -> Path | None:
    del teacher_root
    resolved = Path(checkpoint).expanduser().resolve()
    metadata = checkpoint_resume_metadata(resolved)
    if not metadata["resumable"]:
        raise RuntimeError(
            f"resume checkpoint is finalized and not resumable: {resolved}. "
            "Use a saved epoch*.pt or last_resume.pt checkpoint instead."
        )
    return resolved


def resolve_resume_argument(resume: Any, *, teacher_name: str, teacher_root: Path) -> bool | str:
    if not resume:
        return False
    if isinstance(resume, bool):
        return str(resolve_latest_resumable_checkpoint(teacher_root, teacher_name=teacher_name))
    if isinstance(resume, Path):
        return str(resolve_resume_checkpoint_path(resume, teacher_root=teacher_root))
    if isinstance(resume, str):
        normalized = resume.strip()
        if not normalized:
            return False
        if normalized.lower() == "latest":
            return str(resolve_latest_resumable_checkpoint(teacher_root, teacher_name=teacher_name))
        return str(resolve_resume_checkpoint_path(Path(normalized), teacher_root=teacher_root))
    raise TypeError("resume must be false, true, 'latest', or an exact checkpoint path")


def extract_run_dir(train_result: Any, fallback_dir: Path) -> Path:
    for candidate in (
        getattr(train_result, "save_dir", None),
        getattr(getattr(train_result, "trainer", None), "save_dir", None),
    ):
        if candidate:
            return Path(candidate)
    return fallback_dir


_checkpoint_resume_metadata = checkpoint_resume_metadata
_coerce_weights_name = coerce_weights_name
_extract_run_dir = extract_run_dir
_resolve_latest_resumable_checkpoint = resolve_latest_resumable_checkpoint
_resolve_resume_argument = resolve_resume_argument
_resolve_resume_checkpoint_path = resolve_resume_checkpoint_path
