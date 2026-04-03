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
    "resolve_resume_argument",
    "resolve_resume_checkpoint_path",
]


def coerce_weights_name(model_size: str, weights: str | None) -> str:
    if weights:
        return weights
    size = model_size.strip().lower() or "n"
    return f"yolo26{size}.pt"


def _load_checkpoint_payload(checkpoint: Path) -> dict[str, Any] | None:
    if torch is None or not checkpoint.is_file():
        return None
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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


def _infer_teacher_root_from_checkpoint(checkpoint: Path) -> Path | None:
    for parent in checkpoint.parents:
        if (parent / "latest_run.json").is_file():
            return parent
    return None


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


def _find_latest_resumable_checkpoint(teacher_root: Path) -> Path | None:
    candidates: set[Path] = set()
    for pattern in ("**/last_resume*.pt", "**/last*.pt", "**/epoch*.pt"):
        candidates.update(path for path in teacher_root.glob(pattern) if path.is_file())
    resumable = [path for path in candidates if checkpoint_resume_metadata(path)["resumable"]]
    if not resumable:
        return None
    return max(resumable, key=_resume_candidate_sort_key)


def resolve_resume_checkpoint_path(checkpoint: Path, *, teacher_root: Path | None = None) -> Path | None:
    if not checkpoint.is_file():
        return None
    if checkpoint_resume_metadata(checkpoint)["resumable"]:
        return checkpoint

    search_root = teacher_root or _infer_teacher_root_from_checkpoint(checkpoint)
    if search_root is not None:
        fallback = _find_latest_resumable_checkpoint(search_root)
        if fallback is not None:
            return fallback

    sibling_candidates = [path for path in checkpoint.parent.glob("last_resume*.pt") if path.is_file()]
    sibling_candidates.extend(path for path in checkpoint.parent.glob("epoch*.pt") if path.is_file())
    resumable = [path for path in sibling_candidates if checkpoint_resume_metadata(path)["resumable"]]
    if resumable:
        return max(resumable, key=_resume_candidate_sort_key)
    return checkpoint


def _find_latest_teacher_checkpoint(teacher_root: Path) -> Path | None:
    resumable = _find_latest_resumable_checkpoint(teacher_root)
    if resumable is not None:
        return resumable
    candidates = [path for path in teacher_root.glob("**/last*.pt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def resolve_resume_argument(resume: Any, *, teacher_name: str, teacher_root: Path) -> bool | str:
    if not resume:
        return False
    if isinstance(resume, Path):
        resolved = resolve_resume_checkpoint_path(resume, teacher_root=teacher_root)
        return str(resolved) if resolved is not None else str(resume)
    if isinstance(resume, str):
        normalized = resume.strip()
        if not normalized:
            return False
        resolved = resolve_resume_checkpoint_path(Path(normalized), teacher_root=teacher_root)
        return str(resolved) if resolved is not None else normalized

    checkpoint = _find_latest_teacher_checkpoint(teacher_root)
    if checkpoint is None:
        raise FileNotFoundError(
            f"resume requested for teacher '{teacher_name}' but no last.pt exists under {teacher_root}"
        )
    return str(checkpoint)


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
_resolve_resume_argument = resolve_resume_argument
_resolve_resume_checkpoint_path = resolve_resume_checkpoint_path
