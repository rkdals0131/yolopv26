from __future__ import annotations

from pathlib import Path


def artifact_paths_for_checkpoint(checkpoint_path: Path) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    artifact_path = checkpoint_path.with_suffix(".torchscript.pt")
    meta_path = artifact_path.with_suffix(".meta.json")
    return artifact_path, meta_path


def ensure_writable_output(path: Path, *, overwrite: bool) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path} (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
