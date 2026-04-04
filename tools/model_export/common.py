from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


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


def resolve_torch_device(device_name: str) -> torch.device:
    token = str(device_name).strip().lower()
    if token in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if token == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(token)


def resolve_ultralytics_device(device_name: str) -> str:
    device = resolve_torch_device(device_name)
    if device.type == "cuda":
        return str(device.index if device.index is not None else 0)
    return "cpu"


def image_size_pair(imgsz: int | tuple[int, int] | list[int]) -> list[int]:
    if isinstance(imgsz, int):
        size = int(imgsz)
        if size <= 0:
            raise ValueError("imgsz must be positive")
        return [size, size]
    if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        height = int(imgsz[0])
        width = int(imgsz[1])
        if height <= 0 or width <= 0:
            raise ValueError("imgsz dimensions must be positive")
        return [height, width]
    raise TypeError("imgsz must be an int or a (height, width) pair")


def safe_unlink(path: Path) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        path.unlink()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
