from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    artifact_paths_for_checkpoint,
    ensure_writable_output,
    image_size_pair,
    json_ready,
    resolve_ultralytics_device,
    safe_unlink,
)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency absence is exercised via tests.
    YOLO = None


def _export_metadata(
    *,
    teacher_name: str,
    checkpoint_path: Path,
    artifact_path: Path,
    class_names: tuple[str, ...],
    imgsz: int | tuple[int, int] | list[int],
    export_device: str,
    export_result_path: str,
) -> dict[str, Any]:
    height, width = image_size_pair(imgsz)
    return {
        "format_version": 1,
        "artifact_type": "teacher_yolo_torchscript",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_name": teacher_name,
        "source_checkpoint": str(checkpoint_path),
        "artifact_path": str(artifact_path),
        "class_names": list(class_names),
        "input": {
            "layout": "NCHW",
            "dtype": "float32",
            "range": [0.0, 1.0],
            "height": height,
            "width": width,
        },
        "export": {
            "backend": "ultralytics",
            "format": "torchscript",
            "device": export_device,
            "result_path": export_result_path,
        },
    }


def export_teacher_torchscript(
    *,
    teacher_name: str,
    checkpoint_path: Path,
    class_names: tuple[str, ...],
    imgsz: int | tuple[int, int] | list[int],
    output_path: Path | None = None,
    meta_path: Path | None = None,
    device_name: str = "auto",
    overwrite: bool = True,
) -> dict[str, Any]:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    default_output_path, default_meta_path = artifact_paths_for_checkpoint(checkpoint_path)
    output_path = Path(output_path).expanduser().resolve() if output_path is not None else default_output_path
    meta_path = Path(meta_path).expanduser().resolve() if meta_path is not None else default_meta_path
    ensure_writable_output(output_path, overwrite=overwrite)
    ensure_writable_output(meta_path, overwrite=overwrite)
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

    export_device = resolve_ultralytics_device(device_name)
    model = YOLO(str(checkpoint_path))
    export_kwargs = {
        "format": "torchscript",
        "imgsz": image_size_pair(imgsz),
        "device": export_device,
        "optimize": False,
        "simplify": False,
    }
    with tempfile.TemporaryDirectory(prefix=f"{teacher_name}_torchscript_export_") as temp_dir:
        temp_root = Path(temp_dir)
        exported_path = Path(model.export(project=str(temp_root), name="export", **export_kwargs)).expanduser().resolve()
        if not exported_path.is_file():
            raise FileNotFoundError(f"ultralytics export did not produce an artifact: {exported_path}")
        safe_unlink(output_path)
        shutil.move(str(exported_path), str(output_path))

    metadata = _export_metadata(
        teacher_name=teacher_name,
        checkpoint_path=checkpoint_path,
        artifact_path=output_path,
        class_names=class_names,
        imgsz=imgsz,
        export_device=export_device,
        export_result_path=str(output_path),
    )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    return {
        "teacher_name": teacher_name,
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(output_path),
        "meta_path": str(meta_path),
        "class_names": list(class_names),
        "imgsz": image_size_pair(imgsz),
        "device": export_device,
    }
