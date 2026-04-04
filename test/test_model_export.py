from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.model_export.common import artifact_paths_for_checkpoint
from tools.model_export import pv26_torchscript as pv26_exporter
from tools.model_export import teacher_torchscript as teacher_exporter


def test_artifact_paths_for_checkpoint_are_adjacent() -> None:
    checkpoint = Path("/tmp/example/best.pt")

    artifact_path, meta_path = artifact_paths_for_checkpoint(checkpoint)

    assert artifact_path == Path("/tmp/example/best.torchscript.pt")
    assert meta_path == Path("/tmp/example/best.torchscript.meta.json")


def test_resolve_trunk_weights_uses_checkpoint_variant_when_available(tmp_path: Path) -> None:
    repo_root = tmp_path / "yolopv26"
    repo_root.mkdir()
    (repo_root / "yolo26n.pt").write_text("n", encoding="utf-8")
    s_weights = repo_root / "yolo26s.pt"
    s_weights.write_text("s", encoding="utf-8")

    assert (
        pv26_exporter.resolve_trunk_weights(
            repo_root,
            None,
            checkpoint_variant="s",
        )
        == s_weights.resolve()
    )


def test_resolve_trunk_weights_rejects_explicit_variant_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path / "yolopv26"
    repo_root.mkdir()
    explicit = repo_root / "yolo26n.pt"
    explicit.write_text("n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint expects a YOLO26 s backbone"):
        pv26_exporter.resolve_trunk_weights(
            repo_root,
            explicit,
            checkpoint_variant="s",
        )


def test_infer_backbone_variant_from_head_channels_supports_yolo26s() -> None:
    assert pv26_exporter.infer_backbone_variant_from_head_channels((128, 256, 512)) == "s"


def test_build_example_input_random_seed_is_reproducible() -> None:
    first, first_info = pv26_exporter.build_example_input(
        input_height=32,
        input_width=48,
        device=torch.device("cpu"),
        example_image=None,
        seed=1234,
    )
    second, second_info = pv26_exporter.build_example_input(
        input_height=32,
        input_width=48,
        device=torch.device("cpu"),
        example_image=None,
        seed=1234,
    )

    assert torch.allclose(first, second)
    assert first_info == second_info == {"kind": "random", "seed": 1234}


def test_letterbox_example_image_rejects_invalid_image_shape() -> None:
    with pytest.raises(ValueError):
        pv26_exporter.letterbox_example_image(
            np.zeros((0, 0, 3), dtype=np.uint8),
            input_height=608,
            input_width=800,
        )


def test_teacher_export_writes_adjacent_artifact_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")

    class FakeYOLO:
        def __init__(self, source: str) -> None:
            self.source = source

        def export(self, **kwargs: object) -> str:
            project = Path(str(kwargs["project"]))
            name = str(kwargs["name"])
            output_path = project / name / "best.torchscript.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("artifact", encoding="utf-8")
            return str(output_path)

    monkeypatch.setattr(teacher_exporter, "YOLO", FakeYOLO)

    result = teacher_exporter.export_teacher_torchscript(
        teacher_name="mobility",
        checkpoint_path=checkpoint,
        class_names=("vehicle", "bike", "pedestrian"),
        imgsz=640,
        device_name="cpu",
        overwrite=True,
    )

    artifact_path = Path(result["artifact_path"])
    meta_path = Path(result["meta_path"])
    assert artifact_path == checkpoint.with_suffix(".torchscript.pt")
    assert artifact_path.read_text(encoding="utf-8") == "artifact"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["teacher_name"] == "mobility"
    assert metadata["source_checkpoint"] == str(checkpoint.resolve())
    assert metadata["class_names"] == ["vehicle", "bike", "pedestrian"]
    assert metadata["input"]["height"] == 640
    assert metadata["input"]["width"] == 640
