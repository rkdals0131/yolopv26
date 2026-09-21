from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.pv26_schema import (
    AIHUB_LANE_DATASET_KEY,
    EXHAUSTIVE_DATASET_KEY_BY_SOURCE,
    AIHUB_TRAFFIC_DATASET_KEY,
)
from PIL import Image
from pv26_prepared_dataset_fixture import create_prepared_pv26_dataset
from tools import pv26_eval_harness as harness


def _write_supported_meta(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "output_names": ["det", "tl_attr", "lane", "stop_line", "crosswalk"],
                "tl_bits": ["red", "yellow", "green", "arrow"],
                "outputs": {
                    "det": {"shape": ["batch", 1, 12]},
                    "tl_attr": {"shape": ["batch", 1, 4]},
                    "lane": {"shape": ["batch", 12, 54]},
                    "stop_line": {"shape": ["batch", 6, 9]},
                    "crosswalk": {"shape": ["batch", 8, 33]},
                },
            }
        ),
        encoding="utf-8",
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, _image_bgr):
        self.calls += 1
        return (
            SimpleNamespace(
                detections=[
                    {
                        "box_xyxy": [120.0, 80.0, 180.0, 220.0],
                        "score": 0.95,
                        "class_id": 5,
                        "class_name": "traffic_light",
                        "tl_attr_scores": {"red": 0.95, "yellow": 0.05, "green": 0.05, "arrow": 0.90},
                    },
                    {
                        "box_xyxy": [360.0, 120.0, 500.0, 260.0],
                        "score": 0.90,
                        "class_id": 6,
                        "class_name": "sign",
                        "tl_attr_scores": {"red": 0.0, "yellow": 0.0, "green": 0.0, "arrow": 0.0},
                    },
                ],
                lanes=[
                    {
                        "score": 0.9,
                        "class_name": "white_lane",
                        "lane_type": "solid",
                        "points_xy": [[240.0, 700.0], [260.0, 520.0], [280.0, 340.0]],
                    },
                    {
                        "score": 0.9,
                        "class_name": "yellow_lane",
                        "lane_type": "dotted",
                        "points_xy": [[980.0, 700.0], [960.0, 520.0], [940.0, 340.0]],
                    },
                ],
                stop_lines=[{"score": 0.9, "points_xy": [[260.0, 620.0], [1000.0, 620.0]]}],
                crosswalks=[
                    {
                        "score": 0.9,
                        "points_xy": [[330.0, 650.0], [470.0, 650.0], [500.0, 710.0], [300.0, 710.0]],
                    }
                ],
            ),
            {"preprocess_ms": 0.25, "forward_ms": 1.0, "postprocess_ms": 0.5},
        )


def test_model_contract_status_requires_tl_attr_output() -> None:
    supported, reason = harness.model_contract_status(
        {
            "output_names": ["det", "lane", "stop_line"],
            "outputs": {
                "det": {"shape": ["batch", 1, 12]},
                "lane": {"shape": ["batch", 12, 54]},
                "stop_line": {"shape": ["batch", 6, 9]},
            },
        }
    )

    assert not supported
    assert reason is not None and "tl_attr" in reason


def test_build_candidates_pairs_model_names_weights_and_meta(tmp_path: Path) -> None:
    weights_a = tmp_path / "spade_best.torchscript.pt"
    weights_b = tmp_path / "run_best.torchscript.pt"
    meta_a = tmp_path / "spade_best.torchscript.meta.json"
    meta_b = tmp_path / "run_best.torchscript.meta.json"

    candidates = harness.build_candidates(
        [str(weights_a), str(weights_b)],
        [str(meta_a), str(meta_b)],
        spade_root=tmp_path,
        names=["spade/best", "run/run_done/best"],
    )

    assert [item.name for item in candidates] == ["spade_best", "run_run_done_best"]
    assert candidates[0].weights == weights_a.resolve()
    assert candidates[0].model_meta == meta_a.resolve()
    assert candidates[1].weights == weights_b.resolve()
    assert candidates[1].model_meta == meta_b.resolve()


def test_build_candidates_suffixes_duplicate_stem_fallback_names(tmp_path: Path) -> None:
    first = tmp_path / "a" / "best.torchscript.pt"
    second = tmp_path / "b" / "best.torchscript.pt"

    candidates = harness.build_candidates(
        [str(first), str(second)],
        [],
        spade_root=tmp_path,
    )

    assert [item.name for item in candidates] == ["best.torchscript", "best.torchscript_2"]
    assert candidates[0].model_meta == (tmp_path / "a" / "best.torchscript.meta.json").resolve()
    assert candidates[1].model_meta == (tmp_path / "b" / "best.torchscript.meta.json").resolve()


def test_build_candidates_rejects_unpaired_model_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require paired --weights"):
        harness.build_candidates([], [], spade_root=tmp_path, names=["spade/best"])


def test_runtime_prediction_preserves_tl_attr_scores(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), "#202020").save(image_path)
    prediction = harness.run_runtime_predictions(
        FakeRuntime(),
        [
            {
                "meta": {
                    "sample_id": "sample",
                    "image_path": str(image_path),
                }
            }
        ],
    )

    assert prediction[0]["detections"][0]["tl_attr_scores"]["red"] == 0.95


def test_composite_score_averages_task_representatives_equally() -> None:
    assert harness.composite_score({"detector": 1.0, "traffic_light": 0.5, "lane": 0.0}) == 0.5


def test_evaluate_models_writes_summary_predictions_plot_and_overlay(tmp_path: Path) -> None:
    dataset_root = create_prepared_pv26_dataset(
        tmp_path / "dataset",
        splits=("val",),
        dataset_keys=(
            EXHAUSTIVE_DATASET_KEY_BY_SOURCE[AIHUB_TRAFFIC_DATASET_KEY],
            AIHUB_LANE_DATASET_KEY,
        ),
    )
    weights = tmp_path / "model.torchscript.pt"
    weights.write_bytes(b"fake")
    meta = tmp_path / "model.torchscript.meta.json"
    _write_supported_meta(meta)
    candidate = harness.ModelCandidate("fake_model", weights, meta)
    options = harness.HarnessOptions(
        dataset_root=dataset_root,
        output_root=tmp_path / "out",
        candidates=(candidate,),
        spade_root=tmp_path,
        pv26_repo_root=Path(__file__).resolve().parents[1],
        device_name="cpu",
        split="val",
        overlay_count=1,
    )

    summary = harness.evaluate_models(options, runtime_factory=lambda _candidate, _options: FakeRuntime())

    summary_path = Path(summary["summary_json"])
    assert summary_path.is_file()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    model = saved["models"]["fake_model"]
    assert model["status"] == "evaluated"
    assert model["sample_count"] == 2
    assert "traffic_light" in model["representative_scores"]
    assert Path(model["predictions_jsonl"]).is_file()
    first_prediction = json.loads(Path(model["predictions_jsonl"]).read_text(encoding="utf-8").splitlines()[0])
    assert first_prediction["predictions"]["detections"][0]["tl_attr_scores"]["red"] == 0.95
    assert first_prediction["timing_ms"]["forward_ms"] == 1.0
    assert first_prediction["timing_ms"]["total_ms"] == 1.75
    assert model["timing_summary"]["forward_ms"]["count"] == 2
    assert model["timing_summary"]["forward_ms"]["mean"] == 1.0
    assert model["timing_summary"]["total_ms"]["mean"] == 1.75
    assert model["sample_score_summary"]["count"] == 2
    assert model["sample_score_summary"]["worst_samples"]
    assert model["histogram_summary"]["detector"]["prediction_confidence"]["count"] > 0
    assert saved["plots"]
    plot_names = {Path(plot_path).name for plot_path in saved["plots"]}
    assert {
        "summary_metrics.png",
        "detector_per_class.png",
        "task_error_distributions.png",
        "sample_score_distribution.png",
        "latency.png",
    }.issubset(plot_names)
    assert all(Path(plot_path).is_file() for plot_path in saved["plots"])
    assert "_plot_histograms" not in model
    assert "_plot_sample_records" not in model
    assert model["overlays"]
    assert Path(model["overlays"][0]).is_file()


def test_supervised_task_names_uses_tl_attr_validity() -> None:
    raw_batch = {
        "source_mask": [{"det": True, "tl_attr": True, "lane": False, "stop_line": False, "crosswalk": False}],
        "valid_mask": [{"tl_attr": torch.tensor([False], dtype=torch.bool)}],
    }

    assert harness.supervised_task_names(raw_batch) == ["detector"]
