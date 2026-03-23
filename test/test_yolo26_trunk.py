from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


class _DummyDetect(nn.Module):
    pass


class _DummyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1)),
                nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1)),
                _DummyDetect(),
            ]
        )


class _DummyYOLO:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.model = _DummyCore()


class YOLO26TrunkTests(unittest.TestCase):
    def test_pretrained_adapter_requires_yolo26_capable_ultralytics(self) -> None:
        from model.trunk.ultralytics_yolo26 import build_yolo26n_trunk

        with patch("model.trunk.ultralytics_yolo26.ULTRALYTICS_VERSION", "8.3.78"):
            with self.assertRaisesRegex(RuntimeError, "8.4.0"):
                build_yolo26n_trunk()

    def test_pretrained_adapter_extracts_trunk_and_detect_head(self) -> None:
        from model.trunk.ultralytics_yolo26 import build_yolo26n_trunk

        with patch("model.trunk.ultralytics_yolo26.ULTRALYTICS_VERSION", "8.4.2"):
            with patch("model.trunk.ultralytics_yolo26.YOLO", _DummyYOLO):
                adapter = build_yolo26n_trunk()

        self.assertEqual(adapter.weights, "yolo26n.pt")
        self.assertEqual(adapter.ultralytics_version, "8.4.2")
        self.assertEqual(len(adapter.trunk), 2)
        self.assertEqual(adapter.detect_head.__class__.__name__, "_DummyDetect")
        self.assertEqual(adapter.detect_head_index, 2)
        self.assertEqual(adapter.feature_source_indices, ())
        self.assertTrue(all(parameter.requires_grad for parameter in adapter.trunk.parameters()))

    def test_partial_loader_only_applies_matching_shapes(self) -> None:
        from model.trunk.ultralytics_yolo26 import load_matching_state_dict

        module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        source = {
            "0.weight": torch.ones((4, 4)),
            "0.bias": torch.zeros(4),
            "1.weight": torch.ones((3, 3)),
            "missing.bias": torch.zeros(1),
        }

        summary = load_matching_state_dict(module, source)

        self.assertEqual(summary["loaded_count"], 2)
        self.assertIn("1.weight", summary["skipped_shape_keys"])
        self.assertIn("missing.bias", summary["missing_target_keys"])
        self.assertTrue(torch.allclose(module[0].weight, torch.ones((4, 4))))
        self.assertTrue(torch.allclose(module[0].bias, torch.zeros(4)))

    def test_summary_reports_trunk_and_detect_metadata(self) -> None:
        from model.trunk.ultralytics_yolo26 import UltralyticsYOLO26TrunkAdapter, summarize_trunk_adapter

        raw_model = _DummyCore()
        adapter = UltralyticsYOLO26TrunkAdapter(
            weights="yolo26n.pt",
            ultralytics_version="8.4.25",
            raw_model=raw_model,
            trunk=nn.Sequential(*list(raw_model.model.children())[:-1]),
            detect_head=raw_model.model[-1],
            detect_head_index=2,
        )

        summary = summarize_trunk_adapter(adapter)

        self.assertEqual(summary["weights"], "yolo26n.pt")
        self.assertEqual(summary["raw_model_class"], "_DummyCore")
        self.assertEqual(summary["raw_layer_count"], 3)
        self.assertEqual(summary["trunk_layer_count"], 2)
        self.assertEqual(summary["detect_head_class"], "_DummyDetect")
        self.assertEqual(summary["detect_head_index"], 2)
        self.assertEqual(summary["feature_source_indices"], [])


if __name__ == "__main__":
    unittest.main()
