from __future__ import annotations

import unittest

from tools.od_bootstrap.sweep.schema import BoxProvenance, RunManifest, TeacherJobManifest


class ODBootstrapSchemaTests(unittest.TestCase):
    def test_box_provenance_serializes_required_fields(self) -> None:
        provenance = BoxProvenance(
            label_origin="bootstrap",
            teacher_name="signal",
            confidence=0.82,
            model_version="signal_yolov26n_bootstrap_v1",
            run_id="20260327_000000_od_bootstrap",
            created_at="2026-03-27T10:00:00",
        )

        self.assertEqual(
            provenance.to_dict(),
            {
                "label_origin": "bootstrap",
                "teacher_name": "signal",
                "confidence": 0.82,
                "model_version": "signal_yolov26n_bootstrap_v1",
                "run_id": "20260327_000000_od_bootstrap",
                "created_at": "2026-03-27T10:00:00",
            },
        )

    def test_manifests_serialize_expected_fields(self) -> None:
        run_manifest = RunManifest(
            run_id="run_001",
            created_at="2026-03-27T10:00:00",
            scenario_path="/tmp/bootstrap.yaml",
            execution_mode="model-centric",
            dry_run=True,
            run_dir="/tmp/runs/od_bootstrap/run_001",
            image_pool_manifest="/tmp/image_pool.jsonl",
            image_count=3,
            teacher_names=("mobility", "signal", "obstacle"),
        )
        teacher_manifest = TeacherJobManifest(
            run_id="run_001",
            created_at="2026-03-27T10:00:00",
            teacher_name="obstacle",
            base_model="yolov26n",
            model_version="obstacle_v1",
            checkpoint_path="/tmp/obstacle.pt",
            classes=("traffic_cone", "obstacle"),
            image_count=3,
            predictions_path="/tmp/runs/od_bootstrap/run_001/teachers/obstacle/predictions.jsonl",
            dry_run=True,
        )

        self.assertEqual(run_manifest.to_dict()["teacher_names"], ("mobility", "signal", "obstacle"))
        self.assertEqual(teacher_manifest.to_dict()["teacher_name"], "obstacle")
        self.assertEqual(teacher_manifest.to_dict()["classes"], ("traffic_cone", "obstacle"))
