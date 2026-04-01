from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.od_bootstrap.build.checkpoint_audit import TeacherCheckpointSpec, audit_teacher_checkpoints
from tools.od_bootstrap.build.review import canonical_scene_to_overlay_scene, render_overlay, render_review_bundle, select_review_rows
from tools.od_bootstrap.build.sample_manifest import select_sample_entries, summarize_entries
from tools.od_bootstrap.build.image_list import ImageListEntry


class _FakeCheckpointModel(torch.nn.Module):
    def __init__(self, scale: str, nc: int) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.yaml = {"scale": scale, "nc": nc}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


class ODBootstrapSampleHelpersTests(unittest.TestCase):
    def test_select_sample_entries_applies_dataset_split_quotas(self) -> None:
        entries = []
        root = Path("/tmp/sample_manifest")
        counts = {
            ("bdd100k_det_100k", "train"): 3,
            ("bdd100k_det_100k", "val"): 2,
            ("bdd100k_det_100k", "test"): 4,
            ("aihub_traffic_seoul", "train"): 2,
            ("aihub_traffic_seoul", "val"): 2,
            ("aihub_obstacle_seoul", "train"): 2,
            ("aihub_obstacle_seoul", "val"): 2,
        }
        for (dataset_key, split), total in counts.items():
            for index in range(total):
                sample_id = f"{dataset_key}_{split}_{index}"
                entries.append(
                    ImageListEntry(
                        sample_id=sample_id,
                        sample_uid=f"{dataset_key}__{split}__{sample_id}",
                        image_path=root / dataset_key / split / f"{sample_id}.jpg",
                        scene_path=root / dataset_key / split / f"{sample_id}.json",
                        dataset_root=root / dataset_key,
                        dataset_key=dataset_key,
                        split=split,
                        det_path=None,
                        source_name="source",
                    )
                )

        selected = select_sample_entries(
            entries,
            quotas={
                "bdd100k_det_100k": {"train": 2, "val": 1},
                "aihub_traffic_seoul": {"train": 1, "val": 1},
                "aihub_obstacle_seoul": {"train": 1, "val": 1},
            },
        )

        summary = summarize_entries(selected)
        self.assertEqual(summary["image_count"], 7)
        self.assertEqual(
            summary["split_counts"],
            {
                "aihub_obstacle_seoul::train": 1,
                "aihub_obstacle_seoul::val": 1,
                "aihub_traffic_seoul::train": 1,
                "aihub_traffic_seoul::val": 1,
                "bdd100k_det_100k::train": 2,
                "bdd100k_det_100k::val": 1,
            },
        )
        self.assertFalse(any(entry.split == "test" for entry in selected))

    def test_audit_teacher_checkpoints_reports_scale_and_alias_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mobility_target = root / "mobility" / "best.pt"
            signal_target = root / "signal" / "best.pt"
            signal_alias = root / "signal_alias" / "best.pt"

            mobility_target.parent.mkdir(parents=True, exist_ok=True)
            signal_target.parent.mkdir(parents=True, exist_ok=True)
            signal_alias.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": 85,
                    "best_fitness": 0.42,
                    "train_args": {"model": "yolo26s.pt", "name": "mobility_run", "project": "/tmp/mobility"},
                    "model": _FakeCheckpointModel(scale="s", nc=3),
                    "optimizer": {"state": {}},
                },
                mobility_target,
            )
            torch.save(
                {
                    "epoch": 61,
                    "best_fitness": 0.51,
                    "train_args": {"model": "yolo26s.pt", "name": "signal_run", "project": "/tmp/signal"},
                    "model": _FakeCheckpointModel(scale="s", nc=2),
                    "optimizer": {"state": {}},
                },
                signal_target,
            )
            torch.save(
                {
                    "epoch": 98,
                    "best_fitness": 0.33,
                    "train_args": {"model": "yolo26n.pt", "name": "signal_old", "project": "/tmp/signal_old"},
                    "model": _FakeCheckpointModel(scale="n", nc=2),
                    "optimizer": {"state": {}},
                },
                signal_alias,
            )

            summary = audit_teacher_checkpoints(
                specs=(
                    TeacherCheckpointSpec(
                        teacher_name="mobility",
                        checkpoint_path=mobility_target,
                        alias_checkpoint_path=root / "mobility_alias" / "best.pt",
                    ),
                    TeacherCheckpointSpec(
                        teacher_name="signal",
                        checkpoint_path=signal_target,
                        alias_checkpoint_path=signal_alias,
                    ),
                )
            )

            by_teacher = {item["teacher_name"]: item for item in summary["teachers"]}
            self.assertTrue(summary["all_targets_match_expected_scale"])
            self.assertEqual(by_teacher["mobility"]["checkpoint"]["scale"], "s")
            self.assertFalse(by_teacher["mobility"]["alias_checkpoint"]["exists"])
            self.assertEqual(by_teacher["signal"]["checkpoint"]["scale"], "s")
            self.assertEqual(by_teacher["signal"]["alias_checkpoint"]["scale"], "n")
            self.assertFalse(by_teacher["signal"]["alias_checkpoint"]["matches_expected_scale"])
            self.assertFalse(by_teacher["signal"]["alias_checkpoint"]["same_as_target"])

    def test_canonical_scene_to_overlay_scene_promotes_signal_detections_without_duplicates(self) -> None:
        scene = {
            "detections": [
                {"class_name": "traffic_light", "bbox": [10, 20, 30, 40]},
                {"class_name": "sign", "bbox": {"x1": 50, "y1": 60, "x2": 70, "y2": 80}},
                {"class_name": "obstacle", "bbox": [90, 100, 120, 150]},
            ],
            "traffic_lights": [
                {"bbox": [10, 20, 30, 40]},
            ],
            "lanes": [
                {"class_name": "white_lane", "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
            ],
            "crosswalks": [
                {"points": [[5, 6], [7, 8], [9, 10]]},
            ],
        }

        overlay_scene = canonical_scene_to_overlay_scene(scene, image_path=Path("/tmp/source.jpg"))

        self.assertEqual(overlay_scene["image"]["source_path"], str(Path("/tmp/source.jpg").resolve()))
        self.assertEqual(len(overlay_scene["traffic_lights"]), 1)
        self.assertEqual(len(overlay_scene["traffic_signs"]), 1)
        self.assertEqual(overlay_scene["detections"], [{"class_name": "obstacle", "bbox": [90.0, 100.0, 120.0, 150.0]}])
        self.assertEqual(overlay_scene["lanes"][0]["points"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(overlay_scene["crosswalks"][0]["points"]), 3)

    def test_render_review_bundle_writes_index_for_requested_dataset_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "meta" / "final_dataset_manifest.json"
            output_root = root / "review"
            samples = []
            dataset_keys = [
                "pv26_exhaustive_bdd100k_det_100k",
                "pv26_exhaustive_aihub_traffic_seoul",
                "pv26_exhaustive_aihub_obstacle_seoul",
                "aihub_lane_seoul",
            ]
            for dataset_key in dataset_keys:
                sample_id = f"{dataset_key}_sample"
                scene_path = root / "labels_scene" / "val" / f"{sample_id}.json"
                image_path = root / "images" / "val" / f"{sample_id}.jpg"
                _write_text(image_path, "img")
                _write_json(
                    scene_path,
                    {
                        "image": {"file_name": image_path.name, "width": 640, "height": 480},
                        "source": {"dataset": dataset_key, "split": "val"},
                        "detections": [{"class_name": "vehicle", "bbox": [10, 20, 30, 40]}],
                        "traffic_lights": [],
                        "traffic_signs": [],
                        "lanes": [],
                        "stop_lines": [],
                        "crosswalks": [],
                    },
                )
                samples.append(
                    {
                        "final_sample_id": sample_id,
                        "source_dataset_key": dataset_key,
                        "split": "val",
                        "scene_path": str(scene_path),
                        "image_path": str(image_path),
                    }
                )
            _write_json(
                manifest_path,
                {
                    "version": "pv26-exhaustive-od-lane-v2",
                    "samples": samples,
                },
            )

            def _fake_render(scene: dict, output_path: Path) -> None:
                self.assertIn("image", scene)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"png")

            with patch("tools.od_bootstrap.build.review.render_overlay", side_effect=_fake_render):
                summary = render_review_bundle(
                    manifest_path=manifest_path,
                    output_root=output_root,
                    quotas={dataset_key: 1 for dataset_key in dataset_keys},
                )

            self.assertEqual(summary["image_count"], 4)
            self.assertTrue((output_root / "index.json").is_file())
            index_payload = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index_payload["entries"]), 4)

    def test_select_review_rows_applies_seeded_dataset_sampling(self) -> None:
        samples = []
        dataset_key = "pv26_exhaustive_bdd100k_det_100k"
        for index in range(6):
            samples.append(
                {
                    "final_sample_id": f"{dataset_key}_sample_{index}",
                    "source_dataset_key": dataset_key,
                    "split": "val",
                    "scene_path": f"/tmp/{dataset_key}_{index}.json",
                    "image_path": f"/tmp/{dataset_key}_{index}.jpg",
                }
            )
        manifest = {"samples": samples}

        default_rows = select_review_rows(manifest, quotas={dataset_key: 3})
        seeded_rows_a = select_review_rows(manifest, quotas={dataset_key: 3}, seed=1)
        seeded_rows_b = select_review_rows(manifest, quotas={dataset_key: 3}, seed=1)
        seeded_rows_c = select_review_rows(manifest, quotas={dataset_key: 3}, seed=2)

        self.assertEqual(
            [row["final_sample_id"] for row in default_rows],
            [
                f"{dataset_key}_sample_0",
                f"{dataset_key}_sample_1",
                f"{dataset_key}_sample_2",
            ],
        )
        self.assertEqual(
            [row["final_sample_id"] for row in seeded_rows_a],
            [row["final_sample_id"] for row in seeded_rows_b],
        )
        self.assertNotEqual(
            [row["final_sample_id"] for row in default_rows],
            [row["final_sample_id"] for row in seeded_rows_a],
        )
        self.assertNotEqual(
            [row["final_sample_id"] for row in seeded_rows_a],
            [row["final_sample_id"] for row in seeded_rows_c],
        )


if __name__ == "__main__":
    unittest.main()
