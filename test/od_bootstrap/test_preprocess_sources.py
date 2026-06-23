from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.od_bootstrap import main as od_bootstrap_main
from tools.od_bootstrap.cli import _load_json as _load_cli_json
from tools.od_bootstrap.source.prepare import (
    AIHUB_LANE_DIRNAME,
    AIHUB_OBSTACLE_DIRNAME,
    AIHUB_TRAFFIC_DIRNAME,
    prepare_od_bootstrap_sources,
)
from tools.od_bootstrap.source.types import SourcePrepConfig, SourceRoots
from tools.od_bootstrap.presets import build_default_source_preset


class ODBootstrapSourcePrepTests(unittest.TestCase):
    def test_cli_load_json_rejects_non_mapping_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payload.json"
            self._write_text(path, "[]\n")

            with self.assertRaisesRegex(TypeError, "JSON root must be a mapping"):
                _load_cli_json(path)

    def test_prepare_od_bootstrap_sources_calls_existing_canonicalizers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "BDD100K"
            aihub_root = root / "AIHUB"
            output_root = root / "od_bootstrap"
            for path in [
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
                aihub_root / AIHUB_LANE_DIRNAME,
                aihub_root / AIHUB_OBSTACLE_DIRNAME,
                aihub_root / AIHUB_TRAFFIC_DIRNAME,
                aihub_root / "docs",
            ]:
                path.mkdir(parents=True, exist_ok=True)

            config = SourcePrepConfig(
                roots=SourceRoots(
                    bdd_root=bdd_root,
                    bdd_images_root=bdd_root / "bdd100k_images_100k" / "100k",
                    bdd_labels_root=bdd_root / "bdd100k_labels" / "100k",
                    aihub_root=aihub_root,
                ),
                output_root=output_root,
                workers=3,
                force_reprocess=True,
                write_source_readmes=False,
                debug_vis_count=0,
            )

            fake_bdd_outputs = {"output_root": output_root / "canonical" / "bdd100k_det_100k"}
            fake_aihub_outputs = {"output_root": output_root / "canonical" / "aihub_standardized"}
            with (
                patch(
                    "tools.od_bootstrap.source.prepare.run_bdd_standardization",
                    return_value=fake_bdd_outputs,
                ) as mock_bdd,
                patch(
                    "tools.od_bootstrap.source.prepare.run_aihub_standardization",
                    return_value=fake_aihub_outputs,
                ) as mock_aihub,
            ):
                result = prepare_od_bootstrap_sources(config)

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.bundle.bootstrap_source_keys, ("bdd100k_det_100k", "aihub_traffic_seoul", "aihub_obstacle_seoul"))
            self.assertEqual(result.bundle.excluded_source_keys, ("aihub_lane_seoul",))
            self.assertEqual(manifest["bootstrap_source_keys"], ["bdd100k_det_100k", "aihub_traffic_seoul", "aihub_obstacle_seoul"])
            self.assertEqual(
                manifest["raw_roots"],
                {
                    "bdd_root": str(bdd_root.resolve()),
                    "aihub_root": str(aihub_root.resolve()),
                    "aihub_lane_root": str((aihub_root / AIHUB_LANE_DIRNAME).resolve()),
                    "aihub_obstacle_root": str((aihub_root / AIHUB_OBSTACLE_DIRNAME).resolve()),
                    "aihub_traffic_root": str((aihub_root / AIHUB_TRAFFIC_DIRNAME).resolve()),
                },
            )
            self.assertEqual(
                manifest["canonical_roots"],
                {
                    "bdd_root": str((output_root / "canonical" / "bdd100k_det_100k").resolve()),
                    "aihub_root": str((output_root / "canonical" / "aihub_standardized").resolve()),
                },
            )
            self.assertTrue(result.image_list_manifest_path.is_file())
            self.assertEqual(
                sorted(result.canonical_debug_vis_manifest_paths.keys()),
                ["aihub_standardized", "bdd100k_det_100k"],
            )

            mock_bdd.assert_called_once()
            mock_aihub.assert_called_once()
            self.assertEqual(mock_bdd.call_args.kwargs["bdd_root"], bdd_root.resolve())
            self.assertEqual(mock_bdd.call_args.kwargs["output_root"], (output_root / "canonical" / "bdd100k_det_100k").resolve())
            self.assertEqual(mock_aihub.call_args.kwargs["lane_root"], (aihub_root / AIHUB_LANE_DIRNAME).resolve())
            self.assertEqual(mock_aihub.call_args.kwargs["obstacle_root"], (aihub_root / AIHUB_OBSTACLE_DIRNAME).resolve())
            self.assertEqual(mock_aihub.call_args.kwargs["traffic_root"], (aihub_root / AIHUB_TRAFFIC_DIRNAME).resolve())
            self.assertFalse(mock_aihub.call_args.kwargs["write_dataset_readmes"])

    def test_prepare_od_bootstrap_sources_writes_flat_debug_vis_under_each_canonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bdd_root = root / "BDD100K"
            aihub_root = root / "AIHUB"
            output_root = root / "od_bootstrap"
            for path in [
                bdd_root / "bdd100k_images_100k" / "100k",
                bdd_root / "bdd100k_labels" / "100k",
                aihub_root / AIHUB_LANE_DIRNAME,
                aihub_root / AIHUB_OBSTACLE_DIRNAME,
                aihub_root / AIHUB_TRAFFIC_DIRNAME,
                aihub_root / "docs",
            ]:
                path.mkdir(parents=True, exist_ok=True)

            canonical_root = output_root / "canonical"
            bdd_canonical_root = canonical_root / "bdd100k_det_100k"
            aihub_canonical_root = canonical_root / "aihub_standardized"
            self._make_image(bdd_canonical_root / "images" / "val" / "bdd_val_001.jpg", 64, 48, "#222222")
            self._make_image(aihub_canonical_root / "images" / "val" / "traffic_val_001.png", 64, 48, "#444444")
            self._make_image(aihub_canonical_root / "images" / "val" / "lane_val_001.png", 64, 48, "#555555")
            self._write_json(
                bdd_canonical_root / "labels_scene" / "val" / "bdd_val_001.json",
                {
                    "image": {"file_name": "bdd_val_001.jpg", "width": 64, "height": 48},
                    "source": {"dataset": "bdd100k_det_100k", "split": "val"},
                    "detections": [{"class_name": "vehicle", "bbox": [10, 10, 30, 30]}],
                },
            )
            self._write_text(
                bdd_canonical_root / "labels_det" / "val" / "bdd_val_001.txt",
                "0 0.312500 0.416667 0.312500 0.416667\n",
            )
            self._write_json(
                aihub_canonical_root / "labels_scene" / "val" / "traffic_val_001.json",
                {
                    "image": {"file_name": "traffic_val_001.png", "width": 64, "height": 48},
                    "source": {"dataset": "aihub_traffic_seoul", "split": "val"},
                    "traffic_lights": [{"bbox": [20, 5, 28, 18]}],
                },
            )
            self._write_json(
                aihub_canonical_root / "labels_scene" / "val" / "lane_val_001.json",
                {
                    "image": {"file_name": "lane_val_001.png", "width": 64, "height": 48},
                    "source": {"dataset": "aihub_lane_seoul", "split": "val"},
                    "lanes": [{"class_name": "white_lane", "points": [[1, 2], [3, 4]]}],
                },
            )
            self._write_text(
                aihub_canonical_root / "labels_det" / "val" / "traffic_val_001.txt",
                "5 0.375000 0.239583 0.125000 0.270833\n",
            )

            config = SourcePrepConfig(
                roots=SourceRoots(
                    bdd_root=bdd_root,
                    bdd_images_root=bdd_root / "bdd100k_images_100k" / "100k",
                    bdd_labels_root=bdd_root / "bdd100k_labels" / "100k",
                    aihub_root=aihub_root,
                ),
                output_root=output_root,
                workers=1,
                force_reprocess=False,
                write_source_readmes=False,
                debug_vis_count=2,
                debug_vis_seed=26,
            )

            fake_bdd_outputs = {"output_root": bdd_canonical_root}
            fake_aihub_outputs = {"output_root": aihub_canonical_root}
            with (
                patch("tools.od_bootstrap.source.prepare.run_bdd_standardization", return_value=fake_bdd_outputs),
                patch("tools.od_bootstrap.source.prepare.run_aihub_standardization", return_value=fake_aihub_outputs),
            ):
                result = prepare_od_bootstrap_sources(config)

            image_list_rows = [
                json.loads(line)
                for line in result.image_list_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                sorted(row["dataset_key"] for row in image_list_rows),
                ["aihub_traffic_seoul", "bdd100k_det_100k"],
            )
            self.assertNotIn("aihub_lane_seoul", {row["dataset_key"] for row in image_list_rows})
            rows_by_key = {row["dataset_key"]: row for row in image_list_rows}
            self.assertEqual(rows_by_key["bdd100k_det_100k"]["sample_uid"], "bdd100k_det_100k__val__bdd_val_001")
            self.assertEqual(rows_by_key["bdd100k_det_100k"]["sample_id"], "bdd_val_001")
            self.assertEqual(rows_by_key["bdd100k_det_100k"]["split"], "val")
            self.assertEqual(rows_by_key["bdd100k_det_100k"]["source_name"], "bdd100k_det_100k")
            self.assertEqual(rows_by_key["bdd100k_det_100k"]["dataset_root"], str(bdd_canonical_root.resolve()))
            self.assertEqual(
                rows_by_key["bdd100k_det_100k"]["det_path"],
                str((bdd_canonical_root / "labels_det" / "val" / "bdd_val_001.txt").resolve()),
            )
            self.assertEqual(
                rows_by_key["aihub_traffic_seoul"]["sample_uid"],
                "aihub_traffic_seoul__val__traffic_val_001",
            )
            self.assertEqual(rows_by_key["aihub_traffic_seoul"]["sample_id"], "traffic_val_001")
            self.assertEqual(rows_by_key["aihub_traffic_seoul"]["split"], "val")
            self.assertEqual(rows_by_key["aihub_traffic_seoul"]["source_name"], "aihub_standardized")
            self.assertEqual(rows_by_key["aihub_traffic_seoul"]["dataset_root"], str(aihub_canonical_root.resolve()))
            self.assertEqual(
                rows_by_key["aihub_traffic_seoul"]["det_path"],
                str((aihub_canonical_root / "labels_det" / "val" / "traffic_val_001.txt").resolve()),
            )

            bdd_manifest = json.loads(
                result.canonical_debug_vis_manifest_paths["bdd100k_det_100k"].read_text(encoding="utf-8")
            )
            aihub_manifest = json.loads(
                result.canonical_debug_vis_manifest_paths["aihub_standardized"].read_text(encoding="utf-8")
            )
            self.assertEqual(bdd_manifest["selection_count"], 1)
            self.assertEqual(aihub_manifest["selection_count"], 1)
            self.assertEqual(aihub_manifest["items"][0]["dataset_key"], "aihub_traffic_seoul")
            bdd_debug_vis_dir = bdd_canonical_root / "meta" / "debug_vis"
            aihub_debug_vis_dir = aihub_canonical_root / "meta" / "debug_vis"
            self.assertEqual(len(sorted(bdd_debug_vis_dir.glob("*.png"))), 1)
            self.assertEqual(len(sorted(aihub_debug_vis_dir.glob("*.png"))), 1)
            self.assertEqual(len(list(bdd_debug_vis_dir.iterdir())), 1)
            self.assertEqual(len(list(aihub_debug_vis_dir.iterdir())), 1)
            self.assertTrue(Path(bdd_manifest["items"][0]["overlay_path"]).is_file())
            self.assertTrue(Path(aihub_manifest["items"][0]["overlay_path"]).is_file())

    def test_default_prepare_sources_preset_resolves_repo_seg_dataset_paths(self) -> None:
        config = build_default_source_preset()
        self.assertTrue(str(config.roots.bdd_root).endswith("/seg_dataset/BDD100K"))
        self.assertTrue(str(config.roots.aihub_root).endswith("/seg_dataset/AIHUB"))
        self.assertTrue(str(config.output_root).endswith("/seg_dataset/pv26_od_bootstrap"))

    def test_default_prepare_sources_preset_reads_shared_user_config_yaml(self) -> None:
        with patch(
            "tools.od_bootstrap.presets.load_user_paths_config",
            return_value={
                "od_bootstrap": {
                    "raw_sources": {
                        "bdd_root": "custom_data/bdd",
                        "bdd_images_root": "custom_data/bdd/images",
                        "bdd_labels_root": "custom_data/bdd/labels",
                        "aihub_root": "custom_data/aihub",
                    },
                    "outputs": {
                        "bootstrap_root": "custom_outputs/bootstrap",
                    },
                }
            },
        ), patch(
            "tools.od_bootstrap.presets.load_user_hyperparameters_config",
            return_value={
                "od_bootstrap": {
                    "source_prep": {
                        "workers": 7,
                        "force_reprocess": True,
                        "write_source_readmes": True,
                        "debug_vis_count": 12,
                        "debug_vis_seed": 99,
                    }
                }
            },
        ):
            config = build_default_source_preset()

        self.assertTrue(str(config.roots.bdd_root).endswith("/custom_data/bdd"))
        self.assertTrue(str(config.roots.bdd_images_root).endswith("/custom_data/bdd/images"))
        self.assertTrue(str(config.roots.bdd_labels_root).endswith("/custom_data/bdd/labels"))
        self.assertTrue(str(config.roots.aihub_root).endswith("/custom_data/aihub"))
        self.assertTrue(str(config.output_root).endswith("/custom_outputs/bootstrap"))
        self.assertEqual(config.workers, 7)
        self.assertTrue(config.force_reprocess)
        self.assertTrue(config.write_source_readmes)
        self.assertEqual(config.debug_vis_count, 12)
        self.assertEqual(config.debug_vis_seed, 99)

    def test_entrypoints_use_default_configs_without_overrides(self) -> None:
        captured: dict[str, object] = {}

        def _fake_prepare(config):
            captured["prepare_config"] = config
            return SimpleNamespace(
                bundle=SimpleNamespace(
                    bdd_root=Path("/tmp/bdd"),
                    aihub_root=Path("/tmp/aihub"),
                    output_root=Path("/tmp/out"),
                    bootstrap_source_keys=("bdd100k_det_100k",),
                    excluded_source_keys=("aihub_lane_seoul",),
                ),
                manifest_path=Path("/tmp/source_prep_manifest.json"),
                image_list_manifest_path=Path("/tmp/bootstrap_image_list.jsonl"),
                canonical_debug_vis_manifest_paths={"bdd100k_det_100k": Path("/tmp/bdd_debug_vis_manifest.json")},
                bdd_outputs={"output_root": Path("/tmp/bdd")},
                aihub_outputs={"output_root": Path("/tmp/aihub")},
            )

        def _fake_build(
            bundle,
            output_root,
            copy_images=False,
            workers=1,
            log_every=250,
            debug_vis_count=0,
            debug_vis_seed=26,
            log_fn=None,
        ):
            captured["build_call"] = {
                "bundle": bundle,
                "output_root": output_root,
                "copy_images": copy_images,
                "workers": workers,
                "log_every": log_every,
                "debug_vis_count": debug_vis_count,
                "debug_vis_seed": debug_vis_seed,
                "log_fn": log_fn,
            }
            return {
                "mobility": SimpleNamespace(
                    dataset_root=Path("/tmp/mobility"),
                    manifest_path=Path("/tmp/mobility_manifest.json"),
                    debug_vis_manifest_path=Path("/tmp/mobility_debug_vis_manifest.json"),
                    sample_count=1,
                    detection_count=1,
                    class_counts={"vehicle": 1},
                )
            }

        def _fake_signal_attr_materialize(canonical_root: Path, output_root: Path, **kwargs):
            captured["signal_attr_canonical_root"] = canonical_root
            captured["signal_attr_output_root"] = output_root
            captured["signal_attr_workers"] = kwargs.get("workers")
            captured["signal_attr_log_fn"] = kwargs.get("log_fn")
            return {"status": "ready", "accepted_count": 1, "rejected_count": 0}

        with (
            patch("tools.od_bootstrap.cli.prepare_od_bootstrap_sources") as mock_prepare,
            patch("tools.od_bootstrap.cli.build_teacher_datasets") as mock_build,
            patch(
                "tools.od_bootstrap.cli.materialize_aihub_signal_attr_crop_dataset_from_canonical_root",
                side_effect=_fake_signal_attr_materialize,
            ),
        ):
            mock_prepare.side_effect = _fake_prepare
            mock_build.side_effect = _fake_build
            od_bootstrap_main(["prepare-sources"])
            od_bootstrap_main(["build-teacher-datasets"])

        repo_root = Path(__file__).resolve().parents[2]
        prepare_config = captured["prepare_config"]
        build_call = captured["build_call"]
        self.assertTrue(str(prepare_config.roots.bdd_root).endswith("/seg_dataset/BDD100K"))
        self.assertTrue(str(prepare_config.output_root).endswith("/seg_dataset/pv26_od_bootstrap"))
        self.assertTrue(str(build_call["bundle"].output_root).endswith("/seg_dataset/pv26_od_bootstrap"))
        self.assertTrue(str(build_call["output_root"]).endswith("/seg_dataset/pv26_od_bootstrap/teacher_datasets"))
        self.assertEqual(build_call["workers"], 8)
        self.assertEqual(build_call["log_every"], 500)
        self.assertEqual(build_call["debug_vis_count"], 20)
        self.assertEqual(build_call["debug_vis_seed"], 26)
        self.assertIsNotNone(build_call["log_fn"])
        self.assertTrue(str(captured["signal_attr_canonical_root"]).endswith("/seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized"))
        self.assertTrue(str(captured["signal_attr_output_root"]).endswith("/seg_dataset/pv26_od_bootstrap/teacher_datasets/signal_attr"))
        self.assertEqual(captured["signal_attr_workers"], 8)
        self.assertIsNotNone(captured["signal_attr_log_fn"])

    def test_signal_attr_dataset_entrypoint_uses_canonical_root(self) -> None:
        captured: dict[str, object] = {}

        def _fake_materialize(canonical_root: Path, output_root: Path, **kwargs):
            captured["canonical_root"] = canonical_root
            captured["output_root"] = output_root
            captured["log_fn"] = kwargs.get("log_fn")
            return {"status": "ready", "accepted_count": 1, "rejected_count": 0}

        with patch(
            "tools.od_bootstrap.cli.materialize_aihub_signal_attr_crop_dataset_from_canonical_root",
            side_effect=_fake_materialize,
        ):
            od_bootstrap_main(["build-signal-attr-dataset"])

        self.assertTrue(str(captured["canonical_root"]).endswith("/seg_dataset/pv26_od_bootstrap/canonical/aihub_standardized"))
        self.assertTrue(str(captured["output_root"]).endswith("/seg_dataset/pv26_od_bootstrap/teacher_datasets/signal_attr"))
        self.assertIsNotNone(captured["log_fn"])

    def test_signal_attr_train_and_eval_entrypoints_use_default_roots(self) -> None:
        train_calls: list[dict[str, object]] = []
        eval_calls: list[dict[str, object]] = []

        def _fake_train(dataset_root, output_root, *, train_config, model_config, threshold_policy=None, log_fn=None):
            train_calls.append(
                {
                    "dataset_root": dataset_root,
                    "output_root": output_root,
                    "train_config": train_config,
                    "model_config": model_config,
                    "log_fn": log_fn,
                }
            )
            return {"status": "ok"}

        def _fake_eval(dataset_root, checkpoint_path, output_root, *, split, batch_size, device, num_workers, log_fn=None):
            eval_calls.append(
                {
                    "dataset_root": dataset_root,
                    "checkpoint_path": checkpoint_path,
                    "output_root": output_root,
                    "split": split,
                    "log_fn": log_fn,
                }
            )
            return {"status": "ok"}

        with (
            patch("tools.od_bootstrap.cli.train_signal_attr_classifier", side_effect=_fake_train),
            patch("tools.od_bootstrap.cli.evaluate_signal_attr_checkpoint", side_effect=_fake_eval),
        ):
            od_bootstrap_main(
                [
                    "train",
                    "--teacher",
                    "signal_attr",
                    "--epochs",
                    "1",
                    "--batch",
                    "2",
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--no-pin-memory",
                    "--no-persistent-workers",
                    "--prefetch-factor",
                    "2",
                ]
            )
            od_bootstrap_main(["eval", "--teacher", "signal_attr", "--split", "val", "--batch", "2", "--device", "cpu", "--num-workers", "0"])
            od_bootstrap_main(["train-signal-attr", "--epochs", "1", "--batch", "2", "--device", "cpu", "--num-workers", "0"])
            od_bootstrap_main(["eval-signal-attr", "--split", "val", "--batch", "2", "--device", "cpu", "--num-workers", "0"])

        self.assertEqual(len(train_calls), 2)
        self.assertEqual(len(eval_calls), 2)
        for index, call in enumerate(train_calls):
            self.assertTrue(str(call["dataset_root"]).endswith("/seg_dataset/pv26_od_bootstrap/teacher_datasets/signal_attr"))
            self.assertTrue(str(call["output_root"]).endswith("/runs/od_bootstrap/train/signal_attr"))
            self.assertEqual(call["train_config"].epochs, 1)
            self.assertEqual(call["train_config"].batch_size, 2)
            if index == 0:
                self.assertFalse(call["train_config"].pin_memory)
                self.assertFalse(call["train_config"].persistent_workers)
                self.assertEqual(call["train_config"].prefetch_factor, 2)
            self.assertIsNotNone(call["log_fn"])
        for call in eval_calls:
            self.assertTrue(str(call["dataset_root"]).endswith("/seg_dataset/pv26_od_bootstrap/teacher_datasets/signal_attr"))
            self.assertTrue(str(call["checkpoint_path"]).endswith("/runs/od_bootstrap/train/signal_attr/best_signal_attr.pt"))
            self.assertTrue(str(call["output_root"]).endswith("/runs/od_bootstrap/eval/signal_attr"))
            self.assertEqual(call["split"], "val")
            self.assertIsNotNone(call["log_fn"])

    def test_exhaustive_od_entrypoint_auto_discovers_signal_attr_sidecar(self) -> None:
        captured_paths: list[Path | None] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_root = root / "train"
            eval_root = root / "eval"
            checkpoint_path = train_root / "signal_attr" / "best_signal_attr.pt"
            eval_report_path = eval_root / "signal_attr" / "signal_attr_eval_report.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eval_report_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")
            eval_report_path.write_text("{}\n", encoding="utf-8")

            def _fake_run_sweep(scenario, *, scenario_path, signal_attr_checkpoint_path=None):
                captured_paths.append(signal_attr_checkpoint_path)
                return {"status": "ok"}

            with (
                patch("tools.od_bootstrap.cli.build_sweep_preset", return_value=SimpleNamespace(run=SimpleNamespace(output_root=root / "sweep"))),
                patch("tools.od_bootstrap.cli.build_teacher_train_preset", return_value=SimpleNamespace(run=SimpleNamespace(output_root=train_root))),
                patch("tools.od_bootstrap.cli.build_teacher_eval_preset", return_value=SimpleNamespace(run=SimpleNamespace(output_root=eval_root))),
                patch("tools.od_bootstrap.cli.run_model_centric_sweep_scenario", side_effect=_fake_run_sweep),
            ):
                od_bootstrap_main(["build-exhaustive-od"])
                od_bootstrap_main(["build-exhaustive-od", "--no-signal-attr-sidecar"])

        self.assertEqual(captured_paths[0], checkpoint_path.resolve())
        self.assertIsNone(captured_paths[1])

    def test_lane_val_odpseudo_entrypoint_generates_missing_sample_results_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "lane_val_eval"
            checkpoint_path = root / "weights" / "best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
            calls: dict[str, object] = {}

            def _fake_generate(**kwargs):
                calls["generate"] = kwargs
                kwargs["sample_results_path"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["sample_results_path"].write_text("{}\n", encoding="utf-8")
                return {
                    "sample_results_path": str(kwargs["sample_results_path"]),
                    "sample_count": 1,
                    "accepted_detection_count": 0,
                    "rejected_candidate_count": 0,
                    "prediction_count_by_teacher": {},
                    "output_root": str(output_root),
                    "sample_results_manifest_path": str(output_root / "meta" / "sample_results_manifest.json"),
                    "run_id": "test_run",
                }

            def _fake_load(path: Path):
                calls["load_path"] = path
                return [{"sample_id": "lane_a"}]

            def _fake_build(**kwargs):
                calls["build"] = kwargs
                return {
                    "output_root": str(kwargs["output_root"]),
                    "manifest_path": str(kwargs["output_root"] / "meta" / "final_dataset_manifest.json"),
                    "rejected_candidates_path": str(kwargs["output_root"] / "meta" / "rejected_detections.jsonl"),
                    "sample_count": 1,
                    "nonfinite_candidate_count": 0,
                }

            sweep = SimpleNamespace(
                run=SimpleNamespace(output_root=root / "runs", device="cpu", batch_size=2),
                teachers=(
                    SimpleNamespace(name="mobility", checkpoint_path=checkpoint_path),
                    SimpleNamespace(name="signal", checkpoint_path=checkpoint_path),
                    SimpleNamespace(name="obstacle", checkpoint_path=checkpoint_path),
                ),
                class_policy={},
            )
            with (
                patch("tools.od_bootstrap.cli.build_teacher_dataset_preset", return_value=SimpleNamespace(canonical_root=root / "bootstrap")),
                patch("tools.od_bootstrap.cli.build_sweep_preset", return_value=sweep),
                patch("tools.od_bootstrap.cli.run_lane_val_odpseudo_teacher_sample_results", side_effect=_fake_generate),
                patch("tools.od_bootstrap.cli._load_lane_val_sample_results", side_effect=_fake_load),
                patch("tools.od_bootstrap.cli.build_lane_val_odpseudo_eval_root", side_effect=_fake_build),
            ):
                self.assertEqual(
                    od_bootstrap_main(
                        [
                            "build-lane-val-odpseudo",
                            "--output-root",
                            str(output_root),
                            "--expected-base-count",
                            "1",
                        ]
                    ),
                    0,
                )

            self.assertEqual(calls["load_path"], output_root / "meta" / "sample_results.jsonl")
            self.assertEqual(calls["generate"]["expected_base_count"], 1)
            self.assertEqual(calls["build"]["sample_results"], [{"sample_id": "lane_a"}])

    def test_calibration_entrypoint_passes_preset_scenario_path(self) -> None:
        with patch("tools.od_bootstrap.cli.calibrate_class_policy_scenario") as mock_calibrate:
            self.assertEqual(od_bootstrap_main(["calibrate"]), 0)

        mock_calibrate.assert_called_once()
        self.assertEqual(mock_calibrate.call_args.kwargs["scenario_path"], Path("preset_calibration"))

    @staticmethod
    def _make_image(path: Path, width: int, height: int, color: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (width, height), color).save(path)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_text(path: Path, contents: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
