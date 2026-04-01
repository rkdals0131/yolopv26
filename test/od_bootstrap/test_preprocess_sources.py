from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.od_bootstrap.preprocess.run_build_teacher_datasets import main as build_teacher_datasets_main
from tools.od_bootstrap.preprocess.run_prepare_sources import DEFAULT_CONFIG_PATH, _resolve_config, main as prepare_sources_main
from tools.od_bootstrap.preprocess.sources import (
    AIHUB_LANE_DIRNAME,
    AIHUB_OBSTACLE_DIRNAME,
    AIHUB_TRAFFIC_DIRNAME,
    SourcePrepConfig,
    SourceRoots,
    prepare_od_bootstrap_sources,
)


class ODBootstrapSourcePrepTests(unittest.TestCase):
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
                    "tools.od_bootstrap.preprocess.sources.run_bdd_standardization",
                    return_value=fake_bdd_outputs,
                ) as mock_bdd,
                patch(
                    "tools.od_bootstrap.preprocess.sources.run_aihub_standardization",
                    return_value=fake_aihub_outputs,
                ) as mock_aihub,
            ):
                result = prepare_od_bootstrap_sources(config)

            self.assertEqual(result.bundle.bootstrap_source_keys, ("bdd100k_det_100k", "aihub_traffic_seoul", "aihub_obstacle_seoul"))
            self.assertEqual(result.bundle.excluded_source_keys, ("aihub_lane_seoul",))
            self.assertEqual(json.loads(result.manifest_path.read_text(encoding="utf-8"))["bootstrap_source_keys"], ["bdd100k_det_100k", "aihub_traffic_seoul", "aihub_obstacle_seoul"])
            self.assertTrue(result.image_list_manifest_path.is_file())

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
                patch("tools.od_bootstrap.preprocess.sources.run_bdd_standardization", return_value=fake_bdd_outputs),
                patch("tools.od_bootstrap.preprocess.sources.run_aihub_standardization", return_value=fake_aihub_outputs),
            ):
                result = prepare_od_bootstrap_sources(config)

            bdd_manifest = json.loads(
                result.canonical_debug_vis_manifest_paths["bdd100k_det_100k"].read_text(encoding="utf-8")
            )
            aihub_manifest = json.loads(
                result.canonical_debug_vis_manifest_paths["aihub_standardized"].read_text(encoding="utf-8")
            )
            self.assertEqual(bdd_manifest["selection_count"], 1)
            self.assertEqual(aihub_manifest["selection_count"], 1)
            bdd_debug_vis_dir = bdd_canonical_root / "meta" / "debug_vis"
            aihub_debug_vis_dir = aihub_canonical_root / "meta" / "debug_vis"
            self.assertEqual(len(sorted(bdd_debug_vis_dir.glob("*.png"))), 1)
            self.assertEqual(len(sorted(aihub_debug_vis_dir.glob("*.png"))), 1)
            self.assertEqual(len(list(bdd_debug_vis_dir.iterdir())), 1)
            self.assertEqual(len(list(aihub_debug_vis_dir.iterdir())), 1)
            self.assertTrue(Path(bdd_manifest["items"][0]["overlay_path"]).is_file())
            self.assertTrue(Path(aihub_manifest["items"][0]["overlay_path"]).is_file())

    def test_default_prepare_sources_config_resolves_repo_seg_dataset_paths(self) -> None:
        args = SimpleNamespace(
            config=DEFAULT_CONFIG_PATH,
            bdd_root=None,
            aihub_root=None,
            output_root=None,
            workers=None,
            force_reprocess=False,
        )
        config = _resolve_config(args)
        repo_root = Path(__file__).resolve().parents[2]
        self.assertEqual(config.roots.bdd_root, (repo_root / "seg_dataset" / "BDD100K").resolve())
        self.assertEqual(config.roots.aihub_root, (repo_root / "seg_dataset" / "AIHUB").resolve())
        self.assertEqual(config.output_root, (repo_root / "seg_dataset" / "pv26_od_bootstrap").resolve())

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

        with (
            patch("tools.od_bootstrap.preprocess.run_prepare_sources.prepare_od_bootstrap_sources") as mock_prepare,
            patch("tools.od_bootstrap.preprocess.run_build_teacher_datasets.build_teacher_datasets") as mock_build,
        ):
            mock_prepare.side_effect = _fake_prepare
            mock_build.side_effect = _fake_build
            prepare_sources_main([])
            build_teacher_datasets_main([])

        repo_root = Path(__file__).resolve().parents[2]
        prepare_config = captured["prepare_config"]
        build_call = captured["build_call"]
        self.assertEqual(prepare_config.roots.bdd_root, (repo_root / "seg_dataset" / "BDD100K").resolve())
        self.assertEqual(prepare_config.output_root, (repo_root / "seg_dataset" / "pv26_od_bootstrap").resolve())
        self.assertEqual(build_call["bundle"].output_root, (repo_root / "seg_dataset" / "pv26_od_bootstrap").resolve())
        self.assertEqual(build_call["output_root"], (repo_root / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets").resolve())
        self.assertEqual(build_call["workers"], 8)
        self.assertEqual(build_call["log_every"], 500)
        self.assertEqual(build_call["debug_vis_count"], 20)
        self.assertEqual(build_call["debug_vis_seed"], 26)
        self.assertIsNotNone(build_call["log_fn"])

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
