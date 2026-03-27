from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

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
                bdd_outputs={"output_root": Path("/tmp/bdd")},
                aihub_outputs={"output_root": Path("/tmp/aihub")},
            )

        def _fake_build(bundle, output_root, copy_images=False):
            captured["build_call"] = {"bundle": bundle, "output_root": output_root, "copy_images": copy_images}
            return {
                "mobility": SimpleNamespace(
                    dataset_root=Path("/tmp/mobility"),
                    manifest_path=Path("/tmp/mobility_manifest.json"),
                    sample_count=1,
                    detection_count=1,
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
