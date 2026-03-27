from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
