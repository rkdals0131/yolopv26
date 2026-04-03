from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.od_bootstrap.build.debug_vis import generate_canonical_debug_vis
from tools.od_bootstrap.build.image_list import ImageListEntry, build_sample_uid, write_image_list


def _make_image(path: Path, width: int, height: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


class ODBootstrapBuildDebugVisTests(unittest.TestCase):
    def test_generate_canonical_debug_vis_writes_per_dataset_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical_root = root / "canonical"
            bdd_root = canonical_root / "bdd100k_det_100k"
            aihub_root = canonical_root / "aihub_standardized"

            bdd_entry = self._make_canonical_entry(
                dataset_root=bdd_root,
                dataset_key="bdd100k_det_100k",
                split="train",
                sample_id="bdd_train_001",
                image_name="bdd_train_001.jpg",
                color="#111111",
            )
            aihub_entry = self._make_canonical_entry(
                dataset_root=aihub_root,
                dataset_key="aihub_traffic_seoul",
                split="val",
                sample_id="aihub_traffic_val_001",
                image_name="aihub_traffic_val_001.png",
                color="#222222",
            )

            image_list_path = root / "image_list.jsonl"
            write_image_list(image_list_path, (bdd_entry, aihub_entry))

            outputs = generate_canonical_debug_vis(
                image_list_manifest_path=image_list_path,
                canonical_root=canonical_root,
                debug_vis_count=1,
                debug_vis_seed=7,
            )

            self.assertEqual(sorted(outputs), ["aihub_standardized", "bdd100k_det_100k"])
            for dataset_name, dataset_key in (
                ("bdd100k_det_100k", "bdd100k_det_100k"),
                ("aihub_standardized", "aihub_traffic_seoul"),
            ):
                manifest = json.loads(outputs[dataset_name]["debug_vis_manifest"].read_text(encoding="utf-8"))
                self.assertEqual(outputs[dataset_name]["selection_count"], 1)
                self.assertEqual(manifest["selection_count"], 1)
                self.assertEqual(manifest["items"][0]["dataset_key"], dataset_key)
                self.assertTrue(Path(manifest["items"][0]["overlay_path"]).is_file())

    def _make_canonical_entry(
        self,
        *,
        dataset_root: Path,
        dataset_key: str,
        split: str,
        sample_id: str,
        image_name: str,
        color: str,
    ) -> ImageListEntry:
        image_path = dataset_root / "images" / split / image_name
        scene_path = dataset_root / "labels_scene" / split / f"{sample_id}.json"
        _make_image(image_path, 640, 360, color)
        _write_json(
            scene_path,
            {
                "version": "test",
                "image": {
                    "file_name": image_name,
                    "width": 640,
                    "height": 360,
                },
                "source": {
                    "dataset": dataset_key,
                    "split": split,
                },
                "detections": [{"class_name": "vehicle", "bbox": [10, 20, 120, 160]}],
                "lanes": [{"class_name": "white_lane", "points": [[0, 0], [100, 100]]}],
            },
        )
        return ImageListEntry(
            sample_id=sample_id,
            sample_uid=build_sample_uid(dataset_key=dataset_key, split=split, sample_id=sample_id),
            image_path=image_path,
            scene_path=scene_path,
            dataset_root=dataset_root,
            dataset_key=dataset_key,
            split=split,
        )


if __name__ == "__main__":
    unittest.main()
