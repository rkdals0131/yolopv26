import tempfile
import unittest
from pathlib import Path

from pv26.dataset.wod_acquire import (
    WOD_DOWNLOAD_STATUS_DOWNLOADED,
    WOD_DOWNLOAD_STATUS_FAILED,
    WOD_DOWNLOAD_STATUS_RAW_DELETED,
    find_wod_acquire_context_entry,
    iter_contexts_for_download,
    parse_gcloud_storage_ls_long,
    reconcile_wod_acquire_state,
)


def _touch(path: Path, *, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class TestWodAcquireParsing(unittest.TestCase):
    def test_parse_gcloud_storage_ls_long_ignores_dirs_and_non_matching_lines(self):
        text = """
         6  2024-02-27T23:11:03Z  gs://waymo_open_dataset_v_2_0_1/training/camera_segmentation/
   1829960  2024-02-27T23:11:10Z  gs://waymo_open_dataset_v_2_0_1/training/camera_segmentation/10017090168044687777_6380_000_6400_000.parquet
TOTAL: 2 objects, 1829966 bytes (1.7 MiB)
"""
        items = parse_gcloud_storage_ls_long(text, component="camera_segmentation")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].context_name, "10017090168044687777_6380_000_6400_000")
        self.assertEqual(items[0].size_bytes, 1829960)


class TestWodAcquireState(unittest.TestCase):
    def test_reconcile_marks_downloaded_when_local_required_components_exist(self):
        remote_text_img = (
            "  100  2024-02-27T23:11:10Z  "
            "gs://waymo_open_dataset_v_2_0_1/training/camera_image/ctx_a.parquet\n"
        )
        remote_text_seg = (
            "  50  2024-02-27T23:11:10Z  "
            "gs://waymo_open_dataset_v_2_0_1/training/camera_segmentation/ctx_a.parquet\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp) / "training"
            _touch(training_root / "camera_image" / "ctx_a.parquet", size=100)
            _touch(training_root / "camera_segmentation" / "ctx_a.parquet", size=50)
            state = reconcile_wod_acquire_state(
                training_root=training_root,
                remote_objects_by_component={
                    "camera_image": parse_gcloud_storage_ls_long(remote_text_img, component="camera_image"),
                    "camera_segmentation": parse_gcloud_storage_ls_long(remote_text_seg, component="camera_segmentation"),
                    "camera_box": [],
                },
            )
            ctx = find_wod_acquire_context_entry(state, "ctx_a")
            self.assertTrue(ctx["remote_processable"])
            self.assertEqual(ctx["download_status"], WOD_DOWNLOAD_STATUS_DOWNLOADED)
            self.assertTrue(ctx["local_has_image"])
            self.assertTrue(ctx["local_has_segmentation"])

    def test_reconcile_preserves_raw_deleted_after_bulk_completion(self):
        prior_state = {
            "contexts": [
                {
                    "context_name": "ctx_done",
                    "camera_image_url": "gs://bucket/training/camera_image/ctx_done.parquet",
                    "camera_segmentation_url": "gs://bucket/training/camera_segmentation/ctx_done.parquet",
                    "remote_has_image": True,
                    "remote_has_segmentation": True,
                    "download_status": WOD_DOWNLOAD_STATUS_RAW_DELETED,
                }
            ]
        }
        bulk_state = {
            "contexts": [
                {
                    "context_name": "ctx_done",
                    "status": "completed",
                    "output_num_rows": 95,
                    "output_has_det_rows": 95,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            state = reconcile_wod_acquire_state(
                training_root=Path(tmp) / "training",
                prior_state=prior_state,
                bulk_state=bulk_state,
            )
        ctx = find_wod_acquire_context_entry(state, "ctx_done")
        self.assertEqual(ctx["download_status"], WOD_DOWNLOAD_STATUS_RAW_DELETED)
        self.assertEqual(ctx["bulk_status"], "completed")

    def test_iter_contexts_for_download_includes_failed_when_requested(self):
        state = {
            "contexts": [
                {
                    "context_name": "ctx_a",
                    "remote_processable": True,
                    "download_status": "remote_only",
                },
                {
                    "context_name": "ctx_b",
                    "remote_processable": True,
                    "download_status": WOD_DOWNLOAD_STATUS_FAILED,
                },
                {
                    "context_name": "ctx_c",
                    "remote_processable": True,
                    "download_status": WOD_DOWNLOAD_STATUS_DOWNLOADED,
                },
            ]
        }
        only_pending = iter_contexts_for_download(state, include_failed=False)
        self.assertEqual([row["context_name"] for row in only_pending], ["ctx_a"])
        with_failed = iter_contexts_for_download(state, include_failed=True)
        self.assertEqual([row["context_name"] for row in with_failed], ["ctx_a", "ctx_b"])


if __name__ == "__main__":
    unittest.main()
