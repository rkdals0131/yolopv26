import tempfile
import unittest
from pathlib import Path

from tools.data_analysis.wod.run_wod_pv26_interactive import (
    _build_acquire_command,
    _build_acquire_progress_line,
    _build_run_target_plan,
    _load_acquire_progress_snapshot,
    _render_target_progress_line,
    _summarize_target_progress,
)


class TestWodInteractiveHelpers(unittest.TestCase):
    def test_build_run_target_plan_filters_failed_and_applies_limit(self):
        state = {
            "contexts": [
                {
                    "context_name": "ctx_a",
                    "status": "pending",
                    "processable_now": True,
                },
                {
                    "context_name": "ctx_b",
                    "status": "failed",
                    "processable_now": True,
                },
                {
                    "context_name": "ctx_c",
                    "status": "completed",
                    "processable_now": True,
                },
            ]
        }

        from unittest import mock

        with mock.patch(
            "tools.data_analysis.wod.run_wod_pv26_interactive.load_wod_bulk_state",
            return_value=state,
        ), mock.patch(
            "tools.data_analysis.wod.run_wod_pv26_interactive.reconcile_wod_bulk_state",
            return_value=state,
        ):
            plan = _build_run_target_plan(
                training_root=Path("/tmp/training"),
                shards_root=Path("/tmp/shards"),
                state_path=Path("/tmp/state.json"),
                selected_contexts_csv="",
                retry_failed=True,
                max_contexts=1,
            )

        self.assertEqual(plan.context_names, ["ctx_a"])

    def test_summarize_target_progress_counts_rows_and_statuses(self):
        state = {
            "contexts": [
                {
                    "context_name": "ctx_a",
                    "status": "completed",
                    "output_num_rows": 100,
                    "output_has_det_rows": 80,
                },
                {
                    "context_name": "ctx_b",
                    "status": "in_progress",
                    "output_num_rows": 5,
                    "output_has_det_rows": 5,
                },
                {
                    "context_name": "ctx_c",
                    "status": "failed",
                    "output_num_rows": 0,
                    "output_has_det_rows": 0,
                },
            ]
        }
        summary = _summarize_target_progress(state, ["ctx_a", "ctx_b", "ctx_c"])
        self.assertEqual(summary["target_total"], 3)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["in_progress"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["output_num_rows"], 105)
        self.assertEqual(summary["output_has_det_rows"], 85)
        self.assertEqual(summary["in_progress_names"], ["ctx_b"])

    def test_render_target_progress_line_mentions_active_context(self):
        line = _render_target_progress_line(
            {
                "target_total": 4,
                "completed": 1,
                "failed": 1,
                "in_progress": 1,
                "pending": 1,
                "blocked": 0,
                "output_num_rows": 250,
                "output_has_det_rows": 200,
                "in_progress_names": ["ctx_live"],
            }
        )
        self.assertIn("2/4 (50.0%)", line)
        self.assertIn("완료 1", line)
        self.assertIn("실패 1", line)
        self.assertIn("현재 ctx_live", line)

    def test_load_acquire_progress_snapshot_filters_target_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "acquire.json"
            state_path.write_text(
                '{"contexts": ['
                '{"context_name": "ctx_a", "download_status": "downloaded"},'
                '{"context_name": "ctx_b", "download_status": "failed"},'
                '{"context_name": "ctx_c", "download_status": "remote_only"}'
                ']}',
                encoding="utf-8",
            )
            snapshot = _load_acquire_progress_snapshot(state_path, target_contexts=["ctx_a", "ctx_b"])
            self.assertEqual(snapshot.total, 2)
            self.assertEqual(snapshot.downloaded, 1)
            self.assertEqual(snapshot.failed, 1)
            self.assertEqual(snapshot.remote_only, 0)

    def test_build_acquire_command_includes_download_options(self):
        cmd = _build_acquire_command(
            subcommand="download",
            training_root=Path("/tmp/training"),
            shards_root=Path("/tmp/shards"),
            state_path=Path("/tmp/acquire.json"),
            summary_csv_path=Path("/tmp/acquire.csv"),
            contexts="ctx_a,ctx_b",
            retry_failed=True,
            max_contexts=3,
            jobs=5,
            include_box=True,
        )
        self.assertIn("--jobs", cmd)
        self.assertIn("5", cmd)
        self.assertIn("--include-box", cmd)
        self.assertIn("--retry-failed", cmd)
        self.assertIn("ctx_a,ctx_b", cmd)

    def test_build_acquire_progress_line_mentions_active_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "acquire.json"
            state_path.write_text(
                '{"contexts": ['
                '{"context_name": "ctx_live", "download_status": "downloading"},'
                '{"context_name": "ctx_done", "download_status": "downloaded"}'
                ']}',
                encoding="utf-8",
            )
            line = _build_acquire_progress_line(state_path, ["ctx_live", "ctx_done"])
            self.assertIn("1/2 (50.0%)", line)
            self.assertIn("진행중 1", line)
            self.assertIn("현재 ctx_live", line)


if __name__ == "__main__":
    unittest.main()
