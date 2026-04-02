from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tools.od_bootstrap.source import aihub_lane_worker, aihub_obstacle_worker, aihub_traffic_worker


class AIHubWorkerModuleCleanupTests(unittest.TestCase):
    @staticmethod
    def _imported_names(module_path: Path) -> dict[str, tuple[str, ...]]:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_names: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names[node.module] = tuple(alias.name for alias in node.names)
        return imported_names

    def test_worker_modules_expose_public_aliases_for_entry_points(self) -> None:
        self.assertIs(aihub_lane_worker.lane_worker, aihub_lane_worker._lane_worker)
        self.assertIs(aihub_traffic_worker.combo_name, aihub_traffic_worker._combo_name)
        self.assertIs(aihub_traffic_worker.traffic_worker, aihub_traffic_worker._traffic_worker)
        self.assertIs(
            aihub_obstacle_worker.prepare_debug_scene_for_overlay,
            aihub_obstacle_worker._prepare_debug_scene_for_overlay,
        )
        self.assertIs(aihub_obstacle_worker.obstacle_worker, aihub_obstacle_worker._obstacle_worker)

    def test_aihub_workers_imports_public_worker_symbols(self) -> None:
        module_path = Path(__file__).resolve().parents[2] / "tools/od_bootstrap/source/aihub_workers.py"
        imported_names = self._imported_names(module_path)

        self.assertEqual(imported_names["aihub_lane_worker"], ("lane_worker",))
        self.assertEqual(
            imported_names["aihub_obstacle_worker"],
            ("obstacle_worker", "prepare_debug_scene_for_overlay"),
        )
        self.assertEqual(imported_names["aihub_traffic_worker"], ("combo_name", "traffic_worker"))

    def test_support_modules_use_public_shared_helpers(self) -> None:
        source_root = Path(__file__).resolve().parents[2] / "tools/od_bootstrap/source"
        module_expectations = {
            "aihub_lane_worker.py": {
                "shared_io": ("link_or_copy", "load_json", "write_json"),
                "shared_raw": ("extract_annotations", "extract_attribute_map", "extract_points", "normalize_text", "safe_slug"),
                "shared_scene": ("DEFAULT_SCENE_VERSION", "build_base_scene", "sample_id"),
                "shared_summary": ("counter_to_dict",),
            },
            "aihub_obstacle_worker.py": {
                "shared_io": ("link_or_copy", "load_json", "write_json", "write_text"),
                "shared_raw": ("extract_annotations", "normalize_text", "safe_slug"),
                "shared_scene": ("DEFAULT_SCENE_VERSION", "bbox_to_yolo_line", "build_base_scene", "sample_id"),
                "shared_summary": ("counter_to_dict",),
            },
            "aihub_traffic_worker.py": {
                "shared_io": ("link_or_copy", "load_json", "write_json", "write_text"),
                "shared_raw": ("extract_annotations", "extract_bbox", "extract_tl_state", "normalize_text", "safe_slug"),
                "shared_scene": ("DEFAULT_SCENE_VERSION", "bbox_to_yolo_line", "build_base_scene", "sample_id"),
                "shared_summary": ("counter_to_dict",),
            },
            "aihub_workers.py": {
                "shared_io": ("load_json",),
                "shared_raw": ("normalize_text", "safe_slug"),
                "shared_resume": ("count_held_annotation_reasons", "load_existing_scene_output"),
                "shared_scene": ("sample_id",),
                "shared_summary": ("counter_to_dict",),
            },
            "bdd100k.py": {
                "shared_debug": ("generate_debug_vis",),
                "shared_io": ("link_or_copy", "load_json", "write_json", "write_text"),
                "shared_parallel": (
                    "PARALLEL_INFLIGHT_CHUNKS_PER_WORKER",
                    "PARALLEL_SUBMIT_LOG_INTERVAL",
                    "PARALLEL_WAIT_HEARTBEAT_SECONDS",
                    "LiveLogger",
                    "default_workers",
                    "iter_task_chunks",
                    "parallel_chunk_size",
                ),
                "shared_raw": ("env_path", "now_iso", "probe_image_size", "repo_root", "safe_slug", "seg_dataset_root"),
                "shared_resume": ("count_held_annotation_reasons", "load_existing_scene_output"),
                "shared_reports": ("det_class_map_yaml",),
                "shared_scene": ("bbox_to_yolo_line",),
                "shared_source_meta": (
                    "bdd_readme",
                    "bdd_source_inventory_markdown",
                    "build_bdd_inventory",
                    "build_bdd_source_inventory",
                ),
                "shared_summary": ("counter_to_dict",),
            },
        }

        for file_name, expected_imports in module_expectations.items():
            imported_names = self._imported_names(source_root / file_name)
            for module_name, expected_names in expected_imports.items():
                self.assertEqual(imported_names[module_name], expected_names, msg=f"{file_name}:{module_name}")

    def test_bdd100k_no_longer_imports_aihub_private_runtime_helpers(self) -> None:
        module_path = Path(__file__).resolve().parents[2] / "tools/od_bootstrap/source/bdd100k.py"
        imported_names = self._imported_names(module_path)
        self.assertNotIn("aihub", imported_names)
