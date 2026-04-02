from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tools.od_bootstrap.source import aihub_lane_worker, aihub_obstacle_worker, aihub_traffic_worker


class AIHubWorkerModuleCleanupTests(unittest.TestCase):
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
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_names: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names[node.module] = tuple(alias.name for alias in node.names)

        self.assertEqual(imported_names["aihub_lane_worker"], ("lane_worker",))
        self.assertEqual(
            imported_names["aihub_obstacle_worker"],
            ("obstacle_worker", "prepare_debug_scene_for_overlay"),
        )
        self.assertEqual(imported_names["aihub_traffic_worker"], ("combo_name", "traffic_worker"))
