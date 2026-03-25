from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.viz.overlay import render_overlay


class OverlayRenderTests(unittest.TestCase):
    def test_render_overlay_draws_generic_detections_without_duplicate_signal_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "source.jpg"
            output_path = root / "debug" / "overlay.png"
            source_image.write_bytes(b"")
            scene = {
                "image": {
                    "source_path": str(source_image),
                },
                "detections": [
                    {"class_name": "vehicle", "bbox": [10, 20, 30, 40]},
                    {"class_name": "traffic_light", "bbox": [50, 60, 70, 80]},
                    {"class_name": "sign", "bbox": [90, 100, 110, 120]},
                    {"class_name": "pedestrian", "bbox": [130, 140, 150, 160]},
                ],
                "traffic_lights": [
                    {"bbox": [50, 60, 70, 80]},
                ],
                "traffic_signs": [
                    {"bbox": [90, 100, 110, 120]},
                ],
            }

            with patch("model.viz.overlay.subprocess.run") as mocked:
                render_overlay(scene, output_path)

            command = mocked.call_args.args[0]
            draw_ops = [item for item in command if isinstance(item, str) and item.startswith("rectangle ")]

            self.assertIn("rectangle 10,20 30,40", draw_ops)
            self.assertIn("rectangle 50,60 70,80", draw_ops)
            self.assertIn("rectangle 90,100 110,120", draw_ops)
            self.assertIn("rectangle 130,140 150,160", draw_ops)
            self.assertEqual(draw_ops.count("rectangle 50,60 70,80"), 1)
            self.assertEqual(draw_ops.count("rectangle 90,100 110,120"), 1)


if __name__ == "__main__":
    unittest.main()
