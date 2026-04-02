from __future__ import annotations

import unittest

from model.engine.loss import build_loss_spec, render_loss_spec_markdown


class PV26LossSpecTests(unittest.TestCase):
    def test_loss_spec_matches_approved_multitask_design(self) -> None:
        spec = build_loss_spec()

        self.assertEqual(spec["version"], "pv26-loss-v6")
        self.assertEqual(
            spec["model_contract"]["od_classes"],
            ["vehicle", "bike", "pedestrian", "traffic_cone", "obstacle", "traffic_light", "sign"],
        )
        self.assertEqual(spec["model_contract"]["tl_bits"], ["red", "yellow", "green", "arrow"])
        self.assertIn("Q_det", spec["heads"]["det"]["shape"])
        self.assertIn("Q_det", spec["heads"]["tl_attr"]["shape"])
        self.assertEqual(spec["heads"]["lane"]["target_encoding"]["polyline_points"], 16)
        self.assertEqual(spec["heads"]["lane"]["query_count"], 12)
        self.assertEqual(spec["heads"]["stop_line"]["query_count"], 6)
        self.assertEqual(spec["heads"]["crosswalk"]["query_count"], 4)
        self.assertEqual(spec["heads"]["stop_line"]["target_encoding"]["polyline_points"], 4)
        self.assertEqual(spec["heads"]["crosswalk"]["target_encoding"]["polygon_points"], 8)
        self.assertIn("N_gt_det", spec["sample_contract"]["det_targets"]["boxes_xyxy"])
        self.assertEqual(spec["sample_contract"]["naming"]["Q_det"], "number of detector prediction slots in one image")
        self.assertEqual(spec["sample_contract"]["image"]["shape"], [3, 608, 800])
        self.assertEqual(spec["transform_contract"]["dataset_raw_hw"], "variable")
        self.assertEqual(spec["transform_contract"]["vehicle_camera_raw_hw"], [600, 800])
        self.assertEqual(spec["transform_contract"]["network_hw"], [608, 800])
        self.assertEqual(spec["transform_contract"]["interpolation"], "bilinear")
        self.assertEqual(spec["transform_contract"]["padding_fill_uint8"], 114)
        self.assertEqual(spec["dataset_masking"]["bdd100k"]["det"], 1)
        self.assertEqual(spec["dataset_masking"]["aihub_obstacle"]["det"], 1)
        self.assertEqual(spec["dataset_masking"]["aihub_lane"]["crosswalk"], 1)
        self.assertEqual(spec["sampler"]["ratios"]["aihub_obstacle"], 0.15)
        self.assertIn("arrow_only -> [0,0,0,1]", spec["tl_attr_policy"]["valid_examples"])
        self.assertIn("non_car_traffic_light", spec["tl_attr_policy"]["masked_cases"])
        self.assertNotIn("arrow_without_base_color", spec["tl_attr_policy"]["masked_cases"])
        self.assertIn(
            "Each detector positive reads TL bits from its matched GT index.",
            spec["tl_attr_policy"]["training_binding"],
        )
        self.assertEqual(spec["inference_contract"]["raw_model_output"]["det"], "float32[B, Q_det, 12]")
        self.assertEqual(len(spec["training_schedule"]), 4)
        self.assertEqual(spec["training_schedule"][0]["loss_weights"]["lane"], 2.0)
        self.assertEqual(spec["training_schedule"][-1]["name"], "stage_4_lane_family_finetune")
        self.assertEqual(spec["training_schedule"][-1]["freeze"], "trunk_and_detector_tl_heads_frozen")
        self.assertEqual(spec["training_schedule"][-1]["loss_weights"]["det"], 0.0)
        self.assertEqual(spec["training_schedule"][-1]["loss_weights"]["tl_attr"], 0.0)

    def test_markdown_renderer_exposes_key_sections(self) -> None:
        markdown = render_loss_spec_markdown()
        self.assertIn("# PV26 Loss Design Spec", markdown)
        self.assertIn("## TL Attribute Policy", markdown)
        self.assertIn("## Sample Contract", markdown)
        self.assertIn("Q_det=number of detector prediction slots in one image", markdown)
        self.assertIn("traffic_light", markdown)
        self.assertIn("stage_3_end_to_end_finetune", markdown)
        self.assertIn("stage_4_lane_family_finetune", markdown)


if __name__ == "__main__":
    unittest.main()
