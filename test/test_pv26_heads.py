from __future__ import annotations

import unittest

import torch
from model.engine.loss import build_loss_spec


SPEC = build_loss_spec()
LANE_QUERY_COUNT = int(SPEC["heads"]["lane"]["query_count"])
LANE_VECTOR_DIM = int(SPEC["heads"]["lane"]["shape"].split(" x ")[-1])
STOP_LINE_QUERY_COUNT = int(SPEC["heads"]["stop_line"]["query_count"])
STOP_LINE_VECTOR_DIM = int(SPEC["heads"]["stop_line"]["shape"].split(" x ")[-1])
CROSSWALK_QUERY_COUNT = int(SPEC["heads"]["crosswalk"]["query_count"])
CROSSWALK_VECTOR_DIM = int(SPEC["heads"]["crosswalk"]["shape"].split(" x ")[-1])


class PV26HeadsTests(unittest.TestCase):
    def test_heads_produce_documented_output_shapes(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 64, 128, 256))
        features = [
            torch.randn(2, 64, 152, 200),
            torch.randn(2, 64, 76, 100),
            torch.randn(2, 128, 38, 50),
            torch.randn(2, 256, 19, 25),
        ]

        outputs = heads(features)

        self.assertEqual(tuple(outputs["det"].shape), (2, 9975, 12))
        self.assertEqual(tuple(outputs["tl_attr"].shape), (2, 9975, 4))
        self.assertEqual(tuple(outputs["lane"].shape), (2, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        self.assertEqual(tuple(outputs["stop_line"].shape), (2, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM))
        self.assertEqual(tuple(outputs["crosswalk"].shape), (2, CROSSWALK_QUERY_COUNT, CROSSWALK_VECTOR_DIM))
        self.assertEqual(outputs["det_feature_shapes"], [(76, 100), (38, 50), (19, 25)])
        self.assertEqual(outputs["det_feature_strides"], [8, 16, 32])

    def test_heads_expose_feature_contract_metadata(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 64, 128, 256))
        summary = heads.describe()

        self.assertEqual(heads.lane_head_mode, "seg_first")
        self.assertEqual(summary["feature_channels"], [64, 64, 128, 256])
        self.assertEqual(summary["feature_strides"], [4, 8, 16, 32])
        self.assertEqual(summary["det_feature_channels"], [64, 128, 256])
        self.assertEqual(summary["det_feature_strides"], [8, 16, 32])
        self.assertEqual(summary["det_dim"], 12)
        self.assertEqual(summary["tl_attr_dim"], 4)
        self.assertEqual(summary["lane_queries"], LANE_QUERY_COUNT)
        self.assertEqual(summary["stop_line_queries"], STOP_LINE_QUERY_COUNT)
        self.assertEqual(summary["crosswalk_queries"], CROSSWALK_QUERY_COUNT)
        self.assertEqual(summary["roadmark"]["lane_head_mode"], "seg_first")
        self.assertEqual(summary["roadmark"]["lane_family_shared_adapter"], "disabled")

    def test_heads_can_enable_lane_family_shared_adapter(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(
            in_channels=(64, 64, 128, 256),
            lane_family_shared_adapter_enabled=True,
        )
        features = [
            torch.randn(1, 64, 152, 200),
            torch.randn(1, 64, 76, 100),
            torch.randn(1, 128, 38, 50),
            torch.randn(1, 256, 19, 25),
        ]

        outputs = heads(features, encoded={})
        adapter_modules = heads.roadmark_heads.lane_family_adapter_modules()

        self.assertEqual(heads.describe()["roadmark"]["lane_family_shared_adapter"], "zero_init_residual_p2_p3_p4")
        self.assertEqual(len(adapter_modules), 1)
        self.assertTrue(torch.isfinite(outputs["lane"]).all())
        self.assertTrue(torch.isfinite(outputs["stop_line"]).all())
        self.assertTrue(torch.isfinite(outputs["crosswalk"]).all())

    def test_heads_can_enable_lane_family_task_adapters(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(
            in_channels=(64, 64, 128, 256),
            lane_family_task_adapter_enabled=True,
        )
        features = [
            torch.randn(1, 64, 152, 200),
            torch.randn(1, 64, 76, 100),
            torch.randn(1, 128, 38, 50),
            torch.randn(1, 256, 19, 25),
        ]

        outputs = heads(features, encoded={})
        adapter_modules = heads.roadmark_heads.lane_family_adapter_modules()

        self.assertEqual(heads.describe()["roadmark"]["lane_family_task_adapter"], "zero_init_residual_per_task_p2_p3_p4")
        self.assertEqual(len(adapter_modules), 1)
        self.assertTrue(torch.isfinite(outputs["lane"]).all())
        self.assertTrue(torch.isfinite(outputs["stop_line"]).all())
        self.assertTrue(torch.isfinite(outputs["crosswalk"]).all())

    def test_heads_reject_wrong_feature_count(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 64, 128, 256))
        with self.assertRaisesRegex(ValueError, "4 feature maps"):
            heads([torch.randn(1, 64, 76, 100)])

    def test_heads_support_yolo26s_channel_contract(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(128, 128, 256, 512))
        features = [
            torch.randn(2, 128, 152, 200),
            torch.randn(2, 128, 76, 100),
            torch.randn(2, 256, 38, 50),
            torch.randn(2, 512, 19, 25),
        ]

        outputs = heads(features)

        self.assertEqual(tuple(outputs["det"].shape), (2, 9975, 12))
        self.assertEqual(tuple(outputs["lane"].shape), (2, LANE_QUERY_COUNT, LANE_VECTOR_DIM))
        self.assertEqual(outputs["det_feature_shapes"], [(76, 100), (38, 50), (19, 25)])

    def test_roadmark_native_outputs_are_finite(self) -> None:
        from model.net import PV26Heads

        heads = PV26Heads(in_channels=(64, 64, 128, 256))
        features = [
            torch.randn(1, 64, 152, 200),
            torch.randn(1, 64, 76, 100),
            torch.randn(1, 128, 38, 50),
            torch.randn(1, 256, 19, 25),
        ]

        outputs = heads(features, encoded={})

        self.assertTrue(torch.isfinite(outputs["lane"]).all())
        self.assertTrue(torch.isfinite(outputs["stop_line"]).all())
        self.assertTrue(torch.isfinite(outputs["crosswalk"]).all())

    def test_stopline_head_builds_denoise_queries_from_encoded_segments(self) -> None:
        from model.net.stopline_head_line import StopLineDenseLocalHead

        head = StopLineDenseLocalHead((16, 16), hidden_dim=16, output_queries=4)
        head.train()
        encoded_stop_line = torch.zeros((1, STOP_LINE_QUERY_COUNT, STOP_LINE_VECTOR_DIM), dtype=torch.float32)
        encoded_stop_line[0, 0, 0] = 1.0
        encoded_stop_line[0, 0, 1:] = torch.tensor(
            [100.0, 500.0, 180.0, 500.0, 260.0, 500.0, 340.0, 500.0],
            dtype=torch.float32,
        )
        encoded = {
            "stop_line": encoded_stop_line,
            "mask": {
                "stop_line_source": torch.ones(1, dtype=torch.bool),
                "stop_line_valid": torch.zeros((1, STOP_LINE_QUERY_COUNT), dtype=torch.bool),
            },
        }
        encoded["mask"]["stop_line_valid"][0, 0] = True
        outputs = head(
            (
                torch.randn(1, 16, 16, 20),
                torch.randn(1, 16, 8, 10),
            ),
            encoded=encoded,
        )

        self.assertEqual(tuple(outputs["stop_line_segment_denoise_logits"].shape), (1, 4))
        self.assertEqual(tuple(outputs["stop_line_segment_denoise_points"].shape), (1, 4, 2, 2))
        self.assertEqual(tuple(outputs["stop_line_endpoint_logits"].shape), (1, 2, 16, 20))
        self.assertEqual(tuple(outputs["stop_line_endpoint_offset"].shape), (1, 4, 16, 20))
        self.assertEqual(tuple(outputs["stop_line_axis_segment_logits"].shape), (1, 4))
        self.assertEqual(tuple(outputs["stop_line_axis_segment_points"].shape), (1, 4, 2, 2))
        self.assertGreater(int(outputs["stop_line_segment_denoise_valid"].sum().item()), 0)
        self.assertTrue(torch.isfinite(outputs["stop_line_segment_denoise_points"]).all())
        self.assertTrue(torch.isfinite(outputs["stop_line_endpoint_offset"]).all())
        self.assertTrue(torch.isfinite(outputs["stop_line_axis_segment_points"]).all())


if __name__ == "__main__":
    unittest.main()
