from __future__ import annotations

import unittest


class PV26EnginePublicApiTests(unittest.TestCase):
    def test_engine_package_reexports_curated_public_surface(self) -> None:
        import model.engine as engine
        from model.engine.batch import augment_lane_family_metrics, move_batch_to_device, raw_batch_for_metrics
        from model.engine.evaluator import PV26Evaluator
        from model.engine.loss import PV26DetAssignmentUnavailable, PV26MultiTaskLoss
        from model.engine.metrics import PV26MetricConfig, summarize_pv26_metrics
        from model.engine.postprocess import PV26PostprocessConfig, postprocess_pv26_batch
        from model.engine.spec import SPEC_VERSION, build_loss_spec, render_loss_spec_markdown
        from model.engine.trainer import (
            PV26Trainer,
            STAGE_NAMES,
            TIMING_KEYS,
            TENSORBOARD_LOSS_KEYS,
            build_pv26_optimizer,
            build_pv26_scheduler,
            configure_pv26_train_stage,
            run_pv26_tiny_overfit,
        )

        expected = {
            "PV26DetAssignmentUnavailable": PV26DetAssignmentUnavailable,
            "PV26Evaluator": PV26Evaluator,
            "PV26MetricConfig": PV26MetricConfig,
            "PV26MultiTaskLoss": PV26MultiTaskLoss,
            "PV26PostprocessConfig": PV26PostprocessConfig,
            "PV26Trainer": PV26Trainer,
            "SPEC_VERSION": SPEC_VERSION,
            "STAGE_NAMES": STAGE_NAMES,
            "TIMING_KEYS": TIMING_KEYS,
            "TENSORBOARD_LOSS_KEYS": TENSORBOARD_LOSS_KEYS,
            "augment_lane_family_metrics": augment_lane_family_metrics,
            "build_loss_spec": build_loss_spec,
            "build_pv26_optimizer": build_pv26_optimizer,
            "build_pv26_scheduler": build_pv26_scheduler,
            "configure_pv26_train_stage": configure_pv26_train_stage,
            "move_batch_to_device": move_batch_to_device,
            "postprocess_pv26_batch": postprocess_pv26_batch,
            "raw_batch_for_metrics": raw_batch_for_metrics,
            "render_loss_spec_markdown": render_loss_spec_markdown,
            "run_pv26_tiny_overfit": run_pv26_tiny_overfit,
            "summarize_pv26_metrics": summarize_pv26_metrics,
        }

        self.assertEqual(set(engine.__all__), set(expected))
        self.assertTrue(all(not name.startswith("_") for name in engine.__all__))
        for name, exported in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(engine, name), exported)


if __name__ == "__main__":
    unittest.main()
