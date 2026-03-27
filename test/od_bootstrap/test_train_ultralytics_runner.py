from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.od_bootstrap.train.ultralytics_runner import _make_teacher_trainer


class _FakePbar:
    def __init__(self) -> None:
        self.desc = "1/100      2.71G      2.231      4.309   0.005974        400        640"
        self.values: list[str] = []

    def set_description(self, value: str) -> None:
        self.desc = value
        self.values.append(value)


class _FakeTrainLoader:
    def __len__(self) -> int:
        return 100


class UltralyticsRunnerTests(unittest.TestCase):
    def test_profile_postfix_updates_before_log_interval(self) -> None:
        runtime_params = {
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "log_every_n_steps": 20,
            "profile_window": 20,
            "profile_device_sync": False,
        }
        trainer_cls, callbacks = _make_teacher_trainer(runtime_params=runtime_params, log_fn=lambda message: None)
        self.assertIsNotNone(trainer_cls)
        del trainer_cls

        with tempfile.TemporaryDirectory() as temp_dir:
            pbar = _FakePbar()
            trainer = SimpleNamespace(
                device=SimpleNamespace(type="cpu"),
                od_batch_started_at=10.050,
                od_last_batch_end_at=10.000,
                od_pending_wait_sec=0.050,
                od_epoch_step=0,
                od_global_step=0,
                od_epoch_timing_window=deque(maxlen=runtime_params["profile_window"]),
                train_loader=_FakeTrainLoader(),
                od_epoch_started_at=10.000,
                od_profile_log_path=Path(temp_dir) / "profile_log.jsonl",
                od_tensorboard_writer=None,
                od_pbar=pbar,
                od_log=lambda message: None,
                epoch=0,
                epochs=100,
                tloss={"box_loss": 1.0},
                label_loss_items=lambda losses, prefix="train": {f"{prefix}/box_loss": float(losses["box_loss"])},
            )

            with patch("tools.od_bootstrap.train.ultralytics_runner.time.perf_counter", return_value=10.150):
                callbacks["on_train_batch_end"](trainer)

            self.assertEqual(trainer.od_epoch_step, 1)
            self.assertEqual(trainer.od_global_step, 1)
            self.assertEqual(len(pbar.values), 1)
            self.assertIn(" | elapsed=", pbar.values[0])
            self.assertIn("iter=", pbar.values[0])
            self.assertIn("wait=", pbar.values[0])
            self.assertIn("compute=", pbar.values[0])
            self.assertFalse(trainer.od_profile_log_path.exists())


if __name__ == "__main__":
    unittest.main()
