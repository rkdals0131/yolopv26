from __future__ import annotations

from collections import deque
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.od_bootstrap.train.ultralytics_runner import (
    _install_ultralytics_postfix_renderer,
    _make_teacher_trainer,
    _resolve_resume_argument,
)


class _FakePbar:
    def __init__(self) -> None:
        self.desc = "1/100      2.71G      2.231      4.309   0.005974        400        640"
        self.values: list[str] = []

    def set_description(self, value: str) -> None:
        self.desc = value
        self.values.append(value)


class _FakeUltralyticsPbar:
    MIN_RATE_CALC_INTERVAL = 0.01
    RATE_SMOOTHING_FACTOR = 0.3
    MAX_SMOOTHED_RATE = 1000000

    def __init__(self) -> None:
        self.desc = "      1/100      2.89G      2.237      4.319    0.00586        423        640"
        self.disable = False
        self.closed = False
        self.noninteractive = False
        self.file = io.StringIO()
        self.total = 4375
        self.n = 58
        self.last_print_n = 38
        self.start_t = 100.0
        self.last_print_t = 123.7
        self.last_rate = 0.0
        self.is_bytes = False

    def _should_update(self, dt: float, dn: int) -> bool:
        del dt, dn
        return True

    def _format_num(self, num: int | float) -> str:
        return str(int(num))

    def _format_time(self, seconds: float) -> str:
        return f"{seconds:.1f}s"

    def _format_rate(self, rate: float) -> str:
        return f"{rate:.1f}it/s" if rate > 0 else ""

    def _generate_bar(self) -> str:
        return "────────────"

    def _display(self, final: bool = False) -> None:
        del final


class _FakeTrainLoader:
    def __len__(self) -> int:
        return 100


def _write_checkpoint(path: Path, *, epoch: int, total_epochs: int, resumable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch if resumable else -1,
            "optimizer": {"state": {}, "param_groups": []} if resumable else None,
            "train_args": {
                "data": "dataset.yaml",
                "epochs": total_epochs,
                "batch": 2,
                "imgsz": 640,
            },
        },
        path,
    )


class UltralyticsRunnerTests(unittest.TestCase):
    def test_ultralytics_postfix_renders_at_line_end(self) -> None:
        pbar = _FakeUltralyticsPbar()
        _install_ultralytics_postfix_renderer(pbar)
        with patch("tools.od_bootstrap.train.ultralytics_runner.time.time", return_value=123.9):
            pbar.set_bootstrap_postfix("elapsed=00:24 eta=20:25 iter=283.7ms wait=0.3ms compute=283.7ms")
        rendered = pbar.file.getvalue()
        self.assertIn("58/4375", rendered)
        self.assertIn("it/s", rendered)
        self.assertIn("\n\033[Kelapsed=00:24 eta=20:25 iter=283.7ms wait=0.3ms compute=283.7ms", rendered)

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

    def test_resolve_resume_argument_prefers_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            teacher_root = Path(temp_dir) / "signal"
            stripped_last = teacher_root / "20260328_040148" / "weights" / "last.pt"
            resumable_epoch = teacher_root / "20260328_040148" / "weights" / "epoch70.pt"
            latest_run = teacher_root / "latest_run.json"
            _write_checkpoint(stripped_last, epoch=71, total_epochs=72, resumable=False)
            _write_checkpoint(resumable_epoch, epoch=70, total_epochs=72, resumable=True)
            latest_run.parent.mkdir(parents=True, exist_ok=True)
            latest_run.write_text("{}", encoding="utf-8")
            resolved = _resolve_resume_argument(True, teacher_name="signal", teacher_root=teacher_root)
            self.assertEqual(resolved, str(resumable_epoch))

    def test_teacher_trainer_check_resume_allows_epoch_extension(self) -> None:
        runtime_params = {
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "log_every_n_steps": 20,
            "profile_window": 20,
            "profile_device_sync": False,
        }
        trainer_cls, _ = _make_teacher_trainer(runtime_params=runtime_params, log_fn=lambda message: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "signal" / "20260328_040148" / "weights" / "epoch70.pt"
            data_yaml = root / "dataset.yaml"
            data_yaml.write_text("path: dataset\n", encoding="utf-8")
            _write_checkpoint(checkpoint, epoch=70, total_epochs=72, resumable=True)

            trainer = object.__new__(trainer_cls)
            trainer.args = SimpleNamespace(resume=str(checkpoint), data=str(data_yaml))
            trainer.resume = False

            trainer.check_resume({"epochs": 150, "batch": 4, "device": "cpu"})

            self.assertTrue(trainer.resume)
            self.assertEqual(trainer.args.resume, str(checkpoint))
            self.assertEqual(trainer.args.model, str(checkpoint))
            self.assertEqual(trainer.args.epochs, 150)
            self.assertEqual(trainer.args.batch, 4)
            self.assertEqual(trainer.args.device, "cpu")

    def test_extended_resume_scheduler_preserves_lr_continuity(self) -> None:
        runtime_params = {
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "log_every_n_steps": 20,
            "profile_window": 20,
            "profile_device_sync": False,
        }
        trainer_cls, _ = _make_teacher_trainer(runtime_params=runtime_params, log_fn=lambda message: None)
        trainer = object.__new__(trainer_cls)
        trainer.resume = True
        trainer.od_resume_base_epochs = 72
        trainer.od_resume_start_epoch = 71
        trainer.epochs = 150
        trainer.args = SimpleNamespace(cos_lr=False, lrf=0.01)

        lf = trainer._build_extended_resume_lf()

        self.assertIsNotNone(lf)
        self.assertAlmostEqual(lf(71), 0.02375, places=8)
        self.assertLess(lf(72), lf(71))
        self.assertAlmostEqual(lf(150), 0.01, places=8)


if __name__ == "__main__":
    unittest.main()
