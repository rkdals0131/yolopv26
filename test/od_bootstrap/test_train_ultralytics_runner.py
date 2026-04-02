from __future__ import annotations

from collections import deque
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from common.scalars import flatten_scalar_tree
from tools.od_bootstrap.teacher.runtime_progress import install_ultralytics_postfix_renderer
from tools.od_bootstrap.teacher.runtime_resume import resolve_resume_argument
from tools.od_bootstrap.teacher.runtime_tensorboard import (
    build_epoch_tensorboard_payload,
    build_train_step_tensorboard_payload,
)
from tools.od_bootstrap.teacher.ultralytics_runner import _make_teacher_trainer


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
        install_ultralytics_postfix_renderer(pbar)
        with patch("tools.od_bootstrap.teacher.ultralytics_runner.time.time", return_value=123.9):
            pbar.set_bootstrap_postfix(
                "elapsed=00:24  |  eta=20:25  |  iter=283.7ms  |  wait=0.3ms  |  compute=283.7ms"
            )
        rendered = pbar.file.getvalue()
        self.assertIn("58/4375", rendered)
        self.assertIn("it/s", rendered)
        self.assertIn(
            "\n\033[Kelapsed=00:24  |  eta=20:25  |  iter=283.7ms  |  wait=0.3ms  |  compute=283.7ms",
            rendered,
        )

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

            with patch("tools.od_bootstrap.teacher.ultralytics_runner.time.perf_counter", return_value=10.150):
                callbacks["on_train_batch_end"](trainer)

            self.assertEqual(trainer.od_epoch_step, 1)
            self.assertEqual(trainer.od_global_step, 1)
            self.assertEqual(len(pbar.values), 1)
            self.assertIn("  |  elapsed=", pbar.values[0])
            self.assertIn("iter=", pbar.values[0])
            self.assertIn("wait=", pbar.values[0])
            self.assertIn("compute=", pbar.values[0])
            self.assertFalse(trainer.od_profile_log_path.exists())

    def test_epoch_tensorboard_payload_keeps_only_requested_tags(self) -> None:
        payload = build_epoch_tensorboard_payload(
            losses={
                "train/box_loss": 1.25,
                "train/cls_loss": 2.5,
                "train/dfl_loss": 0.75,
                "train/extra_loss": 99.0,
            },
            profile_summary={
                "window_size": 20,
                "iteration_sec": {"mean": 0.31, "p50": 0.3, "p99": 0.4},
                "wait_sec": {"mean": 0.02, "p50": 0.01, "p99": 0.05},
                "compute_sec": {"mean": 0.29, "p50": 0.28, "p99": 0.38},
            },
            lr_values={"lr/pg0": 0.001, "lr/pg1": 0.002, "lr/pg2": 0.003},
            metrics={
                "metrics/precision(B)": 0.5,
                "metrics/recall(B)": 0.25,
                "metrics/mAP50(B)": 0.75,
                "metrics/mAP50-95(B)": 0.4,
                "val/box_loss": 1.1,
                "val/cls_loss": 0.9,
                "val/dfl_loss": 0.2,
                "ignored_metric": 123.0,
            },
        )

        scalar_names = {name for name, _ in flatten_scalar_tree("epoch", payload)}

        self.assertEqual(
            scalar_names,
            {
                "epoch/train/box_loss",
                "epoch/train/cls_loss",
                "epoch/train/dfl_loss",
                "epoch/lr/pg0",
                "epoch/profile_sec/iteration_mean",
                "epoch/profile_sec/wait_mean",
                "epoch/profile_sec/compute_mean",
                "epoch/precision",
                "epoch/recall",
                "epoch/f1",
                "epoch/mAP50",
                "epoch/mAP50_95",
                "epoch/val/box_loss",
                "epoch/val/cls_loss",
                "epoch/val/dfl_loss",
            },
        )

    def test_train_step_tensorboard_payload_keeps_only_requested_tags(self) -> None:
        payload = build_train_step_tensorboard_payload(
            losses={
                "train/box_loss": 1.0,
                "train/cls_loss": 2.0,
                "train/dfl_loss": 3.0,
                "train/unused": 4.0,
            },
            profile_summary={
                "iteration_sec": {"mean": 0.31, "p50": 0.3, "p99": 0.4},
                "wait_sec": {"mean": 0.02, "p50": 0.01, "p99": 0.05},
                "compute_sec": {"mean": 0.29, "p50": 0.28, "p99": 0.38},
            },
            elapsed_sec=12.5,
        )

        scalar_names = {name for name, _ in flatten_scalar_tree("train_step", payload)}

        self.assertEqual(
            scalar_names,
            {
                "train_step/loss/box_loss",
                "train_step/loss/cls_loss",
                "train_step/loss/dfl_loss",
                "train_step/profile_sec/iteration_mean",
                "train_step/profile_sec/iteration_p50",
                "train_step/profile_sec/iteration_p99",
                "train_step/profile_sec/wait_mean",
                "train_step/profile_sec/compute_mean",
                "train_step/elapsed_sec",
            },
        )

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
            resolved = resolve_resume_argument(True, teacher_name="signal", teacher_root=teacher_root)
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
