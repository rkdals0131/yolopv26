from __future__ import annotations

import types
import unittest

import numpy as np

from common.train_runtime import (
    build_progress_status,
    format_duration,
    join_status_segments,
    progress_meter,
    sync_timing_device,
    timing_profile,
    write_tensorboard_histograms,
    write_tensorboard_scalars,
)


class _FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.histograms: list[tuple[str, object, int]] = []

    def add_scalar(self, name: str, value: float, *, global_step: int) -> None:
        self.scalars.append((name, value, global_step))

    def add_histogram(self, name: str, value: object, *, global_step: int) -> None:
        self.histograms.append((name, value, global_step))


class _FakeCudaModule:
    def __init__(self, *, available: bool = True, fail_with_device: bool = False) -> None:
        self._available = available
        self._fail_with_device = fail_with_device
        self.calls: list[object] = []
        self.cuda = self

    def is_available(self) -> bool:
        return self._available

    def synchronize(self, device: object | None = None) -> None:
        self.calls.append(device)
        if self._fail_with_device and device is not None:
            raise RuntimeError("device-specific sync unavailable")


class CommonTrainRuntimeTests(unittest.TestCase):
    def test_format_duration_supports_custom_unavailable_labels(self) -> None:
        self.assertEqual(format_duration(None), "n/a")
        self.assertEqual(format_duration(float("nan")), "n/a")
        self.assertEqual(format_duration(None, unavailable="unknown"), "unknown")
        self.assertEqual(format_duration(61.2), "01:01")

    def test_timing_profile_accepts_custom_value_resolver(self) -> None:
        profile = timing_profile(
            [{"timing": {"wait_sec": 0.1, "iteration_sec": 0.5}}, {"timing": {"wait_sec": 0.3, "iteration_sec": 0.9}}],
            keys=("wait_sec", "iteration_sec"),
            value_resolver=lambda item, key: item["timing"][key],
        )

        self.assertEqual(profile["window_size"], 2)
        self.assertAlmostEqual(profile["wait_sec"]["mean"], 0.2)
        self.assertAlmostEqual(profile["iteration_sec"]["p50"], 0.7)

    def test_sync_timing_device_handles_cuda_only_and_generic_fallback(self) -> None:
        fake_torch = _FakeCudaModule(fail_with_device=True)

        sync_timing_device(fake_torch, types.SimpleNamespace(type="cuda"), True)
        self.assertEqual(len(fake_torch.calls), 2)
        self.assertEqual(fake_torch.calls[0], types.SimpleNamespace(type="cuda"))
        self.assertIsNone(fake_torch.calls[1])

        cpu_torch = _FakeCudaModule()
        sync_timing_device(cpu_torch, types.SimpleNamespace(type="cpu"), True)
        self.assertEqual(cpu_torch.calls, [])

    def test_write_tensorboard_scalars_flattens_payload(self) -> None:
        writer = _FakeWriter()

        count = write_tensorboard_scalars(
            writer,
            "train",
            {"loss": {"total": 1.25}, "lr": 0.001},
            7,
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            writer.scalars,
            [
                ("train/loss/total", 1.25, 7),
                ("train/lr", 0.001, 7),
            ],
        )

    def test_write_tensorboard_histograms_flattens_payload(self) -> None:
        writer = _FakeWriter()

        count = write_tensorboard_histograms(
            writer,
            "epoch/val",
            {
                "detector": {
                    "prediction_confidence": [0.9, 0.8],
                    "per_class_confidence": {
                        "traffic_light": [0.95],
                    },
                },
                "lane": {
                    "mean_point_distance": [1.2, 0.8],
                },
            },
            3,
        )

        self.assertEqual(count, 3)
        self.assertEqual(
            [name for name, _, _ in writer.histograms],
            [
                "epoch/val/detector/prediction_confidence",
                "epoch/val/detector/per_class_confidence/traffic_light",
                "epoch/val/lane/mean_point_distance",
            ],
        )
        self.assertTrue(all(isinstance(value, np.ndarray) for _, value, _ in writer.histograms))

    def test_progress_helpers_build_shared_status_segments(self) -> None:
        self.assertEqual(join_status_segments("elapsed=00:10", None, "", "eta=00:20"), "elapsed=00:10  |  eta=00:20")
        self.assertEqual(progress_meter(3, 10, width=5), "[##...]  30%")
        self.assertEqual(
            build_progress_status(current=3, total=10, width=5, segments=("elapsed=00:10", "eta=00:20")),
            "[##...]  30%  |  elapsed=00:10  |  eta=00:20",
        )


if __name__ == "__main__":
    unittest.main()
