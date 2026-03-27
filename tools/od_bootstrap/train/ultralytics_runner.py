from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    tqdm = None

try:
    import torch
    from torch.utils.data import distributed
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    torch = None
    distributed = None

try:
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.data.build import ContiguousDistributedSampler, InfiniteDataLoader, RANK, seed_worker
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import torch_distributed_zero_first
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    YOLO = None
    DEFAULT_CFG = None
    DetectionTrainer = None
    InfiniteDataLoader = None
    ContiguousDistributedSampler = None
    seed_worker = None
    torch_distributed_zero_first = None
    RANK = -1


def _coerce_weights_name(model_size: str, weights: str | None) -> str:
    if weights:
        return weights
    size = model_size.strip().lower() or "n"
    return f"yolo26{size}.pt"


def _extract_run_dir(train_result: Any, fallback_dir: Path) -> Path:
    for candidate in (
        getattr(train_result, "save_dir", None),
        getattr(getattr(train_result, "trainer", None), "save_dir", None),
    ):
        if candidate:
            return Path(candidate)
    return fallback_dir


def _sync_timing_device(device: Any, enabled: bool) -> None:
    if not enabled or torch is None or not torch.cuda.is_available():
        return
    device_type = getattr(device, "type", None)
    if device_type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        torch.cuda.synchronize()


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0.0, min(float(len(ordered) - 1), float(len(ordered) - 1) * float(fraction)))
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    ratio = index - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _profile_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "mean": sum(float(item) for item in values) / len(values),
        "p50": _quantile(values, 0.5),
        "p99": _quantile(values, 0.99),
    }


def _timing_profile(window: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "window_size": len(window),
        "iteration_sec": _profile_stats([item["iteration_sec"] for item in window]),
        "wait_sec": _profile_stats([item["wait_sec"] for item in window]),
        "compute_sec": _profile_stats([item["compute_sec"] for item in window]),
    }


def _flatten_scalar_tree(prefix: str, payload: Any) -> list[tuple[str, float]]:
    scalars: list[tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            scalars.extend(_flatten_scalar_tree(next_prefix, value))
        return scalars
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}/{index}" if prefix else str(index)
            scalars.extend(_flatten_scalar_tree(next_prefix, value))
        return scalars
    if isinstance(payload, bool):
        return [(prefix, 1.0 if payload else 0.0)]
    if isinstance(payload, (int, float)):
        numeric = float(payload)
        if math.isfinite(numeric):
            return [(prefix, numeric)]
    return scalars


def _maybe_build_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - optional dependency.
        return None, {
            "enabled": False,
            "status": "unavailable",
            "error": str(exc),
            "log_dir": str(log_dir),
        }
    try:
        writer = SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # pragma: no cover
        return None, {
            "enabled": False,
            "status": "init_failed",
            "error": str(exc),
            "log_dir": str(log_dir),
        }
    return writer, {
        "enabled": True,
        "status": "active",
        "error": None,
        "log_dir": str(log_dir),
    }


def _write_tensorboard_scalars(writer: Any, prefix: str, payload: dict[str, Any], step: int) -> int:
    if writer is None:
        return 0
    count = 0
    for name, value in _flatten_scalar_tree(prefix, payload):
        writer.add_scalar(name, value, global_step=int(step))
        count += 1
    return count


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _emit_log(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
        return
    print(message, flush=True)


def _loader_profile_payload(loader: Any) -> dict[str, Any]:
    return {
        "batch_size": int(getattr(loader, "batch_size", 0) or 0),
        "num_workers": int(getattr(loader, "num_workers", 0) or 0),
        "pin_memory": bool(getattr(loader, "pin_memory", False)),
        "persistent_workers": bool(getattr(loader, "persistent_workers", False)),
        "prefetch_factor": getattr(loader, "prefetch_factor", None),
        "dataset_size": int(len(getattr(loader, "dataset", []))),
        "num_batches": int(len(loader)),
    }


def _build_teacher_dataloader(
    dataset: Any,
    *,
    batch: int,
    workers: int,
    shuffle: bool,
    rank: int,
    drop_last: bool,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
):
    if torch is None or InfiniteDataLoader is None or seed_worker is None:
        raise RuntimeError("ultralytics dataloader dependencies are not available")
    batch = min(batch, len(dataset))
    device_count = torch.cuda.device_count()
    worker_count = min(os.cpu_count() or 1, max(0, int(workers)))
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle and distributed is not None
        else ContiguousDistributedSampler(dataset)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + int(RANK))
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch,
        "shuffle": shuffle and sampler is None,
        "num_workers": worker_count,
        "sampler": sampler,
        "pin_memory": device_count > 0 and bool(pin_memory),
        "collate_fn": getattr(dataset, "collate_fn", None),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": drop_last and len(dataset) % batch != 0,
    }
    if worker_count > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return InfiniteDataLoader(**loader_kwargs)


def _make_teacher_trainer(
    *,
    runtime_params: dict[str, Any],
    log_fn: Callable[[str], None],
):
    if DetectionTrainer is None or torch_distributed_zero_first is None:
        raise RuntimeError("ultralytics trainer dependencies are not available")

    class TeacherDetectionTrainer(DetectionTrainer):
        def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
            self.od_runtime_params = dict(runtime_params)
            self.od_log = log_fn
            self.od_profile_log_path: Path | None = None
            self.od_profile_summary_path: Path | None = None
            self.od_tensorboard_dir: Path | None = None
            self.od_tensorboard_writer = None
            self.od_tensorboard_status: dict[str, Any] = {
                "enabled": False,
                "status": "not_initialized",
                "error": None,
                "log_dir": None,
            }
            self.od_profile_history: list[dict[str, Any]] = []
            self.od_epoch_timing_window: deque[dict[str, float]] = deque(
                maxlen=max(1, int(runtime_params["profile_window"]))
            )
            self.od_epoch_started_at = 0.0
            self.od_last_batch_end_at: float | None = None
            self.od_batch_started_at: float | None = None
            self.od_pending_wait_sec = 0.0
            self.od_epoch_step = 0
            self.od_global_step = 0

        def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
            assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            shuffle = mode == "train"
            if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
                LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
                shuffle = False
            return _build_teacher_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
                pin_memory=bool(runtime_params["pin_memory"]),
                persistent_workers=bool(runtime_params["persistent_workers"]),
                prefetch_factor=runtime_params["prefetch_factor"],
            )

    try:
        import numpy as np
        from ultralytics.utils import LOGGER
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics runtime dependencies are not available") from exc

    def on_train_start(trainer: Any) -> None:
        trainer.od_profile_log_path = Path(trainer.save_dir) / "profile_log.jsonl"
        trainer.od_profile_summary_path = Path(trainer.save_dir) / "profile_summary.json"
        trainer.od_tensorboard_dir = Path(trainer.save_dir) / "tensorboard"
        trainer.od_tensorboard_dir.mkdir(parents=True, exist_ok=True)
        trainer.od_tensorboard_writer, trainer.od_tensorboard_status = _maybe_build_summary_writer(trainer.od_tensorboard_dir)
        train_loader_profile = _loader_profile_payload(trainer.train_loader)
        val_loader_profile = _loader_profile_payload(trainer.test_loader)
        trainer.od_log(
            f"[od_bootstrap.train] tensorboard status={trainer.od_tensorboard_status['status']} "
            f"log_dir={trainer.od_tensorboard_status['log_dir']}"
        )
        trainer.od_log(
            f"[od_bootstrap.train] loader train batch={train_loader_profile['batch_size']} "
            f"workers={train_loader_profile['num_workers']} pin_memory={train_loader_profile['pin_memory']} "
            f"persistent_workers={train_loader_profile['persistent_workers']} "
            f"prefetch_factor={train_loader_profile['prefetch_factor']} batches={train_loader_profile['num_batches']}"
        )
        trainer.od_log(
            f"[od_bootstrap.train] loader val batch={val_loader_profile['batch_size']} "
            f"workers={val_loader_profile['num_workers']} pin_memory={val_loader_profile['pin_memory']} "
            f"persistent_workers={val_loader_profile['persistent_workers']} "
            f"prefetch_factor={val_loader_profile['prefetch_factor']} batches={val_loader_profile['num_batches']}"
        )
        _write_tensorboard_scalars(
            trainer.od_tensorboard_writer,
            "loader/train",
            train_loader_profile,
            step=0,
        )
        _write_tensorboard_scalars(
            trainer.od_tensorboard_writer,
            "loader/val",
            val_loader_profile,
            step=0,
        )
        _write_tensorboard_scalars(
            trainer.od_tensorboard_writer,
            "runtime",
            {key: value for key, value in runtime_params.items() if key != "profile_device_sync"} | {"profile_device_sync": runtime_params["profile_device_sync"]},
            step=0,
        )
        if trainer.od_tensorboard_writer is not None:
            trainer.od_tensorboard_writer.flush()

    def on_train_epoch_start(trainer: Any) -> None:
        trainer.od_epoch_started_at = time.perf_counter()
        trainer.od_last_batch_end_at = trainer.od_epoch_started_at
        trainer.od_batch_started_at = None
        trainer.od_pending_wait_sec = 0.0
        trainer.od_epoch_step = 0
        trainer.od_epoch_timing_window = deque(maxlen=max(1, int(runtime_params["profile_window"])))

    def on_train_batch_start(trainer: Any) -> None:
        _sync_timing_device(trainer.device, bool(runtime_params["profile_device_sync"]))
        now = time.perf_counter()
        last_batch_end_at = trainer.od_last_batch_end_at or now
        trainer.od_pending_wait_sec = max(0.0, now - last_batch_end_at)
        trainer.od_batch_started_at = now

    def on_train_batch_end(trainer: Any) -> None:
        _sync_timing_device(trainer.device, bool(runtime_params["profile_device_sync"]))
        ended_at = time.perf_counter()
        started_at = trainer.od_batch_started_at or ended_at
        iteration_sec = max(0.0, ended_at - started_at)
        wait_sec = max(0.0, float(trainer.od_pending_wait_sec))
        compute_sec = max(0.0, iteration_sec)
        trainer.od_last_batch_end_at = ended_at
        trainer.od_epoch_step += 1
        trainer.od_global_step += 1
        timing_row = {
            "iteration_sec": iteration_sec,
            "wait_sec": wait_sec,
            "compute_sec": compute_sec,
        }
        trainer.od_epoch_timing_window.append(timing_row)

        step_index = int(trainer.od_epoch_step)
        total_steps = int(len(trainer.train_loader))
        log_every_n_steps = max(1, int(runtime_params["log_every_n_steps"]))
        should_log = step_index % log_every_n_steps == 0 or step_index == total_steps
        if not should_log:
            return

        elapsed_sec = max(0.0, ended_at - trainer.od_epoch_started_at)
        profile_summary = _timing_profile(list(trainer.od_epoch_timing_window))
        remaining_steps = max(0, total_steps - step_index)
        eta_sec = float(profile_summary["iteration_sec"]["mean"]) * float(remaining_steps)
        losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        payload = {
            "epoch": int(trainer.epoch) + 1,
            "epoch_total": int(trainer.epochs),
            "step": step_index,
            "total_steps": total_steps,
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
            "losses": losses,
            "profile": profile_summary,
        }
        if trainer.od_profile_log_path is not None:
            _append_jsonl(trainer.od_profile_log_path, payload)
        _write_tensorboard_scalars(
            trainer.od_tensorboard_writer,
            "train_step",
            {
                "progress": {
                    "epoch": payload["epoch"],
                    "step": step_index,
                    "total_steps": total_steps,
                    "elapsed_sec": elapsed_sec,
                    "eta_sec": eta_sec,
                },
                "loss": {key.split("/", 1)[-1]: value for key, value in losses.items()},
                "profile_sec": {
                    "iteration_mean": profile_summary["iteration_sec"]["mean"],
                    "iteration_p50": profile_summary["iteration_sec"]["p50"],
                    "iteration_p99": profile_summary["iteration_sec"]["p99"],
                    "wait_mean": profile_summary["wait_sec"]["mean"],
                    "compute_mean": profile_summary["compute_sec"]["mean"],
                },
            },
            step=trainer.od_global_step,
        )
        loss_items = " ".join(f"{key.split('/', 1)[-1]}={float(value):.4f}" for key, value in losses.items())
        trainer.od_log(
            "\n".join(
                [
                    (
                        f"[od_bootstrap.train] profile epoch={payload['epoch']}/{payload['epoch_total']} "
                        f"step={step_index}/{total_steps}"
                    ),
                    (
                        f"  progress: elapsed={_format_duration(elapsed_sec)} "
                        f"eta={_format_duration(eta_sec)}"
                    ),
                    f"  losses: {loss_items}",
                    (
                        "  iter_ms: "
                        f"mean={profile_summary['iteration_sec']['mean'] * 1000.0:.3f} "
                        f"p50={profile_summary['iteration_sec']['p50'] * 1000.0:.3f} "
                        f"p99={profile_summary['iteration_sec']['p99'] * 1000.0:.3f}"
                    ),
                    (
                        "  timing_ms: "
                        f"wait={profile_summary['wait_sec']['mean'] * 1000.0:.3f} "
                        f"compute={profile_summary['compute_sec']['mean'] * 1000.0:.3f}"
                    ),
                ]
            )
        )

    def on_train_epoch_end(trainer: Any) -> None:
        epoch_profile = {
            "epoch": int(trainer.epoch) + 1,
            "timing_profile": _timing_profile(list(trainer.od_epoch_timing_window)),
            "loader": _loader_profile_payload(trainer.train_loader),
        }
        trainer.od_profile_history.append(epoch_profile)

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch_index = int(trainer.epoch) + 1
        epoch_payload = {
            "train": {
                "loss": {
                    key.split("/", 1)[-1]: value
                    for key, value in trainer.label_loss_items(trainer.tloss, prefix="train").items()
                },
                "profile_sec": trainer.od_profile_history[-1]["timing_profile"] if trainer.od_profile_history else {},
            },
            "lr": dict(getattr(trainer, "lr", {})),
            "val": dict(getattr(trainer, "metrics", {})),
        }
        _write_tensorboard_scalars(
            trainer.od_tensorboard_writer,
            "epoch",
            epoch_payload,
            step=epoch_index,
        )
        if trainer.od_tensorboard_writer is not None:
            trainer.od_tensorboard_writer.flush()

    def on_train_end(trainer: Any) -> None:
        payload = {
            "runtime": dict(runtime_params),
            "tensorboard": dict(trainer.od_tensorboard_status),
            "train_loader": _loader_profile_payload(trainer.train_loader),
            "val_loader": _loader_profile_payload(trainer.test_loader),
            "epochs": trainer.od_profile_history,
        }
        if trainer.od_profile_summary_path is not None:
            trainer.od_profile_summary_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.od_profile_summary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        if trainer.od_tensorboard_writer is not None:
            trainer.od_tensorboard_writer.flush()
            trainer.od_tensorboard_writer.close()

    TeacherDetectionTrainer._od_bootstrap_runtime_params = dict(runtime_params)
    TeacherDetectionTrainer._od_bootstrap_log_fn = log_fn
    callbacks = {
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_batch_start": on_train_batch_start,
        "on_train_batch_end": on_train_batch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    TeacherDetectionTrainer._od_bootstrap_callbacks = callbacks
    return TeacherDetectionTrainer, callbacks


def train_teacher_with_ultralytics(
    *,
    teacher_name: str,
    dataset_yaml: Path,
    output_root: Path,
    model_size: str,
    weights: str | None,
    train_params: dict[str, Any],
    runtime_params: dict[str, Any],
    exist_ok: bool,
) -> dict[str, Any]:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")

    resolved_weights = _coerce_weights_name(model_size, weights)
    model = YOLO(resolved_weights)
    log_fn = _emit_log
    trainer_cls, callbacks = _make_teacher_trainer(runtime_params=runtime_params, log_fn=log_fn)
    if hasattr(model, "add_callback"):
        for event_name, callback in callbacks.items():
            model.add_callback(event_name, callback)

    train_kwargs = {
        "data": str(dataset_yaml),
        "project": str(output_root),
        "name": teacher_name,
        "exist_ok": bool(exist_ok),
        "pretrained": True,
        **train_params,
    }
    train_result = model.train(trainer=trainer_cls, **train_kwargs)
    run_dir = _extract_run_dir(train_result, output_root / teacher_name)
    trainer = getattr(model, "trainer", None)
    best_checkpoint = run_dir / "weights" / "best.pt"
    last_checkpoint = run_dir / "weights" / "last.pt"
    profile_log_path = run_dir / "profile_log.jsonl"
    profile_summary_path = run_dir / "profile_summary.json"
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_status = dict(
        getattr(
            trainer,
            "od_tensorboard_status",
            {
                "enabled": False,
                "status": "unknown_no_callbacks",
                "error": None,
                "log_dir": str(tensorboard_dir),
            },
        )
    )
    tensorboard_event_files = sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents*")) if tensorboard_dir.is_dir() else []

    summary = {
        "teacher_name": teacher_name,
        "weights": resolved_weights,
        "dataset_yaml": str(dataset_yaml),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "profile_log_path": str(profile_log_path),
        "profile_summary_path": str(profile_summary_path),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_status": tensorboard_status,
        "tensorboard_event_files": tensorboard_event_files,
        "runtime": dict(runtime_params),
        "train_kwargs": train_kwargs,
        "train_result_type": type(train_result).__name__,
    }
    summary_path = run_dir / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary
