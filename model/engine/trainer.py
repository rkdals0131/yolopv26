"""Step-based training for the focused traffic-light and road-marking model."""

from __future__ import annotations

import gc
import os
import random
import signal
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from model.data.dataset import slice_focused_batch


@dataclass(frozen=True)
class FocusedTrainerConfig:
    output_dir: Path
    device: str = "cuda"
    precision: str = "bf16"
    microbatch_size: int = 4
    min_microbatch_size: int = 1
    checkpoint_interval_sec: float = 600.0
    max_consecutive_failures: int = 3
    grad_clip_norm: float | None = None
    stage: str = "joint"

    def __post_init__(self) -> None:
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16, or fp32")
        if self.microbatch_size < 1 or self.min_microbatch_size < 1:
            raise ValueError("microbatch sizes must be positive")
        if self.min_microbatch_size > self.microbatch_size:
            raise ValueError("min_microbatch_size exceeds microbatch_size")
        if self.checkpoint_interval_sec <= 0 or self.max_consecutive_failures < 1:
            raise ValueError("checkpoint interval and failure limit must be positive")


class FocusedBatchAdapter:
    """The three supervised terms of the focused detector/roadmark batch."""

    @staticmethod
    def slice_batch(batch: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
        return slice_focused_batch(batch, start, stop)

    @staticmethod
    def term_counts(batch: dict[str, Any]) -> dict[str, int]:
        positive = ((batch["roadmark_target"] > 0) & batch["roadmark_valid"]).flatten(2).any(dim=2)
        return {
            "det": int(batch["det_labeled"].sum().item()),
            "roadmark_bce": int(batch["roadmark_valid"].sum().item()),
            "roadmark_dice": int(positive.sum().item()),
        }

    @staticmethod
    def weighted_terms(losses: dict[str, torch.Tensor], criterion: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            "det": losses["det"] * float(getattr(criterion, "det_weight", 1.0)),
            "roadmark_bce": losses["roadmark_bce"] * float(getattr(criterion, "roadmark_weight", 1.0)),
            "roadmark_dice": losses["roadmark_dice"] * float(getattr(criterion, "roadmark_weight", 1.0)),
        }

    @staticmethod
    def sample_ids(batch: dict[str, Any]) -> list[str]:
        return [str(meta.get("sample_id", meta)) for meta in batch.get("meta", ())]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as output:
            temporary = output.name
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _model_config(model: torch.nn.Module) -> dict[str, Any] | None:
    describe = getattr(model, "model_config", None)
    if callable(describe):
        return dict(describe())
    config = getattr(model, "config", None)
    if is_dataclass(config):
        return asdict(config)
    return dict(config) if isinstance(config, dict) else None


def _snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # Forward may update BatchNorm buffers before an OOM or nonfinite loss.
    return {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()}


@torch.no_grad()
def _restore_buffers(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    for name, buffer in model.named_buffers():
        buffer.copy_(snapshot[name].to(device=buffer.device))


class FocusedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        sampler: Any,
        config: FocusedTrainerConfig,
        *,
        run_metadata: dict[str, Any] | None = None,
        batch_adapter: Any | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        set_stage = getattr(model, "set_train_stage", None)
        if callable(set_stage):
            set_stage(config.stage)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.batch_adapter = batch_adapter if batch_adapter is not None else FocusedBatchAdapter()
        self.run_metadata = dict(run_metadata or {})
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        self.global_step = 0
        self.skipped_updates = 0
        self.oom_retries = 0
        self.consecutive_failures = 0
        self.best_metric: float | None = None
        self.best_mode = "max"
        self.best_step: int | None = None
        self.microbatch_size = config.microbatch_size
        self._stop_requested = False
        self._unsafe_state = False
        self._last_checkpoint_time = time.monotonic()
        self._last_saved_position: int | None = None
        self._last_failure_reason = ""
        self.last_step_losses: dict[str, float] = {}
        self.planned_steps: int | None = None
        self._progress_at: tuple[int, int] | None = None

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.config.output_dir) / "checkpoints"

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def previous_path(self) -> Path:
        return self.checkpoint_dir / "previous.pt"

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "rng": _rng_state(),
            "sampler": self.sampler.state_dict(),
            "global_step": self.global_step,
            "skipped_updates": self.skipped_updates,
            "oom_retries": self.oom_retries,
            "consecutive_failures": self.consecutive_failures,
            "microbatch_size": self.microbatch_size,
            "stage": self.config.stage,
            "precision": self.config.precision,
            "best_metric": self.best_metric,
            "best_mode": self.best_mode,
            "best_step": self.best_step,
            "model_config": _model_config(self.model),
            "run_metadata": self.run_metadata,
            "planned_steps": self.planned_steps,
        }

    def save_checkpoint(self) -> Path:
        """Publish a complete, clean optimizer-boundary state."""
        if self._unsafe_state:
            raise RuntimeError("optimizer state is uncertain; reload a normal checkpoint")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.latest_path.exists():
            os.replace(self.latest_path, self.previous_path)
            _fsync_directory(self.checkpoint_dir)
        try:
            _atomic_save(self._checkpoint_state(), self.latest_path)
        except BaseException:
            if not self.latest_path.exists() and self.previous_path.exists():
                os.replace(self.previous_path, self.latest_path)
                _fsync_directory(self.checkpoint_dir)
            raise
        self._last_checkpoint_time = time.monotonic()
        self._last_saved_position = int(self.sampler.position)
        return self.latest_path

    def load_checkpoint(self, path: str | Path | None = None) -> Path:
        """Restore full training state; create a fresh DataLoader after this call."""
        candidate = Path(path) if path is not None else self.latest_path
        if not candidate.is_file() and path is None and self.previous_path.is_file():
            candidate = self.previous_path
        # RNG byte tensors must remain on CPU even when model training is on CUDA.
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        if checkpoint["format_version"] != 1:
            raise RuntimeError(f"unsupported focused checkpoint: {candidate}")
        if checkpoint["stage"] != self.config.stage or checkpoint["precision"] != self.config.precision:
            raise RuntimeError("checkpoint stage or precision differs from this training run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.criterion.load_state_dict(checkpoint.get("criterion", {}))
        if self.scheduler is not None:
            if checkpoint["scheduler"] is None:
                raise RuntimeError("checkpoint has no scheduler state")
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler.is_enabled():
            if checkpoint["scaler"] is None:
                raise RuntimeError("checkpoint has no GradScaler state")
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.sampler.load_state_dict(checkpoint["sampler"])
        _restore_rng(checkpoint["rng"])
        self.global_step = int(checkpoint["global_step"])
        self.skipped_updates = int(checkpoint["skipped_updates"])
        self.oom_retries = int(checkpoint.get("oom_retries", 0))
        self.consecutive_failures = int(checkpoint["consecutive_failures"])
        self.microbatch_size = int(checkpoint["microbatch_size"])
        self.best_metric = checkpoint["best_metric"]
        self.best_mode = checkpoint["best_mode"]
        self.best_step = checkpoint["best_step"]
        self.run_metadata = dict(checkpoint.get("run_metadata") or {})
        self.planned_steps = checkpoint.get("planned_steps")
        if self.planned_steps is not None:
            self.planned_steps = int(self.planned_steps)
            self._set_progress(self.planned_steps)
        if self.best_path.is_file():
            best = torch.load(self.best_path, map_location="cpu", weights_only=False)
            if best.get("stage") == self.config.stage:
                metric = float(best["metric"])
                mode = str(best["mode"])
                better = self.best_metric is None or (
                    metric > self.best_metric if mode == "max" else metric < self.best_metric
                )
                if better:
                    self.best_metric = metric
                    self.best_mode = mode
                    self.best_step = int(best["global_step"])
        self._unsafe_state = False
        self._last_checkpoint_time = time.monotonic()
        self._last_saved_position = int(self.sampler.position)
        return candidate

    def update_best(self, metric: float, *, mode: str = "max") -> bool:
        """Store a single weights-only selection artifact for an evaluated step."""
        if self._unsafe_state:
            raise RuntimeError("cannot select weights after uncertain optimizer update")
        if mode not in {"max", "min"} or not np.isfinite(metric):
            raise ValueError("best metric must be finite and mode must be max or min")
        if self.best_metric is not None:
            if mode != self.best_mode:
                raise ValueError("best metric direction changed within a run")
            if not ((metric > self.best_metric) if mode == "max" else (metric < self.best_metric)):
                return False
        _atomic_save(
            {"model": self.model.state_dict(), "metric": float(metric), "mode": mode,
             "global_step": self.global_step, "stage": self.config.stage,
             "model_config": _model_config(self.model),
             "run_metadata": self.run_metadata},
            self.best_path,
        )
        self.best_metric = float(metric)
        self.best_mode = mode
        self.best_step = self.global_step
        self.save_checkpoint()
        return True

    def _to_device(self, micro: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in micro.items()
        }

    def _autocast(self):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.config.precision)
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None)

    def _attempt_update(self, batch: dict[str, Any], physical_size: int) -> bool:
        sample_count = int(batch["image"].shape[0])
        total_counts = self.batch_adapter.term_counts(batch)
        if not any(total_counts.values()):
            self._last_failure_reason = "no_labeled_task"
            return False
        self.optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        for start in range(0, sample_count, physical_size):
            stop = min(sample_count, start + physical_size)
            micro_cpu = self.batch_adapter.slice_batch(batch, start, stop)
            micro_counts = self.batch_adapter.term_counts(micro_cpu)
            micro = self._to_device(micro_cpu)
            with self._autocast():
                forward = getattr(self.model, "forward_for_loss", self.model)
                outputs = forward(micro["image"])
            with torch.autocast(device_type=self.device.type, enabled=False):
                losses = self.criterion(outputs, micro)
                weighted_terms = self.batch_adapter.weighted_terms(losses, self.criterion)
                scaled_terms = {
                    key: term * (micro_counts[key] / total_counts[key] if total_counts[key] else 0.0)
                    for key, term in weighted_terms.items()
                }
                loss = sum(scaled_terms.values())
            if not torch.isfinite(loss.detach()).item():
                invalid_terms = [
                    key for key, term in scaled_terms.items() if not torch.isfinite(term.detach()).item()
                ]
                self._last_failure_reason = "nonfinite_loss:" + ",".join(invalid_terms)
                return False
            for key, term in scaled_terms.items():
                detached = term.detach()
                accumulated[key] = accumulated[key] + detached if key in accumulated else detached
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            del outputs, losses, weighted_terms, scaled_terms, loss, micro
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        gradients_finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item() if parameters else True
        if not gradients_finite:
            self._last_failure_reason = "nonfinite_gradient"
            if self.scaler.is_enabled():
                self.scaler.update()
            return False
        if self.config.grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip_norm)
            if not torch.isfinite(norm).item():
                self._last_failure_reason = "nonfinite_gradient_norm"
                if self.scaler.is_enabled():
                    self.scaler.update(new_scale=self.scaler.get_scale() / 2)
                return False
        try:
            if self.scaler.is_enabled():
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < old_scale:
                    self._last_failure_reason = "fp16_scaler_overflow"
                    return False
            else:
                self.optimizer.step()
        except BaseException as error:
            self._unsafe_state = True
            if _is_oom(error):
                raise RuntimeError(
                    f"optimizer-step OOM; state may be partial. Reload {self.latest_path} "
                    f"(or {self.previous_path}) before continuing."
                ) from error
            raise
        self.global_step += 1
        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except BaseException:
                self._unsafe_state = True
                raise
        values = torch.stack([accumulated[key] for key in accumulated]).float().cpu().tolist()
        self.last_step_losses = dict(zip(accumulated, values))
        self.last_step_losses["total"] = sum(values)
        return True

    def train_batch(self, batch: dict[str, Any]) -> bool:
        """Consume one logical batch, retrying recoverable forward/backward OOM."""
        if self._unsafe_state:
            raise RuntimeError("optimizer state is uncertain; reload a normal checkpoint")
        sample_count = int(batch["image"].shape[0])
        if sample_count < 1:
            raise ValueError("empty logical batch")
        rng = _rng_state()
        buffers = _snapshot_buffers(self.model)
        physical_size = min(self.microbatch_size, sample_count)
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            _restore_rng(rng)
            _restore_buffers(self.model, buffers)
            try:
                completed = self._attempt_update(batch, physical_size)
                break
            except BaseException as error:
                if not _is_oom(error) or self._unsafe_state:
                    self._unsafe_state = True
                    self.optimizer.zero_grad(set_to_none=True)
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                if physical_size <= self.config.min_microbatch_size:
                    self._unsafe_state = True
                    raise RuntimeError(
                        f"OOM at minimum physical microbatch size; samples={self.batch_adapter.sample_ids(batch)}"
                    ) from error
                self.oom_retries += 1
                physical_size = max(self.config.min_microbatch_size, physical_size // 2)
                self.microbatch_size = physical_size
            # The caught exception's traceback can retain failed autograd tensors
            # until the except suite exits.
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        if completed:
            self._commit(sample_count)
            self.consecutive_failures = 0
            if time.monotonic() - self._last_checkpoint_time >= self.config.checkpoint_interval_sec:
                self.save_checkpoint()
        else:
            _restore_buffers(self.model, buffers)
            self._commit(sample_count)
            self.skipped_updates += 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.config.max_consecutive_failures:
                raise RuntimeError(
                    f"{self.consecutive_failures} consecutive rejected updates; "
                    f"reason={self._last_failure_reason}; "
                    f"task_counts={self.batch_adapter.term_counts(batch)}; "
                    f"samples={self.batch_adapter.sample_ids(batch)}; "
                    f"last normal checkpoint: {self.latest_path}"
                )
        return completed

    def _commit(self, sample_count: int) -> None:
        try:
            self.sampler.commit(sample_count)
        except BaseException:
            self._unsafe_state = True
            raise

    def _set_progress(self, total_steps: int) -> None:
        progress = getattr(self.criterion, "set_progress", None)
        marker = (self.global_step, total_steps)
        if callable(progress) and self.config.stage != "roadmark" and self._progress_at != marker:
            progress(*marker)
            self._progress_at = marker

    def fit(
        self,
        loader: Iterable[dict[str, Any]],
        *,
        max_steps: int,
        planned_steps: int | None = None,
        on_step: Callable[["FocusedTrainer", dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train to a completed optimizer-step count, saving at safe stop points."""
        if max_steps < self.global_step:
            raise ValueError("max_steps is earlier than resumed global_step")
        planned_steps = max_steps if planned_steps is None else planned_steps
        if planned_steps < max_steps:
            raise ValueError("planned_steps cannot be earlier than max_steps")
        if self.planned_steps is not None and planned_steps != self.planned_steps:
            raise ValueError("planned_steps differs from the resumed learning schedule")
        self.planned_steps = planned_steps
        self._stop_requested = False
        self.model.train()
        self._set_progress(planned_steps)
        if not self.latest_path.exists():
            self.save_checkpoint()
        previous_handlers: dict[signal.Signals, Any] = {}

        def request_stop(_signum: int, _frame: Any) -> None:
            self._stop_requested = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        try:
            iterator = iter(loader)
            while self.global_step < max_steps and not self._stop_requested:
                batch_wait_started = time.monotonic()
                batch = next(iterator)
                batch_wait_sec = time.monotonic() - batch_wait_started
                self.model.train()
                update_started = time.monotonic()
                completed = self.train_batch(batch)
                update_wall_sec = time.monotonic() - update_started
                if completed:
                    self._set_progress(planned_steps)
                if completed and on_step is not None:
                    on_step(self, {
                        "global_step": self.global_step,
                        "sampler_position": int(self.sampler.position),
                        "microbatch_size": self.microbatch_size,
                        "skipped_updates": self.skipped_updates,
                        "oom_retries": self.oom_retries,
                        "losses": dict(self.last_step_losses),
                        "batch_wait_sec": batch_wait_sec,
                        "update_wall_sec": update_wall_sec,
                    })
                if self._stop_requested:
                    break
            if not self._unsafe_state and self.consecutive_failures == 0:
                if self._last_saved_position != int(self.sampler.position):
                    self.save_checkpoint()
            return {
                "global_step": self.global_step,
                "skipped_updates": self.skipped_updates,
                "oom_retries": self.oom_retries,
                "stopped_by_signal": self._stop_requested,
                "latest_checkpoint": str(self.latest_path),
                "best_weights": str(self.best_path) if self.best_path.is_file() else None,
                "microbatch_size": self.microbatch_size,
            }
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
