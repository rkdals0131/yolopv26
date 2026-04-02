from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from common.io import write_json
from common.scalars import flatten_scalar_tree
from . import runtime_progress
from .runtime_artifacts import refresh_latest_teacher_artifacts
from .runtime_callbacks import TeacherRuntimeSupport, build_teacher_runtime_callbacks
from .runtime_resume import (
    checkpoint_resume_metadata,
    coerce_weights_name,
    extract_run_dir,
    resolve_resume_argument,
    resolve_resume_checkpoint_path,
)
from .runtime_tensorboard import (
    build_epoch_tensorboard_payload,
    build_train_step_tensorboard_payload,
    maybe_build_summary_writer,
    write_tensorboard_scalars,
)

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
    from ultralytics.engine import trainer as ultra_trainer
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import torch_distributed_zero_first
except ImportError:  # pragma: no cover - dependency absence handled in tests with patching.
    YOLO = None
    DEFAULT_CFG = None
    ultra_trainer = None
    DetectionTrainer = None
    InfiniteDataLoader = None
    ContiguousDistributedSampler = None
    seed_worker = None
    torch_distributed_zero_first = None
    RANK = -1

# Backwards-compatible aliases for existing tests/callers while teacher runtime
# modules expose a clearer public helper surface.
_append_jsonl = runtime_progress.append_jsonl
_build_epoch_tensorboard_payload = build_epoch_tensorboard_payload
_build_live_postfix = runtime_progress.build_live_postfix
_build_rich_progress_bar = runtime_progress.build_rich_progress_bar
_build_train_step_tensorboard_payload = build_train_step_tensorboard_payload
_checkpoint_resume_metadata = checkpoint_resume_metadata
_coerce_weights_name = coerce_weights_name
_emit_log = runtime_progress.emit_log
_extract_run_dir = extract_run_dir
_flatten_scalar_tree = flatten_scalar_tree
_format_duration = runtime_progress.format_duration
_install_ultralytics_postfix_renderer = runtime_progress.install_ultralytics_postfix_renderer
_loader_profile_payload = runtime_progress.loader_profile_payload
_maybe_build_summary_writer = maybe_build_summary_writer
_refresh_latest_teacher_artifacts = refresh_latest_teacher_artifacts
_resolve_resume_argument = resolve_resume_argument
_resolve_resume_checkpoint_path = resolve_resume_checkpoint_path
_set_progress_postfix = runtime_progress.set_progress_postfix
_timing_profile = runtime_progress.timing_profile
_timestamp_token = runtime_progress.timestamp_token
_write_tensorboard_scalars = write_tensorboard_scalars


def _sync_timing_device(device: Any, enabled: bool) -> None:
    runtime_progress.sync_timing_device(torch, device, enabled)


def _build_teacher_dataloader_kwargs(
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
) -> dict[str, Any]:
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
    return loader_kwargs


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
    loader_kwargs = _build_teacher_dataloader_kwargs(
        dataset,
        batch=batch,
        workers=workers,
        shuffle=shuffle,
        rank=rank,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return InfiniteDataLoader(**loader_kwargs)


def _build_teacher_callbacks(
    *,
    runtime_params: dict[str, Any],
) -> dict[str, Callable[[Any], None]]:
    support = TeacherRuntimeSupport(
        time_module=time,
        deque_type=deque,
        append_jsonl_fn=_append_jsonl,
        sync_timing_device_fn=_sync_timing_device,
        timing_profile_fn=_timing_profile,
        build_live_postfix_fn=_build_live_postfix,
        set_progress_postfix_fn=_set_progress_postfix,
        loader_profile_payload_fn=_loader_profile_payload,
        maybe_build_summary_writer_fn=maybe_build_summary_writer,
        write_tensorboard_scalars_fn=write_tensorboard_scalars,
        build_train_step_tensorboard_payload_fn=build_train_step_tensorboard_payload,
        build_epoch_tensorboard_payload_fn=build_epoch_tensorboard_payload,
    )
    return build_teacher_runtime_callbacks(
        runtime_params=runtime_params,
        support=support,
    )


def _build_teacher_train_kwargs(
    *,
    dataset_yaml: Path,
    teacher_root: Path,
    run_name: str,
    train_params: dict[str, Any],
    exist_ok: bool,
) -> dict[str, Any]:
    return {
        "data": str(dataset_yaml),
        "project": str(teacher_root),
        "name": run_name,
        "exist_ok": bool(exist_ok),
        "pretrained": True,
        **train_params,
    }


def _teacher_runtime_artifact_paths(run_dir: Path) -> dict[str, Path]:
    tensorboard_dir = run_dir / "tensorboard"
    return {
        "best_checkpoint": run_dir / "weights" / "best.pt",
        "last_checkpoint": run_dir / "weights" / "last.pt",
        "profile_log": run_dir / "profile_log.jsonl",
        "profile_summary": run_dir / "profile_summary.json",
        "tensorboard_dir": tensorboard_dir,
        "train_summary": run_dir / "train_summary.json",
    }


def _resolve_tensorboard_status(trainer: Any, tensorboard_dir: Path) -> dict[str, Any]:
    return dict(
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


def _build_teacher_train_summary(
    *,
    teacher_name: str,
    resolved_weights: str,
    dataset_yaml: Path,
    teacher_root: Path,
    run_dir: Path,
    runtime_params: dict[str, Any],
    train_kwargs: dict[str, Any],
    train_result: Any,
    trainer: Any,
) -> dict[str, Any]:
    artifact_paths = _teacher_runtime_artifact_paths(run_dir)
    tensorboard_dir = artifact_paths["tensorboard_dir"]
    tensorboard_status = _resolve_tensorboard_status(trainer, tensorboard_dir)
    tensorboard_event_files = (
        sorted(path.name for path in tensorboard_dir.glob("events.out.tfevents*"))
        if tensorboard_dir.is_dir()
        else []
    )
    return {
        "teacher_name": teacher_name,
        "weights": resolved_weights,
        "dataset_yaml": str(dataset_yaml),
        "teacher_root": str(teacher_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(artifact_paths["best_checkpoint"]),
        "last_checkpoint": str(artifact_paths["last_checkpoint"]),
        "profile_log_path": str(artifact_paths["profile_log"]),
        "profile_summary_path": str(artifact_paths["profile_summary"]),
        "tensorboard_dir": str(tensorboard_dir),
        "tensorboard_status": tensorboard_status,
        "tensorboard_event_files": tensorboard_event_files,
        "runtime": dict(runtime_params),
        "train_kwargs": train_kwargs,
        "train_result_type": type(train_result).__name__,
    }


def _make_teacher_trainer(
    *,
    runtime_params: dict[str, Any],
    log_fn: Callable[[str], None],
):
    if DetectionTrainer is None or torch_distributed_zero_first is None or ultra_trainer is None:
        raise RuntimeError("ultralytics trainer dependencies are not available")

    class TeacherDetectionTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            self.od_resume_base_epochs = None
            self.od_resume_start_epoch = None
            resume = self.args.resume
            if resume:
                try:
                    exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                    requested = Path(ultra_trainer.check_file(resume) if exists else ultra_trainer.get_latest_run())
                    last = resolve_resume_checkpoint_path(requested) or requested
                    metadata = checkpoint_resume_metadata(last)
                    ckpt_args = dict(metadata["train_args"])
                    if not ckpt_args:
                        raise FileNotFoundError(f"resume checkpoint is missing train_args: {last}")
                    if not metadata["resumable"]:
                        raise FileNotFoundError(
                            f"resume checkpoint is finalized and not resumable: {last}. "
                            "Use a saved epoch*.pt or last_resume.pt checkpoint instead."
                        )
                    if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                        ckpt_args["data"] = self.args.data

                    resume = True
                    self.od_resume_base_epochs = int(ckpt_args.get("epochs") or 0) or None
                    self.od_resume_start_epoch = int(metadata["epoch"]) + 1
                    self.args = ultra_trainer.get_cfg(ckpt_args)
                    self.args.model = self.args.resume = str(last)
                    for k in (
                        "epochs",
                        "imgsz",
                        "batch",
                        "device",
                        "close_mosaic",
                        "augmentations",
                        "save_period",
                        "workers",
                        "cache",
                        "patience",
                        "time",
                        "freeze",
                        "val",
                        "plots",
                    ):
                        if k in overrides:
                            setattr(self.args, k, overrides[k])

                    if ckpt_args.get("augmentations") is not None:
                        ultra_trainer.LOGGER.warning(
                            "Custom Albumentations transforms were used in the original training run but are not "
                            "being restored. To preserve custom augmentations when resuming, you need to pass the "
                            "'augmentations' parameter again to get expected results. Example: \n"
                            f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                        )
                except Exception as e:
                    raise FileNotFoundError(
                        "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                        "i.e. 'yolo train resume model=path/to/last.pt'"
                    ) from e
            self.resume = resume

        def _build_extended_resume_lf(self):
            base_epochs = getattr(self, "od_resume_base_epochs", None)
            start_epoch = getattr(self, "od_resume_start_epoch", None)
            if not self.resume or base_epochs is None or start_epoch is None or self.epochs <= base_epochs:
                return None

            lrf = float(self.args.lrf)
            total_epochs = int(self.epochs)
            resume_epoch = int(start_epoch)

            def _base_linear(epoch_index: int) -> float:
                return max(1.0 - float(epoch_index) / float(base_epochs), 0.0) * (1.0 - lrf) + lrf

            def _base_cosine(epoch_index: int) -> float:
                progress = max(float(epoch_index), 0.0) * math.pi / float(base_epochs)
                return ((1.0 - math.cos(progress)) / 2.0) * (lrf - 1.0) + 1.0

            base_fn = _base_cosine if self.args.cos_lr else _base_linear
            start_factor = float(base_fn(resume_epoch))
            remaining_epochs = max(total_epochs - resume_epoch, 1)

            if self.args.cos_lr:
                def _extended_lf(epoch_index: int) -> float:
                    if epoch_index <= resume_epoch:
                        return float(base_fn(epoch_index))
                    progress = min(max(float(epoch_index - resume_epoch) / float(remaining_epochs), 0.0), 1.0)
                    return ((1.0 - math.cos(progress * math.pi)) / 2.0) * (lrf - start_factor) + start_factor
            else:
                def _extended_lf(epoch_index: int) -> float:
                    if epoch_index <= resume_epoch:
                        return float(base_fn(epoch_index))
                    progress = min(max(float(epoch_index - resume_epoch) / float(remaining_epochs), 0.0), 1.0)
                    return (1.0 - progress) * start_factor + progress * lrf

            return _extended_lf

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
            self.od_pbar = None

        def _setup_scheduler(self):
            extended_lf = self._build_extended_resume_lf()
            if extended_lf is not None:
                self.lf = extended_lf
                self.scheduler = ultra_trainer.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
                return
            super()._setup_scheduler()

        def save_model(self):
            super().save_model()
            resume_last = self.wdir / "last_resume.pt"
            if self.last.is_file():
                shutil.copy2(self.last, resume_last)

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

        def _do_train(self):
            if self.world_size > 1:
                self._setup_ddp()
            self._setup_train()

            nb = len(self.train_loader)
            nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
            last_opt_step = -1
            self.epoch_time = None
            self.epoch_time_start = ultra_trainer.time.time()
            self.train_time_start = ultra_trainer.time.time()
            self.run_callbacks("on_train_start")
            ultra_trainer.LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
                f"Logging results to {ultra_trainer.colorstr('bold', self.save_dir)}\n"
                f"Starting training for "
                + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )
            if self.args.close_mosaic:
                base_idx = (self.epochs - self.args.close_mosaic) * nb
                self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            epoch = self.start_epoch
            self.optimizer.zero_grad()
            self._oom_retries = 0
            while True:
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with ultra_trainer.warnings.catch_warnings():
                    ultra_trainer.warnings.simplefilter("ignore")
                    self.scheduler.step()

                self._model_train()
                if ultra_trainer.RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                self.od_pbar = None
                if ultra_trainer.RANK in {-1, 0}:
                    ultra_trainer.LOGGER.info(self.progress_string())
                    rich_pbar = _build_rich_progress_bar(
                        enumerate(self.train_loader),
                        total=nb,
                        description=self.progress_string(),
                    )
                    if rich_pbar is not None:
                        pbar = rich_pbar
                        self.od_pbar = rich_pbar
                self.tloss = None
                try:
                    for i, batch in pbar:
                        self.run_callbacks("on_train_batch_start")
                        ni = i + nb * epoch
                        if ni <= nw:
                            xi = [0, nw]
                            self.accumulate = max(
                                1,
                                int(ultra_trainer.np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()),
                            )
                            for x in self.optimizer.param_groups:
                                x["lr"] = ultra_trainer.np.interp(
                                    ni,
                                    xi,
                                    [
                                        self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                        x["initial_lr"] * self.lf(epoch),
                                    ],
                                )
                                if "momentum" in x:
                                    x["momentum"] = ultra_trainer.np.interp(
                                        ni,
                                        xi,
                                        [self.args.warmup_momentum, self.args.momentum],
                                    )

                        try:
                            with ultra_trainer.autocast(self.amp):
                                batch = self.preprocess_batch(batch)
                                if self.args.compile:
                                    preds = self.model(batch["img"])
                                    loss, self.loss_items = ultra_trainer.unwrap_model(self.model).loss(batch, preds)
                                else:
                                    loss, self.loss_items = self.model(batch)
                                self.loss = loss.sum()
                                if ultra_trainer.RANK != -1:
                                    self.loss *= self.world_size
                                self.tloss = (
                                    self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                                )
                            self.scaler.scale(self.loss).backward()
                        except torch.cuda.OutOfMemoryError:
                            if epoch > self.start_epoch or self._oom_retries >= 3 or ultra_trainer.RANK != -1:
                                raise
                            self._oom_retries += 1
                            old_batch = self.batch_size
                            self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                            ultra_trainer.LOGGER.warning(
                                f"CUDA out of memory with batch={old_batch}. "
                                f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                            )
                            self._clear_memory()
                            self._build_train_pipeline()
                            self.scheduler.last_epoch = self.start_epoch - 1
                            nb = len(self.train_loader)
                            nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                            last_opt_step = -1
                            self.optimizer.zero_grad()
                            break
                        if ni - last_opt_step >= self.accumulate:
                            self.optimizer_step()
                            last_opt_step = ni
                            if self.args.time:
                                self.stop = (ultra_trainer.time.time() - self.train_time_start) > (self.args.time * 3600)
                                if ultra_trainer.RANK != -1:
                                    broadcast_list = [self.stop if ultra_trainer.RANK == 0 else None]
                                    ultra_trainer.dist.broadcast_object_list(broadcast_list, 0)
                                    self.stop = broadcast_list[0]
                                if self.stop:
                                    break

                        if ultra_trainer.RANK in {-1, 0}:
                            loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                            if hasattr(pbar, "set_description"):
                                pbar.set_description(
                                    ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                                    % (
                                        f"{epoch + 1}/{self.epochs}",
                                        f"{self._get_memory():.3g}G",
                                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                                        batch["cls"].shape[0],
                                        batch["img"].shape[-1],
                                    )
                                )
                            self.run_callbacks("on_batch_end")
                            if self.args.plots and ni in self.plot_idx:
                                self.plot_training_samples(batch, ni)

                        self.run_callbacks("on_train_batch_end")
                        if self.stop:
                            break
                    else:
                        self._oom_retries = 0
                finally:
                    if hasattr(pbar, "close"):
                        pbar.close()

                self.od_pbar = None
                if self._oom_retries and not self.stop:
                    continue

                if hasattr(ultra_trainer.unwrap_model(self.model).criterion, "update"):
                    ultra_trainer.unwrap_model(self.model).criterion.update()

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
                self.run_callbacks("on_train_epoch_end")
                if ultra_trainer.RANK in {-1, 0}:
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                final_epoch = epoch + 1 >= self.epochs
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)
                    self.metrics, self.fitness = self.validate()

                if self._handle_nan_recovery(epoch):
                    continue

                self.nan_recovery_attempts = 0
                if ultra_trainer.RANK in {-1, 0}:
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (ultra_trainer.time.time() - self.train_time_start) > (self.args.time * 3600)
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                t = ultra_trainer.time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = ultra_trainer.math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch
                    self.stop |= epoch >= self.epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(0.5)

                if ultra_trainer.RANK != -1:
                    broadcast_list = [self.stop if ultra_trainer.RANK == 0 else None]
                    ultra_trainer.dist.broadcast_object_list(broadcast_list, 0)
                    self.stop = broadcast_list[0]
                if self.stop:
                    break
                epoch += 1

            seconds = ultra_trainer.time.time() - self.train_time_start
            ultra_trainer.LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if ultra_trainer.RANK in {-1, 0}:
                if self.args.plots:
                    self.plot_metrics()
                self.run_callbacks("on_train_end")
            self._clear_memory()
            ultra_trainer.unset_deterministic()
            self.run_callbacks("teardown")

    try:
        import numpy as np
        from ultralytics.utils import LOGGER
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics runtime dependencies are not available") from exc

    TeacherDetectionTrainer._od_bootstrap_runtime_params = dict(runtime_params)
    TeacherDetectionTrainer._od_bootstrap_log_fn = log_fn
    callbacks = _build_teacher_callbacks(runtime_params=runtime_params)
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

    resolved_weights = coerce_weights_name(model_size, weights)
    log_fn = _emit_log
    trainer_cls, callbacks = _make_teacher_trainer(runtime_params=runtime_params, log_fn=log_fn)
    teacher_root = output_root / teacher_name
    train_params = dict(train_params)
    train_params["resume"] = resolve_resume_argument(
        train_params.get("resume", False),
        teacher_name=teacher_name,
        teacher_root=teacher_root,
    )
    model_source = train_params["resume"] if train_params["resume"] else resolved_weights
    model = YOLO(model_source)
    if hasattr(model, "add_callback"):
        for event_name, callback in callbacks.items():
            model.add_callback(event_name, callback)
    run_name = _timestamp_token()
    train_kwargs = _build_teacher_train_kwargs(
        dataset_yaml=dataset_yaml,
        teacher_root=teacher_root,
        run_name=run_name,
        train_params=train_params,
        exist_ok=exist_ok,
    )
    train_result = model.train(trainer=trainer_cls, **train_kwargs)
    run_dir = extract_run_dir(train_result, teacher_root / run_name)
    trainer = getattr(model, "trainer", None)
    summary = _build_teacher_train_summary(
        teacher_name=teacher_name,
        resolved_weights=resolved_weights,
        dataset_yaml=dataset_yaml,
        teacher_root=teacher_root,
        run_dir=run_dir,
        runtime_params=runtime_params,
        train_kwargs=train_kwargs,
        train_result=train_result,
        trainer=trainer,
    )
    latest_artifacts = refresh_latest_teacher_artifacts(teacher_root=teacher_root, run_dir=run_dir, summary=summary)
    summary["latest_artifacts"] = latest_artifacts

    artifact_paths = _teacher_runtime_artifact_paths(run_dir)
    summary_path = artifact_paths["train_summary"]
    write_json(summary_path, summary)
    latest_summary_path = Path(latest_artifacts["train_summary_path"])
    write_json(latest_summary_path, summary)
    return summary
