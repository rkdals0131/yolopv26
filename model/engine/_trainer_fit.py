from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable


def _best_metric_path(best_metric: str | None, *, has_val_loader: bool) -> str:
    return best_metric or ("val.losses.total.mean" if has_val_loader else "train.losses.total.mean")


def _build_run_summary(
    *,
    trainer: Any,
    epochs: int,
    output_dir: Path,
    history_dir: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
    best_metric_path: str,
    best_metric_value: float | None,
    best_epoch: int | None,
    best_checkpoint_path: Path | None,
    run_started_at: float,
    tensorboard_status: dict[str, Any],
    auto_resume: bool,
    start_epoch: int,
    early_exit_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "stage": trainer.stage,
        "epochs": int(epochs),
        "completed_epochs": len(trainer.epoch_history),
        "global_step": int(trainer.global_step),
        "run_dir": str(output_dir),
        "best_metric_path": best_metric_path,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "last_epoch": trainer.epoch_history[-1] if trainer.epoch_history else None,
        "skipped_steps": int(trainer.skipped_steps),
        "history_paths": {
            "train_steps": str(history_dir / "train_steps.jsonl"),
            "epochs": str(history_dir / "epochs.jsonl"),
        },
        "checkpoint_paths": {
            "last": str(checkpoint_dir / "last.pt"),
            "best": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        },
        "tensorboard": dict(tensorboard_status),
        "manifest_path": str(manifest_path),
        "duration_sec": time.perf_counter() - run_started_at,
        "auto_resumed": bool(auto_resume and start_epoch > 1),
        "resume_start_epoch": int(start_epoch),
    }
    if early_exit_state is not None:
        summary["early_exit"] = early_exit_state
    return summary


def _build_run_manifest(
    *,
    trainer: Any,
    run_started_at_iso: str,
    log_every_n_steps: int,
    profile_window: int,
    profile_device_sync: bool,
    tensorboard_status: dict[str, Any],
    output_dir: Path,
    run_summary: dict[str, Any],
    run_manifest_extra: dict[str, Any] | None,
    now_iso_fn: Callable[[], str],
    optimizer_group_hparams_fn: Callable[[Any], dict[str, float]],
    json_ready_fn: Callable[[Any], Any],
    run_manifest_version: str,
) -> dict[str, Any]:
    return {
        "version": run_manifest_version,
        "created_at": run_started_at_iso,
        "updated_at": now_iso_fn(),
        "stage": trainer.stage,
        "device": str(trainer.device),
        "optimizer": optimizer_group_hparams_fn(trainer.optimizer),
        "trainer": {
            "amp_enabled": bool(trainer.amp_enabled),
            "accumulate_steps": int(trainer.accumulate_steps),
            "grad_clip_norm": trainer.grad_clip_norm,
            "skip_non_finite_loss": bool(trainer.skip_non_finite_loss),
            "oom_guard": bool(trainer.oom_guard),
            "log_every_n_steps": int(log_every_n_steps),
            "profile_window": int(profile_window),
            "profile_device_sync": bool(profile_device_sync),
        },
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "history": run_summary["history_paths"],
            "checkpoints": run_summary["checkpoint_paths"],
            "tensorboard": dict(tensorboard_status),
        },
        "run_state": json_ready_fn(run_summary),
        "extra": json_ready_fn(run_manifest_extra or {}),
    }


def _restore_resume_state(
    *,
    trainer: Any,
    resume_candidate: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    best_metric: str | None,
    has_val_loader: bool,
    resolve_summary_path_fn: Callable[[dict[str, Any], str], float],
) -> tuple[int, float | None, int | None, Path | None]:
    checkpoint = trainer.load_checkpoint(resume_candidate, map_location=trainer.device)
    extra_state = checkpoint.get("extra_state", {}) if isinstance(checkpoint.get("extra_state"), dict) else {}
    restored_epoch = int(extra_state.get("epoch", len(trainer.epoch_history)))
    start_epoch = restored_epoch + 1
    best_metric_value: float | None = None
    best_epoch: int | None = None
    best_checkpoint_path: Path | None = None
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior_summary.get("best_metric_value") is not None:
            best_metric_value = float(prior_summary["best_metric_value"])
        if prior_summary.get("best_epoch") is not None:
            best_epoch = int(prior_summary["best_epoch"])
    if best_metric_value is None and trainer.epoch_history:
        best_metric_value = resolve_summary_path_fn(
            trainer.epoch_history[-1],
            _best_metric_path(best_metric, has_val_loader=has_val_loader),
        )
        best_epoch = int(trainer.epoch_history[-1]["epoch"])
    if (checkpoint_dir / "best.pt").is_file():
        best_checkpoint_path = checkpoint_dir / "best.pt"
    return start_epoch, best_metric_value, best_epoch, best_checkpoint_path


def run_fit(
    trainer: Any,
    train_loader: Any,
    *,
    epochs: int,
    phase_index: int | None = None,
    phase_count: int | None = None,
    phase_name: str | None = None,
    val_loader: Any = None,
    run_dir: str | Path | None = None,
    val_every: int = 1,
    checkpoint_every: int = 1,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    best_metric: str | None = None,
    best_mode: str = "min",
    auto_resume: bool = False,
    resume_path: str | Path | None = None,
    enable_tensorboard: bool = True,
    early_exit_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    run_manifest_extra: dict[str, Any] | None = None,
    log_every_n_steps: int = 1,
    profile_window: int = 20,
    profile_device_sync: bool = False,
    default_run_dir_fn: Callable[[], Path] | None = None,
    now_iso_fn: Callable[[], str] | None = None,
    write_json_fn: Callable[[str | Path, dict[str, Any]], Path] | None = None,
    json_ready_fn: Callable[[Any], Any] | None = None,
    maybe_build_summary_writer_fn: Callable[..., Any] | None = None,
    optimizer_group_hparams_fn: Callable[[Any], dict[str, float]] | None = None,
    resolve_summary_path_fn: Callable[[dict[str, Any], str], float] | None = None,
    is_better_fn: Callable[[float, float | None, str], bool] | None = None,
    write_tensorboard_scalars_fn: Callable[[Any, str, dict[str, Any], int], None] | None = None,
    tensorboard_epoch_payload_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    run_manifest_version: str = "pv26-train-run-v1",
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("fit requires epochs > 0")
    if log_every_n_steps <= 0:
        raise ValueError("log_every_n_steps must be > 0")
    if profile_window <= 0:
        raise ValueError("profile_window must be > 0")
    if (
        default_run_dir_fn is None
        or now_iso_fn is None
        or write_json_fn is None
        or json_ready_fn is None
        or maybe_build_summary_writer_fn is None
        or optimizer_group_hparams_fn is None
        or resolve_summary_path_fn is None
        or is_better_fn is None
        or write_tensorboard_scalars_fn is None
        or tensorboard_epoch_payload_fn is None
    ):
        raise ValueError("run_fit requires helper function dependencies")

    output_dir = Path(run_dir) if run_dir is not None else default_run_dir_fn()
    history_dir = output_dir / "history"
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    best_metric_path = _best_metric_path(best_metric, has_val_loader=val_loader is not None)
    best_metric_value: float | None = None
    best_epoch: int | None = None
    best_checkpoint_path: Path | None = None
    run_started_at = time.perf_counter()
    run_started_at_iso = now_iso_fn()
    run_summary: dict[str, Any] = {}
    early_exit_state: dict[str, Any] | None = None
    start_epoch = 1
    manifest_path = output_dir / "run_manifest.json"
    previous_writer = trainer.tensorboard_writer
    previous_status = dict(trainer.tensorboard_status)
    previous_tb_step = int(trainer._tensorboard_train_step)
    tensorboard_purge_step: int | None = None
    resumed_from_checkpoint = False
    if auto_resume:
        resume_candidate = Path(resume_path) if resume_path is not None else checkpoint_dir / "last.pt"
        if resume_candidate.is_file():
            resumed_from_checkpoint = True
            start_epoch, best_metric_value, best_epoch, best_checkpoint_path = _restore_resume_state(
                trainer=trainer,
                resume_candidate=resume_candidate,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                best_metric=best_metric,
                has_val_loader=val_loader is not None,
                resolve_summary_path_fn=resolve_summary_path_fn,
            )
    trainer._tensorboard_train_step = len(trainer.history)
    if resumed_from_checkpoint:
        tensorboard_purge_step = max(1, trainer._tensorboard_train_step + 1)
    if enable_tensorboard:
        trainer.tensorboard_writer, trainer.tensorboard_status = maybe_build_summary_writer_fn(
            tensorboard_dir,
            purge_step=tensorboard_purge_step,
        )
    else:
        trainer.tensorboard_writer = None
        trainer.tensorboard_status = {
            "enabled": False,
            "status": "disabled_by_config",
            "error": None,
            "log_dir": str(tensorboard_dir),
            "purge_step": tensorboard_purge_step,
        }
    evaluator = trainer.build_evaluator() if val_loader is not None else None
    try:
        if start_epoch > epochs:
            run_summary = _build_run_summary(
                trainer=trainer,
                epochs=epochs,
                output_dir=output_dir,
                history_dir=history_dir,
                checkpoint_dir=checkpoint_dir,
                manifest_path=manifest_path,
                best_metric_path=best_metric_path,
                best_metric_value=best_metric_value,
                best_epoch=best_epoch,
                best_checkpoint_path=best_checkpoint_path,
                run_started_at=run_started_at,
                tensorboard_status=trainer.tensorboard_status,
                auto_resume=True,
                start_epoch=start_epoch,
            )
            write_json_fn(output_dir / "summary.json", run_summary)
            write_json_fn(
                manifest_path,
                _build_run_manifest(
                    trainer=trainer,
                    run_started_at_iso=run_started_at_iso,
                    log_every_n_steps=log_every_n_steps,
                    profile_window=profile_window,
                    profile_device_sync=profile_device_sync,
                    tensorboard_status=trainer.tensorboard_status,
                    output_dir=output_dir,
                    run_summary=run_summary,
                    run_manifest_extra=run_manifest_extra,
                    now_iso_fn=now_iso_fn,
                    optimizer_group_hparams_fn=optimizer_group_hparams_fn,
                    json_ready_fn=json_ready_fn,
                    run_manifest_version=run_manifest_version,
                ),
            )
            return run_summary

        for epoch in range(start_epoch, epochs + 1):
            epoch_summary: dict[str, Any] = {
                "epoch": int(epoch),
                "stage": trainer.stage,
                "epoch_started_at": now_iso_fn(),
                "train": trainer.train_epoch(
                    train_loader,
                    epoch=epoch,
                    epoch_total=epochs,
                    phase_index=phase_index,
                    phase_count=phase_count,
                    phase_name=phase_name,
                    max_batches=max_train_batches,
                    step_log_path=history_dir / "train_steps.jsonl",
                    log_every_n_steps=log_every_n_steps,
                    profile_window=profile_window,
                    profile_device_sync=profile_device_sync,
                ),
            }
            if val_loader is not None and epoch % val_every == 0:
                epoch_summary["val"] = trainer.validate_epoch(
                    val_loader,
                    epoch=epoch,
                    epoch_total=epochs,
                    phase_index=phase_index,
                    phase_count=phase_count,
                    phase_name=phase_name,
                    evaluator=evaluator,
                    max_batches=max_val_batches,
                    log_every_n_steps=log_every_n_steps,
                    profile_window=profile_window,
                    profile_device_sync=profile_device_sync,
                )
            if trainer.scheduler is not None:
                trainer.scheduler.step()
                epoch_summary["scheduler_lrs"] = [float(group["lr"]) for group in trainer.optimizer.param_groups]

            metric_value = resolve_summary_path_fn(epoch_summary, best_metric_path)
            epoch_summary["selection"] = {
                "best_metric_path": best_metric_path,
                "best_metric_value": metric_value,
                "best_mode": best_mode,
            }
            is_best = is_better_fn(metric_value, best_metric_value, best_mode)
            if is_best:
                best_metric_value = metric_value
                best_epoch = epoch

            early_exit_state = None
            if early_exit_callback is not None:
                callback_result = early_exit_callback(epoch_summary)
                if callback_result is not None:
                    if not isinstance(callback_result, dict):
                        raise TypeError("early_exit_callback must return dict[str, Any] | None")
                    early_exit_state = dict(callback_result)
                    early_exit_state.setdefault("should_stop", True)

            trainer.epoch_history.append(epoch_summary)
            last_checkpoint_path = trainer.save_checkpoint(
                checkpoint_dir / "last.pt",
                extra_state={"epoch": epoch, "epoch_summary": epoch_summary},
            )
            epoch_summary["checkpoint_last"] = str(last_checkpoint_path)
            if epoch % checkpoint_every == 0:
                epoch_checkpoint_path = trainer.save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    extra_state={"epoch": epoch, "epoch_summary": epoch_summary},
                )
                epoch_summary["checkpoint_epoch"] = str(epoch_checkpoint_path)
            if is_best:
                best_checkpoint_path = trainer.save_checkpoint(
                    checkpoint_dir / "best.pt",
                    extra_state={"epoch": epoch, "epoch_summary": epoch_summary},
                )
                epoch_summary["checkpoint_best"] = str(best_checkpoint_path)

            trainer.save_history_jsonl(history_dir / "train_steps.jsonl")
            trainer.save_epoch_history_jsonl(history_dir / "epochs.jsonl")

            if trainer.tensorboard_writer is not None:
                write_tensorboard_scalars_fn(
                    trainer.tensorboard_writer,
                    "epoch",
                    tensorboard_epoch_payload_fn(epoch_summary),
                    epoch,
                )
                trainer.tensorboard_writer.flush()

            serialized_early_exit = json_ready_fn(early_exit_state) if early_exit_state is not None else None
            run_summary = _build_run_summary(
                trainer=trainer,
                epochs=epochs,
                output_dir=output_dir,
                history_dir=history_dir,
                checkpoint_dir=checkpoint_dir,
                manifest_path=manifest_path,
                best_metric_path=best_metric_path,
                best_metric_value=best_metric_value,
                best_epoch=best_epoch,
                best_checkpoint_path=best_checkpoint_path,
                run_started_at=run_started_at,
                tensorboard_status=trainer.tensorboard_status,
                auto_resume=auto_resume,
                start_epoch=start_epoch,
                early_exit_state=serialized_early_exit,
            )
            write_json_fn(output_dir / "summary.json", run_summary)
            write_json_fn(
                manifest_path,
                _build_run_manifest(
                    trainer=trainer,
                    run_started_at_iso=run_started_at_iso,
                    log_every_n_steps=log_every_n_steps,
                    profile_window=profile_window,
                    profile_device_sync=profile_device_sync,
                    tensorboard_status=trainer.tensorboard_status,
                    output_dir=output_dir,
                    run_summary=run_summary,
                    run_manifest_extra=run_manifest_extra,
                    now_iso_fn=now_iso_fn,
                    optimizer_group_hparams_fn=optimizer_group_hparams_fn,
                    json_ready_fn=json_ready_fn,
                    run_manifest_version=run_manifest_version,
                ),
            )
            if early_exit_state is not None and bool(early_exit_state.get("should_stop", True)):
                break
        return run_summary
    finally:
        if trainer.tensorboard_writer is not None:
            trainer.tensorboard_writer.flush()
            trainer.tensorboard_writer.close()
        trainer.tensorboard_writer = previous_writer
        trainer.tensorboard_status = previous_status
        trainer._tensorboard_train_step = previous_tb_step
