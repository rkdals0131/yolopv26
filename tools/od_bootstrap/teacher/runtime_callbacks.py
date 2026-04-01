from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from common.io import write_json


def build_teacher_runtime_callbacks(
    *,
    runtime_params: dict[str, Any],
    time_module: Any,
    deque_type: Any,
    append_jsonl_fn: Callable[[Path, dict[str, Any]], None],
    sync_timing_device_fn: Callable[[Any, bool], None],
    timing_profile_fn: Callable[[list[dict[str, float]]], dict[str, Any]],
    build_live_postfix_fn: Callable[..., str],
    set_progress_postfix_fn: Callable[[Any, str], bool],
    loader_profile_payload_fn: Callable[[Any], dict[str, Any]],
    maybe_build_summary_writer_fn: Callable[[Path], tuple[Any, dict[str, Any]]],
    write_tensorboard_scalars_fn: Callable[[Any, str, dict[str, Any], int], int],
    build_train_step_tensorboard_payload_fn: Callable[..., dict[str, Any]],
    build_epoch_tensorboard_payload_fn: Callable[..., dict[str, Any]],
) -> dict[str, Callable[[Any], None]]:
    def on_train_start(trainer: Any) -> None:
        trainer.od_profile_log_path = Path(trainer.save_dir) / "profile_log.jsonl"
        trainer.od_profile_summary_path = Path(trainer.save_dir) / "profile_summary.json"
        trainer.od_tensorboard_dir = Path(trainer.save_dir) / "tensorboard"
        trainer.od_tensorboard_dir.mkdir(parents=True, exist_ok=True)
        trainer.od_tensorboard_writer, trainer.od_tensorboard_status = maybe_build_summary_writer_fn(trainer.od_tensorboard_dir)
        train_loader_profile = loader_profile_payload_fn(trainer.train_loader)
        val_loader_profile = loader_profile_payload_fn(trainer.test_loader)
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
        if trainer.od_tensorboard_writer is not None:
            trainer.od_tensorboard_writer.flush()

    def on_train_epoch_start(trainer: Any) -> None:
        trainer.od_epoch_started_at = time_module.perf_counter()
        trainer.od_last_batch_end_at = trainer.od_epoch_started_at
        trainer.od_batch_started_at = None
        trainer.od_pending_wait_sec = 0.0
        trainer.od_epoch_step = 0
        trainer.od_epoch_timing_window = deque_type(maxlen=max(1, int(runtime_params["profile_window"])))

    def on_train_batch_start(trainer: Any) -> None:
        sync_timing_device_fn(trainer.device, bool(runtime_params["profile_device_sync"]))
        now = time_module.perf_counter()
        last_batch_end_at = trainer.od_last_batch_end_at or now
        trainer.od_pending_wait_sec = max(0.0, now - last_batch_end_at)
        trainer.od_batch_started_at = now

    def on_train_batch_end(trainer: Any) -> None:
        sync_timing_device_fn(trainer.device, bool(runtime_params["profile_device_sync"]))
        ended_at = time_module.perf_counter()
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
        elapsed_sec = max(0.0, ended_at - trainer.od_epoch_started_at)
        profile_summary = timing_profile_fn(list(trainer.od_epoch_timing_window))
        remaining_steps = max(0, total_steps - step_index)
        eta_sec = float(profile_summary["iteration_sec"]["mean"]) * float(remaining_steps)
        set_progress_postfix_fn(
            trainer.od_pbar,
            build_live_postfix_fn(
                elapsed_sec=elapsed_sec,
                eta_sec=eta_sec,
                profile_summary=profile_summary,
            ),
        )

        should_log = step_index % log_every_n_steps == 0 or step_index == total_steps
        if not should_log:
            return

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
            append_jsonl_fn(trainer.od_profile_log_path, payload)
        write_tensorboard_scalars_fn(
            trainer.od_tensorboard_writer,
            "train_step",
            build_train_step_tensorboard_payload_fn(
                losses=losses,
                profile_summary=profile_summary,
                elapsed_sec=elapsed_sec,
            ),
            step=trainer.od_global_step,
        )
        if trainer.od_pbar is None:
            trainer.od_log(
                f"[od_bootstrap.train] profile epoch={payload['epoch']}/{payload['epoch_total']} "
                f"step={step_index}/{total_steps} "
                f"{build_live_postfix_fn(elapsed_sec=elapsed_sec, eta_sec=eta_sec, profile_summary=profile_summary)}"
            )

    def on_train_epoch_end(trainer: Any) -> None:
        epoch_profile = {
            "epoch": int(trainer.epoch) + 1,
            "timing_profile": timing_profile_fn(list(trainer.od_epoch_timing_window)),
            "loader": loader_profile_payload_fn(trainer.train_loader),
        }
        trainer.od_profile_history.append(epoch_profile)

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch_index = int(trainer.epoch) + 1
        epoch_payload = build_epoch_tensorboard_payload_fn(
            losses=trainer.label_loss_items(trainer.tloss, prefix="train"),
            profile_summary=trainer.od_profile_history[-1]["timing_profile"] if trainer.od_profile_history else {},
            lr_values=dict(getattr(trainer, "lr", {})),
            metrics=dict(getattr(trainer, "metrics", {})),
        )
        write_tensorboard_scalars_fn(
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
            "train_loader": loader_profile_payload_fn(trainer.train_loader),
            "val_loader": loader_profile_payload_fn(trainer.test_loader),
            "epochs": trainer.od_profile_history,
        }
        if trainer.od_profile_summary_path is not None:
            write_json(trainer.od_profile_summary_path, payload)
        if trainer.od_tensorboard_writer is not None:
            trainer.od_tensorboard_writer.flush()
            trainer.od_tensorboard_writer.close()

    return {
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_batch_start": on_train_batch_start,
        "on_train_batch_end": on_train_batch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
