from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import MethodType
from typing import Any, Callable

from common.scalars import flatten_scalar_tree as _flatten_scalar_tree

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


def _coerce_weights_name(model_size: str, weights: str | None) -> str:
    if weights:
        return weights
    size = model_size.strip().lower() or "n"
    return f"yolo26{size}.pt"


def _load_checkpoint_payload(checkpoint: Path) -> dict[str, Any] | None:
    if torch is None or not checkpoint.is_file():
        return None
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_resume_metadata(checkpoint: Path) -> dict[str, Any]:
    payload = _load_checkpoint_payload(checkpoint)
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    optimizer = payload.get("optimizer") if isinstance(payload, dict) else None
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    return {
        "epoch": int(epoch) if isinstance(epoch, int) else -1,
        "train_args": dict(train_args) if isinstance(train_args, dict) else {},
        "resumable": isinstance(optimizer, dict) and isinstance(epoch, int) and int(epoch) >= 0,
    }


def _infer_teacher_root_from_checkpoint(checkpoint: Path) -> Path | None:
    for parent in checkpoint.parents:
        if (parent / "latest_run.json").is_file():
            return parent
    return None


def _resume_candidate_sort_key(checkpoint: Path) -> tuple[int, int, int, int, str]:
    metadata = _checkpoint_resume_metadata(checkpoint)
    name = checkpoint.name
    kind_priority = 2 if name.startswith("last_resume") else 1 if name.startswith("last") else 0
    return (
        int(metadata["resumable"]),
        kind_priority,
        int(checkpoint.stat().st_mtime_ns),
        int(metadata["epoch"]),
        str(checkpoint),
    )


def _find_latest_resumable_checkpoint(teacher_root: Path) -> Path | None:
    candidates: set[Path] = set()
    for pattern in ("**/last_resume*.pt", "**/last*.pt", "**/epoch*.pt"):
        candidates.update(path for path in teacher_root.glob(pattern) if path.is_file())
    resumable = [path for path in candidates if _checkpoint_resume_metadata(path)["resumable"]]
    if not resumable:
        return None
    return max(resumable, key=_resume_candidate_sort_key)


def _resolve_resume_checkpoint_path(checkpoint: Path, *, teacher_root: Path | None = None) -> Path | None:
    if not checkpoint.is_file():
        return None
    if _checkpoint_resume_metadata(checkpoint)["resumable"]:
        return checkpoint

    search_root = teacher_root or _infer_teacher_root_from_checkpoint(checkpoint)
    if search_root is not None:
        fallback = _find_latest_resumable_checkpoint(search_root)
        if fallback is not None:
            return fallback

    sibling_candidates = [path for path in checkpoint.parent.glob("last_resume*.pt") if path.is_file()]
    sibling_candidates.extend(path for path in checkpoint.parent.glob("epoch*.pt") if path.is_file())
    resumable = [path for path in sibling_candidates if _checkpoint_resume_metadata(path)["resumable"]]
    if resumable:
        return max(resumable, key=_resume_candidate_sort_key)
    return checkpoint


def _find_latest_teacher_checkpoint(teacher_root: Path) -> Path | None:
    resumable = _find_latest_resumable_checkpoint(teacher_root)
    if resumable is not None:
        return resumable
    candidates = [path for path in teacher_root.glob("**/last*.pt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _resolve_resume_argument(resume: Any, *, teacher_name: str, teacher_root: Path) -> bool | str:
    if not resume:
        return False
    if isinstance(resume, Path):
        resolved = _resolve_resume_checkpoint_path(resume, teacher_root=teacher_root)
        return str(resolved) if resolved is not None else str(resume)
    if isinstance(resume, str):
        normalized = resume.strip()
        if not normalized:
            return False
        resolved = _resolve_resume_checkpoint_path(Path(normalized), teacher_root=teacher_root)
        return str(resolved) if resolved is not None else normalized

    checkpoint = _find_latest_teacher_checkpoint(teacher_root)
    if checkpoint is None:
        raise FileNotFoundError(
            f"resume requested for teacher '{teacher_name}' but no last.pt exists under {teacher_root}"
        )
    return str(checkpoint)


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


def _coerce_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _first_scalar(source: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in source:
            numeric = _coerce_scalar(source[key])
            if numeric is not None:
                return numeric
    return None


def _train_loss_payload(losses: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for source_key, target_key in (
        ("train/box_loss", "box_loss"),
        ("train/cls_loss", "cls_loss"),
        ("train/dfl_loss", "dfl_loss"),
    ):
        value = _first_scalar(losses, source_key, target_key)
        if value is not None:
            payload[target_key] = value
    return payload


def _epoch_lr_payload(lr_values: dict[str, Any]) -> dict[str, float]:
    value = _first_scalar(lr_values, "lr/pg0", "pg0")
    if value is None:
        return {}
    return {"pg0": value}


def _epoch_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    precision = None
    recall = None
    for target_key, candidates in (
        ("precision", ("metrics/precision(B)", "metrics/precision")),
        ("recall", ("metrics/recall(B)", "metrics/recall")),
        ("mAP50", ("metrics/mAP50(B)", "metrics/mAP50")),
        ("mAP50_95", ("metrics/mAP50-95(B)", "metrics/mAP50-95")),
    ):
        value = _first_scalar(metrics, *candidates)
        if value is not None:
            payload[target_key] = value
            if target_key == "precision":
                precision = value
            elif target_key == "recall":
                recall = value

    if precision is not None and recall is not None and (precision + recall) > 0.0:
        payload["f1"] = (2.0 * precision * recall) / (precision + recall)

    val_payload: dict[str, float] = {}
    for source_key, target_key in (
        ("val/box_loss", "box_loss"),
        ("val/cls_loss", "cls_loss"),
        ("val/dfl_loss", "dfl_loss"),
    ):
        value = _first_scalar(metrics, source_key, target_key)
        if value is not None:
            val_payload[target_key] = value
    if val_payload:
        payload["val"] = val_payload
    return payload


def _epoch_profile_payload(profile_summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for target_key, source_group in (
        ("iteration_mean", "iteration_sec"),
        ("wait_mean", "wait_sec"),
        ("compute_mean", "compute_sec"),
    ):
        if isinstance(profile_summary.get(source_group), dict):
            value = _first_scalar(profile_summary[source_group], "mean")
            if value is not None:
                payload[target_key] = value
    return payload


def _train_step_profile_payload(profile_summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for target_key, source_group, stat_key in (
        ("iteration_mean", "iteration_sec", "mean"),
        ("iteration_p50", "iteration_sec", "p50"),
        ("iteration_p99", "iteration_sec", "p99"),
        ("wait_mean", "wait_sec", "mean"),
        ("compute_mean", "compute_sec", "mean"),
    ):
        if isinstance(profile_summary.get(source_group), dict):
            value = _first_scalar(profile_summary[source_group], stat_key)
            if value is not None:
                payload[target_key] = value
    return payload


def _build_epoch_tensorboard_payload(
    *,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
    lr_values: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    train_loss = _train_loss_payload(losses)
    if train_loss:
        payload["train"] = train_loss

    lr_payload = _epoch_lr_payload(lr_values)
    if lr_payload:
        payload["lr"] = lr_payload

    profile_payload = _epoch_profile_payload(profile_summary)
    if profile_payload:
        payload["profile_sec"] = profile_payload

    payload.update(_epoch_metric_payload(metrics))
    return payload


def _build_train_step_tensorboard_payload(
    *,
    losses: dict[str, Any],
    profile_summary: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    loss_payload = _train_loss_payload(losses)
    if loss_payload:
        payload["loss"] = loss_payload

    profile_payload = _train_step_profile_payload(profile_summary)
    if profile_payload:
        payload["profile_sec"] = profile_payload

    payload["elapsed_sec"] = float(elapsed_sec)
    return payload


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


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_live_postfix(
    *,
    elapsed_sec: float,
    eta_sec: float | None,
    profile_summary: dict[str, Any],
) -> str:
    return " ".join(
        [
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(eta_sec)}",
            f"iter={profile_summary['iteration_sec']['mean'] * 1000.0:.1f}ms",
            f"wait={profile_summary['wait_sec']['mean'] * 1000.0:.1f}ms",
            f"compute={profile_summary['compute_sec']['mean'] * 1000.0:.1f}ms",
        ]
    )


def _set_progress_postfix(pbar: Any, postfix: str) -> bool:
    if pbar is None:
        return False
    if hasattr(pbar, "set_bootstrap_postfix"):
        pbar.set_bootstrap_postfix(postfix)
        return True
    if hasattr(pbar, "set_postfix_str"):
        pbar.set_postfix_str(postfix, refresh=True)
        return True
    description = str(getattr(pbar, "desc", "") or "")
    base_description = description.split(" | ")[0] if " | " in description else description
    merged_description = f"{base_description} | {postfix}" if base_description else postfix
    if hasattr(pbar, "set_description"):
        pbar.set_description(merged_description)
        return True
    if hasattr(pbar, "set_postfix"):
        pbar.set_postfix(profile=postfix)
        return True
    return False


def _render_progress_line(pbar: Any, *, final: bool = False) -> str | None:
    if pbar.disable or (pbar.closed and not final):
        return None

    current_time = time.time()
    dt = current_time - pbar.last_print_t
    dn = pbar.n - pbar.last_print_n

    if not final and not pbar._should_update(dt, dn):
        return None

    if dt > pbar.MIN_RATE_CALC_INTERVAL:
        rate = dn / dt if dt else 0.0
        if rate < pbar.MAX_SMOOTHED_RATE:
            pbar.last_rate = pbar.RATE_SMOOTHING_FACTOR * rate + (1 - pbar.RATE_SMOOTHING_FACTOR) * pbar.last_rate
            rate = pbar.last_rate
    else:
        rate = pbar.last_rate

    if pbar.total and pbar.n >= pbar.total:
        overall_elapsed = current_time - pbar.start_t
        if overall_elapsed > 0:
            rate = pbar.n / overall_elapsed

    pbar.last_print_n = pbar.n
    pbar.last_print_t = current_time
    elapsed = current_time - pbar.start_t

    remaining_str = ""
    if pbar.total and 0 < pbar.n < pbar.total and elapsed > 0:
        est_rate = rate or (pbar.n / elapsed)
        remaining_str = f"<{pbar._format_time((pbar.total - pbar.n) / est_rate)}"

    if pbar.total:
        percent = (pbar.n / pbar.total) * 100
        n_str = pbar._format_num(pbar.n)
        t_str = pbar._format_num(pbar.total)
        if pbar.is_bytes and len(n_str) >= 2 and len(t_str) >= 2 and n_str[-2] == t_str[-2]:
            n_str = n_str.rstrip("KMGTPB")
    else:
        percent = 0.0
        n_str, t_str = pbar._format_num(pbar.n), "?"

    elapsed_str = pbar._format_time(elapsed)
    rate_str = pbar._format_rate(rate) or (pbar._format_rate(pbar.n / elapsed) if elapsed > 0 else "")
    bar = pbar._generate_bar()

    if pbar.total:
        if pbar.is_bytes and pbar.n >= pbar.total:
            progress_str = f"{pbar.desc}: {percent:.0f}% {bar} {t_str} {rate_str} {elapsed_str}"
        else:
            progress_str = f"{pbar.desc}: {percent:.0f}% {bar} {n_str}/{t_str} {rate_str} {elapsed_str}{remaining_str}"
    else:
        progress_str = f"{pbar.desc}: {bar} {n_str} {rate_str} {elapsed_str}"

    return progress_str


def _install_ultralytics_postfix_renderer(pbar: Any) -> Any:
    if pbar is None or getattr(pbar, "_od_bootstrap_renderer_installed", False):
        return pbar
    if not hasattr(pbar, "_display"):
        return pbar

    def _display_with_bootstrap_postfix(self: Any, final: bool = False) -> None:
        progress_str = _render_progress_line(self, final=final)
        if progress_str is None:
            return
        postfix = str(getattr(self, "_od_bootstrap_postfix", "") or "").strip()
        try:
            if self.noninteractive:
                if postfix:
                    self.file.write(f"{progress_str} | {postfix}")
                else:
                    self.file.write(progress_str)
            else:
                prior_line_count = int(getattr(self, "_od_bootstrap_rendered_lines", 1))
                if prior_line_count > 1:
                    self.file.write("\r\033[1A\r\033[K")
                else:
                    self.file.write("\r\033[K")
                self.file.write(progress_str)
                if postfix:
                    self.file.write(f"\n\033[K{postfix}")
                    self._od_bootstrap_rendered_lines = 2
                else:
                    self._od_bootstrap_rendered_lines = 1
            self.file.flush()
        except Exception:
            pass

    def _set_bootstrap_postfix(self: Any, postfix: str) -> None:
        self._od_bootstrap_postfix = str(postfix)
        if not self.disable:
            self._display()

    pbar._od_bootstrap_postfix = ""
    pbar._od_bootstrap_renderer_installed = True
    pbar._od_bootstrap_rendered_lines = 1
    pbar._display = MethodType(_display_with_bootstrap_postfix, pbar)
    pbar.set_bootstrap_postfix = MethodType(_set_bootstrap_postfix, pbar)
    return pbar


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _remove_path(destination)
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        try:
            os.symlink(source, destination)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def _refresh_latest_teacher_artifacts(
    *,
    teacher_root: Path,
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    latest_summary_path = teacher_root / "train_summary.json"
    latest_summary_path.parent.mkdir(parents=True, exist_ok=True)
    latest_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    alias_weights_root = teacher_root / "weights"
    alias_actions: dict[str, str] = {}
    for checkpoint_name in ("best.pt", "last.pt", "last_resume.pt"):
        source_path = run_dir / "weights" / checkpoint_name
        if not source_path.is_file():
            continue
        alias_actions[checkpoint_name] = _link_or_copy_file(source_path, alias_weights_root / checkpoint_name)

    latest_run_payload = {
        "teacher_root": str(teacher_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "alias_actions": alias_actions,
    }
    latest_run_path = teacher_root / "latest_run.json"
    latest_run_path.write_text(json.dumps(latest_run_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "train_summary_path": str(latest_summary_path),
        "latest_run_path": str(latest_run_path),
        "weights_root": str(alias_weights_root),
        "alias_actions": alias_actions,
    }


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
                    last = _resolve_resume_checkpoint_path(requested) or requested
                    metadata = _checkpoint_resume_metadata(last)
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
                    pbar = ultra_trainer.TQDM(enumerate(self.train_loader), total=nb)
                    _install_ultralytics_postfix_renderer(pbar)
                    self.od_pbar = pbar
                self.tloss = None
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
        elapsed_sec = max(0.0, ended_at - trainer.od_epoch_started_at)
        profile_summary = _timing_profile(list(trainer.od_epoch_timing_window))
        remaining_steps = max(0, total_steps - step_index)
        eta_sec = float(profile_summary["iteration_sec"]["mean"]) * float(remaining_steps)
        _set_progress_postfix(
            trainer.od_pbar,
            _build_live_postfix(
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
            _append_jsonl(trainer.od_profile_log_path, payload)
        _write_tensorboard_scalars(
            trainer.od_tensorboard_writer,
            "train_step",
            _build_train_step_tensorboard_payload(
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
                f"{_build_live_postfix(elapsed_sec=elapsed_sec, eta_sec=eta_sec, profile_summary=profile_summary)}"
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
        epoch_payload = _build_epoch_tensorboard_payload(
            losses=trainer.label_loss_items(trainer.tloss, prefix="train"),
            profile_summary=trainer.od_profile_history[-1]["timing_profile"] if trainer.od_profile_history else {},
            lr_values=dict(getattr(trainer, "lr", {})),
            metrics=dict(getattr(trainer, "metrics", {})),
        )
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
    log_fn = _emit_log
    trainer_cls, callbacks = _make_teacher_trainer(runtime_params=runtime_params, log_fn=log_fn)
    teacher_root = output_root / teacher_name
    train_params = dict(train_params)
    train_params["resume"] = _resolve_resume_argument(
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
    train_kwargs = {
        "data": str(dataset_yaml),
        "project": str(teacher_root),
        "name": run_name,
        "exist_ok": bool(exist_ok),
        "pretrained": True,
        **train_params,
    }
    train_result = model.train(trainer=trainer_cls, **train_kwargs)
    run_dir = _extract_run_dir(train_result, teacher_root / run_name)
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
        "teacher_root": str(teacher_root),
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
    latest_artifacts = _refresh_latest_teacher_artifacts(teacher_root=teacher_root, run_dir=run_dir, summary=summary)
    summary["latest_artifacts"] = latest_artifacts

    summary_path = run_dir / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    latest_summary_path = Path(latest_artifacts["train_summary_path"])
    latest_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary
