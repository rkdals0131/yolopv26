#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.constants import DET_CLASSES_CANONICAL
from pv26.criterion import PV26Criterion
from pv26.multitask_model import PV26MultiHead, PV26MultiHeadYOLO26
from pv26.torch_dataset import AugmentSpec, Pv26ManifestDataset, Pv26Sample
from tools.train.common import (
    compute_map50,
    cxcywh_to_xyxy,
    decode_det_predictions,
    format_loss_line,
    mean_losses,
    resolve_device,
    seed_everything,
    update_binary_iou,
)

DEFAULT_DATASET_ROOT = Path("/home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PV26 practical train/val pipeline")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"PV26 dataset root with meta/split_manifest.csv (default: {DEFAULT_DATASET_ROOT})",
    )
    p.add_argument("--arch", type=str, default="yolo26n", choices=["yolo26n", "stub"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when workers > 0")
    p.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=True,
        help="Keep DataLoader workers alive across epochs (default: on)",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help="If >0, print N-step averaged train stage timings (data_wait/h2d/fwd/loss/bwd/opt).",
    )
    p.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around profiling timers for more accurate stage timings (adds overhead).",
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:N")
    p.add_argument("--amp", dest="amp", action="store_true", default=True, help="Enable AMP (CUDA only, default: on)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    p.add_argument("--max-batches", type=int, default=0, help="Max batches for both train/val (0=all)")
    p.add_argument("--max-train-batches", type=int, default=0, help="Max train batches (0=all or --max-batches)")
    p.add_argument("--max-val-batches", type=int, default=0, help="Max val batches (0=all or --max-batches)")
    p.add_argument("--run-name", type=str, default="", help="Run directory name under --out-dir")
    p.add_argument("--out-dir", type=Path, default=Path("runs/pv26_train"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    p.add_argument("--resume-latest", action="store_true", help="Resume from <run>/checkpoints/latest.pt")
    p.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=True)
    p.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    p.add_argument("--progress", dest="progress", action="store_true", default=True)
    p.add_argument("--no-progress", dest="progress", action="store_false")
    p.add_argument("--log-every", type=int, default=10, help="Console print interval when tqdm is disabled")
    p.add_argument("--augment", dest="augment", action="store_true", default=True, help="Enable train-time augmentation")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable train-time augmentation")
    p.add_argument("--aug-hflip", type=float, default=0.5, help="Horizontal flip probability for train set")
    p.add_argument("--aug-brightness", type=float, default=0.2, help="Brightness jitter delta (0 disables)")
    p.add_argument("--aug-contrast", type=float, default=0.2, help="Contrast jitter delta (0 disables)")
    p.add_argument("--aug-saturation", type=float, default=0.2, help="Saturation jitter delta (0 disables)")
    return p


def _resolve_max_batches(args: argparse.Namespace) -> tuple[int, int]:
    max_train = int(args.max_train_batches)
    max_val = int(args.max_val_batches)
    max_both = int(args.max_batches)
    if max_train <= 0 and max_both > 0:
        max_train = max_both
    if max_val <= 0 and max_both > 0:
        max_val = max_both
    return max_train, max_val


def _limit_steps(loader_len: int, max_batches: int) -> int:
    if max_batches <= 0:
        return loader_len
    return min(loader_len, max_batches)


def _require_tqdm(enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Progress bars require 'tqdm'. Install it with: uv pip install tqdm "
            "or run with --no-progress."
        ) from exc


def _build_tb_writer(enabled: bool, log_dir: Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorBoard logging requires 'tensorboard'. Install it with: uv pip install tensorboard "
            "or run with --no-tensorboard."
        ) from exc


_SCOPE_TO_CODE = {"full": 0, "subset": 1, "none": 2}


def _collate_with_images(samples: List[Pv26Sample]) -> Tuple[torch.Tensor, List[Pv26Sample], Dict[str, Any]]:
    # Build image batch/targets in worker process so main thread can focus on GPU steps.
    images = torch.stack([s.image for s in samples], dim=0)
    det_yolo_list = [s.det_yolo for s in samples]
    det_scope_list = [str(s.det_label_scope).strip().lower() for s in samples]
    det_scope_code_list: List[int] = []
    for scope in det_scope_list:
        if scope not in _SCOPE_TO_CODE:
            raise ValueError(f"invalid det_label_scope in batch: {scope}")
        det_scope_code_list.append(int(_SCOPE_TO_CODE[scope]))

    has_det = torch.tensor([s.has_det for s in samples], dtype=torch.long)
    has_da = torch.tensor([s.has_da for s in samples], dtype=torch.long)
    has_rm = torch.tensor(
        [[s.has_rm_lane_marker, s.has_rm_road_marker_non_lane, s.has_rm_stop_line] for s in samples],
        dtype=torch.long,
    )
    da_mask = torch.stack([s.da_mask for s in samples], dim=0)
    rm_mask = torch.stack([s.rm_mask for s in samples], dim=0)
    det_scope_code = torch.tensor(det_scope_code_list, dtype=torch.long)

    det_tgt_batch_idx_parts: List[torch.Tensor] = []
    det_tgt_cls_parts: List[torch.Tensor] = []
    det_tgt_box_parts: List[torch.Tensor] = []
    for i, gt in enumerate(det_yolo_list):
        if gt.numel() == 0:
            continue
        if gt.ndim != 2 or gt.shape[-1] != 5:
            raise ValueError("det_yolo per sample must be [N,5]")
        det_tgt_batch_idx_parts.append(torch.full((gt.shape[0],), int(i), dtype=torch.long))
        det_tgt_cls_parts.append(gt[:, 0].to(dtype=torch.float32))
        det_tgt_box_parts.append(gt[:, 1:5].to(dtype=torch.float32))

    if det_tgt_batch_idx_parts:
        det_tgt_batch_idx = torch.cat(det_tgt_batch_idx_parts, dim=0)
        det_tgt_cls = torch.cat(det_tgt_cls_parts, dim=0)
        det_tgt_bboxes = torch.cat(det_tgt_box_parts, dim=0)
    else:
        det_tgt_batch_idx = torch.zeros((0,), dtype=torch.long)
        det_tgt_cls = torch.zeros((0,), dtype=torch.float32)
        det_tgt_bboxes = torch.zeros((0, 4), dtype=torch.float32)

    target_batch: Dict[str, Any] = {
        "_pv26_prepared": True,
        "det_yolo": det_yolo_list,
        "det_label_scope": det_scope_list,
        "det_scope_code": det_scope_code,
        "det_tgt_batch_idx": det_tgt_batch_idx,
        "det_tgt_cls": det_tgt_cls,
        "det_tgt_bboxes": det_tgt_bboxes,
        "has_det": has_det,
        "has_da": has_da,
        "has_rm": has_rm,
        "da_mask": da_mask,
        "rm_mask": rm_mask,
    }
    return images, list(samples), target_batch


def _move_prepared_batch_to_device(batch: Dict[str, Any], *, device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _make_run_dir(base: Path, run_name: str, arch: str) -> Path:
    if run_name.strip():
        return base / run_name.strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{arch}_{stamp}"


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: Optional[float],
    best_epoch: Optional[int],
    args: argparse.Namespace,
) -> None:
    args_serialized: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_serialized[k] = str(v)
        else:
            args_serialized[k] = v

    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "args": args_serialized,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, Optional[float], Optional[int]]:
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our checkpoint stores optimizer/scaler/args metadata, so we explicitly
    # request full load for local, trusted checkpoints.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions.
        ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = ckpt.get("best_score", None)
        best_epoch = ckpt.get("best_epoch", None)
        return start_epoch, best_score, best_epoch

    # Fallback for raw state_dict checkpoints.
    model.load_state_dict(ckpt)
    return 0, None, None


def _choose_best_score(val_losses: Dict[str, float], map50: Optional[float]) -> tuple[float, str]:
    if map50 is not None:
        return float(map50), "map50"
    return -float(val_losses["total"]), "neg_val_total_loss"


def _gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024 ** 3)


def _format_cuda_mem_stats(*, device: torch.device) -> str:
    free_b, total_b = torch.cuda.mem_get_info(device)
    alloc_b = torch.cuda.memory_allocated(device)
    reserved_b = torch.cuda.memory_reserved(device)
    peak_alloc_b = torch.cuda.max_memory_allocated(device)
    peak_reserved_b = torch.cuda.max_memory_reserved(device)
    return (
        f"cuda_mem alloc={_gib(alloc_b):.2f}GiB resv={_gib(reserved_b):.2f}GiB "
        f"peak_alloc={_gib(peak_alloc_b):.2f}GiB peak_resv={_gib(peak_reserved_b):.2f}GiB "
        f"free={_gib(free_b):.2f}/{_gib(total_b):.2f}GiB"
    )


def _query_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.0,
        ).strip()
    except Exception:
        return ""
    if not out:
        return ""
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 6:
        return ""
    u_gpu, u_mem, mem_used, mem_total, power_w, temp_c = parts
    return (
        f"smi util={u_gpu}% memutil={u_mem}% mem={mem_used}/{mem_total}MiB "
        f"pwr={power_w}W temp={temp_c}C"
    )


def _format_cpu_stats() -> str:
    rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_mib = rss_kib / 1024.0
    if hasattr(os, "getloadavg"):
        l1, l5, l15 = os.getloadavg()
        return f"cpu rss={rss_mib:.0f}MiB load={l1:.1f},{l5:.1f},{l15:.1f}"
    return f"cpu rss={rss_mib:.0f}MiB"


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    qq = max(0.0, min(1.0, float(q)))
    idx = int((len(values) - 1) * qq)
    return values[idx]


def _prepare_images_for_model(images_cpu: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    images = images_cpu.to(device=device, non_blocking=True)
    if images.dtype != torch.float32:
        # Keep dataset/collate tensors compact on CPU (uint8), normalize on device.
        images = images.to(dtype=torch.float32)
        images.mul_(1.0 / 255.0)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def train_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: PV26Criterion,
    loader: DataLoader[Tuple[torch.Tensor, List[Pv26Sample], Dict[str, Any]]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
    show_progress: bool,
    tqdm_fn,
    log_every: int,
    profile_every: int,
    profile_sync_cuda: bool,
) -> Dict[str, float]:
    model.train()
    losses_sum = {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0}
    steps = 0
    prof_every = max(0, int(profile_every))
    prof_sync = bool(profile_sync_cuda and device.type == "cuda")
    prof_count = 0
    prof_sum_total = 0.0
    prof_sum_wait = 0.0
    prof_sum_h2d = 0.0
    prof_sum_fwd = 0.0
    prof_sum_loss = 0.0
    prof_sum_bwd = 0.0
    prof_sum_opt = 0.0
    prof_sum_batch = 0
    wait_hist: deque[float] = deque(maxlen=200)
    min_free_b_window: Optional[int] = None
    step_cap = _limit_steps(len(loader), max_batches)
    iterator = enumerate(loader)
    if show_progress and tqdm_fn is not None:
        iterator = tqdm_fn(iterator, total=step_cap, desc="train", leave=False)

    last_iter_end = time.perf_counter()
    for batch_idx, packed in iterator:
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images_cpu, batch, target_batch_cpu = packed
        t_iter_enter = time.perf_counter()
        t_wait = t_iter_enter - last_iter_end
        wait_hist.append(float(t_wait))
        t0 = t_iter_enter
        images = _prepare_images_for_model(images_cpu, device=device)
        if device.type == "cuda":
            free_b, _ = torch.cuda.mem_get_info(device)
            if min_free_b_window is None or free_b < min_free_b_window:
                min_free_b_window = int(free_b)
        target_batch = _move_prepared_batch_to_device(target_batch_cpu, device=device)
        if prof_sync:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        t2 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(images)
            if prof_sync:
                torch.cuda.synchronize(device)
            t3 = time.perf_counter()
            losses = criterion(preds, target_batch)
            if prof_sync:
                torch.cuda.synchronize(device)
            t4 = time.perf_counter()
        scaler.scale(losses["total"]).backward()
        if prof_sync:
            torch.cuda.synchronize(device)
        t5 = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        if prof_sync:
            torch.cuda.synchronize(device)
        t6 = time.perf_counter()
        # Mark compute boundary here so "wait" reflects loader latency only.
        last_iter_end = t6

        steps += 1
        for key in losses_sum:
            losses_sum[key] += float(losses[key].detach().item())

        if prof_every > 0:
            prof_count += 1
            prof_sum_total += (t6 - t0 + t_wait)
            prof_sum_wait += t_wait
            prof_sum_h2d += (t1 - t0)
            prof_sum_fwd += (t3 - t2)
            prof_sum_loss += (t4 - t3)
            prof_sum_bwd += (t5 - t4)
            prof_sum_opt += (t6 - t5)
            prof_sum_batch += int(images_cpu.shape[0])
            if prof_count >= prof_every:
                avg_total_ms = 1000.0 * prof_sum_total / float(prof_count)
                avg_wait_ms = 1000.0 * prof_sum_wait / float(prof_count)
                avg_h2d_ms = 1000.0 * prof_sum_h2d / float(prof_count)
                avg_fwd_ms = 1000.0 * prof_sum_fwd / float(prof_count)
                avg_loss_ms = 1000.0 * prof_sum_loss / float(prof_count)
                avg_bwd_ms = 1000.0 * prof_sum_bwd / float(prof_count)
                avg_opt_ms = 1000.0 * prof_sum_opt / float(prof_count)
                wait_pct = (100.0 * avg_wait_ms / avg_total_ms) if avg_total_ms > 0 else 0.0
                sync_tag = "sync" if prof_sync else "async"
                total_batch = max(1, int(prof_sum_batch))
                thr = float(total_batch) / max(1e-9, prof_sum_total)
                compute_s = max(1e-9, prof_sum_total - prof_sum_wait)
                thr_no_wait = float(total_batch) / compute_s
                wait_vals = sorted(wait_hist)
                wait_p50_ms = 1000.0 * _quantile(wait_vals, 0.50)
                wait_p90_ms = 1000.0 * _quantile(wait_vals, 0.90)
                wait_p99_ms = 1000.0 * _quantile(wait_vals, 0.99)
                extra_parts = [
                    f"thr={thr:.1f}img/s",
                    f"thr_no_wait={thr_no_wait:.1f}img/s",
                    f"wait_p50={wait_p50_ms:.1f}ms",
                    f"wait_p90={wait_p90_ms:.1f}ms",
                    f"wait_p99={wait_p99_ms:.1f}ms",
                    _format_cpu_stats(),
                ]
                if device.type == "cuda":
                    extra_parts.append(_format_cuda_mem_stats(device=device))
                    if min_free_b_window is not None:
                        extra_parts.append(f"min_free={_gib(min_free_b_window):.2f}GiB")
                    smi_line = _query_nvidia_smi()
                    if smi_line:
                        extra_parts.append(smi_line)
                    if amp_enabled:
                        extra_parts.append(f"amp_scale={float(scaler.get_scale()):.0f}")
                print(
                    "[profile][train] "
                    f"step={batch_idx + 1} avg{prof_count}({sync_tag}) "
                    f"total={avg_total_ms:.1f}ms wait={avg_wait_ms:.1f}ms({wait_pct:.1f}%) "
                    f"h2d={avg_h2d_ms:.1f}ms fwd={avg_fwd_ms:.1f}ms loss={avg_loss_ms:.1f}ms "
                    f"bwd={avg_bwd_ms:.1f}ms opt={avg_opt_ms:.1f}ms "
                    + " ".join(extra_parts),
                    flush=True,
                )
                prof_count = 0
                prof_sum_total = 0.0
                prof_sum_wait = 0.0
                prof_sum_h2d = 0.0
                prof_sum_fwd = 0.0
                prof_sum_loss = 0.0
                prof_sum_bwd = 0.0
                prof_sum_opt = 0.0
                prof_sum_batch = 0
                min_free_b_window = None

        if show_progress and tqdm_fn is not None:
            iterator.set_postfix(loss=float(losses["total"].detach().item()))
        elif (batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0:
            step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
            print(format_loss_line(f"[train] step={batch_idx + 1}", step_losses), flush=True)

    return mean_losses(losses_sum, steps)


def validate(
    *,
    model: torch.nn.Module,
    criterion: PV26Criterion,
    loader: DataLoader[Tuple[torch.Tensor, List[Pv26Sample], Dict[str, Any]]],
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
    show_progress: bool,
    tqdm_fn,
    log_every: int,
) -> tuple[Dict[str, float], Dict[str, Optional[float]], Optional[float]]:
    model.eval()
    losses_sum = {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0}
    steps = 0

    metric_stats: Dict[str, Dict[str, int]] = {
        "da": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_marker": {"inter": 0, "union": 0, "supervised": 0},
        "rm_road_marker_non_lane": {"inter": 0, "union": 0, "supervised": 0},
        "rm_stop_line": {"inter": 0, "union": 0, "supervised": 0},
    }

    num_det_classes = len(DET_CLASSES_CANONICAL)
    det_preds_by_class: Dict[int, List[tuple[float, str, torch.Tensor]]] = {c: [] for c in range(num_det_classes)}
    gt_by_img_class: Dict[tuple[str, int], torch.Tensor] = {}
    det_conf_thres = 0.01
    det_nms_iou = 0.5
    det_max_per_image = 200

    step_cap = _limit_steps(len(loader), max_batches)
    iterator = enumerate(loader)
    if show_progress and tqdm_fn is not None:
        iterator = tqdm_fn(iterator, total=step_cap, desc="val", leave=False)

    with torch.no_grad():
        for batch_idx, packed in iterator:
            if max_batches > 0 and batch_idx >= max_batches:
                break

            images_cpu, batch, target_batch_cpu = packed
            images = _prepare_images_for_model(images_cpu, device=device)
            target_batch = _move_prepared_batch_to_device(target_batch_cpu, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                preds = model(images)
                losses = criterion(preds, target_batch)

            steps += 1
            for key in losses_sum:
                losses_sum[key] += float(losses[key].detach().item())

            if show_progress and tqdm_fn is not None:
                iterator.set_postfix(loss=float(losses["total"].detach().item()))
            elif (batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0:
                step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
                print(format_loss_line(f"[val] step={batch_idx + 1}", step_losses), flush=True)

            for i, sample in enumerate(batch):
                if sample.has_da:
                    update_binary_iou(metric_stats["da"], preds.da[i, 0], sample.da_mask.to(device=device))

                rm_channels = [
                    ("rm_lane_marker", sample.has_rm_lane_marker),
                    ("rm_road_marker_non_lane", sample.has_rm_road_marker_non_lane),
                    ("rm_stop_line", sample.has_rm_stop_line),
                ]
                for c_idx, (name, has_flag) in enumerate(rm_channels):
                    if has_flag:
                        update_binary_iou(metric_stats[name], preds.rm[i, c_idx], sample.rm_mask[c_idx].to(device=device))

                if sample.has_det and str(sample.det_label_scope) == "full":
                    gt = sample.det_yolo
                    if gt.numel():
                        gt_cls = gt[:, 0].to(dtype=torch.long)
                        gt_boxes_norm = cxcywh_to_xyxy(gt[:, 1:5].to(dtype=torch.float32)).clamp(0.0, 1.0)
                        if isinstance(preds.det, tuple):
                            _, h, w = tuple(sample.image.shape)
                            gt_boxes = gt_boxes_norm.clone()
                            gt_boxes[:, 0::2] *= float(w)
                            gt_boxes[:, 1::2] *= float(h)
                        else:
                            gt_boxes = gt_boxes_norm
                        for c in range(num_det_classes):
                            m = gt_cls == c
                            if bool(m.any()):
                                gt_by_img_class[(sample.sample_id, c)] = gt_boxes[m].detach().cpu()

                    if isinstance(preds.det, tuple):
                        y = preds.det[0]
                        det_i = y[i]
                        keep = det_i[:, 4] > float(det_conf_thres)
                        det_i = det_i[keep]
                        if det_i.numel():
                            det_boxes = det_i[:, 0:4].detach().cpu()
                            det_scores = det_i[:, 4].detach().cpu()
                            det_cls = det_i[:, 5].to(dtype=torch.long).detach().cpu()
                            for j in range(int(det_scores.shape[0])):
                                c = int(det_cls[j].item())
                                if 0 <= c < num_det_classes:
                                    det_preds_by_class[c].append((float(det_scores[j].item()), sample.sample_id, det_boxes[j]))
                    else:
                        det_boxes, det_scores, det_cls = decode_det_predictions(
                            preds.det[i].detach(),
                            conf_thres=det_conf_thres,
                            nms_iou=det_nms_iou,
                            max_det=det_max_per_image,
                        )
                        det_boxes = det_boxes.detach().cpu()
                        det_scores = det_scores.detach().cpu()
                        det_cls = det_cls.detach().cpu()
                        for j in range(int(det_scores.shape[0])):
                            c = int(det_cls[j].item())
                            if 0 <= c < num_det_classes:
                                det_preds_by_class[c].append((float(det_scores[j].item()), sample.sample_id, det_boxes[j]))

    val_losses = mean_losses(losses_sum, steps)
    iou_metrics: Dict[str, Optional[float]] = {}
    for name, stats in metric_stats.items():
        if stats["supervised"] == 0:
            iou_metrics[name] = None
        elif stats["union"] == 0:
            iou_metrics[name] = 0.0
        else:
            iou_metrics[name] = float(stats["inter"]) / float(stats["union"])

    map50, _ap_by_class = compute_map50(
        preds_by_class=det_preds_by_class,
        gt_by_img_class=gt_by_img_class,
        num_classes=num_det_classes,
        iou_thres=0.5,
    )
    return val_losses, iou_metrics, map50


def main() -> int:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        # Throughput-oriented defaults for Ampere+ training workloads.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cpu":
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
        )
    seed_everything(args.seed, device=device)
    tqdm_fn = _require_tqdm(args.progress)

    run_dir = _make_run_dir(args.out_dir, args.run_name, args.arch)
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = _build_tb_writer(args.tensorboard, tb_dir)

    train_aug = None
    if bool(args.augment):
        train_aug = AugmentSpec(
            hflip_prob=max(0.0, min(1.0, float(args.aug_hflip))),
            brightness=max(0.0, float(args.aug_brightness)),
            contrast=max(0.0, float(args.aug_contrast)),
            saturation=max(0.0, float(args.aug_saturation)),
        )

    train_ds = Pv26ManifestDataset(dataset_root=args.dataset_root, splits=("train",), augment=train_aug)
    val_ds = Pv26ManifestDataset(dataset_root=args.dataset_root, splits=("val",))
    if len(train_ds) == 0:
        raise RuntimeError("train split is empty")
    if len(val_ds) == 0:
        raise RuntimeError("val split is empty")

    dl_gen = torch.Generator().manual_seed(int(args.seed))
    num_workers = max(0, int(args.workers))
    loader_perf_kwargs = {}
    if num_workers > 0:
        pf = max(2, int(args.prefetch_factor))
        loader_perf_kwargs = {
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": pf,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_with_images,
        generator=dl_gen,
        pin_memory=(device.type == "cuda"),
        **loader_perf_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_with_images,
        pin_memory=(device.type == "cuda"),
        **loader_perf_kwargs,
    )

    if args.arch == "stub":
        model = PV26MultiHead(num_det_classes=len(DET_CLASSES_CANONICAL)).to(device)
        criterion = PV26Criterion(num_det_classes=len(DET_CLASSES_CANONICAL)).to(device)
    else:
        model = PV26MultiHeadYOLO26(num_det_classes=len(DET_CLASSES_CANONICAL), yolo26_cfg="yolo26n.yaml").to(device)
        criterion = PV26Criterion(
            num_det_classes=len(DET_CLASSES_CANONICAL),
            od_loss_impl="ultralytics_e2e",
            ultra_det_model=model.det_model,
        ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    resume_path = args.resume
    if args.resume_latest:
        resume_path = ckpt_dir / "latest.pt"
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        start_epoch, best_score, best_epoch = _load_checkpoint(
            ckpt_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        print(f"[pv26] resumed from {resume_path} (start_epoch={start_epoch + 1})")

    max_train_batches, max_val_batches = _resolve_max_batches(args)
    print(f"[pv26] run_dir={run_dir}")
    print(f"[pv26] device={device} seed={int(args.seed)} amp={amp_enabled}")
    print(f"[pv26] train_samples={len(train_ds)} val_samples={len(val_ds)}")

    for epoch in range(start_epoch, int(args.epochs)):
        print(f"\n[pv26] epoch {epoch + 1}/{args.epochs}")
        train_losses = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=max_train_batches,
            show_progress=args.progress,
            tqdm_fn=tqdm_fn,
            log_every=args.log_every,
            profile_every=args.profile_every,
            profile_sync_cuda=args.profile_sync_cuda,
        )
        print(format_loss_line("[train] mean", train_losses))

        val_losses, ious, map50 = validate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=max_val_batches,
            show_progress=args.progress,
            tqdm_fn=tqdm_fn,
            log_every=args.log_every,
        )
        print(format_loss_line("[val] mean", val_losses))
        for name, iou in ious.items():
            if iou is None:
                print(f"[val] iou_{name}=skipped(no_supervision)")
            else:
                print(f"[val] iou_{name}={iou:.4f}")
        if map50 is None:
            print("[val] map50=skipped(no_det_gt)")
        else:
            print(f"[val] map50={map50:.4f}")

        if writer is not None:
            writer.add_scalar("train/loss_total", train_losses["total"], epoch + 1)
            writer.add_scalar("train/loss_od", train_losses["od"], epoch + 1)
            writer.add_scalar("train/loss_da", train_losses["da"], epoch + 1)
            writer.add_scalar("train/loss_rm", train_losses["rm"], epoch + 1)
            writer.add_scalar("val/loss_total", val_losses["total"], epoch + 1)
            writer.add_scalar("val/loss_od", val_losses["od"], epoch + 1)
            writer.add_scalar("val/loss_da", val_losses["da"], epoch + 1)
            writer.add_scalar("val/loss_rm", val_losses["rm"], epoch + 1)
            if map50 is not None:
                writer.add_scalar("val/map50", map50, epoch + 1)
            for name, iou in ious.items():
                if iou is not None:
                    writer.add_scalar(f"val/iou_{name}", iou, epoch + 1)
            writer.flush()

        score, score_name = _choose_best_score(val_losses, map50)
        is_best = best_score is None or score > float(best_score)
        if is_best:
            best_score = score
            best_epoch = epoch + 1
            _save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                args=args,
            )
            print(f"[pv26] new best checkpoint ({score_name}={score:.6f}) at epoch {best_epoch}")

        _save_checkpoint(
            path=ckpt_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            args=args,
        )
        print(f"[pv26] saved latest checkpoint: {ckpt_dir / 'latest.pt'}")

    if writer is not None:
        writer.close()
    print(f"[pv26] finished. latest={ckpt_dir / 'latest.pt'} best={ckpt_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
