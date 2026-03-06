#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.constants import DET_CLASSES_CANONICAL
from pv26.criterion import PV26Criterion, PV26PreparedBatch
from pv26.multitask_model import PV26MultiHead, PV26MultiHeadYOLO26
from pv26.torch_dataset import AugmentSpec, Pv26ManifestDataset, Pv26Sample
from tools.train.common import (
    compute_map50,
    cxcywh_to_xyxy,
    decode_det_predictions,
    format_loss_line,
    resolve_device,
    seed_everything,
)

DEFAULT_DATASET_ROOT = Path("/home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full")


@dataclass(frozen=True)
class TrainPv26ScriptDefaults:
    # Edit this block when you run train_pv26.py directly from the repo.
    # Any CLI argument still overrides the value here, so Modal wrappers remain compatible.
    dataset_root: Path          = DEFAULT_DATASET_ROOT  # 직접 실행용 데이터셋 루트
    arch: str                   = "yolo26n"            # 모델 백본/헤드 조합
    epochs: int                 = 5                    # 총 학습 epoch 수
    batch_size: int             = 16                   # train/val 공통 배치 크기
    seg_output_stride: int      = 2                    # segmentation logits stride
    workers: int                = 6                    # DataLoader worker 수
    prefetch_factor: int        = 4                    # worker당 prefetch 배치 수
    persistent_workers: bool    = True                 # epoch 사이 worker 유지
    lr: float                   = 0.0                  # 0 이하면 optimizer별 자동 LR
    optimizer: str              = "adamw"              # adamw | adam | sgd
    weight_decay: float         = 1e-4                 # optimizer weight decay
    momentum: float             = 0.937                # SGD momentum
    scheduler: str              = "cosine"             # cosine | none
    warmup_epochs: int          = 3                    # warmup epoch 수
    warmup_start_factor: float  = 0.1                  # warmup 시작 LR 비율
    min_lr_ratio: float         = 0.05                 # cosine eta_min 비율
    compile: bool               = False                # model torch.compile on/off
    compile_mode: str           = "default"            # compile mode
    compile_fullgraph: bool     = False                # fullgraph compile on/off
    compile_seg_loss: bool      = True                 # seg loss block만 compile
    profile_every: int          = 20                   # train profile 출력 주기
    profile_sync_cuda: bool     = False                # profile 시 CUDA sync 여부
    profile_system: bool        = False                # 시스템 메모리/GPU 통계 포함
    device: str                 = "auto"               # auto | cpu | cuda | cuda:N
    amp: bool                   = True                 # CUDA AMP 사용 여부
    eval_map_every: int         = 1                    # val mAP 계산 주기
    train_drop_last: bool       = False                # 마지막 ragged batch drop 여부
    validate_masks: bool        = False                # dataset mask strict validation
    tensorboard: bool           = True                 # TensorBoard writer 사용 여부
    progress: bool              = False                # tqdm progress bar 사용 여부
    log_every: int              = 10                   # no-progress일 때 로그 주기
    augment: bool               = True                 # train augmentation on/off
    aug_hflip: float            = 0.5                  # horizontal flip 확률
    aug_brightness: float       = 0.2                  # brightness jitter 강도
    aug_contrast: float         = 0.2                  # contrast jitter 강도
    aug_saturation: float       = 0.2                  # saturation jitter 강도


SCRIPT_DEFAULTS = TrainPv26ScriptDefaults()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PV26 practical train/val pipeline")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DEFAULTS.dataset_root,
        help=(
            "PV26 dataset root with meta/split_manifest.csv "
            f"(default: {SCRIPT_DEFAULTS.dataset_root})"
        ),
    )
    p.add_argument("--arch", type=str, default=SCRIPT_DEFAULTS.arch, choices=["yolo26n", "stub"])
    p.add_argument("--epochs", type=int, default=SCRIPT_DEFAULTS.epochs)
    p.add_argument("--batch-size", type=int, default=SCRIPT_DEFAULTS.batch_size)
    p.add_argument(
        "--seg-output-stride",
        type=int,
        default=SCRIPT_DEFAULTS.seg_output_stride,
        choices=[1, 2],
        help=(
            "Segmentation output stride relative to input resolution "
            f"(default: {SCRIPT_DEFAULTS.seg_output_stride})."
        ),
    )
    p.add_argument("--workers", type=int, default=SCRIPT_DEFAULTS.workers)
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=SCRIPT_DEFAULTS.prefetch_factor,
        help="DataLoader prefetch factor when workers > 0",
    )
    p.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs",
    )
    p.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=SCRIPT_DEFAULTS.lr,
        help="Base LR. Set <=0 to use optimizer-specific auto LR (adam/adamw=1e-3, sgd=1e-2).",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default=SCRIPT_DEFAULTS.optimizer,
        choices=["adamw", "adam", "sgd"],
        help="Optimizer for all trainable params (default: adamw).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=SCRIPT_DEFAULTS.weight_decay,
        help="Weight decay (used by adamw/adam/sgd).",
    )
    p.add_argument(
        "--momentum",
        type=float,
        default=SCRIPT_DEFAULTS.momentum,
        help="Momentum for SGD optimizer.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default=SCRIPT_DEFAULTS.scheduler,
        choices=["cosine", "none"],
        help="LR scheduler policy (default: cosine).",
    )
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=SCRIPT_DEFAULTS.warmup_epochs,
        help="Warmup epochs before cosine phase (0 disables warmup).",
    )
    p.add_argument(
        "--warmup-start-factor",
        type=float,
        default=SCRIPT_DEFAULTS.warmup_start_factor,
        help="Warmup start LR factor for LinearLR (0<factor<=1).",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=SCRIPT_DEFAULTS.min_lr_ratio,
        help="eta_min ratio for cosine scheduler (eta_min = lr * ratio).",
    )
    p.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Enable torch.compile on CUDA.",
    )
    p.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile.",
    )
    p.add_argument(
        "--compile-mode",
        type=str,
        default=SCRIPT_DEFAULTS.compile_mode,
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode.",
    )
    p.add_argument(
        "--compile-fullgraph",
        dest="compile_fullgraph",
        action="store_true",
        help="Enable fullgraph mode in torch.compile (useful for graph-break A/B).",
    )
    p.add_argument(
        "--no-compile-fullgraph",
        dest="compile_fullgraph",
        action="store_false",
        help="Disable fullgraph mode in torch.compile.",
    )
    p.add_argument(
        "--compile-seg-loss",
        dest="compile_seg_loss",
        action="store_true",
        help="Enable torch.compile on the DA/RM seg loss block only.",
    )
    p.add_argument(
        "--no-compile-seg-loss",
        dest="compile_seg_loss",
        action="store_false",
        help="Disable seg loss torch.compile.",
    )
    p.add_argument(
        "--det-pretrained",
        type=Path,
        default=None,
        help="Optional path to detection trunk checkpoint/state_dict to load before training.",
    )
    p.add_argument(
        "--profile-every",
        type=int,
        default=SCRIPT_DEFAULTS.profile_every,
        help="Print N-step averaged train stage timings (default: off; set >0 to enable).",
    )
    p.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around profiling timers for more accurate stage timings (adds overhead).",
    )
    p.add_argument(
        "--profile-system",
        action="store_true",
        help="Include CPU/CUDA memory + nvidia-smi in profile logs (higher overhead).",
    )
    p.add_argument("--device", type=str, default=SCRIPT_DEFAULTS.device, help="auto|cpu|cuda|cuda:N")
    p.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP (CUDA only)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    p.add_argument("--max-batches", type=int, default=0, help="Max batches for both train/val (0=all)")
    p.add_argument("--max-train-batches", type=int, default=0, help="Max train batches (0=all or --max-batches)")
    p.add_argument("--max-val-batches", type=int, default=0, help="Max val batches (0=all or --max-batches)")
    p.add_argument("--run-name", type=str, default="", help="Run directory name under --out-dir")
    p.add_argument("--out-dir", type=Path, default=Path("runs/pv26_train"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    p.add_argument("--resume-latest", action="store_true", help="Resume from <run>/checkpoints/latest.pt")
    p.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    p.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    p.add_argument("--progress", dest="progress", action="store_true")
    p.add_argument("--no-progress", dest="progress", action="store_false")
    p.add_argument(
        "--log-every",
        type=int,
        default=SCRIPT_DEFAULTS.log_every,
        help="Console print interval when tqdm is disabled",
    )
    p.add_argument(
        "--eval-map-every",
        type=int,
        default=SCRIPT_DEFAULTS.eval_map_every,
        help="Compute validation mAP every N epochs (final epoch always computes mAP).",
    )
    p.add_argument(
        "--train-drop-last",
        action="store_true",
        help="Drop incomplete final train batch (useful for throughput/compile-shape benchmarks).",
    )
    p.add_argument(
        "--validate-masks",
        action="store_true",
        help="Enable strict mask value validation in dataset __getitem__ (debug only; adds CPU overhead).",
    )
    p.add_argument("--augment", dest="augment", action="store_true", help="Enable train-time augmentation")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable train-time augmentation")
    p.add_argument(
        "--aug-hflip",
        type=float,
        default=SCRIPT_DEFAULTS.aug_hflip,
        help="Horizontal flip probability for train set",
    )
    p.add_argument(
        "--aug-brightness",
        type=float,
        default=SCRIPT_DEFAULTS.aug_brightness,
        help="Brightness jitter delta (0 disables)",
    )
    p.add_argument(
        "--aug-contrast",
        type=float,
        default=SCRIPT_DEFAULTS.aug_contrast,
        help="Contrast jitter delta (0 disables)",
    )
    p.add_argument(
        "--aug-saturation",
        type=float,
        default=SCRIPT_DEFAULTS.aug_saturation,
        help="Saturation jitter delta (0 disables)",
    )
    p.set_defaults(
        persistent_workers=SCRIPT_DEFAULTS.persistent_workers,
        compile=SCRIPT_DEFAULTS.compile,
        compile_fullgraph=SCRIPT_DEFAULTS.compile_fullgraph,
        compile_seg_loss=SCRIPT_DEFAULTS.compile_seg_loss,
        profile_sync_cuda=SCRIPT_DEFAULTS.profile_sync_cuda,
        profile_system=SCRIPT_DEFAULTS.profile_system,
        amp=SCRIPT_DEFAULTS.amp,
        tensorboard=SCRIPT_DEFAULTS.tensorboard,
        progress=SCRIPT_DEFAULTS.progress,
        train_drop_last=SCRIPT_DEFAULTS.train_drop_last,
        validate_masks=SCRIPT_DEFAULTS.validate_masks,
        augment=SCRIPT_DEFAULTS.augment,
    )
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


def _collate_train(samples: List[Pv26Sample], *, seg_output_stride: int) -> Tuple[torch.Tensor, PV26PreparedBatch]:
    # Build image batch/targets in worker process so main thread can focus on GPU steps.
    images = torch.stack([s.image for s in samples], dim=0)
    return images, PV26PreparedBatch.from_samples(samples, seg_output_stride=int(seg_output_stride))


def _collate_eval(samples: List[Pv26Sample], *, seg_output_stride: int) -> Tuple[torch.Tensor, PV26PreparedBatch]:
    images = torch.stack([s.image for s in samples], dim=0)
    return images, PV26PreparedBatch.from_samples(
        samples,
        include_sample_id=True,
        include_fullres_masks=True,
        seg_output_stride=int(seg_output_stride),
    )


def _move_prepared_batch_to_device(batch: PV26PreparedBatch, *, device: torch.device) -> PV26PreparedBatch:
    return batch.to_device(device=device)


def _resize_seg_logits_for_eval(logits: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == out_size:
        return logits
    return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: Optional[float],
    best_epoch: Optional[int],
    best_det_score: Optional[float] = None,
    best_det_epoch: Optional[int] = None,
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
        "model_state": _unwrap_compiled_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_det_score": best_det_score,
        "best_det_epoch": best_det_epoch,
        "args": args_serialized,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, Optional[float], Optional[int], Optional[float], Optional[int]]:
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our checkpoint stores optimizer/scaler/args metadata, so we explicitly
    # request full load for local, trusted checkpoints.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions.
        ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        _unwrap_compiled_model(model).load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = ckpt.get("best_score", None)
        best_epoch = ckpt.get("best_epoch", None)
        best_det_score = ckpt.get("best_det_score", None)
        best_det_epoch = ckpt.get("best_det_epoch", None)
        return start_epoch, best_score, best_epoch, best_det_score, best_det_epoch

    # Fallback for raw state_dict checkpoints.
    _unwrap_compiled_model(model).load_state_dict(ckpt)
    return 0, None, None, None, None


def _choose_best_total_score(val_losses: Dict[str, float]) -> float:
    # Multi-task default: select best checkpoint by the aggregate validation loss.
    return -float(val_losses["total"])


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


def _unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    base = model
    while hasattr(base, "_orig_mod"):
        base = getattr(base, "_orig_mod")
    return base


def _resolve_base_lr(*, args: argparse.Namespace) -> tuple[float, str]:
    lr = float(args.lr)
    if lr > 0.0:
        return lr, "manual"
    opt_name = str(args.optimizer).strip().lower()
    if opt_name == "sgd":
        return 1e-2, "auto"
    return 1e-3, "auto"


def _build_optimizer(*, model: torch.nn.Module, args: argparse.Namespace, base_lr: float) -> torch.optim.Optimizer:
    # Param groups:
    # - Separate detection trunk vs PV26 heads when using --det-pretrained (helps stability).
    # - Exclude bias/1D params (norm scales, biases) from weight decay.
    det_lr_mult = 1.0
    if getattr(args, "det_pretrained", None) is not None and hasattr(model, "det_model"):
        det_lr_mult = 0.1
    trunk_lr = float(base_lr) * float(det_lr_mult)
    head_lr = float(base_lr)

    trunk_decay: List[torch.nn.Parameter] = []
    trunk_no_decay: List[torch.nn.Parameter] = []
    head_decay: List[torch.nn.Parameter] = []
    head_no_decay: List[torch.nn.Parameter] = []

    def _is_no_decay(n: str, p: torch.nn.Parameter) -> bool:
        return n.endswith(".bias") or int(p.ndim) < 2

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_trunk = name.startswith("det_model.") or name.startswith("det_backend.det_model.")
        is_no_decay = _is_no_decay(name, p)
        if is_trunk and is_no_decay:
            trunk_no_decay.append(p)
        elif is_trunk:
            trunk_decay.append(p)
        elif is_no_decay:
            head_no_decay.append(p)
        else:
            head_decay.append(p)

    groups: List[Dict[str, Any]] = []
    if trunk_decay:
        groups.append({"params": trunk_decay, "lr": trunk_lr, "weight_decay": float(args.weight_decay), "name": "trunk_decay"})
    if trunk_no_decay:
        groups.append({"params": trunk_no_decay, "lr": trunk_lr, "weight_decay": 0.0, "name": "trunk_no_decay"})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": float(args.weight_decay), "name": "head_decay"})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "head_no_decay"})
    if not groups:
        raise RuntimeError("no trainable parameters found")
    opt_name = str(args.optimizer).strip().lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(groups, lr=base_lr)
    if opt_name == "adam":
        return torch.optim.Adam(groups, lr=base_lr)
    if opt_name == "sgd":
        return torch.optim.SGD(groups, lr=base_lr, momentum=float(args.momentum), nesterov=True)
    raise ValueError(f"unsupported optimizer: {opt_name}")


def _build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_epochs: int,
    base_lr: float,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    sched_name = str(args.scheduler).strip().lower()
    warmup_epochs = max(0, int(args.warmup_epochs))
    start_factor = float(args.warmup_start_factor)
    if start_factor <= 0.0 or start_factor > 1.0:
        raise ValueError("--warmup-start-factor must be in (0, 1]")

    if sched_name == "none":
        if warmup_epochs > 0:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
        return None
    if sched_name == "cosine":
        eta_min = float(base_lr) * max(0.0, float(args.min_lr_ratio))
        total_epochs_i = max(1, int(total_epochs))
        if warmup_epochs > 0 and warmup_epochs < total_epochs_i:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs_i - warmup_epochs),
                eta_min=eta_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        if warmup_epochs > 0:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs_i,
            eta_min=eta_min,
        )
    raise ValueError(f"unsupported scheduler: {sched_name}")


def _extract_state_dict_candidates(payload: Any) -> List[Dict[str, torch.Tensor]]:
    candidates: List[Dict[str, torch.Tensor]] = []
    if isinstance(payload, torch.nn.Module):
        candidates.append({k: v for k, v in payload.state_dict().items() if isinstance(v, torch.Tensor)})
        return candidates
    if isinstance(payload, dict):
        tensor_entries = {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        if tensor_entries:
            candidates.append(tensor_entries)
        for key in ("state_dict", "model_state", "model", "ema"):
            if key in payload:
                candidates.extend(_extract_state_dict_candidates(payload[key]))
    return candidates


def _select_best_det_state_dict(
    *,
    candidates: List[Dict[str, torch.Tensor]],
    target_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    best: Dict[str, torch.Tensor] = {}
    best_count = 0
    prefixes = ("", "model.", "module.", "_orig_mod.")
    for cand in candidates:
        for prefix in prefixes:
            matched: Dict[str, torch.Tensor] = {}
            for k, v in cand.items():
                if prefix:
                    if not k.startswith(prefix):
                        continue
                    kk = k[len(prefix) :]
                else:
                    kk = k
                tgt = target_state.get(kk, None)
                if tgt is None:
                    continue
                if tuple(v.shape) != tuple(tgt.shape):
                    continue
                matched[kk] = v
            if len(matched) > best_count:
                best = matched
                best_count = len(matched)
    return best


def _maybe_load_det_pretrained(*, model: torch.nn.Module, det_pretrained: Optional[Path], device: torch.device) -> None:
    if det_pretrained is None:
        return
    p = Path(det_pretrained).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"--det-pretrained not found: {p}")
    model_base = _unwrap_compiled_model(model)
    det_model = getattr(model_base, "det_model", None)
    if det_model is None:
        print(f"[pv26] warning: det_pretrained ignored (model has no det_model): {p}", flush=True)
        return
    try:
        payload = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=device)
    candidates = _extract_state_dict_candidates(payload)
    if not candidates:
        raise RuntimeError(f"no state_dict-like payload found in {p}")
    target_state = det_model.state_dict()
    matched = _select_best_det_state_dict(candidates=candidates, target_state=target_state)
    if not matched:
        raise RuntimeError(
            f"failed to match detection trunk state_dict keys from {p}; "
            f"target_keys={len(target_state)} candidates={len(candidates)}"
        )
    casted: Dict[str, torch.Tensor] = {}
    for k, v in matched.items():
        tgt = target_state[k]
        casted[k] = v.detach().to(dtype=tgt.dtype)
    det_model.load_state_dict(casted, strict=False)
    print(
        f"[pv26] loaded detection pretrained weights: matched={len(casted)}/{len(target_state)} from {p}",
        flush=True,
    )


def _maybe_compile_model(
    *,
    model: torch.nn.Module,
    device: torch.device,
    enable_compile: bool,
    compile_mode: str,
    compile_fullgraph: bool,
) -> torch.nn.Module:
    if not bool(enable_compile):
        print("[pv26] torch.compile disabled", flush=True)
        return model
    if device.type != "cuda":
        print("[pv26] torch.compile skipped (device is not CUDA)", flush=True)
        return model
    if not hasattr(torch, "compile"):
        print("[pv26] torch.compile unavailable on this torch build", flush=True)
        return model
    try:
        compiled = torch.compile(model, mode=str(compile_mode), fullgraph=bool(compile_fullgraph))
        print(
            f"[pv26] torch.compile enabled (mode={compile_mode}, fullgraph={bool(compile_fullgraph)})",
            flush=True,
        )
        return compiled
    except Exception as exc:
        print(f"[pv26] warning: torch.compile failed, fallback to eager ({exc})", flush=True)
        return model


def _maybe_compile_seg_loss(
    *,
    criterion: PV26Criterion,
    device: torch.device,
    enable_compile: bool,
    compile_mode: str,
    compile_fullgraph: bool,
) -> PV26Criterion:
    criterion.disable_compile_seg_loss()
    if not bool(enable_compile):
        print("[pv26] seg-loss compile disabled", flush=True)
        return criterion
    if device.type != "cuda":
        print("[pv26] seg-loss compile skipped (device is not CUDA)", flush=True)
        return criterion
    if not hasattr(torch, "compile"):
        print("[pv26] seg-loss compile unavailable on this torch build", flush=True)
        return criterion
    try:
        criterion.enable_compile_seg_loss(
            compile_mode=str(compile_mode),
            compile_fullgraph=bool(compile_fullgraph),
        )
        print(
            f"[pv26] seg-loss compile enabled (mode={compile_mode}, fullgraph={bool(compile_fullgraph)})",
            flush=True,
        )
        return criterion
    except Exception as exc:
        criterion.disable_compile_seg_loss()
        print(f"[pv26] warning: seg-loss compile failed, fallback to eager ({exc})", flush=True)
        return criterion


def train_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: PV26Criterion,
    loader: DataLoader[Tuple[torch.Tensor, PV26PreparedBatch]],
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
    profile_system: bool,
) -> Dict[str, float]:
    model.train()
    losses_sum = {
        k: torch.zeros((), device=device, dtype=torch.float32)
        for k in ("total", "od", "da", "rm", "rm_lane_subclass")
    }
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
    loader_it = iter(loader)
    step_iter = range(step_cap)
    if show_progress and tqdm_fn is not None:
        step_iter = tqdm_fn(step_iter, total=step_cap, desc="train", leave=False)

    for batch_idx in step_iter:
        t_wait0 = time.perf_counter()
        try:
            images_cpu, target_batch_cpu = next(loader_it)
        except StopIteration:
            break
        t_wait = time.perf_counter() - t_wait0
        if prof_every > 0:
            wait_hist.append(float(t_wait))
        t0 = time.perf_counter()
        images = _prepare_images_for_model(images_cpu, device=device)
        if prof_every > 0 and profile_system and device.type == "cuda":
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

        steps += 1
        for key in losses_sum:
            losses_sum[key] += losses[key].detach()

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
                ]
                if profile_system:
                    extra_parts.append(_format_cpu_stats())
                    if device.type == "cuda":
                        extra_parts.append(_format_cuda_mem_stats(device=device))
                        if min_free_b_window is not None:
                            extra_parts.append(f"min_free={_gib(min_free_b_window):.2f}GiB")
                        smi_line = _query_nvidia_smi()
                        if smi_line:
                            extra_parts.append(smi_line)
                if device.type == "cuda" and amp_enabled:
                    extra_parts.append(f"amp_scale={float(scaler.get_scale()):.0f}")
                print(
                    "[profile][train] "
                    f"step={batch_idx + 1}/{step_cap} avg{prof_count}({sync_tag}) "
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

        if (not show_progress) and ((batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0):
            step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
            print(format_loss_line(f"[train] step={batch_idx + 1}/{step_cap}", step_losses), flush=True)

    den = max(1, steps)
    return {k: float((v / float(den)).detach().cpu()) for k, v in losses_sum.items()}


def validate(
    *,
    model: torch.nn.Module,
    criterion: PV26Criterion,
    loader: DataLoader[Tuple[torch.Tensor, PV26PreparedBatch]],
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
    show_progress: bool,
    tqdm_fn,
    log_every: int,
    profile_every: int,
    compute_map: bool,
) -> tuple[Dict[str, float], Dict[str, Optional[float]], Optional[float], bool]:
    model.eval()
    t_val_start = time.perf_counter()
    losses_sum = {
        k: torch.zeros((), device=device, dtype=torch.float32)
        for k in ("total", "od", "da", "rm", "rm_lane_subclass")
    }
    steps = 0

    metric_stats: Dict[str, Dict[str, int]] = {
        "da": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_marker": {"inter": 0, "union": 0, "supervised": 0},
        "rm_road_marker_non_lane": {"inter": 0, "union": 0, "supervised": 0},
        "rm_stop_line": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_subclass_white_solid": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_subclass_white_dashed": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_subclass_yellow_solid": {"inter": 0, "union": 0, "supervised": 0},
        "rm_lane_subclass_yellow_dashed": {"inter": 0, "union": 0, "supervised": 0},
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

            images_cpu, target_batch_cpu = packed
            images = _prepare_images_for_model(images_cpu, device=device)
            target_batch = _move_prepared_batch_to_device(target_batch_cpu, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                preds = model(images)
                losses = criterion(preds, target_batch)

            steps += 1
            for key in losses_sum:
                losses_sum[key] += losses[key].detach()

            if (not show_progress) and ((batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0):
                step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
                print(format_loss_line(f"[val] step={batch_idx + 1}/{step_cap}", step_losses), flush=True)

            sample_ids: List[str] = list(target_batch_cpu.sample_id)
            has_det_cpu: torch.Tensor = target_batch_cpu.has_det
            det_scope_list: List[str] = list(target_batch_cpu.det_label_scope)
            det_yolo_list: List[torch.Tensor] = list(target_batch_cpu.det_yolo)
            img_h = int(images_cpu.shape[-2])
            img_w = int(images_cpu.shape[-1])

            da_mask_dev: torch.Tensor = (
                target_batch.da_mask_fullres if target_batch.da_mask_fullres is not None else target_batch.da_mask
            )
            rm_mask_dev: torch.Tensor = (
                target_batch.rm_mask_fullres if target_batch.rm_mask_fullres is not None else target_batch.rm_mask
            )
            has_da_dev: torch.Tensor = target_batch.has_da.to(dtype=torch.bool)
            has_rm_dev: torch.Tensor = target_batch.has_rm.to(dtype=torch.bool)
            rm_lane_subclass_mask_dev: torch.Tensor = (
                target_batch.rm_lane_subclass_mask_fullres
                if target_batch.rm_lane_subclass_mask_fullres is not None
                else target_batch.rm_lane_subclass_mask
            )
            has_rm_lane_subclass_dev: torch.Tensor = target_batch.has_rm_lane_subclass.to(dtype=torch.bool)

            pred_da_logits = _resize_seg_logits_for_eval(preds.da, da_mask_dev.shape[-2:])
            pred_rm_logits = _resize_seg_logits_for_eval(preds.rm, rm_mask_dev.shape[-2:])
            pred_lane_subclass_logits = _resize_seg_logits_for_eval(
                preds.rm_lane_subclass,
                rm_lane_subclass_mask_dev.shape[-2:],
            )

            valid_da = (da_mask_dev != 255) & has_da_dev[:, None, None]
            supervised_da = valid_da.reshape(valid_da.shape[0], -1).any(dim=1)
            pred_da = pred_da_logits[:, 0] > 0
            tgt_da = da_mask_dev == 1
            metric_stats["da"]["inter"] += int((((pred_da & tgt_da) & valid_da).sum()).item())
            metric_stats["da"]["union"] += int((((pred_da | tgt_da) & valid_da).sum()).item())
            metric_stats["da"]["supervised"] += int(supervised_da.sum().item())

            rm_channels = [
                "rm_lane_marker",
                "rm_road_marker_non_lane",
                "rm_stop_line",
            ]
            valid_rm = (rm_mask_dev != 255) & has_rm_dev[:, :, None, None]
            pred_rm = pred_rm_logits > 0
            tgt_rm = rm_mask_dev == 1
            for c_idx, name in enumerate(rm_channels):
                valid_c = valid_rm[:, c_idx]
                supervised_c = valid_c.reshape(valid_c.shape[0], -1).any(dim=1)
                metric_stats[name]["inter"] += int((((pred_rm[:, c_idx] & tgt_rm[:, c_idx]) & valid_c).sum()).item())
                metric_stats[name]["union"] += int((((pred_rm[:, c_idx] | tgt_rm[:, c_idx]) & valid_c).sum()).item())
                metric_stats[name]["supervised"] += int(supervised_c.sum().item())

            lane_subclass_specs = [
                (1, "rm_lane_subclass_white_solid"),
                (2, "rm_lane_subclass_white_dashed"),
                (3, "rm_lane_subclass_yellow_solid"),
                (4, "rm_lane_subclass_yellow_dashed"),
            ]
            pred_lane_subclass = torch.argmax(pred_lane_subclass_logits, dim=1)
            valid_lane_subclass = (rm_lane_subclass_mask_dev != 255) & has_rm_lane_subclass_dev[:, None, None]
            supervised_lane_subclass = valid_lane_subclass.reshape(valid_lane_subclass.shape[0], -1).any(dim=1)
            for cls_id, cls_name in lane_subclass_specs:
                pred_c = pred_lane_subclass == int(cls_id)
                tgt_c = rm_lane_subclass_mask_dev == int(cls_id)
                metric_stats[cls_name]["inter"] += int((((pred_c & tgt_c) & valid_lane_subclass).sum()).item())
                metric_stats[cls_name]["union"] += int((((pred_c | tgt_c) & valid_lane_subclass).sum()).item())
                metric_stats[cls_name]["supervised"] += int(supervised_lane_subclass.sum().item())

            for i in range(int(images_cpu.shape[0])):
                sid = sample_ids[i] if i < len(sample_ids) else f"{batch_idx}:{i}"
                scope = str(det_scope_list[i]).strip().lower()
                if int(has_det_cpu[i].item()) != 0 and scope == "full":
                    gt = det_yolo_list[i]
                    if gt.numel():
                        gt_cls = gt[:, 0].to(dtype=torch.long)
                        gt_boxes_norm = cxcywh_to_xyxy(gt[:, 1:5].to(dtype=torch.float32)).clamp(0.0, 1.0)
                        if isinstance(preds.det, tuple):
                            gt_boxes = gt_boxes_norm.clone()
                            gt_boxes[:, 0::2] *= float(img_w)
                            gt_boxes[:, 1::2] *= float(img_h)
                        else:
                            gt_boxes = gt_boxes_norm
                        for c in range(num_det_classes):
                            m = gt_cls == c
                            if bool(m.any()):
                                gt_by_img_class[(sid, c)] = gt_boxes[m].detach().cpu()

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
                                    det_preds_by_class[c].append((float(det_scores[j].item()), sid, det_boxes[j]))
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
                                det_preds_by_class[c].append((float(det_scores[j].item()), sid, det_boxes[j]))

    t_loop_end = time.perf_counter()
    den = max(1, steps)
    val_losses = {k: float((v / float(den)).detach().cpu()) for k, v in losses_sum.items()}
    iou_metrics: Dict[str, Optional[float]] = {}
    for name, stats in metric_stats.items():
        if stats["supervised"] == 0:
            iou_metrics[name] = None
        elif stats["union"] == 0:
            iou_metrics[name] = 0.0
        else:
            iou_metrics[name] = float(stats["inter"]) / float(stats["union"])

    lane_subclass_metric_names = [
        "rm_lane_subclass_white_solid",
        "rm_lane_subclass_white_dashed",
        "rm_lane_subclass_yellow_solid",
        "rm_lane_subclass_yellow_dashed",
    ]
    lane_subclass_ious = [iou_metrics[n] for n in lane_subclass_metric_names if iou_metrics.get(n) is not None]
    iou_metrics["rm_lane_subclass_miou4"] = (
        None if not lane_subclass_ious else float(sum(lane_subclass_ious) / float(len(lane_subclass_ious)))
    )
    lane_subclass_present_ious = [
        float(iou_metrics[n])
        for n in lane_subclass_metric_names
        if iou_metrics.get(n) is not None and metric_stats[n]["union"] > 0
    ]
    iou_metrics["rm_lane_subclass_miou4_present"] = (
        None
        if not lane_subclass_present_ious
        else float(sum(lane_subclass_present_ious) / float(len(lane_subclass_present_ious)))
    )

    t_map_start = time.perf_counter()
    map50: Optional[float]
    map_computed = bool(compute_map)
    if map_computed:
        map50, _ap_by_class = compute_map50(
            preds_by_class=det_preds_by_class,
            gt_by_img_class=gt_by_img_class,
            num_classes=num_det_classes,
            iou_thres=0.5,
        )
    else:
        map50 = None
    t_map_end = time.perf_counter()

    if int(profile_every) > 0:
        loop_ms = 1000.0 * (t_loop_end - t_val_start)
        map_ms = 1000.0 * (t_map_end - t_map_start)
        total_ms = 1000.0 * (t_map_end - t_val_start)
        print(
            "[profile][val] "
            f"steps={steps}/{step_cap} loop={loop_ms:.1f}ms map={map_ms:.1f}ms "
            f"total={total_ms:.1f}ms map_computed={map_computed}",
            flush=True,
        )
    return val_losses, iou_metrics, map50, map_computed


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

    train_ds = Pv26ManifestDataset(
        dataset_root=args.dataset_root,
        splits=("train",),
        augment=train_aug,
        validate_masks=bool(args.validate_masks),
    )
    val_ds = Pv26ManifestDataset(
        dataset_root=args.dataset_root,
        splits=("val",),
        validate_masks=bool(args.validate_masks),
    )
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

    train_collate_fn = partial(
        _collate_eval if args.arch == "stub" else _collate_train,
        seg_output_stride=int(args.seg_output_stride),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        generator=dl_gen,
        pin_memory=(device.type == "cuda"),
        drop_last=bool(args.train_drop_last),
        **loader_perf_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(_collate_eval, seg_output_stride=int(args.seg_output_stride)),
        pin_memory=(device.type == "cuda"),
        **loader_perf_kwargs,
    )

    if args.arch == "stub":
        model = PV26MultiHead(
            num_det_classes=len(DET_CLASSES_CANONICAL),
            num_lane_subclasses=4,
            seg_output_stride=int(args.seg_output_stride),
        ).to(device)
        criterion = PV26Criterion(num_det_classes=len(DET_CLASSES_CANONICAL), num_lane_subclasses=4).to(device)
    else:
        model = PV26MultiHeadYOLO26(
            num_det_classes=len(DET_CLASSES_CANONICAL),
            num_lane_subclasses=4,
            yolo26_cfg="yolo26n.yaml",
            seg_output_stride=int(args.seg_output_stride),
        ).to(device)
        criterion = PV26Criterion(
            num_det_classes=len(DET_CLASSES_CANONICAL),
            num_lane_subclasses=4,
            od_loss_impl="ultralytics_e2e",
            det_loss_adapter=model.build_det_loss_adapter(),
        ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    _maybe_load_det_pretrained(model=model, det_pretrained=args.det_pretrained, device=device)
    base_lr, lr_mode = _resolve_base_lr(args=args)
    optimizer = _build_optimizer(model=model, args=args, base_lr=base_lr)
    scheduler = _build_scheduler(
        optimizer=optimizer,
        args=args,
        total_epochs=int(args.epochs),
        base_lr=base_lr,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    best_det_score: Optional[float] = None
    best_det_epoch: Optional[int] = None
    resume_path = args.resume
    if args.resume_latest:
        resume_path = ckpt_dir / "latest.pt"
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        start_epoch, best_score, best_epoch, best_det_score, best_det_epoch = _load_checkpoint(
            ckpt_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(f"[pv26] resumed from {resume_path} (start_epoch={start_epoch + 1})")

    model = _maybe_compile_model(
        model=model,
        device=device,
        enable_compile=bool(args.compile),
        compile_mode=str(args.compile_mode),
        compile_fullgraph=bool(args.compile_fullgraph),
    )
    criterion = _maybe_compile_seg_loss(
        criterion=criterion,
        device=device,
        enable_compile=bool(args.compile_seg_loss),
        compile_mode=str(args.compile_mode),
        compile_fullgraph=bool(args.compile_fullgraph),
    )

    max_train_batches, max_val_batches = _resolve_max_batches(args)
    print(f"[pv26] run_dir={run_dir}")
    print(f"[pv26] device={device} seed={int(args.seed)} amp={amp_enabled}")
    print(f"[pv26] lr={base_lr:.6g} (mode={lr_mode})")
    print(
        f"[pv26] optimizer={str(args.optimizer).lower()} scheduler={str(args.scheduler).lower()} "
        f"warmup_epochs={int(args.warmup_epochs)} warmup_start_factor={float(args.warmup_start_factor):.3f} "
        f"compile={bool(args.compile)} compile_mode={args.compile_mode} "
        f"compile_fullgraph={bool(args.compile_fullgraph)} "
        f"compile_seg_loss={bool(args.compile_seg_loss)}"
    )
    print(
        f"[pv26] profile_every={int(args.profile_every)} profile_sync_cuda={bool(args.profile_sync_cuda)} "
        f"profile_system={bool(args.profile_system)}"
    )
    print(
        f"[pv26] train_drop_last={bool(args.train_drop_last)} "
        f"validate_masks={bool(args.validate_masks)} "
        f"seg_output_stride={int(args.seg_output_stride)}"
    )
    print(f"[pv26] eval_map_every={max(1, int(args.eval_map_every))}")
    if args.det_pretrained is not None:
        print(f"[pv26] det_pretrained={Path(args.det_pretrained)}")
    print(f"[pv26] train_samples={len(train_ds)} val_samples={len(val_ds)}")

    eval_map_every = max(1, int(args.eval_map_every))
    total_epochs = int(args.epochs)
    for epoch in range(start_epoch, total_epochs):
        epoch_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[pv26] epoch {epoch + 1}/{args.epochs} [{epoch_ts}]")
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
            profile_system=args.profile_system,
        )
        print(format_loss_line("[train] mean", train_losses))

        compute_map = ((epoch + 1) % eval_map_every == 0) or ((epoch + 1) == total_epochs)
        val_losses, ious, map50, map_computed = validate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=max_val_batches,
            show_progress=args.progress,
            tqdm_fn=tqdm_fn,
            log_every=args.log_every,
            profile_every=args.profile_every,
            compute_map=compute_map,
        )
        print(format_loss_line("[val] mean", val_losses))
        for name, iou in ious.items():
            if iou is None:
                print(f"[val] iou_{name}=skipped(no_supervision)")
            else:
                print(f"[val] iou_{name}={iou:.4f}")
        if not map_computed:
            print(f"[val] map50=skipped(eval_map_every={eval_map_every})")
        elif map50 is None:
            print("[val] map50=skipped(no_det_gt)")
        else:
            print(f"[val] map50={map50:.4f}")

        if writer is not None:
            writer.add_scalar("train/loss_total", train_losses["total"], epoch + 1)
            writer.add_scalar("train/loss_od", train_losses["od"], epoch + 1)
            writer.add_scalar("train/loss_da", train_losses["da"], epoch + 1)
            writer.add_scalar("train/loss_rm", train_losses["rm"], epoch + 1)
            writer.add_scalar("train/loss_rm_lane_subclass", train_losses["rm_lane_subclass"], epoch + 1)
            writer.add_scalar("val/loss_total", val_losses["total"], epoch + 1)
            writer.add_scalar("val/loss_od", val_losses["od"], epoch + 1)
            writer.add_scalar("val/loss_da", val_losses["da"], epoch + 1)
            writer.add_scalar("val/loss_rm", val_losses["rm"], epoch + 1)
            writer.add_scalar("val/loss_rm_lane_subclass", val_losses["rm_lane_subclass"], epoch + 1)
            if map_computed and map50 is not None:
                writer.add_scalar("val/map50", map50, epoch + 1)
            for name, iou in ious.items():
                if iou is not None:
                    writer.add_scalar(f"val/iou_{name}", iou, epoch + 1)
            for gi, g in enumerate(optimizer.param_groups):
                g_name = str(g.get("name", f"group{gi}"))
                writer.add_scalar(f"train/lr_{g_name}", float(g["lr"]), epoch + 1)
            writer.flush()

        if scheduler is not None:
            prev_lrs = [(str(g.get("name", f"group{gi}")), float(g["lr"])) for gi, g in enumerate(optimizer.param_groups)]
            scheduler.step()
            next_lrs = [(str(g.get("name", f"group{gi}")), float(g["lr"])) for gi, g in enumerate(optimizer.param_groups)]
            if len(prev_lrs) == 1:
                print(f"[pv26] lr {prev_lrs[0][1]:.6g} -> {next_lrs[0][1]:.6g}")
            else:
                prev_s = " ".join([f"{n}={lr:.6g}" for n, lr in prev_lrs])
                next_s = " ".join([f"{n}={lr:.6g}" for n, lr in next_lrs])
                print(f"[pv26] lr {prev_s} -> {next_s}")

        total_score = _choose_best_total_score(val_losses)
        is_best_total = best_score is None or total_score > float(best_score)
        if is_best_total:
            best_score = total_score
            best_epoch = epoch + 1
            _save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                best_det_score=best_det_score,
                best_det_epoch=best_det_epoch,
                args=args,
            )
            print(f"[pv26] new best checkpoint (neg_val_total_loss={best_score:.6f}) at epoch {best_epoch}")

        if map_computed and map50 is not None:
            det_score = float(map50)
            is_best_det = best_det_score is None or det_score > float(best_det_score)
            if is_best_det:
                best_det_score = det_score
                best_det_epoch = epoch + 1
                _save_checkpoint(
                    path=ckpt_dir / "best_det.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    best_det_score=best_det_score,
                    best_det_epoch=best_det_epoch,
                    args=args,
                )
                print(f"[pv26] new best_det checkpoint (map50={best_det_score:.6f}) at epoch {best_det_epoch}")

        _save_checkpoint(
            path=ckpt_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            best_det_score=best_det_score,
            best_det_epoch=best_det_epoch,
            args=args,
        )
        print(f"[pv26] saved latest checkpoint: {ckpt_dir / 'latest.pt'}")

    if writer is not None:
        writer.close()
    print(
        f"[pv26] finished. latest={ckpt_dir / 'latest.pt'} "
        f"best_total={ckpt_dir / 'best.pt'} best_det={ckpt_dir / 'best_det.pt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
