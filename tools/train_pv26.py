#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional
import warnings

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pv26.constants import DET_CLASSES_CANONICAL
from pv26.criterion import PV26Criterion
from pv26.multitask_model import PV26MultiHead, PV26MultiHeadYOLO26
from pv26.torch_dataset import AugmentSpec, Pv26ManifestDataset, Pv26Sample
from tools.train_pv26_smoke import (
    _compute_map50,
    _cxcywh_to_xyxy,
    _decode_det_predictions,
    _format_loss_line,
    _mean_losses,
    _update_binary_iou,
    resolve_device,
    seed_everything,
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
    p.add_argument("--lr", type=float, default=5e-4)
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


def train_one_epoch(
    *,
    model: torch.nn.Module,
    criterion: PV26Criterion,
    loader: DataLoader[List[Pv26Sample]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
    show_progress: bool,
    tqdm_fn,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    losses_sum = {"total": 0.0, "od": 0.0, "da": 0.0, "rm": 0.0}
    steps = 0
    step_cap = _limit_steps(len(loader), max_batches)
    iterator = enumerate(loader)
    if show_progress and tqdm_fn is not None:
        iterator = tqdm_fn(iterator, total=step_cap, desc="train", leave=False)

    for batch_idx, batch in iterator:
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = torch.stack([s.image for s in batch], dim=0).to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            preds = model(images)
            losses = criterion(preds, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        steps += 1
        for key in losses_sum:
            losses_sum[key] += float(losses[key].detach().item())

        if show_progress and tqdm_fn is not None:
            iterator.set_postfix(loss=float(losses["total"].detach().item()))
        elif (batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0:
            step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
            print(_format_loss_line(f"[train] step={batch_idx + 1}", step_losses), flush=True)

    return _mean_losses(losses_sum, steps)


def validate(
    *,
    model: torch.nn.Module,
    criterion: PV26Criterion,
    loader: DataLoader[List[Pv26Sample]],
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
        for batch_idx, batch in iterator:
            if max_batches > 0 and batch_idx >= max_batches:
                break

            images = torch.stack([s.image for s in batch], dim=0).to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                preds = model(images)
                losses = criterion(preds, batch)

            steps += 1
            for key in losses_sum:
                losses_sum[key] += float(losses[key].detach().item())

            if show_progress and tqdm_fn is not None:
                iterator.set_postfix(loss=float(losses["total"].detach().item()))
            elif (batch_idx + 1) % max(1, log_every) == 0 or batch_idx == 0:
                step_losses = {k: float(v.detach().item()) for k, v in losses.items()}
                print(_format_loss_line(f"[val] step={batch_idx + 1}", step_losses), flush=True)

            for i, sample in enumerate(batch):
                if sample.has_da:
                    _update_binary_iou(metric_stats["da"], preds.da[i, 0], sample.da_mask.to(device=device))

                rm_channels = [
                    ("rm_lane_marker", sample.has_rm_lane_marker),
                    ("rm_road_marker_non_lane", sample.has_rm_road_marker_non_lane),
                    ("rm_stop_line", sample.has_rm_stop_line),
                ]
                for c_idx, (name, has_flag) in enumerate(rm_channels):
                    if has_flag:
                        _update_binary_iou(metric_stats[name], preds.rm[i, c_idx], sample.rm_mask[c_idx].to(device=device))

                if sample.has_det and str(sample.det_label_scope) == "full":
                    gt = sample.det_yolo
                    if gt.numel():
                        gt_cls = gt[:, 0].to(dtype=torch.long)
                        gt_boxes_norm = _cxcywh_to_xyxy(gt[:, 1:5].to(dtype=torch.float32)).clamp(0.0, 1.0)
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
                        det_boxes, det_scores, det_cls = _decode_det_predictions(
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

    mean_losses = _mean_losses(losses_sum, steps)
    iou_metrics: Dict[str, Optional[float]] = {}
    for name, stats in metric_stats.items():
        if stats["supervised"] == 0:
            iou_metrics[name] = None
        elif stats["union"] == 0:
            iou_metrics[name] = 0.0
        else:
            iou_metrics[name] = float(stats["inter"]) / float(stats["union"])

    map50, _ap_by_class = _compute_map50(
        preds_by_class=det_preds_by_class,
        gt_by_img_class=gt_by_img_class,
        num_classes=num_det_classes,
        iou_thres=0.5,
    )
    return mean_losses, iou_metrics, map50


def main() -> int:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
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
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        collate_fn=lambda x: x,
        generator=dl_gen,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        collate_fn=lambda x: x,
        pin_memory=(device.type == "cuda"),
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
        )
        print(_format_loss_line("[train] mean", train_losses))

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
        print(_format_loss_line("[val] mean", val_losses))
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
