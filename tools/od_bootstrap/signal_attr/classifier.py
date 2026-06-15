from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.io import read_json, read_jsonl, write_json, write_jsonl_sorted
from common.pv26_schema import TL_BITS
from common.train_runtime import format_duration, join_status_segments, timing_profile

BASE_COLORS = ("off", "red", "yellow", "green")
BASE_COLOR_TO_INDEX = {name: index for index, name in enumerate(BASE_COLORS)}
INDEX_TO_BASE_COLOR = {index: name for name, index in BASE_COLOR_TO_INDEX.items()}

SIGNAL_ATTR_TEACHER_REASON_VALID = "valid"
SIGNAL_ATTR_TEACHER_REASON_LOW_CONFIDENCE = "signal_attr_teacher_low_confidence"
SIGNAL_ATTR_TEACHER_REASON_AMBIGUOUS_BITS = "signal_attr_teacher_ambiguous_bits"
SIGNAL_ATTR_TEACHER_REASON_NONFINITE_LOGITS = "signal_attr_teacher_nonfinite_logits"


@dataclass(frozen=True)
class SignalAttrClassifierConfig:
    input_size: int = 128
    width: int = 24
    dropout: float = 0.10


@dataclass(frozen=True)
class SignalAttrThresholdPolicy:
    name: str = "signal_attr_v1"
    base_color_min_confidence: float = 0.75
    arrow_threshold: float = 0.50
    arrow_ambiguity_band: float = 0.10


@dataclass(frozen=True)
class SignalAttrTrainConfig:
    epochs: int = 20
    batch_size: int = 384
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    arrow_loss_weight: float = 1.0
    device: str = "cuda:0"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    seed: int = 26
    log_every_n_steps: int = 20


@dataclass(frozen=True)
class SignalAttrPrediction:
    tl_bits: dict[str, int]
    tl_attr_valid: int
    collapse_reason: str
    base_color: str
    arrow: int
    base_color_confidence: float
    arrow_probability: float
    base_color_scores: dict[str, float]


class SignalAttrCropClassifier(nn.Module):
    """Small crop classifier for traffic-light base color plus arrow state."""

    def __init__(self, config: SignalAttrClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or SignalAttrClassifierConfig()
        width = int(self.config.width)
        if width <= 0:
            raise ValueError("classifier width must be > 0")
        dropout = float(self.config.dropout)
        self.backbone = nn.Sequential(
            _conv_block(3, width, stride=2),
            _conv_block(width, width * 2, stride=2),
            _conv_block(width * 2, width * 4, stride=2),
            _conv_block(width * 4, width * 4, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=max(0.0, min(dropout, 0.9))),
        )
        feature_dim = width * 4
        self.base_color_head = nn.Linear(feature_dim, len(BASE_COLORS))
        self.arrow_head = nn.Linear(feature_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {
            "base_color_logits": self.base_color_head(features),
            "arrow_logit": self.arrow_head(features).squeeze(-1),
        }


class SignalAttrCropTorchDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        *,
        split: str,
        input_size: int = 128,
        normalization: str = "imagenet",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.input_size = int(input_size)
        self.normalization = str(normalization)
        label_path = self.dataset_root / "labels" / f"{self.split}.jsonl"
        if not label_path.is_file():
            raise FileNotFoundError(f"signal attr labels not found: {label_path}")
        self.rows = tuple(_normalize_label_row(row, row_index=index, label_path=label_path) for index, row in enumerate(read_jsonl(label_path), start=1))
        if not self.rows:
            raise ValueError(f"signal attr split has no rows: {label_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        crop_path = self.dataset_root / row["crop_path"]
        image = load_signal_attr_crop_tensor(
            crop_path,
            input_size=self.input_size,
            normalization=self.normalization,
        )
        return {
            "image": image,
            "base_color_target": BASE_COLOR_TO_INDEX[row["base_color"]],
            "arrow_target": float(row["arrow"]),
            "row": row,
        }


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


def load_signal_attr_crop_tensor(
    image_path: Path,
    *,
    input_size: int = 128,
    normalization: str = "imagenet",
) -> torch.Tensor:
    with Image.open(image_path) as image:
        return signal_attr_crop_image_to_tensor(
            image,
            input_size=input_size,
            normalization=normalization,
        )


def signal_attr_crop_image_to_tensor(
    image: Image.Image,
    *,
    input_size: int = 128,
    normalization: str = "imagenet",
) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (int(input_size), int(input_size)):
        image = image.resize((int(input_size), int(input_size)), resample=_pil_bilinear())
    width, height = image.size
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    tensor = tensor.to(dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
    if str(normalization).strip().lower() == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


def signal_attr_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "base_color_target": torch.tensor([int(item["base_color_target"]) for item in batch], dtype=torch.long),
        "arrow_target": torch.tensor([float(item["arrow_target"]) for item in batch], dtype=torch.float32),
        "rows": [dict(item["row"]) for item in batch],
    }


def signal_attr_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    arrow_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    base_loss = nn.functional.cross_entropy(outputs["base_color_logits"], batch["base_color_target"])
    arrow_loss = nn.functional.binary_cross_entropy_with_logits(outputs["arrow_logit"], batch["arrow_target"])
    total = base_loss + float(arrow_loss_weight) * arrow_loss
    return {
        "total": total,
        "base_color": base_loss.detach(),
        "arrow": arrow_loss.detach(),
    }


@torch.no_grad()
def signal_attr_prediction_from_logits(
    base_color_logits: torch.Tensor | Sequence[float],
    arrow_logit: torch.Tensor | float,
    *,
    policy: SignalAttrThresholdPolicy = SignalAttrThresholdPolicy(),
) -> SignalAttrPrediction:
    base_logits = torch.as_tensor(base_color_logits, dtype=torch.float32).reshape(-1)
    arrow_value = torch.as_tensor(arrow_logit, dtype=torch.float32).reshape(-1)
    if base_logits.numel() != len(BASE_COLORS) or arrow_value.numel() != 1:
        raise ValueError("signal attr prediction expects 4 base-color logits and 1 arrow logit")
    if not bool(torch.isfinite(base_logits).all()) or not bool(torch.isfinite(arrow_value).all()):
        return _invalid_prediction(SIGNAL_ATTR_TEACHER_REASON_NONFINITE_LOGITS)

    base_scores_tensor = torch.softmax(base_logits, dim=0)
    base_confidence, base_index = torch.max(base_scores_tensor, dim=0)
    base_color = INDEX_TO_BASE_COLOR[int(base_index.item())]
    base_scores = {
        color: float(base_scores_tensor[index].item())
        for index, color in enumerate(BASE_COLORS)
    }
    arrow_probability = float(torch.sigmoid(arrow_value[0]).item())
    base_confidence_value = float(base_confidence.item())

    if base_confidence_value < float(policy.base_color_min_confidence):
        return _invalid_prediction(
            SIGNAL_ATTR_TEACHER_REASON_LOW_CONFIDENCE,
            base_color=base_color,
            base_color_confidence=base_confidence_value,
            arrow_probability=arrow_probability,
            base_color_scores=base_scores,
        )

    half_band = max(0.0, float(policy.arrow_ambiguity_band)) / 2.0
    threshold = float(policy.arrow_threshold)
    if half_band > 0.0 and (threshold - half_band) <= arrow_probability <= (threshold + half_band):
        return _invalid_prediction(
            SIGNAL_ATTR_TEACHER_REASON_AMBIGUOUS_BITS,
            base_color=base_color,
            base_color_confidence=base_confidence_value,
            arrow_probability=arrow_probability,
            base_color_scores=base_scores,
        )

    arrow = int(arrow_probability >= threshold)
    bits = {bit: 0 for bit in TL_BITS}
    if base_color in {"red", "yellow", "green"}:
        bits[base_color] = 1
    bits["arrow"] = arrow
    return SignalAttrPrediction(
        tl_bits=bits,
        tl_attr_valid=1,
        collapse_reason=SIGNAL_ATTR_TEACHER_REASON_VALID,
        base_color=base_color,
        arrow=arrow,
        base_color_confidence=base_confidence_value,
        arrow_probability=arrow_probability,
        base_color_scores=base_scores,
    )


def train_signal_attr_classifier(
    dataset_root: Path,
    output_root: Path,
    *,
    train_config: SignalAttrTrainConfig = SignalAttrTrainConfig(),
    model_config: SignalAttrClassifierConfig = SignalAttrClassifierConfig(),
    threshold_policy: SignalAttrThresholdPolicy = SignalAttrThresholdPolicy(),
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _validate_train_config(train_config)
    torch.manual_seed(int(train_config.seed))
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    crop_config = _load_crop_config(Path(dataset_root), fallback_input_size=model_config.input_size)
    normalization = str(crop_config.get("normalization", "imagenet"))
    input_size = int(crop_config.get("input_size", model_config.input_size))
    train_loader = _build_loader(
        dataset_root,
        split="train",
        train_config=train_config,
        input_size=input_size,
        normalization=normalization,
        shuffle=True,
    )
    val_loader = _build_loader(
        dataset_root,
        split="val",
        train_config=train_config,
        input_size=input_size,
        normalization=normalization,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    device = torch.device(train_config.device if torch.cuda.is_available() or str(train_config.device) == "cpu" else "cpu")
    model = SignalAttrCropClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )

    best_combo_accuracy = -1.0
    best_checkpoint_path = output / "best_signal_attr.pt"
    last_checkpoint_path = output / "last_signal_attr.pt"
    history: list[dict[str, Any]] = []
    if log_fn is not None:
        log_fn(
            f"[teacher:signal_attr] train start dataset={Path(dataset_root).resolve()} "
            f"output={output.resolve()} train_samples={len(train_loader.dataset)} "
            f"val_samples={len(val_loader.dataset)} epochs={train_config.epochs} "
            f"batch={train_config.batch_size} device={device} workers={train_config.num_workers} "
            f"pin_memory={bool(train_config.pin_memory) and _pin_memory_enabled(train_config.device)} "
            f"persistent_workers={train_config.persistent_workers} prefetch_factor={train_config.prefetch_factor} "
            f"val_workers={getattr(val_loader, 'num_workers', 0)}"
        )
    train_started_at = time.monotonic()
    for epoch in range(1, int(train_config.epochs) + 1):
        if log_fn is not None:
            log_fn(f"[teacher:signal_attr] epoch {epoch}/{train_config.epochs} train start")
        train_summary = train_signal_attr_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            arrow_loss_weight=float(train_config.arrow_loss_weight),
            epoch=epoch,
            epoch_total=int(train_config.epochs),
            log_every_n_steps=int(train_config.log_every_n_steps),
            log_fn=log_fn,
        )
        val_summary = evaluate_signal_attr_classifier(model, val_loader, device=device)
        epoch_summary = {"epoch": epoch, "train": train_summary, "val": val_summary}
        history.append(epoch_summary)
        checkpoint_payload = _checkpoint_payload(
            model=model,
            model_config=model_config,
            train_config=train_config,
            threshold_policy=threshold_policy,
            crop_config=crop_config,
            epoch=epoch,
            history=history,
        )
        torch.save(checkpoint_payload, last_checkpoint_path)
        combo_accuracy = float(val_summary["combo_accuracy"])
        best_updated = combo_accuracy >= best_combo_accuracy
        if combo_accuracy >= best_combo_accuracy:
            best_combo_accuracy = combo_accuracy
            torch.save(checkpoint_payload, best_checkpoint_path)
        summary = _train_summary_payload(
            dataset_root=dataset_root,
            output=output,
            model_config=model_config,
            train_config=train_config,
            threshold_policy=threshold_policy,
            crop_config=crop_config,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            best_combo_accuracy=best_combo_accuracy,
            history=history,
        )
        write_json(output / "train_summary.json", summary)
        if log_fn is not None:
            checkpoint_state = "best+last" if best_updated else "last"
            log_fn(
                f"[teacher:signal_attr] epoch {epoch}/{train_config.epochs} done "
                f"train_loss={float(train_summary['loss']):.6f} "
                f"val_combo={float(val_summary['combo_accuracy']):.4f} "
                f"val_base={float(val_summary['base_color_accuracy']):.4f} "
                f"val_arrow={float(val_summary['arrow_accuracy']):.4f} "
                f"checkpoint={checkpoint_state}"
            )

    summary = _train_summary_payload(
        dataset_root=dataset_root,
        output=output,
        model_config=model_config,
        train_config=train_config,
        threshold_policy=threshold_policy,
        crop_config=crop_config,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        best_combo_accuracy=best_combo_accuracy,
        history=history,
    )
    write_json(output / "train_summary.json", summary)
    if log_fn is not None:
        elapsed = max(time.monotonic() - train_started_at, 1.0e-6)
        log_fn(
            f"[teacher:signal_attr] train done epochs={train_config.epochs} "
            f"best_combo={best_combo_accuracy:.4f} elapsed={elapsed:.1f}s "
            f"checkpoint={best_checkpoint_path}"
        )
    return summary


def load_signal_attr_classifier_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    resolved_device = _resolve_device(str(device))
    payload = torch.load(Path(checkpoint_path), map_location=resolved_device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"signal attr checkpoint must be a mapping: {checkpoint_path}")
    if str(payload.get("model_type") or "") != "SignalAttrCropClassifier":
        raise ValueError(f"unsupported signal attr checkpoint model_type: {payload.get('model_type')!r}")
    model_config = SignalAttrClassifierConfig(**dict(payload.get("model_config") or {}))
    threshold_policy = SignalAttrThresholdPolicy(**dict(payload.get("threshold_policy") or {}))
    crop_config = dict(payload.get("crop_config") or {"input_size": model_config.input_size, "normalization": "imagenet"})
    model = SignalAttrCropClassifier(model_config).to(resolved_device)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"signal attr checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "model_config": model_config,
        "threshold_policy": threshold_policy,
        "crop_config": crop_config,
        "payload": dict(payload),
        "device": resolved_device,
    }


@torch.no_grad()
def predict_signal_attr_crop_image(
    model: nn.Module,
    image: Image.Image,
    *,
    crop_config: Mapping[str, Any],
    threshold_policy: SignalAttrThresholdPolicy,
    device: torch.device,
) -> SignalAttrPrediction:
    input_size = int(crop_config.get("input_size", 128))
    normalization = str(crop_config.get("normalization", "imagenet"))
    tensor = signal_attr_crop_image_to_tensor(
        image,
        input_size=input_size,
        normalization=normalization,
    ).unsqueeze(0).to(device=device)
    outputs = model(tensor)
    return signal_attr_prediction_from_logits(
        outputs["base_color_logits"][0].detach().cpu(),
        outputs["arrow_logit"][0].detach().cpu(),
        policy=threshold_policy,
    )


def evaluate_signal_attr_checkpoint(
    dataset_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    *,
    split: str = "val",
    batch_size: int = SignalAttrTrainConfig.batch_size,
    device: str = "cuda:0",
    num_workers: int = SignalAttrTrainConfig.num_workers,
    threshold_policy: SignalAttrThresholdPolicy | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    loaded = load_signal_attr_classifier_checkpoint(checkpoint_path, device=device)
    model = loaded["model"]
    resolved_device = loaded["device"]
    crop_config = dict(loaded["crop_config"])
    policy = threshold_policy or loaded["threshold_policy"]
    input_size = int(crop_config.get("input_size", loaded["model_config"].input_size))
    normalization = str(crop_config.get("normalization", "imagenet"))
    dataset = SignalAttrCropTorchDataset(
        Path(dataset_root),
        split=split,
        input_size=input_size,
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        collate_fn=signal_attr_collate,
        **_loader_throughput_kwargs(
            num_workers=max(0, int(num_workers)),
            pin_memory=resolved_device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=SignalAttrTrainConfig.prefetch_factor,
        ),
    )

    base_correct = 0
    arrow_correct = 0
    combo_correct = 0
    sample_count = 0
    bit_stats = {bit: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for bit in TL_BITS}
    target_combo_counts: dict[str, int] = {}
    predicted_combo_counts: dict[str, int] = {}
    reject_reason_counts: dict[str, int] = {}
    prediction_rows: list[dict[str, Any]] = []
    model.eval()
    if log_fn is not None:
        log_fn(
            f"[teacher:signal_attr] eval start dataset={Path(dataset_root).resolve()} "
            f"checkpoint={Path(checkpoint_path).resolve()} split={split} "
            f"samples={len(dataset)} batch={batch_size} device={resolved_device} workers={num_workers}"
        )
    eval_started_at = time.monotonic()
    total_batches = len(loader)
    completed_batches = 0
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, resolved_device)
            outputs = model(moved["image"])
            base_pred = torch.argmax(outputs["base_color_logits"], dim=1).detach().cpu()
            arrow_prob = torch.sigmoid(outputs["arrow_logit"]).detach().cpu()
            arrow_pred = (arrow_prob >= float(policy.arrow_threshold)).to(dtype=torch.long)
            base_target = moved["base_color_target"].detach().cpu()
            arrow_target = moved["arrow_target"].detach().cpu().to(dtype=torch.long)
            for row_index, row in enumerate(batch["rows"]):
                prediction = signal_attr_prediction_from_logits(
                    outputs["base_color_logits"][row_index].detach().cpu(),
                    outputs["arrow_logit"][row_index].detach().cpu(),
                    policy=policy,
                )
                target_bits = _row_tl_bits(row)
                pred_bits = dict(prediction.tl_bits)
                target_combo = _combo_from_bits(target_bits)
                predicted_combo = _combo_from_bits(pred_bits) if prediction.tl_attr_valid else prediction.collapse_reason
                target_combo_counts[target_combo] = target_combo_counts.get(target_combo, 0) + 1
                predicted_combo_counts[predicted_combo] = predicted_combo_counts.get(predicted_combo, 0) + 1
                reject_reason_counts[prediction.collapse_reason] = reject_reason_counts.get(prediction.collapse_reason, 0) + 1

                base_hit = int(base_pred[row_index].item()) == int(base_target[row_index].item())
                arrow_hit = int(arrow_pred[row_index].item()) == int(arrow_target[row_index].item())
                base_correct += int(base_hit)
                arrow_correct += int(arrow_hit)
                combo_correct += int(prediction.tl_attr_valid and pred_bits == target_bits)
                sample_count += 1
                for bit in TL_BITS:
                    truth = int(target_bits.get(bit, 0))
                    pred = int(pred_bits.get(bit, 0)) if prediction.tl_attr_valid else 0
                    if truth and pred:
                        bit_stats[bit]["tp"] += 1
                    elif not truth and pred:
                        bit_stats[bit]["fp"] += 1
                    elif truth and not pred:
                        bit_stats[bit]["fn"] += 1
                    else:
                        bit_stats[bit]["tn"] += 1
                prediction_rows.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "split": row.get("split"),
                        "crop_path": row.get("crop_path"),
                        "target_bits": target_bits,
                        "predicted_bits": pred_bits,
                        "tl_attr_valid": int(prediction.tl_attr_valid),
                        "collapse_reason": prediction.collapse_reason,
                        "base_color": prediction.base_color,
                        "base_color_confidence": prediction.base_color_confidence,
                        "arrow_probability": prediction.arrow_probability,
                        "target_base_color": row.get("base_color"),
                        "target_arrow": int(row.get("arrow", 0)),
                    }
                )
            completed_batches += 1
            _log_progress(
                log_fn,
                prefix="[teacher:signal_attr] eval",
                completed=completed_batches,
                total=total_batches,
                sample_count=sample_count,
                started_at=eval_started_at,
                log_every=20,
            )

    report = {
        "version": "signal-attr-classifier-eval-v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "split": str(split),
        "sample_count": sample_count,
        "threshold_policy": asdict(policy),
        "crop_config": crop_config,
        "base_color_accuracy": base_correct / max(sample_count, 1),
        "arrow_accuracy": arrow_correct / max(sample_count, 1),
        "combo_accuracy": combo_correct / max(sample_count, 1),
        "bit_metrics": {bit: _binary_metrics(stats) for bit, stats in bit_stats.items()},
        "target_combo_counts": dict(sorted(target_combo_counts.items())),
        "predicted_combo_counts": dict(sorted(predicted_combo_counts.items())),
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "prediction_count": len(prediction_rows),
        "valid_prediction_count": sum(1 for row in prediction_rows if int(row["tl_attr_valid"])),
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "signal_attr_eval_report.json", report)
    write_jsonl_sorted(output / "signal_attr_predictions.jsonl", prediction_rows)
    if log_fn is not None:
        elapsed = max(time.monotonic() - eval_started_at, 1.0e-6)
        log_fn(
            f"[teacher:signal_attr] eval done samples={sample_count} "
            f"valid={report['valid_prediction_count']} combo={report['combo_accuracy']:.4f} "
            f"elapsed={elapsed:.1f}s"
        )
    return report


def train_signal_attr_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    arrow_loss_weight: float,
    epoch: int | None = None,
    epoch_total: int | None = None,
    log_every_n_steps: int = 20,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    sample_count = 0
    started_at = time.perf_counter()
    last_batch_end_at = started_at
    total_batches = len(loader)
    profile_window: list[dict[str, float]] = []
    profile_window_size = max(1, int(log_every_n_steps))
    for batch_index, batch in enumerate(loader, start=1):
        batch_started_at = time.perf_counter()
        wait_sec = max(0.0, batch_started_at - last_batch_end_at)
        stage_started_at = time.perf_counter()
        moved = _move_batch(batch, device)
        preprocess_sec = max(0.0, time.perf_counter() - stage_started_at)
        stage_started_at = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(moved["image"])
        forward_sec = max(0.0, time.perf_counter() - stage_started_at)
        stage_started_at = time.perf_counter()
        losses = signal_attr_loss(outputs, moved, arrow_loss_weight=arrow_loss_weight)
        loss_sec = max(0.0, time.perf_counter() - stage_started_at)
        stage_started_at = time.perf_counter()
        losses["total"].backward()
        backward_sec = max(0.0, time.perf_counter() - stage_started_at)
        stage_started_at = time.perf_counter()
        optimizer.step()
        optimizer_sec = max(0.0, time.perf_counter() - stage_started_at)
        batch_finished_at = time.perf_counter()
        last_batch_end_at = batch_finished_at
        batch_size = int(moved["image"].shape[0])
        total_loss += float(losses["total"].detach().cpu()) * batch_size
        sample_count += batch_size
        profile_window.append(
            {
                "iteration_sec": max(0.0, batch_finished_at - batch_started_at),
                "wait_sec": wait_sec,
                "compute_sec": max(0.0, batch_finished_at - batch_started_at),
                "preprocess_sec": preprocess_sec,
                "forward_sec": forward_sec,
                "loss_sec": loss_sec,
                "backward_sec": backward_sec,
                "optimizer_sec": optimizer_sec,
            }
        )
        if len(profile_window) > profile_window_size:
            profile_window.pop(0)
        _log_progress(
            log_fn,
            prefix=_train_progress_prefix(epoch=epoch, epoch_total=epoch_total),
            completed=batch_index,
            total=total_batches,
            sample_count=sample_count,
            started_at=started_at,
            log_every=max(1, int(log_every_n_steps)),
            profile_summary=_signal_attr_timing_profile(profile_window),
        )
    return {"loss": total_loss / max(sample_count, 1), "sample_count": sample_count}


@torch.no_grad()
def evaluate_signal_attr_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    sample_count = 0
    base_correct = 0
    arrow_correct = 0
    combo_correct = 0
    for batch in loader:
        moved = _move_batch(batch, device)
        outputs = model(moved["image"])
        base_pred = torch.argmax(outputs["base_color_logits"], dim=1)
        arrow_pred = (torch.sigmoid(outputs["arrow_logit"]) >= 0.5).to(dtype=torch.long)
        base_target = moved["base_color_target"]
        arrow_target = moved["arrow_target"].to(dtype=torch.long)
        batch_size = int(base_target.numel())
        sample_count += batch_size
        base_hits = base_pred == base_target
        arrow_hits = arrow_pred == arrow_target
        base_correct += int(base_hits.sum().item())
        arrow_correct += int(arrow_hits.sum().item())
        combo_correct += int((base_hits & arrow_hits).sum().item())
    return {
        "sample_count": sample_count,
        "base_color_accuracy": base_correct / max(sample_count, 1),
        "arrow_accuracy": arrow_correct / max(sample_count, 1),
        "combo_accuracy": combo_correct / max(sample_count, 1),
    }


def _normalize_label_row(row: Mapping[str, Any], *, row_index: int, label_path: Path) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"signal attr label row must be an object: {label_path}:{row_index}")
    crop_path = str(row.get("crop_path") or "").strip()
    if not crop_path:
        raise ValueError(f"signal attr label row missing crop_path: {label_path}:{row_index}")
    base_color = str(row.get("base_color") or "").strip().lower()
    if base_color not in BASE_COLOR_TO_INDEX:
        raise ValueError(f"unsupported signal attr base_color={base_color!r}: {label_path}:{row_index}")
    try:
        arrow = int(row.get("arrow"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"signal attr arrow must be 0/1: {label_path}:{row_index}") from exc
    if arrow not in (0, 1):
        raise ValueError(f"signal attr arrow must be 0/1: {label_path}:{row_index}")
    return {**dict(row), "crop_path": crop_path, "base_color": base_color, "arrow": arrow}


def _invalid_prediction(
    reason: str,
    *,
    base_color: str = "off",
    base_color_confidence: float = 0.0,
    arrow_probability: float = 0.0,
    base_color_scores: Mapping[str, float] | None = None,
) -> SignalAttrPrediction:
    return SignalAttrPrediction(
        tl_bits={bit: 0 for bit in TL_BITS},
        tl_attr_valid=0,
        collapse_reason=reason,
        base_color=base_color,
        arrow=0,
        base_color_confidence=float(base_color_confidence) if math.isfinite(float(base_color_confidence)) else 0.0,
        arrow_probability=float(arrow_probability) if math.isfinite(float(arrow_probability)) else 0.0,
        base_color_scores=dict(base_color_scores or {color: 0.0 for color in BASE_COLORS}),
    )


def _pil_bilinear() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return int(resampling.BILINEAR)


def _build_loader(
    dataset_root: Path,
    *,
    split: str,
    train_config: SignalAttrTrainConfig,
    input_size: int,
    normalization: str,
    shuffle: bool,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
) -> DataLoader:
    dataset = SignalAttrCropTorchDataset(
        Path(dataset_root),
        split=split,
        input_size=input_size,
        normalization=normalization,
    )
    return DataLoader(
        dataset,
        batch_size=int(train_config.batch_size),
        shuffle=shuffle,
        num_workers=int(train_config.num_workers if num_workers is None else num_workers),
        collate_fn=signal_attr_collate,
        **_loader_throughput_kwargs(
            num_workers=int(train_config.num_workers if num_workers is None else num_workers),
            pin_memory=(
                bool(train_config.pin_memory if pin_memory is None else pin_memory)
                and _pin_memory_enabled(train_config.device)
            ),
            persistent_workers=bool(
                train_config.persistent_workers if persistent_workers is None else persistent_workers
            ),
            prefetch_factor=int(train_config.prefetch_factor),
        ),
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "image": batch["image"].to(device=device),
        "base_color_target": batch["base_color_target"].to(device=device),
        "arrow_target": batch["arrow_target"].to(device=device),
        "rows": batch.get("rows", []),
    }


def _load_crop_config(dataset_root: Path, *, fallback_input_size: int) -> dict[str, Any]:
    config_path = dataset_root / "meta" / "crop_config.json"
    if not config_path.is_file():
        return {"input_size": int(fallback_input_size), "normalization": "imagenet"}
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise TypeError(f"signal attr crop config must be an object: {config_path}")
    return dict(payload)


def _validate_train_config(config: SignalAttrTrainConfig) -> None:
    if int(config.epochs) <= 0:
        raise ValueError("signal attr epochs must be > 0")
    if int(config.batch_size) <= 0:
        raise ValueError("signal attr batch_size must be > 0")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("signal attr learning_rate must be > 0")
    if int(config.num_workers) < 0:
        raise ValueError("signal attr num_workers must be >= 0")
    if int(config.prefetch_factor) <= 0:
        raise ValueError("signal attr prefetch_factor must be > 0")
    if int(config.log_every_n_steps) <= 0:
        raise ValueError("signal attr log_every_n_steps must be > 0")


def _loader_throughput_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"pin_memory": bool(pin_memory)}
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _pin_memory_enabled(device: str | torch.device) -> bool:
    return str(device) != "cpu" and torch.cuda.is_available()


def _resolve_device(device: str) -> torch.device:
    requested = str(device)
    if requested != "cpu" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _train_progress_prefix(*, epoch: int | None, epoch_total: int | None) -> str:
    if epoch is None or epoch_total is None:
        return "[teacher:signal_attr] train"
    return f"[teacher:signal_attr] epoch {epoch}/{epoch_total} train"


def _log_progress(
    log_fn: Callable[[str], None] | None,
    *,
    prefix: str,
    completed: int,
    total: int,
    sample_count: int,
    started_at: float,
    log_every: int,
    profile_summary: dict[str, Any] | None = None,
) -> None:
    if log_fn is None:
        return
    if completed != 1 and completed != total and completed % max(1, int(log_every)) != 0:
        return
    elapsed = max(time.perf_counter() - started_at, 1.0e-6)
    rate = sample_count / elapsed
    profile_text = _signal_attr_profile_text(elapsed_sec=elapsed, completed=completed, total=total, profile_summary=profile_summary)
    log_fn(
        f"{prefix} progress {completed}/{total} batches "
        f"({rate:.1f} samples/s, samples={sample_count})"
        f"{profile_text}"
    )


def _signal_attr_timing_profile(window: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    return timing_profile(
        window,
        keys=(
            "iteration_sec",
            "wait_sec",
            "compute_sec",
            "preprocess_sec",
            "forward_sec",
            "loss_sec",
            "backward_sec",
            "optimizer_sec",
        ),
    )


def _signal_attr_profile_text(
    *,
    elapsed_sec: float,
    completed: int,
    total: int,
    profile_summary: dict[str, Any] | None,
) -> str:
    if not profile_summary or "iteration_sec" not in profile_summary:
        return ""
    iteration_mean = _profile_mean(profile_summary, "iteration_sec")
    remaining = max(0, int(total) - int(completed))
    eta_sec = iteration_mean * float(remaining) if remaining else 0.0
    summary = join_status_segments(
        f"elapsed={format_duration(elapsed_sec)}",
        f"eta={format_duration(eta_sec)}",
        f"iter={iteration_mean * 1000.0:.1f}ms",
        f"wait={_profile_mean(profile_summary, 'wait_sec') * 1000.0:.1f}ms",
        f"compute={_profile_mean(profile_summary, 'compute_sec') * 1000.0:.1f}ms",
    )
    stages = join_status_segments(
        f"prep={_profile_mean(profile_summary, 'preprocess_sec') * 1000.0:.1f}ms",
        f"fwd={_profile_mean(profile_summary, 'forward_sec') * 1000.0:.1f}ms",
        f"loss={_profile_mean(profile_summary, 'loss_sec') * 1000.0:.1f}ms",
        f"bwd={_profile_mean(profile_summary, 'backward_sec') * 1000.0:.1f}ms",
        f"opt={_profile_mean(profile_summary, 'optimizer_sec') * 1000.0:.1f}ms",
    )
    return "\n" + "\n".join(segment for segment in (summary, stages) if segment)


def _profile_mean(profile_summary: Mapping[str, Any], key: str) -> float:
    group = profile_summary.get(key)
    if not isinstance(group, Mapping):
        return 0.0
    try:
        return float(group.get("mean", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _row_tl_bits(row: Mapping[str, Any]) -> dict[str, int]:
    raw_bits = row.get("tl_bits")
    if not isinstance(raw_bits, Mapping):
        raw_bits = {}
    return {bit: int(raw_bits.get(bit, 0) or 0) for bit in TL_BITS}


def _combo_from_bits(bits: Mapping[str, int]) -> str:
    active = [bit for bit in TL_BITS if int(bits.get(bit, 0))]
    return "+".join(active) if active else "off"


def _binary_metrics(stats: Mapping[str, int]) -> dict[str, float | int]:
    tp = int(stats.get("tp", 0))
    fp = int(stats.get("fp", 0))
    fn = int(stats.get("fn", 0))
    tn = int(stats.get("tn", 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1.0e-12)
    support = tp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _train_summary_payload(
    *,
    dataset_root: Path,
    output: Path,
    model_config: SignalAttrClassifierConfig,
    train_config: SignalAttrTrainConfig,
    threshold_policy: SignalAttrThresholdPolicy,
    crop_config: Mapping[str, Any],
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    best_combo_accuracy: float,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "signal-attr-classifier-train-v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_root": str(output.resolve()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "threshold_policy": asdict(threshold_policy),
        "crop_config": dict(crop_config),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_combo_accuracy": best_combo_accuracy,
        "history": list(history),
    }


def _checkpoint_payload(
    *,
    model: nn.Module,
    model_config: SignalAttrClassifierConfig,
    train_config: SignalAttrTrainConfig,
    threshold_policy: SignalAttrThresholdPolicy,
    crop_config: Mapping[str, Any],
    epoch: int,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "model_type": "SignalAttrCropClassifier",
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "threshold_policy": asdict(threshold_policy),
        "crop_config": dict(crop_config),
        "base_colors": list(BASE_COLORS),
        "tl_bits": list(TL_BITS),
        "epoch": int(epoch),
        "history": list(history),
    }


__all__ = [
    "BASE_COLORS",
    "BASE_COLOR_TO_INDEX",
    "INDEX_TO_BASE_COLOR",
    "SIGNAL_ATTR_TEACHER_REASON_AMBIGUOUS_BITS",
    "SIGNAL_ATTR_TEACHER_REASON_LOW_CONFIDENCE",
    "SIGNAL_ATTR_TEACHER_REASON_NONFINITE_LOGITS",
    "SIGNAL_ATTR_TEACHER_REASON_VALID",
    "SignalAttrClassifierConfig",
    "SignalAttrCropClassifier",
    "SignalAttrCropTorchDataset",
    "SignalAttrPrediction",
    "SignalAttrThresholdPolicy",
    "SignalAttrTrainConfig",
    "evaluate_signal_attr_checkpoint",
    "evaluate_signal_attr_classifier",
    "load_signal_attr_classifier_checkpoint",
    "load_signal_attr_crop_tensor",
    "predict_signal_attr_crop_image",
    "signal_attr_collate",
    "signal_attr_crop_image_to_tensor",
    "signal_attr_loss",
    "signal_attr_prediction_from_logits",
    "train_signal_attr_classifier",
    "train_signal_attr_epoch",
]
