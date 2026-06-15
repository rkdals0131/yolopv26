from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.io import read_json, read_jsonl, write_json
from common.pv26_schema import TL_BITS

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
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    arrow_loss_weight: float = 1.0
    device: str = "cuda:0"
    num_workers: int = 4
    seed: int = 26


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
    for epoch in range(1, int(train_config.epochs) + 1):
        train_summary = train_signal_attr_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            arrow_loss_weight=float(train_config.arrow_loss_weight),
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
        if combo_accuracy >= best_combo_accuracy:
            best_combo_accuracy = combo_accuracy
            torch.save(checkpoint_payload, best_checkpoint_path)

    summary = {
        "version": "signal-attr-classifier-train-v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_root": str(output.resolve()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "threshold_policy": asdict(threshold_policy),
        "crop_config": crop_config,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_combo_accuracy": best_combo_accuracy,
        "history": history,
    }
    write_json(output / "train_summary.json", summary)
    return summary


def train_signal_attr_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    arrow_loss_weight: float,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        moved = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(moved["image"])
        losses = signal_attr_loss(outputs, moved, arrow_loss_weight=arrow_loss_weight)
        losses["total"].backward()
        optimizer.step()
        batch_size = int(moved["image"].shape[0])
        total_loss += float(losses["total"].detach().cpu()) * batch_size
        sample_count += batch_size
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
        num_workers=int(train_config.num_workers),
        collate_fn=signal_attr_collate,
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
    "evaluate_signal_attr_classifier",
    "load_signal_attr_crop_tensor",
    "signal_attr_collate",
    "signal_attr_loss",
    "signal_attr_prediction_from_logits",
    "train_signal_attr_classifier",
    "train_signal_attr_epoch",
]
