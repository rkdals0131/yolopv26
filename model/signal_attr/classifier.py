"""Internal crop classifier, checkpoint reader, and product-state evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

from common.io import read_jsonl
from .aihub_policy import TL_BITS


BASE_COLORS = ("off", "red", "yellow", "green")
BASE_COLOR_TO_INDEX = {color: index for index, color in enumerate(BASE_COLORS)}
INDEX_TO_BASE_COLOR = {index: color for color, index in BASE_COLOR_TO_INDEX.items()}


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
class SignalAttrPrediction:
    tl_bits: dict[str, int]
    tl_attr_valid: int
    collapse_reason: str
    base_color: str
    arrow: int
    base_color_confidence: float
    arrow_probability: float
    base_color_scores: dict[str, float]


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class SignalAttrCropClassifier(nn.Module):
    """The same parameter layout as the included SignalAttr baseline checkpoint."""

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
        self.base_color_head = nn.Linear(width * 4, len(BASE_COLORS))
        self.arrow_head = nn.Linear(width * 4, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {
            "base_color_logits": self.base_color_head(features),
            "arrow_logit": self.arrow_head(features).squeeze(-1),
        }


def signal_attr_crop_image_to_tensor(
    image: Image.Image, *, input_size: int = 128, normalization: str = "imagenet"
) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (int(input_size), int(input_size)):
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((int(input_size), int(input_size)), resample=resampling.BILINEAR)
    width, height = image.size
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    tensor = tensor.to(dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    if str(normalization).strip().lower() == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


class SignalAttrCropTorchDataset(Dataset):
    def __init__(
        self, dataset_root: Path, *, split: str, input_size: int = 128,
        normalization: str = "imagenet", labels_root: Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.input_size = int(input_size)
        self.normalization = str(normalization)
        labels = Path(labels_root) if labels_root is not None else self.dataset_root / "labels"
        label_path = labels / f"{self.split}.jsonl"
        if not label_path.is_file():
            raise FileNotFoundError(f"signal attr labels not found: {label_path}")
        self.rows = tuple(self._decode_row(row, label_path) for row in read_jsonl(label_path))
        if not self.rows:
            raise ValueError(f"signal attr split has no rows: {label_path}")

    @staticmethod
    def _decode_row(row: Mapping[str, Any], label_path: Path) -> dict[str, Any]:
        crop_path = str(row.get("crop_path") or "")
        base_color = str(row.get("base_color") or "").lower()
        arrow = int(row.get("arrow", -1))
        if not crop_path or base_color not in BASE_COLOR_TO_INDEX or arrow not in (0, 1):
            raise ValueError(f"invalid signal attr crop target: {label_path}")
        return {**row, "crop_path": crop_path, "base_color": base_color, "arrow": arrow}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(self.dataset_root / row["crop_path"]) as image:
            tensor = signal_attr_crop_image_to_tensor(
                image, input_size=self.input_size, normalization=self.normalization
            )
        return {
            "image": tensor,
            "base_color_target": BASE_COLOR_TO_INDEX[row["base_color"]],
            "arrow_target": float(row["arrow"]),
            "arrow_target_valid": float(row.get("light_type") != "pedestrian"),
            "row": row,
        }


def signal_attr_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "base_color_target": torch.tensor([int(item["base_color_target"]) for item in batch], dtype=torch.long),
        "arrow_target": torch.tensor([float(item["arrow_target"]) for item in batch]),
        "arrow_target_valid": torch.tensor([float(item.get("arrow_target_valid", 1)) for item in batch]),
        "rows": [dict(item["row"]) for item in batch],
    }


def _invalid_prediction(
    reason: str, *, base_color: str = "off", base_color_confidence: float = 0.0,
    arrow_probability: float = 0.0, base_color_scores: Mapping[str, float] | None = None,
) -> SignalAttrPrediction:
    return SignalAttrPrediction(
        tl_bits={bit: 0 for bit in TL_BITS}, tl_attr_valid=0, collapse_reason=reason,
        base_color=base_color, arrow=0,
        base_color_confidence=base_color_confidence if math.isfinite(base_color_confidence) else 0.0,
        arrow_probability=arrow_probability if math.isfinite(arrow_probability) else 0.0,
        base_color_scores=dict(base_color_scores or {color: 0.0 for color in BASE_COLORS}),
    )


@torch.no_grad()
def signal_attr_prediction_from_logits(
    base_color_logits: torch.Tensor | Sequence[float], arrow_logit: torch.Tensor | float,
    *, policy: SignalAttrThresholdPolicy = SignalAttrThresholdPolicy(),
) -> SignalAttrPrediction:
    base_logits = torch.as_tensor(base_color_logits, dtype=torch.float32).reshape(-1)
    arrow_value = torch.as_tensor(arrow_logit, dtype=torch.float32).reshape(-1)
    if base_logits.numel() != len(BASE_COLORS) or arrow_value.numel() != 1:
        raise ValueError("signal attr prediction expects 4 base-color logits and 1 arrow logit")
    if not bool(torch.isfinite(base_logits).all()) or not bool(torch.isfinite(arrow_value).all()):
        return _invalid_prediction("signal_attr_nonfinite_logits")
    scores = torch.softmax(base_logits, dim=0)
    confidence, index = scores.max(dim=0)
    base_color = INDEX_TO_BASE_COLOR[int(index.item())]
    base_scores = {color: float(scores[i].item()) for i, color in enumerate(BASE_COLORS)}
    arrow_probability = float(torch.sigmoid(arrow_value[0]).item())
    confidence_value = float(confidence.item())
    if confidence_value < policy.base_color_min_confidence:
        return _invalid_prediction(
            "signal_attr_low_confidence", base_color=base_color,
            base_color_confidence=confidence_value, arrow_probability=arrow_probability,
            base_color_scores=base_scores,
        )
    half_band = max(0.0, policy.arrow_ambiguity_band) / 2.0
    if half_band and policy.arrow_threshold - half_band <= arrow_probability <= policy.arrow_threshold + half_band:
        return _invalid_prediction(
            "signal_attr_ambiguous_bits", base_color=base_color,
            base_color_confidence=confidence_value, arrow_probability=arrow_probability,
            base_color_scores=base_scores,
        )
    arrow = int(arrow_probability >= policy.arrow_threshold)
    bits = {bit: int(bit == base_color) for bit in TL_BITS}
    bits["arrow"] = arrow
    return SignalAttrPrediction(
        bits, 1, "valid", base_color, arrow, confidence_value, arrow_probability, base_scores
    )


@torch.no_grad()
def product_signal_attr_prediction_from_logits(
    base_color_logits: torch.Tensor | Sequence[float], arrow_logit: torch.Tensor | float,
    *, light_type: str, all_off_is_valid: bool,
    policy: SignalAttrThresholdPolicy = SignalAttrThresholdPolicy(),
) -> SignalAttrPrediction:
    if light_type == "pedestrian":
        prediction = signal_attr_prediction_from_logits(
            base_color_logits, arrow_logit, policy=replace(policy, arrow_ambiguity_band=0.0)
        )
        if not prediction.tl_attr_valid:
            return prediction
        if prediction.base_color == "off" and not all_off_is_valid:
            return _invalid_prediction(
                "all_off_unverified", base_color=prediction.base_color,
                base_color_confidence=prediction.base_color_confidence,
                arrow_probability=prediction.arrow_probability,
                base_color_scores=prediction.base_color_scores,
            )
        if prediction.base_color not in ("off", "red", "green"):
            return _invalid_prediction(
                "pedestrian_unsupported_state", base_color=prediction.base_color,
                base_color_confidence=prediction.base_color_confidence,
                arrow_probability=prediction.arrow_probability,
                base_color_scores=prediction.base_color_scores,
            )
        return replace(prediction, tl_bits={bit: int(bit == prediction.base_color) for bit in TL_BITS}, arrow=0)
    if light_type != "car":
        raise ValueError(f"unsupported product signal light_type: {light_type}")
    prediction = signal_attr_prediction_from_logits(base_color_logits, arrow_logit, policy=policy)
    if prediction.tl_attr_valid and prediction.base_color == "off" and not prediction.arrow and not all_off_is_valid:
        return _invalid_prediction(
            "all_off_unverified", base_color=prediction.base_color,
            base_color_confidence=prediction.base_color_confidence,
            arrow_probability=prediction.arrow_probability,
            base_color_scores=prediction.base_color_scores,
        )
    return prediction


def load_signal_attr_classifier_checkpoint(
    checkpoint_path: Path, *, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    payload = torch.load(Path(checkpoint_path), map_location=resolved_device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("model_type") != "SignalAttrCropClassifier":
        raise ValueError(f"unsupported SignalAttr checkpoint: {checkpoint_path}")
    config = SignalAttrClassifierConfig(**dict(payload.get("model_config") or {}))
    policy = SignalAttrThresholdPolicy(**dict(payload.get("threshold_policy") or {}))
    model = SignalAttrCropClassifier(config).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "model": model, "model_config": config, "threshold_policy": policy,
        "crop_config": dict(payload.get("crop_config") or {"input_size": config.input_size, "normalization": "imagenet"}),
        "payload": dict(payload), "device": resolved_device,
    }


def _binary_metrics(stats: Mapping[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(stats.get(key, 0)) for key in ("tp", "fp", "fn", "tn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn,
        "precision": precision, "recall": recall,
        "f1": (2 * precision * recall) / max(precision + recall, 1.0e-12),
    }


@torch.no_grad()
def evaluate_signal_attr_classifier(
    model: nn.Module, loader: Any, *, device: torch.device,
    threshold_policy: SignalAttrThresholdPolicy = SignalAttrThresholdPolicy(),
    all_off_is_valid: bool = False,
) -> dict[str, Any]:
    """One forward per batch; macro-F1 includes truth or predicted positives, not TN-only states."""
    model.eval()
    states = {
        "car": {"red": "red", "yellow": "yellow", "green": "green", "left_arrow": "arrow"},
        "pedestrian": {"red": "red", "green": "green"},
    }
    bit_counts = {
        kind: {name: {key: 0 for key in ("tp", "fp", "fn", "tn")} for name in names}
        for kind, names in states.items()
    }
    type_counts = {kind: {"sample_count": 0, "valid_count": 0} for kind in states}
    base_correct = arrow_correct = arrow_count = combo_correct = sample_count = 0
    for batch in loader:
        images = batch["image"].to(device=device)
        outputs = model(images)
        base_pred = outputs["base_color_logits"].argmax(dim=1).cpu()
        arrow_pred = (torch.sigmoid(outputs["arrow_logit"]) >= threshold_policy.arrow_threshold).cpu()
        for index, row in enumerate(batch["rows"]):
            kind = str(row["light_type"])
            truth_color = str(row["base_color"])
            truth_arrow = int(row["arrow"])
            base_correct += int(int(base_pred[index]) == BASE_COLOR_TO_INDEX[truth_color])
            if kind == "car":
                arrow_correct += int(int(arrow_pred[index]) == truth_arrow)
                arrow_count += 1
            combo_correct += int(
                int(base_pred[index]) == BASE_COLOR_TO_INDEX[truth_color]
                and (kind != "car" or int(arrow_pred[index]) == truth_arrow)
            )
            sample_count += 1
            prediction = product_signal_attr_prediction_from_logits(
                outputs["base_color_logits"][index], outputs["arrow_logit"][index],
                light_type=kind, all_off_is_valid=all_off_is_valid, policy=threshold_policy,
            )
            type_counts[kind]["sample_count"] += 1
            type_counts[kind]["valid_count"] += int(prediction.tl_attr_valid)
            for name, bit in states[kind].items():
                truth = truth_arrow if name == "left_arrow" else int(truth_color == name)
                predicted = int(prediction.tl_bits[bit]) if prediction.tl_attr_valid else 0
                key = "tp" if truth and predicted else "fp" if predicted else "fn" if truth else "tn"
                bit_counts[kind][name][key] += 1
    macro_terms: list[float] = []
    by_light_type: dict[str, Any] = {}
    for kind, counts in type_counts.items():
        measured = {name: _binary_metrics(bit_counts[kind][name]) for name in states[kind]}
        macro_terms.extend(
            float(item["f1"]) for item in measured.values()
            if int(item["tp"]) + int(item["fp"]) + int(item["fn"]) > 0
        )
        by_light_type[kind] = {
            **counts, "valid_coverage": counts["valid_count"] / max(counts["sample_count"], 1),
            "states": measured,
        }
    return {
        "sample_count": sample_count,
        "base_color_accuracy": base_correct / max(sample_count, 1),
        "arrow_accuracy": arrow_correct / max(arrow_count, 1),
        "combo_accuracy": combo_correct / max(sample_count, 1),
        "valid_coverage": sum(item["valid_count"] for item in type_counts.values()) / max(sample_count, 1),
        "macro_state_f1": sum(macro_terms) / len(macro_terms) if macro_terms else 0.0,
        "by_light_type": by_light_type,
    }


__all__ = [
    "BASE_COLORS", "BASE_COLOR_TO_INDEX", "TL_BITS", "SignalAttrClassifierConfig",
    "SignalAttrThresholdPolicy", "SignalAttrCropClassifier", "SignalAttrCropTorchDataset",
    "SignalAttrPrediction", "signal_attr_crop_image_to_tensor", "signal_attr_collate",
    "signal_attr_prediction_from_logits", "product_signal_attr_prediction_from_logits",
    "load_signal_attr_classifier_checkpoint", "evaluate_signal_attr_classifier",
]
