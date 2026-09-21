"""Use the focused step trainer for the internal SignalAttr crop classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.io import read_json
from model.data.dataset import LogicalBatchSampler
from model.engine.trainer import FocusedTrainer, FocusedTrainerConfig, _atomic_save, _fsync_directory

from .classifier import (
    BASE_COLORS,
    SignalAttrClassifierConfig,
    SignalAttrCropClassifier,
    SignalAttrCropTorchDataset,
    SignalAttrThresholdPolicy,
    evaluate_signal_attr_classifier,
    load_signal_attr_classifier_checkpoint,
    signal_attr_collate,
)

DEFAULT_INITIAL_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "signal_attr" / "best_signal_attr.pt"


class SignalAttrBatchAdapter:
    def slice_batch(self, batch: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
        return {
            key: value[start:stop] if isinstance(value, torch.Tensor) else value[start:stop]
            for key, value in batch.items()
        }

    def term_counts(self, batch: Mapping[str, Any]) -> dict[str, int]:
        return {
            "base_color": int(batch["image"].shape[0]),
            "left_arrow": int(batch["arrow_target_valid"].sum().item()),
        }

    def weighted_terms(self, losses: Mapping[str, torch.Tensor], criterion: nn.Module) -> dict[str, torch.Tensor]:
        return {
            "base_color": losses["base_color"],
            "left_arrow": losses["left_arrow"] * float(getattr(criterion, "arrow_loss_weight", 1.0)),
        }

    def sample_ids(self, batch: Mapping[str, Any]) -> list[str]:
        return [str(row.get("sample_id")) for row in batch["rows"]]


class SignalAttrCriterion(nn.Module):
    def __init__(self, *, arrow_loss_weight: float = 1.0) -> None:
        super().__init__()
        self.arrow_loss_weight = float(arrow_loss_weight)

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        base_color = nn.functional.cross_entropy(
            outputs["base_color_logits"].float(), batch["base_color_target"]
        )
        arrow_values = nn.functional.binary_cross_entropy_with_logits(
            outputs["arrow_logit"].float(), batch["arrow_target"].float(), reduction="none"
        )
        mask = batch["arrow_target_valid"].float()
        left_arrow = (arrow_values * mask).sum() / mask.sum().clamp_min(1.0)
        return {"base_color": base_color, "left_arrow": left_arrow}


class _SampledCropDataset(SignalAttrCropTorchDataset):
    def __init__(self, dataset_root: Path, *, split: str, input_size: int, normalization: str,
                 labels_root: Path, sampling: str) -> None:
        super().__init__(dataset_root, split=split, input_size=input_size,
                         normalization=normalization, labels_root=labels_root)
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(self.rows):
            light_type = str(row.get("light_type") or "")
            if light_type not in ("car", "pedestrian"):
                raise ValueError(f"SignalAttr product crop is missing car/pedestrian light_type: {row.get('sample_id')}")
            color = str(row["base_color"])
            key = f"{light_type}:{color}:left={int(row['arrow'])}" if light_type == "car" else f"{light_type}:{color}"
            groups.setdefault(key, []).append(index)
        self.group_counts = dict(sorted((key, len(indices)) for key, indices in groups.items()))
        if sampling == "balanced":
            self.sources = tuple(SimpleNamespace(name=key, weight=1.0) for key in sorted(groups))
            self.indices_by_source = {key: tuple(groups[key]) for key in sorted(groups)}
        else:
            self.sources = (SimpleNamespace(name="signal_attr", weight=1.0),)
            self.indices_by_source = {"signal_attr": tuple(range(len(self)))}

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        sample_index = index[0] if isinstance(index, tuple) else index
        return super().__getitem__(sample_index)


@dataclass
class SignalAttrFocusedRun:
    trainer: FocusedTrainer
    train_loader: DataLoader
    val_loader: DataLoader
    model_config: SignalAttrClassifierConfig
    threshold_policy: SignalAttrThresholdPolicy
    crop_config: dict[str, Any]
    all_off_is_valid: bool
    snapshot_root: Path
    sampling: str
    group_counts: dict[str, int]

    def evaluate(self) -> dict[str, Any]:
        return evaluate_signal_attr_classifier(
            self.trainer.model,
            self.val_loader,
            device=self.trainer.device,
            threshold_policy=self.threshold_policy,
            all_off_is_valid=self.all_off_is_valid,
        )

    def publish_checkpoint(self, output_path: Path, *, best: bool = True) -> Path:
        if best:
            selected = torch.load(self.trainer.best_path, map_location="cpu", weights_only=False)
            state_dict = selected["model"]
            selected_step = int(selected["global_step"])
            model_config = dict(selected["model_config"])
            metadata = dict(selected["run_metadata"])
        else:
            state_dict = {key: value.detach().cpu() for key, value in self.trainer.model.state_dict().items()}
            selected_step = self.trainer.global_step
            model_config = asdict(self.model_config)
            metadata = dict(self.trainer.run_metadata)
        payload = {
            "model_type": "SignalAttrCropClassifier",
            "model_state_dict": state_dict,
            "model_config": model_config,
            "threshold_policy": dict(metadata["threshold_policy"]),
            "crop_config": dict(metadata["crop_config"]),
            "base_colors": list(BASE_COLORS),
            "tl_bits": ["red", "yellow", "green", "arrow"],
            "state_semantics": "left_arrow",
            "all_off_is_valid": bool(metadata["all_off_is_valid"]),
            "sampling": str(metadata.get("sampling", "natural")),
            "global_step": selected_step,
        }
        path = Path(output_path)
        _atomic_save(payload, path)
        return path


def _snapshot_dataset_rows(dataset_root: Path, output_root: Path, *, resume: bool,
                           has_checkpoint: bool) -> Path:
    snapshot = output_root / "data_snapshot"
    if resume and snapshot.is_dir():
        return snapshot
    if resume and has_checkpoint:
        raise FileNotFoundError(f"SignalAttr checkpoint needs its saved data snapshot: {snapshot}")
    if snapshot.exists():
        raise FileExistsError(f"SignalAttr run data snapshot already exists; resume this run: {snapshot}")
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".signal_attr_snapshot_", dir=output_root))
    try:
        files = {
            "train.jsonl": dataset_root / "labels" / "train.jsonl",
            "val.jsonl": dataset_root / "labels" / "val.jsonl",
            "crop_config.json": dataset_root / "meta" / "crop_config.json",
            "manifest.json": dataset_root / "meta" / "signal_attr_dataset_manifest.json",
        }
        for name, source in files.items():
            destination = stage / name
            shutil.copyfile(source, destination)
            with destination.open("r+b") as file:
                os.fsync(file.fileno())
        if read_json(stage / "manifest.json").get("state_semantics") != "left_arrow":
            raise ValueError("SignalAttr product training requires left_arrow crop labels")
        _fsync_directory(stage)
        os.replace(stage, snapshot)
        _fsync_directory(output_root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return snapshot


def build_signal_attr_focused_run(
    dataset_root: Path,
    output_root: Path,
    *,
    logical_batch_size: int,
    microbatch_size: int,
    num_workers: int = 0,
    seed: int = 26,
    device: str = "cuda",
    precision: str = "bf16",
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    arrow_loss_weight: float = 1.0,
    initial_checkpoint: Path | None = DEFAULT_INITIAL_CHECKPOINT,
    checkpoint_interval_sec: float = 600.0,
    resume: bool = False,
    sampling: str = "natural",
) -> SignalAttrFocusedRun:
    if sampling not in ("natural", "balanced"):
        raise ValueError(f"unsupported SignalAttr sampling: {sampling}")
    root = Path(dataset_root)
    output = Path(output_root)
    latest = output / "checkpoints" / "latest.pt"
    previous = output / "checkpoints" / "previous.pt"
    has_checkpoint = resume and (latest.is_file() or previous.is_file())
    snapshot = _snapshot_dataset_rows(root, output, resume=resume, has_checkpoint=has_checkpoint)
    manifest = read_json(snapshot / "manifest.json")
    if resume and manifest.get("state_semantics") != "left_arrow":
        raise ValueError("SignalAttr product training requires left_arrow crop labels")
    crop_config = dict(read_json(snapshot / "crop_config.json"))
    resumed_checkpoint: Mapping[str, Any] | None = None
    if not has_checkpoint:
        torch.manual_seed(seed)
    if has_checkpoint:
        resumed_checkpoint = torch.load(latest if latest.is_file() else previous, map_location="cpu", weights_only=False)
        saved_sampling = str(resumed_checkpoint["run_metadata"].get("sampling", "natural"))
        if sampling != saved_sampling:
            raise ValueError(f"SignalAttr sampling differs from saved run: {saved_sampling}")
        model_config = SignalAttrClassifierConfig(**resumed_checkpoint["model_config"])
        model = SignalAttrCropClassifier(model_config)
        policy = SignalAttrThresholdPolicy(**resumed_checkpoint["run_metadata"]["threshold_policy"])
    elif initial_checkpoint is None:
        model_config = SignalAttrClassifierConfig(input_size=int(crop_config["input_size"]))
        model = SignalAttrCropClassifier(model_config)
        policy = SignalAttrThresholdPolicy()
    else:
        loaded = load_signal_attr_classifier_checkpoint(initial_checkpoint, device="cpu")
        model_config = replace(loaded["model_config"], input_size=int(crop_config["input_size"]))
        model = loaded["model"]
        model.config = model_config
        policy = loaded["threshold_policy"]
    train_dataset = _SampledCropDataset(
        root, split="train", input_size=int(crop_config["input_size"]),
        normalization=str(crop_config["normalization"]), labels_root=snapshot, sampling=sampling,
    )
    val_dataset = SignalAttrCropTorchDataset(
        root, split="val", input_size=int(crop_config["input_size"]),
        normalization=str(crop_config["normalization"]), labels_root=snapshot,
    )
    sampler = LogicalBatchSampler(train_dataset, batch_size=logical_batch_size, seed=seed)
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=num_workers,
        collate_fn=signal_attr_collate, pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=logical_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=signal_attr_collate, pin_memory=device.startswith("cuda"),
    )
    model = model.to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer = FocusedTrainer(
        model,
        SignalAttrCriterion(arrow_loss_weight=arrow_loss_weight),
        optimizer,
        scheduler=None,
        sampler=sampler,
        config=FocusedTrainerConfig(
            output_dir=output, device=device, precision=precision,
            microbatch_size=microbatch_size,
            stage="signal_attr", checkpoint_interval_sec=checkpoint_interval_sec,
        ),
        batch_adapter=SignalAttrBatchAdapter(),
        run_metadata={
            "dataset_root": str(root.resolve()),
            "data_snapshot": str(snapshot.resolve()),
            "state_semantics": "left_arrow",
            "all_off_is_valid": bool(manifest.get("all_off_is_valid", False)),
            "crop_config": crop_config,
            "threshold_policy": asdict(policy),
            "sampling": sampling,
            "sampling_group_counts": train_dataset.group_counts,
        },
    )
    if has_checkpoint:
        trainer.load_checkpoint()
    return SignalAttrFocusedRun(
        trainer=trainer,
        train_loader=loader,
        val_loader=val_loader,
        model_config=model_config,
        threshold_policy=policy,
        crop_config=crop_config,
        all_off_is_valid=bool(manifest.get("all_off_is_valid", False)),
        snapshot_root=snapshot,
        sampling=sampling,
        group_counts=train_dataset.group_counts,
    )


__all__ = [
    "DEFAULT_INITIAL_CHECKPOINT",
    "SignalAttrBatchAdapter",
    "SignalAttrCriterion",
    "SignalAttrFocusedRun",
    "build_signal_attr_focused_run",
]
