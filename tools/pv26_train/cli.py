"""Train and resume the focused PV26 model directly from AIHub sources."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml

from common.io import atomic_write_json, read_json, write_json
from common.paths import REPO_ROOT
from common.train_runtime import training_run_lock
from model.data.dataset import FocusedDataset, FocusedSource, LogicalBatchSampler, collate_focused
from model.engine.evaluation import evaluate_focused
from model.engine.loss import PV26FocusedLoss
from model.engine.trainer import FocusedTrainer, FocusedTrainerConfig
from model.net.pv26 import PV26FocusedModel


DEFAULT_CONFIG = REPO_ROOT / "config/pv26.yaml"
ARTIFACT_ROOT = REPO_ROOT / "runs"


def _path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _output_directory(path: Path) -> Path:
    root = ARTIFACT_ROOT.resolve()
    if not root.is_dir():
        raise RuntimeError(f"artifact storage is unavailable: {ARTIFACT_ROOT}")
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"training output must be under {ARTIFACT_ROOT}")
    if shutil.disk_usage(root).free < 1024 ** 3:
        raise RuntimeError("less than 1 GiB is available for training checkpoints")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _loader(dataset, config: dict, *, sampler=None, batch_size: int = 2) -> DataLoader:
    workers = int(config["num_workers"])
    kwargs = dict(num_workers=workers, pin_memory=bool(config["pin_memory"]), collate_fn=collate_focused)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=int(config["prefetch_factor"]))
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)


def _stop_loader(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def validation_subset(dataset: FocusedDataset, samples_per_source: int) -> Subset:
    """Spread the fixed evaluation budget across each source, independent of batching."""
    selected = []
    for indices in dataset.indices_by_source.values():
        count = len(indices) if samples_per_source <= 0 else min(len(indices), samples_per_source)
        positions = np.linspace(0, len(indices) - 1, count, dtype=int)
        selected.extend(indices[int(position)] for position in positions)
    return Subset(dataset, selected)


def _optimizer_and_scheduler(model: PV26FocusedModel, train_cfg: dict, planned_steps: int):
    head_lr = float(train_cfg["head_lr"])
    groups = []
    for name, module, lr in (
        ("backbone", model.detector.model[:-1], train_cfg["backbone_lr"]),
        ("detector_head", model.detector.model[-1],
         train_cfg.get("detector_head_lr", head_lr)),
        ("roadmark", model.roadmark_decoder,
         train_cfg.get("roadmark_head_lr", head_lr)),
    ):
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(lr), "name": name})

    optimizer_name = str(train_cfg.get("optimizer", "adamw"))
    weight_decay = float(train_cfg["weight_decay"])
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    elif optimizer_name == "schedulefree_adamw":
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(
            groups, weight_decay=weight_decay,
            warmup_steps=int(train_cfg.get("schedulefree_warmup_steps", 0)),
        )
    elif optimizer_name == "prodigy":
        from prodigyopt import Prodigy
        prodigy_groups = [{"params": group["params"], "name": group["name"]} for group in groups]
        optimizer = Prodigy(
            prodigy_groups, lr=1.0, weight_decay=weight_decay,
            d_coef=float(train_cfg.get("prodigy_d_coef", 1.0)),
            slice_p=int(train_cfg.get("prodigy_slice_p", 11)),
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    schedule_name = str(train_cfg.get("lr_schedule", "cosine"))
    if optimizer_name == "schedulefree_adamw" and schedule_name != "constant":
        raise ValueError("schedulefree_adamw requires lr_schedule: constant")
    if schedule_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=planned_steps)
    elif schedule_name == "constant":
        scheduler = None
    else:
        raise ValueError(f"unsupported lr_schedule: {schedule_name}")
    return optimizer, scheduler


@torch.no_grad()
def _recompute_batch_norm(model, loader, *, device: torch.device, precision: str) -> None:
    modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not modules:
        return
    momenta = {module: module.momentum for module in modules}
    was_training = model.training
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    try:
        for cpu_batch in loader:
            image = cpu_batch["image"].to(device, non_blocking=True)
            amp = (
                torch.autocast(device_type=device.type, dtype=dtype)
                if dtype is not None else nullcontext()
            )
            with amp:
                model.forward_for_loss(image)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        model.train(was_training)


def _run_output(args: argparse.Namespace) -> Path:
    if args.resume_run is not None:
        return _output_directory(_path(args.resume_run))
    if args.output_dir is not None:
        return _output_directory(_path(args.output_dir))
    with _path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    stage = args.stage or config["train"]["stage"]
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + stage
    return _output_directory(_path(config["train"]["output_root"]) / name)


def _configuration(args: argparse.Namespace, output: Path) -> dict:
    if args.resume_run is not None:
        if (args.initial_checkpoint is not None or args.stage is not None
                or args.sample_limit is not None or getattr(args, "seed", None) is not None):
            raise ValueError("resume uses the saved stage, dataset, and seed; start a new run to change them")
        cfg = read_json(output / "run_config.json")
    else:
        with _path(args.config).open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        cfg["model"]["weights"] = str(_path(cfg["model"]["weights"]))
        for source in cfg["data"]["sources"]:
            source["root"] = str(_path(source["root"]))
        cfg["data"]["sample_limit_per_source"] = args.sample_limit
        if getattr(args, "seed", None) is not None:
            cfg["data"]["seed"] = args.seed
        if args.stage is not None:
            cfg["train"]["stage"] = args.stage
        if (output / "run_config.json").exists():
            raise FileExistsError(f"run already exists; use --resume-run: {output}")
        if any((output / name).exists() for name in
               ("train_samples.jsonl", "val_samples.jsonl", "checkpoints")):
            raise FileExistsError(f"run artifacts already exist; use a new output directory: {output}")
        cfg["initial_checkpoint"] = str(_path(args.initial_checkpoint)) if args.initial_checkpoint else None
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.microbatch_size is not None:
        cfg["train"]["microbatch_size"] = args.microbatch_size
    if args.resume_run is None:
        atomic_write_json(output / "run_config.json", cfg, ensure_ascii=False)
    return cfg


def train(args: argparse.Namespace) -> dict:
    output = _run_output(args)
    with training_run_lock(output):
        return _train_locked(args, output)


def _train_locked(args: argparse.Namespace, output: Path) -> dict:
    cfg = _configuration(args, output)
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    seed = int(data_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(train_cfg["device"])
    precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(train_cfg["amp_dtype"], train_cfg["amp_dtype"])
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; choose fp16 or fp32 in the training configuration")
    sources = [FocusedSource(**{**item, "root": Path(item["root"])}) for item in data_cfg["sources"]]
    stage = train_cfg["stage"]
    if stage != "joint":
        kind = "traffic" if stage == "detector" else "roadmark"
        sources = [source for source in sources if source.kind == kind]
    image_hw = tuple(cfg["model"]["image_hw"])
    checkpoints = output / "checkpoints"
    has_checkpoint = (checkpoints / "latest.pt").is_file() or (checkpoints / "previous.pt").is_file()
    datasets = []
    for split in ("train", "val"):
        index = output / f"{split}_samples.jsonl"
        if args.resume_run is not None and has_checkpoint and not index.is_file():
            raise FileNotFoundError(f"committed run sample index not found: {index}")
        dataset = FocusedDataset(sources, split=split, image_hw=image_hw,
            seed=seed, augment=bool(data_cfg["augment"] and split == "train"),
            sample_limit_per_source=None if index.is_file() else data_cfg.get("sample_limit_per_source"),
            index_path=index if index.is_file() else None)
        if not index.is_file():
            dataset.save_index(index)
        datasets.append(dataset)
    print(f"data: train={len(datasets[0])}, val={len(datasets[1])}, run={output}", flush=True)
    sampler = LogicalBatchSampler(
        datasets[0], batch_size=int(train_cfg["logical_batch_size"]), seed=seed,
        strategy=str(data_cfg.get("sampling_strategy", "random_with_replacement")),
    )
    initial_checkpoint = cfg.get("initial_checkpoint")
    model = PV26FocusedModel(weights=None if has_checkpoint or initial_checkpoint else cfg["model"]["weights"],
        variant=cfg["model"]["variant"], roadmark_width=int(cfg["model"]["roadmark_width"])).to(device)
    model.set_train_stage(stage)
    if initial_checkpoint and not has_checkpoint:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
    criterion = PV26FocusedLoss(
        model, det_weight=float(train_cfg.get("det_loss_weight", 1.0)),
        roadmark_weight=float(train_cfg.get("roadmark_loss_weight", 1.0)),
    )
    planned_steps = int(train_cfg["max_steps"])
    optimizer, scheduler = _optimizer_and_scheduler(model, train_cfg, planned_steps)
    trainer = FocusedTrainer(model, criterion, optimizer, scheduler, sampler,
        FocusedTrainerConfig(output_dir=output, device=str(device), precision=precision,
            microbatch_size=int(train_cfg["microbatch_size"]), stage=stage,
            checkpoint_interval_sec=float(train_cfg["checkpoint_interval_sec"]),
            max_consecutive_failures=int(train_cfg["max_consecutive_failures"]),
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            gradient_strategy=str(train_cfg.get("gradient_strategy", "sum")),
            gradnorm_alpha=float(train_cfg.get("gradnorm_alpha", 1.5)),
            gradnorm_lr=float(train_cfg.get("gradnorm_lr", 0.025))), run_metadata=cfg)
    if has_checkpoint:
        trainer.load_checkpoint()
        if args.microbatch_size is not None:
            trainer.microbatch_size = args.microbatch_size
    train_loader = _loader(datasets[0], data_cfg, sampler=sampler)
    validation_data = validation_subset(datasets[1], int(train_cfg.get("validation_samples_per_source", 128)))
    val_loader = _loader(validation_data, data_cfg, batch_size=max(1, min(trainer.microbatch_size, 4)))
    bn_loader = None
    if str(train_cfg.get("optimizer", "adamw")) == "schedulefree_adamw":
        calibration_dataset = FocusedDataset(
            sources, split="train", image_hw=image_hw, seed=seed, augment=False,
            index_path=output / "train_samples.jsonl",
        )
        calibration_data = validation_subset(
            calibration_dataset,
            int(train_cfg.get("schedulefree_bn_samples_per_source", 32)),
        )
        bn_loader = _loader(
            calibration_data, data_cfg, batch_size=max(1, min(trainer.microbatch_size, 8))
        )
    stop_at = min(planned_steps, trainer.global_step + args.steps) if args.steps is not None else planned_steps
    started = time.monotonic()
    starting_step = trainer.global_step
    validation = None

    def validate() -> None:
        nonlocal validation
        trainer.begin_evaluation()
        try:
            validation_started = time.monotonic()
            if bn_loader is not None:
                _recompute_batch_norm(model, bn_loader, device=device, precision=precision)
            validation = evaluate_focused(model, criterion, val_loader, device=device, precision=precision,
                                          max_batches=len(val_loader),
                                          geometry_tolerance_px=float(train_cfg.get("geometry_tolerance_px", 8.0)))
            validation["elapsed_sec"] = time.monotonic() - validation_started
            validation["global_step"] = trainer.global_step
            validation["balanced_task_f1"] = 0.5 * (
                float(validation["signal_detection_total"]["f1"])
                + float(validation["roadmark_lines_total"]["f1"])
            )
            write_json(output / "validation.json", validation, ensure_ascii=False)
            trainer.update_best(float(validation["selection_metric"]))
            print(json.dumps({"validation": validation}, ensure_ascii=False), flush=True)
        finally:
            trainer.end_evaluation()

    def on_step(current: FocusedTrainer, summary: dict) -> None:
        progress_every = int(os.environ.get("YOLOPV26_PROGRESS_EVERY", train_cfg["log_every"]))
        if current.global_step % progress_every == 0 or current.global_step == starting_step + 1:
            elapsed = time.monotonic() - started
            print(json.dumps({**summary, "elapsed_sec": elapsed,
                "samples_per_sec": (current.global_step - starting_step) * sampler.batch_size / max(elapsed, 1e-9)},
                ensure_ascii=False), flush=True)
        interval = int(train_cfg["validation_every"])
        if interval > 0 and current.global_step % interval == 0:
            validate()

    try:
        summary = trainer.fit(train_loader, max_steps=stop_at, planned_steps=planned_steps, on_step=on_step)
        if not summary["stopped_by_signal"] and trainer.global_step > starting_step:
            validate()
        summary["run_dir"] = str(output)
        summary["validation"] = validation
        summary["best_weights"] = str(trainer.best_path) if trainer.best_path.exists() else None
        summary["elapsed_sec"] = time.monotonic() - started
        if device.type == "cuda":
            summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write_json(output / "summary.json", summary, ensure_ascii=False)
        return summary
    finally:
        _stop_loader(train_loader)
        _stop_loader(val_loader)
        if bn_loader is not None:
            _stop_loader(bn_loader)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the two-signal and three-roadmark PV26 model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--resume-run", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--stage", choices=("detector", "roadmark", "joint"))
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--steps", type=int, help="Stop after this many additional optimizer updates.")
    parser.add_argument("--sample-limit", type=int, help="Limit samples per source for a short development run.")
    parser.add_argument("--seed", type=int, help="Override the data, augmentation, and initialization seed.")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    result = train(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
