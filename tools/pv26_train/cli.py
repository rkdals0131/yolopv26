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

from common.io import atomic_write_json, read_json
from common.paths import REPO_ROOT
from common.train_runtime import training_run_lock
from model.data.dataset import FocusedDataset, FocusedSource, LogicalBatchSampler, collate_focused
from model.engine.evaluation import evaluate_focused
from model.engine.loss import PV26FocusedLoss
from model.engine.trainer import FocusedTrainer, FocusedTrainerConfig, _atomic_save
from model.net.pv26 import PV26FocusedModel


DEFAULT_CONFIG = REPO_ROOT / "config/pv26.yaml"
ARTIFACT_ROOT = REPO_ROOT / "runs"


def _path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _output_directory(path: Path, artifact_root: Path | None = None) -> Path:
    root_path = artifact_root or ARTIFACT_ROOT
    root = root_path.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"artifact storage is unavailable: {root_path}")
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"training output must be under {root_path}")
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


def optimizer_groups(model: PV26FocusedModel, train_cfg: dict) -> list[dict]:
    head_lr = float(train_cfg["head_lr"])
    groups = []
    for name, module, lr in (
        ("backbone", model.detector.model[:-1], train_cfg["backbone_lr"]),
        ("detector_head", model.detector.model[-1],
         train_cfg.get("detector_head_lr", head_lr)),
        ("roadmark", model.roadmark_decoder,
         train_cfg.get("roadmark_lr", head_lr)),
    ):
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(lr), "name": name})
    return groups


def _evaluate(model, criterion, loader, cfg: dict, *, step: int, scope: str,
              checkpoint: str, roadmark_localization: str = "grid",
              roadmark_threshold: float | tuple[float, float, float] = 0.5) -> dict:
    train = cfg["train"]
    device = torch.device(train["device"])
    precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        train["amp_dtype"], train["amp_dtype"])
    started = time.monotonic()
    last_report = started
    total = len(loader.dataset)

    def progress(samples: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if samples == 0 or samples == total or now - last_report >= 5.0:
            print(json.dumps({"evaluation_progress": {
                "scope": scope, "checkpoint": checkpoint, "global_step": step,
                "samples": samples, "total": total, "elapsed_sec": now - started,
            }}), flush=True)
            last_report = now

    progress(0)
    result = evaluate_focused(
        model, criterion, loader, device=device, precision=precision,
        geometry_tolerance_px=float(train.get("geometry_tolerance_px", 8.0)),
        on_progress=progress, roadmark_localization=roadmark_localization,
        roadmark_threshold=roadmark_threshold,
        roadmark_decode=train.get("roadmark_decode"),
    )
    result.update(global_step=step, scope=scope, checkpoint=checkpoint,
                  elapsed_sec=time.monotonic() - started)
    return result


def evaluate_checkpoint(output: Path, cfg: dict, role: str, *, dataset=None,
                        eval_stage: str | None = None, eval_index_run: Path | None = None,
                        eval_samples_per_source: int | None = None,
                        roadmark_localization: str = "grid",
                        roadmark_threshold: float | tuple[float, float, float] = 0.5,
                        eval_output: Path | None = None) -> dict:
    """Evaluate every saved validation sample, without updating training state."""
    checkpoint = torch.load(output / "checkpoints" / f"{role}.pt", map_location="cpu", weights_only=False)
    stage = eval_stage or checkpoint["stage"]
    name = (f"validation_full_{role}.json" if eval_stage is None and eval_index_run is None
            and roadmark_localization == "grid"
            else f"validation_full_{stage}_{role}_{roadmark_localization}.json")
    destination = (output / eval_output).resolve() if eval_output is not None else output / name
    if eval_output is not None:
        if not destination.is_relative_to(output.resolve()):
            raise ValueError("evaluation output must stay inside its run directory")
        if destination.exists():
            raise FileExistsError(f"evaluation output already exists: {destination}")
    if dataset is None:
        sources = [FocusedSource(**{**s, "root": Path(s["root"])}) for s in cfg["data"]["sources"]]
        if stage != "joint":
            kind = "traffic" if stage == "detector" else "roadmark"
            sources = [source for source in sources if source.kind == kind]
        dataset = FocusedDataset(sources, split="val", image_hw=tuple(cfg["model"]["image_hw"]),
                                 index_path=(eval_index_run or output) / "val_samples.jsonl")
    if eval_samples_per_source is not None:
        dataset = validation_subset(dataset, eval_samples_per_source)
    model = PV26FocusedModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.set_train_stage(stage)
    model.to(torch.device(cfg["train"]["device"]))
    train = cfg["train"]
    criterion = PV26FocusedLoss(
        model, det_weight=float(train.get("det_loss_weight", 1.0)),
        roadmark_weight=float(train.get("roadmark_loss_weight", 1.0)),
        detector_loss_schedule=str(train.get("detector_loss_schedule", "restart")),
    )
    step = int(checkpoint["global_step"])
    criterion.set_progress(step, int(train["max_steps"]))
    del checkpoint
    loader = _loader(dataset, cfg["data"], batch_size=max(1, min(int(cfg["train"]["microbatch_size"]), 4)))
    try:
        result = _evaluate(model, criterion, loader, cfg, step=step, scope="full",
                           checkpoint=role, roadmark_localization=roadmark_localization,
                           roadmark_threshold=roadmark_threshold)
        result.update(evaluation_stage=stage,
                      evaluation_index_run=str(eval_index_run or output),
                      evaluation_samples_per_source=eval_samples_per_source,
                      roadmark_localization=roadmark_localization,
                      roadmark_threshold=roadmark_threshold)
        atomic_write_json(destination, result, ensure_ascii=False)
        print(json.dumps({"validation": result}, ensure_ascii=False), flush=True)
        return result
    finally:
        _stop_loader(loader)


def _optimizer_and_scheduler(model: PV26FocusedModel, train_cfg: dict, planned_steps: int):
    groups = optimizer_groups(model, train_cfg)

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
    artifact_root = _path(args.artifact_root) if getattr(args, "artifact_root", None) else None
    if args.resume_run is not None:
        return _output_directory(_path(args.resume_run), artifact_root)
    if args.output_dir is not None:
        return _output_directory(_path(args.output_dir), artifact_root)
    with _path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    stage = args.stage or config["train"]["stage"]
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + stage
    root = artifact_root or _path(config["train"]["output_root"])
    return _output_directory(root / name, artifact_root)


def _configuration(args: argparse.Namespace, output: Path) -> dict:
    if args.resume_run is not None:
        if (args.initial_checkpoint is not None or args.stage is not None
                or args.sample_limit is not None or getattr(args, "seed", None) is not None
                or getattr(args, "index_run", None) is not None
                or getattr(args, "detector_loss_schedule", None) is not None
                or any(getattr(args, key, None) is not None
                       for key in ("backbone_lr", "head_lr", "roadmark_lr"))):
            raise ValueError("resume uses the saved stage, dataset, and seed; start a new run to change them")
        cfg = read_json(output / "run_config.json")
    else:
        with _path(args.config).open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        cfg["model"]["weights"] = str(_path(cfg["model"]["weights"]))
        for source in cfg["data"]["sources"]:
            source["root"] = str(_path(source["root"]))
        cfg["data"]["sample_limit_per_source"] = args.sample_limit
        index_run = getattr(args, "index_run", None) or cfg["data"].get("index_run")
        if index_run is not None:
            if args.sample_limit is not None:
                raise ValueError("a saved sample index cannot be combined with --sample-limit")
            cfg["data"]["index_run"] = str(_path(index_run))
        if getattr(args, "seed", None) is not None:
            cfg["data"]["seed"] = args.seed
        if args.stage is not None:
            cfg["train"]["stage"] = args.stage
        if getattr(args, "detector_loss_schedule", None) is not None:
            cfg["train"]["detector_loss_schedule"] = args.detector_loss_schedule
        for key in ("backbone_lr", "head_lr", "roadmark_lr"):
            value = getattr(args, key, None)
            if value is not None:
                cfg["train"][key] = value
        if (output / "run_config.json").exists():
            raise FileExistsError(f"run already exists; use --resume-run: {output}")
        if any((output / name).exists() for name in
               ("train_samples.jsonl", "val_samples.jsonl", "checkpoints")):
            raise FileExistsError(f"run artifacts already exist; use a new output directory: {output}")
        initial_checkpoint = args.initial_checkpoint or cfg.get("initial_checkpoint")
        cfg["initial_checkpoint"] = str(_path(initial_checkpoint)) if initial_checkpoint else None
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
    all_sources = [FocusedSource(**{**item, "root": Path(item["root"])}) for item in data_cfg["sources"]]
    sources = all_sources
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
        source_index = (Path(data_cfg["index_run"]) / f"{split}_samples.jsonl"
                        if not index.is_file() and data_cfg.get("index_run") else None)
        dataset = FocusedDataset(all_sources if source_index is not None else sources,
            split=split, image_hw=image_hw,
            seed=seed, augment=bool(data_cfg["augment"] and split == "train"),
            roadmark_target_sigma_cells=(
                float(data_cfg.get("roadmark_target_sigma_cells", 0.0)) if split == "train" else 0.0),
            sample_limit_per_source=None if index.is_file() or source_index is not None
                else data_cfg.get("sample_limit_per_source"),
            index_path=index if index.is_file() else source_index,
            selected_kind=kind if source_index is not None and stage != "joint" else None)
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
        detector_loss_schedule=str(train_cfg.get("detector_loss_schedule", "restart")),
        roadmark_dice=str(train_cfg.get("roadmark_dice", "linear")),
    )
    evaluation_criterion = PV26FocusedLoss(
        model, det_weight=float(train_cfg.get("det_loss_weight", 1.0)),
        roadmark_weight=float(train_cfg.get("roadmark_loss_weight", 1.0)),
        detector_loss_schedule=str(train_cfg.get("detector_loss_schedule", "restart")),
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
    miner = None
    if data_cfg.get("hard_negative_mining"):
        from model.engine.hard_negative_mining import StopNegativeMiner
        mining_config = dict(data_cfg["hard_negative_mining"])
        mining_config["positive_index"] = str(_path(mining_config["positive_index"]))
        # A mined negative is an image whose false positive survives evaluation decoding.
        mining_config["roadmark_decode"] = train_cfg.get("roadmark_decode") or {}
        miner = StopNegativeMiner(datasets[0], sampler, mining_config, output)
    validation = None
    roadmark_best_path = checkpoints / "best_roadmark.pt"
    stop_best_path = checkpoints / "best_stop_line.pt"
    roadmark_best = None
    stop_best = None
    if stage == "joint" and roadmark_best_path.is_file():
        roadmark_best = float(torch.load(roadmark_best_path, map_location="cpu", weights_only=False)["metric"])
    if stage == "joint" and stop_best_path.is_file():
        stop_best = float(torch.load(stop_best_path, map_location="cpu", weights_only=False)["metric"])

    def validate() -> None:
        nonlocal validation, roadmark_best, stop_best
        trainer.begin_evaluation()
        try:
            if bn_loader is not None:
                _recompute_batch_norm(model, bn_loader, device=device, precision=precision)
            evaluation_criterion.set_progress(trainer.global_step, planned_steps)
            validation = _evaluate(
                model, evaluation_criterion, val_loader, cfg, step=trainer.global_step,
                scope="periodic", checkpoint="current",
            )
            validation["balanced_task_f1"] = 0.5 * (
                float(validation["signal_detection_total"]["f1"])
                + float(validation["roadmark_lines_total"]["f1"])
            )
            atomic_write_json(output / "validation.json", validation, ensure_ascii=False)
            # Preserve the signal-priority best.pt policy. Keep the roadmark optimum
            # separately for comparison, never replace best.pt using another metric.
            if stage == "joint":
                score = float(validation["roadmark_lines_total"]["f1"])
                if roadmark_best is None or score > roadmark_best:
                    _atomic_save({"model": model.state_dict(), "metric": score, "mode": "max",
                        "global_step": trainer.global_step, "stage": stage,
                        "model_config": model.model_config(), "run_metadata": cfg}, roadmark_best_path)
                    roadmark_best = score
                stop_score = float(validation["roadmark_lines"]["stop_line"]["f1"])
                if stop_best is None or stop_score > stop_best:
                    _atomic_save({"model": model.state_dict(), "metric": stop_score, "mode": "max",
                        "global_step": trainer.global_step, "stage": stage,
                        "model_config": model.model_config(), "run_metadata": cfg}, stop_best_path)
                    stop_best = stop_score
            selected = trainer.update_best(float(validation["selection_metric"]))
            if bn_loader is not None and not selected:
                # ScheduleFree checkpoints contain evaluation weights. Publish the
                # recalibrated BatchNorm buffers even when best.pt is unchanged.
                trainer.save_checkpoint()
            print(json.dumps({"validation": validation}, ensure_ascii=False), flush=True)
        finally:
            trainer.end_evaluation()

    def on_step(current: FocusedTrainer, summary: dict) -> bool:
        progress_every = int(os.environ.get("YOLOPV26_PROGRESS_EVERY", train_cfg["log_every"]))
        if current.global_step % progress_every == 0 or current.global_step == starting_step + 1:
            elapsed = time.monotonic() - started
            print(json.dumps({**summary, "elapsed_sec": elapsed,
                "samples_per_sec": (current.global_step - starting_step) * sampler.batch_size / max(elapsed, 1e-9)},
                ensure_ascii=False), flush=True)
        interval = int(train_cfg["validation_every"])
        if interval > 0 and current.global_step % interval == 0:
            validate()
        if miner is not None and miner.due(current.global_step) and not current._stop_requested:
            return miner.refresh(current)
        return False

    def on_start(current: FocusedTrainer) -> None:
        if miner is not None and miner.due(current.global_step):
            miner.refresh(current)

    try:
        summary = trainer.fit(train_loader, max_steps=stop_at, planned_steps=planned_steps,
                              on_step=on_step, on_start=on_start)
        if (not summary["stopped_by_signal"] and trainer.global_step > starting_step
                and (validation is None or validation["global_step"] != trainer.global_step)):
            validate()
        summary["run_dir"] = str(output)
        summary["validation"] = validation
        summary["best_weights"] = str(trainer.best_path) if trainer.best_path.exists() else None
        summary["elapsed_sec"] = time.monotonic() - started
        if device.type == "cuda":
            summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        atomic_write_json(output / "summary.json", summary, ensure_ascii=False)
        if not summary["stopped_by_signal"] and trainer.global_step >= planned_steps:
            _stop_loader(train_loader)
            _stop_loader(val_loader)
            if bn_loader is not None:
                _stop_loader(bn_loader)
            # Release the training optimizer and model before loading each
            # evaluation checkpoint. Final scores never enter periodic selection.
            del trainer, optimizer, scheduler, criterion, evaluation_criterion, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            full = {}
            for role in ("latest", "best", "best_roadmark", "best_stop_line"):
                if (checkpoints / f"{role}.pt").is_file():
                    full[role] = evaluate_checkpoint(output, cfg, role, dataset=datasets[1])
            summary["full_validation"] = full
            atomic_write_json(output / "summary.json", summary, ensure_ascii=False)
        return summary
    finally:
        _stop_loader(train_loader)
        _stop_loader(val_loader)
        if bn_loader is not None:
            _stop_loader(bn_loader)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the two-signal and three-roadmark PV26 model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path,
                        help="Existing root containing all outputs for this run.")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--resume-run", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--index-run", type=Path,
                        help="Use and snapshot the train/val sample lists of an earlier run.")
    parser.add_argument("--detector-loss-schedule", choices=("restart", "mature"),
                        help="Choose the detector's one-to-many loss mix for a new run.")
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--roadmark-lr", type=float)
    parser.add_argument("--stage", choices=("detector", "roadmark", "joint"))
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--steps", type=int, help="Stop after this many additional optimizer updates.")
    parser.add_argument("--sample-limit", type=int, help="Limit samples per source for a short development run.")
    parser.add_argument("--evaluate-only", choices=("latest", "best", "best_roadmark", "best_stop_line"),
                        help="With --resume-run, evaluate a checkpoint on its entire saved validation set.")
    parser.add_argument("--eval-index-run", type=Path,
                        help="Evaluate on another run's saved validation list.")
    parser.add_argument("--eval-stage", choices=("joint",),
                        help="Evaluate both tasks even for a roadmark-stage checkpoint.")
    parser.add_argument("--eval-samples-per-source", type=int,
                        help="Use the same evenly spread development subset from each source.")
    parser.add_argument("--roadmark-localization", choices=("grid", "subpixel", "smooth"),
                        help="Compare line coordinates without changing model weights.")
    parser.add_argument("--roadmark-thresholds", type=float, nargs=3,
                        metavar=("WHITE", "YELLOW", "STOP"),
                        help="Compare class thresholds without changing model weights.")
    parser.add_argument("--eval-output", type=Path,
                        help="Write a named evaluation JSON inside the selected run.")
    parser.add_argument("--seed", type=int, help="Override the data, augmentation, and initialization seed.")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.evaluate_only:
        if args.resume_run is None or any(value is not None for value in
                (args.steps, args.stage, args.sample_limit, args.initial_checkpoint,
                 args.output_dir, args.seed)):
            parser.error("--evaluate-only requires --resume-run and cannot change the run or dataset")
        if args.index_run is not None:
            parser.error("--index-run creates a new training run; use --eval-index-run for evaluation")
        if (args.detector_loss_schedule is not None or any(value is not None for value in
                (args.backbone_lr, args.head_lr, args.roadmark_lr))):
            parser.error("training loss and learning-rate overrides require a new run")
        if args.roadmark_thresholds is not None and args.eval_output is None:
            parser.error("--roadmark-thresholds requires --eval-output")
        if args.eval_samples_per_source is not None and args.eval_samples_per_source < 1:
            parser.error("--eval-samples-per-source must be positive")
        output = _run_output(args)
        with training_run_lock(output):
            cfg = _configuration(args, output)
            result = evaluate_checkpoint(output, cfg, args.evaluate_only,
                eval_stage=args.eval_stage,
                eval_index_run=_path(args.eval_index_run) if args.eval_index_run else None,
                eval_samples_per_source=args.eval_samples_per_source,
                roadmark_localization=args.roadmark_localization or "grid",
                roadmark_threshold=tuple(args.roadmark_thresholds) if args.roadmark_thresholds else 0.5,
                eval_output=args.eval_output)
    else:
        if any(value is not None for value in (args.eval_index_run, args.eval_stage,
                args.eval_samples_per_source, args.roadmark_localization,
                args.roadmark_thresholds, args.eval_output)):
            parser.error("evaluation options require --evaluate-only")
        result = train(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
