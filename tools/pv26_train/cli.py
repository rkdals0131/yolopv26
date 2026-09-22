"""Train and resume the focused PV26 model directly from AIHub sources."""

from __future__ import annotations

import argparse
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


def optimizer_groups(model: PV26FocusedModel, config: dict) -> list[dict]:
    groups = []
    for name, module, lr in (
        ("backbone", model.detector.model[:-1], config["backbone_lr"]),
        ("detector_head", model.detector.model[-1], config["head_lr"]),
        ("roadmark", model.roadmark_decoder, config.get("roadmark_lr", config["head_lr"])),
    ):
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": float(lr), "name": name})
    return groups


def _evaluate(model, criterion, loader, cfg: dict, *, step: int, scope: str,
              checkpoint: str) -> dict:
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
        on_progress=progress,
    )
    result.update(global_step=step, scope=scope, checkpoint=checkpoint,
                  elapsed_sec=time.monotonic() - started)
    return result


def evaluate_checkpoint(output: Path, cfg: dict, role: str, *, dataset=None) -> dict:
    """Evaluate every saved validation sample, without updating training state."""
    checkpoint = torch.load(output / "checkpoints" / f"{role}.pt", map_location="cpu", weights_only=False)
    stage = checkpoint["stage"]
    if dataset is None:
        sources = [FocusedSource(**{**s, "root": Path(s["root"])}) for s in cfg["data"]["sources"]]
        if stage != "joint":
            kind = "traffic" if stage == "detector" else "roadmark"
            sources = [source for source in sources if source.kind == kind]
        dataset = FocusedDataset(sources, split="val", image_hw=tuple(cfg["model"]["image_hw"]),
                                 index_path=output / "val_samples.jsonl")
    model = PV26FocusedModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.set_train_stage(stage)
    model.to(torch.device(cfg["train"]["device"]))
    criterion = PV26FocusedLoss(model)
    step = int(checkpoint["global_step"])
    criterion.set_progress(step, int(cfg["train"]["max_steps"]))
    del checkpoint
    loader = _loader(dataset, cfg["data"], batch_size=max(1, min(int(cfg["train"]["microbatch_size"]), 4)))
    try:
        result = _evaluate(model, criterion, loader, cfg, step=step, scope="full", checkpoint=role)
        atomic_write_json(output / f"validation_full_{role}.json", result, ensure_ascii=False)
        print(json.dumps({"validation": result}, ensure_ascii=False), flush=True)
        return result
    finally:
        _stop_loader(loader)


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
        if args.initial_checkpoint is not None or args.stage is not None or args.sample_limit is not None:
            raise ValueError("resume uses the saved stage and dataset; start a new run to change them")
        cfg = read_json(output / "run_config.json")
    else:
        with _path(args.config).open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        cfg["model"]["weights"] = str(_path(cfg["model"]["weights"]))
        for source in cfg["data"]["sources"]:
            source["root"] = str(_path(source["root"]))
        cfg["data"]["sample_limit_per_source"] = args.sample_limit
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
    sampler = LogicalBatchSampler(datasets[0], batch_size=int(train_cfg["logical_batch_size"]), seed=seed)
    initial_checkpoint = cfg.get("initial_checkpoint")
    model = PV26FocusedModel(weights=None if has_checkpoint or initial_checkpoint else cfg["model"]["weights"],
        variant=cfg["model"]["variant"], roadmark_width=int(cfg["model"]["roadmark_width"])).to(device)
    model.set_train_stage(stage)
    if initial_checkpoint and not has_checkpoint:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
    criterion = PV26FocusedLoss(model)
    groups = optimizer_groups(model, train_cfg)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train_cfg["weight_decay"]))
    planned_steps = int(train_cfg["max_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=planned_steps)
    trainer = FocusedTrainer(model, criterion, optimizer, scheduler, sampler,
        FocusedTrainerConfig(output_dir=output, device=str(device), precision=precision,
            microbatch_size=int(train_cfg["microbatch_size"]), stage=stage,
            checkpoint_interval_sec=float(train_cfg["checkpoint_interval_sec"]),
            max_consecutive_failures=int(train_cfg["max_consecutive_failures"]),
            grad_clip_norm=float(train_cfg["grad_clip_norm"])), run_metadata=cfg)
    if has_checkpoint:
        trainer.load_checkpoint()
        if args.microbatch_size is not None:
            trainer.microbatch_size = args.microbatch_size
    train_loader = _loader(datasets[0], data_cfg, sampler=sampler)
    validation_data = validation_subset(datasets[1], int(train_cfg.get("validation_samples_per_source", 128)))
    val_loader = _loader(validation_data, data_cfg, batch_size=max(1, min(trainer.microbatch_size, 4)))
    stop_at = min(planned_steps, trainer.global_step + args.steps) if args.steps is not None else planned_steps
    started = time.monotonic()
    starting_step = trainer.global_step
    validation = None
    roadmark_best_path = checkpoints / "best_roadmark.pt"
    roadmark_best = None
    if stage == "joint" and roadmark_best_path.is_file():
        roadmark_best = float(torch.load(roadmark_best_path, map_location="cpu", weights_only=False)["metric"])

    def validate() -> None:
        nonlocal validation, roadmark_best
        validation = _evaluate(model, criterion, val_loader, cfg, step=trainer.global_step,
                               scope="periodic", checkpoint="current")
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
        trainer.update_best(float(validation["selection_metric"]))
        print(json.dumps({"validation": validation}, ensure_ascii=False), flush=True)

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
            # Release the training optimizer and model before loading each
            # evaluation checkpoint. Final scores never enter periodic selection.
            del trainer, optimizer, scheduler, criterion, model, groups
            if device.type == "cuda":
                torch.cuda.empty_cache()
            full = {}
            for role in ("latest", "best", "best_roadmark"):
                if (checkpoints / f"{role}.pt").is_file():
                    full[role] = evaluate_checkpoint(output, cfg, role, dataset=datasets[1])
            summary["full_validation"] = full
            atomic_write_json(output / "summary.json", summary, ensure_ascii=False)
        return summary
    finally:
        _stop_loader(train_loader)
        _stop_loader(val_loader)


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
    parser.add_argument("--evaluate-only", choices=("latest", "best", "best_roadmark"),
                        help="With --resume-run, evaluate a checkpoint on its entire saved validation set.")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.evaluate_only:
        if args.resume_run is None or any(value is not None for value in
                (args.steps, args.stage, args.sample_limit, args.initial_checkpoint, args.output_dir)):
            parser.error("--evaluate-only requires --resume-run and cannot change the run or dataset")
        output = _run_output(args)
        with training_run_lock(output):
            cfg = _configuration(args, output)
            result = evaluate_checkpoint(output, cfg, args.evaluate_only)
    else:
        result = train(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
