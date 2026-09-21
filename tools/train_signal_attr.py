from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import site
import time

site.addsitedir(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from common.io import atomic_write_json, read_json, write_json
from common.paths import REPO_ROOT
from common.train_runtime import training_run_lock
from model.signal_attr.classifier import signal_attr_collate
from model.signal_attr.crop import SignalAttrCropConfig
from model.signal_attr.training import build_signal_attr_focused_run
from model.signal_attr.dataset import materialize_product_signal_attr_crop_dataset_from_root
from tools.pv26_train.cli import _output_directory, _stop_loader


DEFAULT_CONFIG = REPO_ROOT / "config/signal_attr.yaml"
# Historical values used by signal_config.json files written before YAML owned
# these settings. Resume must not silently adopt changed current defaults.
_LEGACY_RESUME_DEFAULTS = {
    "device": "cuda:0",
    "num_workers": 4,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "arrow_loss_weight": 1.0,
    "checkpoint_interval_sec": 600.0,
    "log_every": 20,
}


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _settings(path: Path) -> dict:
    with _path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _override(args: argparse.Namespace, name: str, fallback):
    value = getattr(args, name, None)
    return fallback if value is None else value


def _train_locked(args: argparse.Namespace, parser: argparse.ArgumentParser, output: Path) -> None:
    if args.resume_run:
        if args.initial_checkpoint is not None:
            parser.error("resume uses the saved initial checkpoint")
        cfg = read_json(output / "signal_config.json")
        if args.dataset is not None and args.dataset.resolve() != Path(cfg["dataset"]):
            parser.error("resume uses the saved crop dataset")
        for key, value in _LEGACY_RESUME_DEFAULTS.items():
            cfg.setdefault(key, value)
    else:
        if (output / "signal_config.json").exists():
            raise FileExistsError(f"run already exists; use --resume-run: {output}")
        if (output / "checkpoints").exists() or (output / "data_snapshot").exists():
            raise FileExistsError(f"run artifacts already exist; use a new output directory: {output}")
        defaults = _settings(getattr(args, "config", DEFAULT_CONFIG))["train"]
        configured_initial = defaults.get("initial_checkpoint")
        initial = _override(args, "initial_checkpoint", configured_initial)
        cfg = {
            "dataset": str(_path(args.dataset)),
            "precision": _override(args, "precision", defaults["precision"]),
            "max_steps": int(_override(args, "max_steps", defaults["max_steps"])),
            "logical_batch_size": int(_override(args, "logical_batch_size", defaults["logical_batch_size"])),
            "microbatch_size": int(_override(args, "microbatch_size", defaults["microbatch_size"])),
            "validation_every": int(_override(args, "validation_every", defaults["validation_every"])),
            "validation_samples": int(_override(args, "validation_samples", defaults["validation_samples"])),
            "seed": int(defaults["seed"]),
            "sampling": _override(args, "sampling", defaults["sampling"]),
            "selection_metric": defaults["selection_metric"],
            "device": _override(args, "device", defaults["device"]),
            "num_workers": int(_override(args, "num_workers", defaults["num_workers"])),
            "learning_rate": float(defaults["learning_rate"]),
            "weight_decay": float(defaults["weight_decay"]),
            "arrow_loss_weight": float(defaults["arrow_loss_weight"]),
            "checkpoint_interval_sec": float(defaults["checkpoint_interval_sec"]),
            "log_every": int(defaults["log_every"]),
            "initial_checkpoint": str(_path(initial)) if initial is not None else None,
        }
        atomic_write_json(output / "signal_config.json", cfg)
    torch.manual_seed(cfg["seed"])
    initial_checkpoint = Path(cfg["initial_checkpoint"]) if cfg.get("initial_checkpoint") else None
    runtime_device = _override(args, "device", cfg["device"])
    runtime_workers = int(_override(args, "num_workers", cfg["num_workers"]))
    runtime_microbatch = int(_override(args, "microbatch_size", cfg["microbatch_size"]))
    run = build_signal_attr_focused_run(Path(cfg["dataset"]), output,
        logical_batch_size=cfg["logical_batch_size"], microbatch_size=runtime_microbatch,
        num_workers=runtime_workers, seed=cfg["seed"], device=runtime_device,
        precision=cfg["precision"], initial_checkpoint=initial_checkpoint,
        sampling=cfg.get("sampling", "natural"),
        learning_rate=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]),
        arrow_loss_weight=float(cfg["arrow_loss_weight"]),
        checkpoint_interval_sec=float(cfg["checkpoint_interval_sec"]),
        resume=bool(args.resume_run))
    if args.resume_run and getattr(args, "microbatch_size", None) is not None:
        run.trainer.microbatch_size = runtime_microbatch
    total = len(run.val_loader.dataset)
    budget = int(cfg["validation_samples"])
    indices = np.linspace(0, total - 1, min(total, budget) if budget > 0 else total, dtype=int).tolist()
    run.val_loader = DataLoader(Subset(run.val_loader.dataset, indices),
        batch_size=cfg["logical_batch_size"], num_workers=runtime_workers,
        collate_fn=signal_attr_collate, pin_memory=runtime_device.startswith("cuda"))
    latest_metrics = None

    def validate():
        nonlocal latest_metrics
        validation_started = time.monotonic()
        latest_metrics = run.evaluate()
        latest_metrics["elapsed_sec"] = time.monotonic() - validation_started
        write_json(output / "validation.json", latest_metrics, ensure_ascii=False)
        run.trainer.update_best(float(latest_metrics[cfg.get("selection_metric", "combo_accuracy")]))
        print(json.dumps({"step": run.trainer.global_step, "validation": latest_metrics}, ensure_ascii=False), flush=True)

    def on_step(trainer, summary):
        progress_every = int(os.environ.get("YOLOPV26_PROGRESS_EVERY", cfg["log_every"]))
        if trainer.global_step % progress_every == 0 or trainer.global_step == 1:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        interval = int(cfg["validation_every"])
        if interval > 0 and trainer.global_step % interval == 0:
            validate()

    stop_at = min(cfg["max_steps"], run.trainer.global_step + args.steps) if args.steps else cfg["max_steps"]
    try:
        result = run.trainer.fit(run.train_loader, max_steps=stop_at,
                                 planned_steps=cfg["max_steps"], on_step=on_step)
        if not result["stopped_by_signal"]:
            validate()
            result["signal_checkpoint"] = str(run.publish_checkpoint(output / "best_signal_attr.pt"))
            result["best_weights"] = str(run.trainer.best_path)
        result["validation"] = latest_metrics
        result["run_dir"] = str(output)
        write_json(output / "summary.json", result, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        _stop_loader(run.train_loader)
        _stop_loader(run.val_loader)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the internal SignalAttr model.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--raw-root", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--all-off-policy", choices=("exclude", "off"), required=True)
    prepare.add_argument("--workers", type=int)
    prepare.add_argument("--sample-limit", type=int)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--dataset", type=Path)
    destination = train.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--resume-run", type=Path)
    train.add_argument("--device")
    train.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    train.add_argument("--max-steps", type=int)
    train.add_argument("--steps", type=int)
    train.add_argument("--logical-batch-size", type=int)
    train.add_argument("--microbatch-size", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--validation-every", type=int)
    train.add_argument("--validation-samples", type=int)
    train.add_argument("--initial-checkpoint", type=Path)
    train.add_argument("--sampling", choices=("balanced", "natural"))
    args = parser.parse_args(argv)
    if args.command == "train" and args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.command == "prepare":
        settings = _settings(args.config)["prepare"]
        raw_root = _path(_override(args, "raw_root", settings["raw_root"]))
        workers = int(_override(args, "workers", settings["workers"]))
        crop_config = SignalAttrCropConfig(**settings["crop"])
        output = _output_directory(args.output_dir)
        with training_run_lock(output):
            if (output / "meta/signal_attr_dataset_manifest.json").exists():
                raise FileExistsError(f"crop dataset already exists: {output}")
            summary = materialize_product_signal_attr_crop_dataset_from_root(
                raw_root, output, all_off_is_valid=args.all_off_policy == "off",
                crop_config=crop_config, workers=workers,
                max_samples_per_split=args.sample_limit, log_fn=print)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return

    if args.resume_run is None and args.dataset is None:
        parser.error("train requires --dataset or --resume-run")
    settings = _settings(args.config)["train"] if args.resume_run is None else None
    output = _output_directory(args.resume_run if args.resume_run else
        args.output_dir or _path(settings["output_root"]) /
        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_signal_attr"))
    with training_run_lock(output):
        _train_locked(args, parser, output)


if __name__ == "__main__":
    main()
