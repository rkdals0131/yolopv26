from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data_yaml import build_teacher_data_yaml, resolve_teacher_dataset_root
from .train_types import TeacherTrainScenario
from .ultralytics_runner import train_teacher_with_ultralytics


def _log_teacher_train(message: str) -> None:
    print(f"[od_bootstrap.train] {message}", flush=True)


def _build_train_params(scenario: TeacherTrainScenario) -> dict[str, Any]:
    train = scenario.train
    return {
        "epochs": train.epochs,
        "imgsz": train.imgsz,
        "batch": train.batch,
        "device": train.device,
        "workers": train.workers,
        "patience": train.patience,
        "cache": train.cache,
        "amp": train.amp,
        "optimizer": train.optimizer,
        "seed": train.seed,
        "resume": train.resume,
        "val": train.val,
        "save_period": train.save_period,
    }


def _build_runtime_params(scenario: TeacherTrainScenario) -> dict[str, Any]:
    train = scenario.train
    return {
        "pin_memory": train.pin_memory,
        "persistent_workers": train.persistent_workers,
        "prefetch_factor": train.prefetch_factor,
        "log_every_n_steps": train.log_every_n_steps,
        "profile_window": train.profile_window,
        "profile_device_sync": train.profile_device_sync,
    }


def run_teacher_train_scenario(scenario: TeacherTrainScenario, *, scenario_path: Path) -> dict[str, Any]:
    dataset_root = resolve_teacher_dataset_root(
        source_root=scenario.dataset.root,
        image_dir=scenario.dataset.image_dir,
        label_dir=scenario.dataset.label_dir,
        train_split=scenario.dataset.train_split,
        val_split=scenario.dataset.val_split,
    )
    data_yaml_path = build_teacher_data_yaml(
        dataset_root=dataset_root,
        class_names=scenario.model.class_names,
        output_path=scenario.run.output_root / scenario.teacher_name / "data.yaml",
        train_split=scenario.dataset.train_split,
        val_split=scenario.dataset.val_split,
    )

    train_summary = train_teacher_with_ultralytics(
        teacher_name=scenario.teacher_name,
        dataset_yaml=data_yaml_path,
        output_root=scenario.run.output_root,
        model_size=scenario.model.model_size,
        weights=scenario.model.weights,
        train_params=_build_train_params(scenario),
        runtime_params=_build_runtime_params(scenario),
        exist_ok=scenario.run.exist_ok,
    )
    summary = {
        "scenario_path": str(scenario_path),
        "teacher_name": scenario.teacher_name,
        "dataset_root": str(dataset_root),
        "data_yaml_path": str(data_yaml_path),
        "class_names": list(scenario.model.class_names),
        "train": asdict(scenario.train),
        "run": asdict(scenario.run),
        "model": asdict(scenario.model),
        "train_summary": train_summary,
    }
    _log_teacher_train(f"trained {scenario.teacher_name} -> {train_summary['best_checkpoint']}")
    summary_path = scenario.run.output_root / scenario.teacher_name / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return summary


__all__ = ["run_teacher_train_scenario"]
