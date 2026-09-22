#!/usr/bin/env python3
"""Run a fixed PV26 method matrix with single-GPU successive halving."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import site
import subprocess
import sys
from typing import Any

import yaml

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from common.io import atomic_write_json
from common.paths import REPO_ROOT


DEFAULT_BASE_CONFIG = REPO_ROOT / "config/pv26.yaml"
DEFAULT_SEARCH_CONFIG = REPO_ROOT / "config/pv26_method_search.yaml"


def _merge(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _score(validation: dict[str, Any]) -> float:
    return 0.5 * (
        float(validation["signal_detection_total"]["f1"])
        + float(validation["roadmark_lines_total"]["f1"])
    )


def _run_to_images(
    *, run_dir: Path, config_path: Path, target_images: int,
    sample_limit: int, seed: int,
) -> dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        batch_size = int(run_config["train"]["logical_batch_size"])
        summary_path = run_dir / "summary.json"
        completed = 0
        if summary_path.is_file():
            completed = int(json.loads(summary_path.read_text(encoding="utf-8"))["global_step"])
        target_steps = target_images // batch_size
        additional = target_steps - completed
        if additional > 0:
            command = [
                sys.executable, str(REPO_ROOT / "tools/run_pv26_train.py"),
                "--resume-run", str(run_dir), "--steps", str(additional),
            ]
        else:
            return json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        batch_size = int(config["train"]["logical_batch_size"])
        target_steps = target_images // batch_size
        command = [
            sys.executable, str(REPO_ROOT / "tools/run_pv26_train.py"),
            "--config", str(config_path), "--output-dir", str(run_dir),
            "--sample-limit", str(sample_limit), "--steps", str(target_steps),
            "--seed", str(seed),
        ]
    if target_images % batch_size:
        raise ValueError(f"resource_images={target_images} is not divisible by batch={batch_size}")
    environment = dict(os.environ)
    environment["YOLOPV26_PROGRESS_EVERY"] = str(max(target_steps, 1))
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def run_search(base_config_path: Path, search_config_path: Path, output: Path) -> dict[str, Any]:
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    search = yaml.safe_load(search_config_path.read_text(encoding="utf-8"))
    experiment = search["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    configs = output / "configs"
    trials_root = output / "trials"
    configs.mkdir(exist_ok=True)
    trials_root.mkdir(exist_ok=True)

    candidates: dict[str, dict[str, Any]] = {}
    for candidate in search["candidates"]:
        name = str(candidate["name"])
        config = deepcopy(base)
        _merge(config, candidate.get("overrides", {}))
        if "source_weights" in candidate:
            weights = candidate["source_weights"]
            for source, weight in zip(config["data"]["sources"], weights, strict=True):
                source["weight"] = float(weight)
        config["data"]["seed"] = int(experiment["seed"])
        config["data"]["sample_limit_per_source"] = int(experiment["sample_limit_per_source"])
        config["train"]["validation_samples_per_source"] = int(
            experiment["validation_samples_per_source"]
        )
        config["train"]["validation_every"] = 0
        batch_size = int(config["train"]["logical_batch_size"])
        max_images = int(experiment["resource_images"][-1])
        if max_images % batch_size:
            raise ValueError(f"maximum resource is not divisible by batch for {name}")
        config["train"]["max_steps"] = max_images // batch_size
        path = configs / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        candidates[name] = {"config": path, "spec": candidate}

    history_path = output / "search_results.json"
    history: dict[str, Any] = {
        "experiment": experiment,
        "ranking_metric": "mean(signal_detection_total.f1, roadmark_lines_total.f1)",
        "rungs": [],
    }
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    active = list(candidates)
    completed_rungs = {int(row["resource_images"]): row for row in history.get("rungs", [])}
    for rung_index, resource_images in enumerate(experiment["resource_images"]):
        resource_images = int(resource_images)
        previous = completed_rungs.get(resource_images)
        if previous is not None:
            active = list(previous["promoted"])
            continue
        results = []
        for name in active:
            run_dir = trials_root / name
            try:
                summary = _run_to_images(
                    run_dir=run_dir, config_path=candidates[name]["config"],
                    target_images=resource_images,
                    sample_limit=int(experiment["sample_limit_per_source"]),
                    seed=int(experiment["seed"]),
                )
                validation = summary["validation"]
                results.append({
                    "name": name,
                    "status": "complete",
                    "resource_images": resource_images,
                    "global_step": int(summary["global_step"]),
                    "score": _score(validation),
                    "signal_f1": float(validation["signal_detection_total"]["f1"]),
                    "roadmark_line_f1": float(validation["roadmark_lines_total"]["f1"]),
                    "roadmark_pixel_f1": float(validation["roadmark_pixels_total"]["f1"]),
                    "elapsed_sec": float(summary["elapsed_sec"]),
                    "run_dir": str(run_dir),
                })
            except subprocess.CalledProcessError as error:
                results.append({
                    "name": name, "status": "failed", "resource_images": resource_images,
                    "returncode": error.returncode, "run_dir": str(run_dir),
                })
        ranked = sorted(
            (row for row in results if row["status"] == "complete"),
            key=lambda row: (row["score"], row["signal_f1"], row["roadmark_line_f1"]),
            reverse=True,
        )
        if rung_index + 1 < len(experiment["resource_images"]):
            keep = max(1, math.ceil(len(ranked) / int(experiment["reduction_factor"])))
            promoted = [row["name"] for row in ranked[:keep]]
        else:
            promoted = [row["name"] for row in ranked]
        history.setdefault("rungs", []).append({
            "resource_images": resource_images,
            "results": results,
            "promoted": promoted,
        })
        atomic_write_json(history_path, history, ensure_ascii=False)
        active = promoted
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    search = yaml.safe_load(args.search_config.read_text(encoding="utf-8"))
    output = args.output_dir or (REPO_ROOT / "runs" / search["experiment"]["name"])
    result = run_search(args.base_config.resolve(), args.search_config.resolve(), output.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
