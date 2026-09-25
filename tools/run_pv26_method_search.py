#!/usr/bin/env python3
"""Run a fixed PV26 method matrix with single-GPU successive halving."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import site
import subprocess
import sys
import time
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


def _run_logged(command: list[str], run_dir: Path, *, environment: dict[str, str] | None = None) -> None:
    """Own the trainer and its loader workers until the command has exited."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "method_search.log").open("a", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            # DataLoader workers inherit this dedicated process group. A worker
            # orphaned by a failed trainer must not outlive the search command.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(50):
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                os.killpg(process.pid, signal.SIGKILL)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def _run_to_images(
    *, run_dir: Path, config_path: Path, target_images: int,
    sample_limit: int | None, seed: int, artifact_root: Path,
) -> dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        batch_size = int(run_config["train"]["logical_batch_size"])
        summary_path = run_dir / "summary.json"
        latest = run_dir / "checkpoints/latest.pt"
        previous = run_dir / "checkpoints/previous.pt"
        completed = 0
        if latest.is_file() or previous.is_file():
            import torch
            checkpoint = torch.load(latest if latest.is_file() else previous,
                                    map_location="cpu", weights_only=False)
            completed = int(checkpoint["global_step"])
        elif summary_path.is_file() and json.loads(summary_path.read_text())["global_step"]:
            raise RuntimeError(f"run has a summary but no resumable checkpoint: {run_dir}")
        target_steps = target_images // batch_size
        additional = target_steps - completed
        if additional < 0:
            raise RuntimeError(f"run passed the requested rung without a saved comparison: {run_dir}")
        if additional > 0:
            command = [
                sys.executable, str(REPO_ROOT / "tools/run_pv26_train.py"),
                "--artifact-root", str(artifact_root),
                "--resume-run", str(run_dir), "--steps", str(additional),
            ]
        elif summary_path.is_file() and int(json.loads(summary_path.read_text())["global_step"]) == target_steps:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            raise RuntimeError(f"checkpoint reached the rung without a matching summary: {run_dir}")
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        batch_size = int(config["train"]["logical_batch_size"])
        target_steps = target_images // batch_size
        command = [
            sys.executable, str(REPO_ROOT / "tools/run_pv26_train.py"),
            "--artifact-root", str(artifact_root),
            "--config", str(config_path), "--output-dir", str(run_dir),
        ]
        if sample_limit is not None:
            command += ["--sample-limit", str(sample_limit)]
        command += ["--steps", str(target_steps), "--seed", str(seed)]
    if target_images % batch_size:
        raise ValueError(f"resource_images={target_images} is not divisible by batch={batch_size}")
    environment = dict(os.environ)
    environment["YOLOPV26_PROGRESS_EVERY"] = str(max(target_steps, 1))
    print(f"training {run_dir.name} -> {target_steps} steps", flush=True)
    _run_logged(command, run_dir, environment=environment)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary["global_step"]) != target_steps or summary.get("stopped_by_signal"):
        raise RuntimeError(f"run stopped before the requested rung: {run_dir}")
    return summary


def _common_validation(run_dir: Path, source_run: Path, resource_images: int,
                       samples_per_source: int, artifact_root: Path) -> dict[str, Any]:
    relative = Path("evaluation") / f"common_{resource_images}.json"
    result_path = run_dir / relative
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    command = [
        sys.executable, str(REPO_ROOT / "tools/run_pv26_train.py"),
        "--artifact-root", str(artifact_root),
        "--resume-run", str(run_dir), "--evaluate-only", "latest",
        "--eval-stage", "joint", "--eval-index-run", str(source_run),
        "--eval-samples-per-source", str(samples_per_source),
        "--eval-output", str(relative),
    ]
    print(f"common validation {run_dir.name}: {samples_per_source}/source", flush=True)
    _run_logged(command, run_dir)
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_search(base_config_path: Path, search_config_path: Path, output: Path,
               *, through_images: int | None = None,
               promote: list[str] | None = None,
               artifact_root: Path | None = None) -> dict[str, Any]:
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    search = yaml.safe_load(search_config_path.read_text(encoding="utf-8"))
    experiment = search["experiment"]
    _merge(base, experiment.get("base_overrides", {}))
    artifact_root = (artifact_root or REPO_ROOT / "runs").expanduser().resolve()
    if not artifact_root.is_dir():
        raise RuntimeError(f"artifact root is unavailable: {artifact_root}")
    output = output.expanduser().resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError(f"search output must be under {artifact_root}")
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
        config["data"]["seed"] = int(candidate.get("seed", experiment["seed"]))
        sample_limit = experiment.get("sample_limit_per_source")
        config["data"]["sample_limit_per_source"] = int(sample_limit) if sample_limit is not None else None
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
        if path.is_file():
            if yaml.safe_load(path.read_text(encoding="utf-8")) != config:
                raise RuntimeError(f"candidate configuration changed within this search: {path}")
        else:
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        candidates[name] = {"config": path, "spec": candidate}

    history_path = output / "search_results.json"
    promotion_policy = str(experiment.get("promotion_policy", "successive_halving"))
    if promotion_policy not in {"successive_halving", "all"}:
        raise ValueError(f"unsupported promotion_policy: {promotion_policy}")
    ranking_metric = ("no automatic winner; compare signal and roadmark separately"
                      if promotion_policy == "all" else
                      "mean(signal_detection_total.f1, roadmark_lines_total.f1)")
    history: dict[str, Any] = {
        "experiment": experiment,
        "ranking_metric": ranking_metric,
        "rungs": [],
    }
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history.get("experiment") != experiment or history.get("ranking_metric") != ranking_metric:
            raise RuntimeError(f"search settings differ from saved results: {history_path}")
    if promote is not None:
        if not history["rungs"] or int(history["rungs"][-1]["resource_images"]) == int(experiment["resource_images"][-1]):
            raise ValueError("promotion requires a completed nonfinal rung")
        last = history["rungs"][-1]
        if through_images is not None and through_images <= int(last["resource_images"]):
            raise ValueError("promotion must precede a later rung")
        eligible = {row["name"] for row in last["results"] if row["status"] == "complete"}
        if not promote or len(set(promote)) != len(promote) or not set(promote) <= eligible:
            raise ValueError("promoted candidates must be distinct completed runs from the last rung")
        last["promoted"] = list(promote)
        atomic_write_json(history_path, history, ensure_ascii=False)

    active = list(candidates)
    completed_rungs = {int(row["resource_images"]): row for row in history.get("rungs", [])}
    for rung_index, resource_images in enumerate(experiment["resource_images"]):
        resource_images = int(resource_images)
        if through_images is not None and resource_images > through_images:
            break
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
                    sample_limit=experiment.get("sample_limit_per_source"),
                    seed=int(candidates[name]["spec"].get("seed", experiment["seed"])),
                    artifact_root=artifact_root,
                )
                source_run = experiment.get("common_eval_run")
                validation = (_common_validation(run_dir,
                    (REPO_ROOT / source_run).resolve(), resource_images,
                    int(experiment["validation_samples_per_source"]), artifact_root)
                    if source_run else summary["validation"])
                result = {
                    "name": name,
                    "status": "complete",
                    "resource_images": resource_images,
                    "global_step": int(summary["global_step"]),
                    "seed": int(candidates[name]["spec"].get("seed", experiment["seed"])),
                    "factors": dict(candidates[name]["spec"].get("factors") or {}),
                    "signal_f1": float(validation["signal_detection_total"]["f1"]),
                    "roadmark_line_f1": float(validation["roadmark_lines_total"]["f1"]),
                    "roadmark_pixel_f1": float(validation["roadmark_pixels_total"]["f1"]),
                    "elapsed_sec": float(summary["elapsed_sec"]),
                    "run_dir": str(run_dir),
                }
                if promotion_policy != "all":
                    result["score"] = _score(validation)
                results.append(result)
                print(f"{name}: signal={result['signal_f1']:.4f} "
                      f"roadmark={result['roadmark_line_f1']:.4f}", flush=True)
            except subprocess.CalledProcessError as error:
                results.append({
                    "name": name, "status": "failed", "resource_images": resource_images,
                    "returncode": error.returncode, "run_dir": str(run_dir),
                })
        if promotion_policy == "all":
            promoted = [row["name"] for row in results if row["status"] == "complete"]
        else:
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
    parser.add_argument("--artifact-root", type=Path,
                        help="Existing root containing the entire search output.")
    parser.add_argument("--through-images", type=int,
                        help="Complete only rungs up to this image-draw budget; resume later.")
    parser.add_argument("--promote", nargs="+",
                        help="Advance selected completed candidates after reviewing a rung.")
    args = parser.parse_args()
    search = yaml.safe_load(args.search_config.read_text(encoding="utf-8"))
    if (args.through_images is not None and args.through_images not in
            search["experiment"]["resource_images"]):
        parser.error("--through-images must name one configured rung")
    artifact_root = (args.artifact_root or REPO_ROOT / "runs").expanduser().resolve()
    output = args.output_dir or (artifact_root / search["experiment"]["name"])
    result = run_search(args.base_config.resolve(), args.search_config.resolve(), output.resolve(),
                        through_images=args.through_images, promote=args.promote,
                        artifact_root=artifact_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
