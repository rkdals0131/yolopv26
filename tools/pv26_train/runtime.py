from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
from pathlib import Path
import time
from typing import Any, Callable

from .config import MetaTrainScenario, PhaseConfig, TrainDefaultsConfig


PHASE_OBJECTIVE_PATH = "selection_metrics.phase_objective"
PHASE_OBJECTIVE_POLICY_VERSION = "phase_objective_v1"
_LANE_FAMILY_SUPPORT_REFS = {
    "lane": 300.0,
    "stop_line": 80.0,
    "crosswalk": 80.0,
}
_PHASE_OBJECTIVE_WEIGHTS = {
    "stage_1_frozen_trunk_warmup": {
        "detector": 0.25,
        "traffic_light": 0.05,
        "lane": 0.45,
        "stop_line": 0.25,
        "crosswalk": 0.0,
    },
    "stage_2_partial_unfreeze": {
        "detector": 0.25,
        "traffic_light": 0.05,
        "lane": 0.40,
        "stop_line": 0.20,
        "crosswalk": 0.10,
    },
    "stage_3_end_to_end_finetune": {
        "detector": 0.30,
        "traffic_light": 0.05,
        "lane": 0.35,
        "stop_line": 0.18,
        "crosswalk": 0.12,
    },
    "stage_4_lane_family_finetune": {
        "detector": 0.0,
        "traffic_light": 0.0,
        "lane": 0.50,
        "stop_line": 0.30,
        "crosswalk": 0.20,
    },
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_tree_value(mapping: dict[str, Any], *parts: str, default: float = 0.0) -> float:
    current: Any = mapping
    for part in parts:
        if not isinstance(current, dict):
            return float(default)
        current = current.get(part)
    if isinstance(current, (int, float)):
        return float(current)
    return float(default)


def _lane_family_support(metric: dict[str, Any]) -> int:
    return int(metric.get("tp", 0)) + int(metric.get("fn", 0))


def _lane_family_reliability(metric: dict[str, Any], *, ref: float) -> float:
    support = _lane_family_support(metric)
    if support <= 0:
        return 0.0
    return min(1.0, math.sqrt(float(support) / float(ref)))


def phase_stop_policy_label(phase: PhaseConfig) -> str:
    if phase.min_delta_abs is not None:
        return f"min_delta_abs={float(phase.min_delta_abs):.4f}"
    return f"min_improvement_pct={float(phase.min_improvement_pct):.3f}"


class PhaseTransitionController:
    def __init__(
        self,
        *,
        phase: PhaseConfig,
        selection: Any,
        resolve_summary_path: Callable[[dict[str, Any], str], Any],
    ) -> None:
        self.phase = phase
        self.selection = selection
        self._resolve_summary_path = resolve_summary_path
        self.best_metric_value: float | None = None
        self.best_epoch: int | None = None
        self.best_phase_objective: float | None = None
        self.best_phase_objective_epoch: int | None = None
        self.plateau_count = 0
        self.last_improvement_abs: float | None = None
        self.last_improvement_pct: float | None = None
        self.last_stop_state: dict[str, Any] | None = None

    def _is_better(self, candidate: float) -> bool:
        if self.best_metric_value is None:
            return True
        if self.selection.mode == "min":
            return candidate < self.best_metric_value
        return candidate > self.best_metric_value

    def _relative_improvement(self, previous_best: float, current_value: float) -> float:
        denominator = max(abs(previous_best), float(self.selection.eps))
        if self.selection.mode == "min":
            return ((previous_best - current_value) / denominator) * 100.0
        return ((current_value - previous_best) / denominator) * 100.0

    def _absolute_improvement(self, previous_best: float, current_value: float) -> float:
        if self.selection.mode == "min":
            return previous_best - current_value
        return current_value - previous_best

    def _uses_absolute_delta(self) -> bool:
        return self.phase.min_delta_abs is not None

    def _current_phase_objective(self, epoch_summary: dict[str, Any]) -> float:
        selection_metrics = epoch_summary.get("selection_metrics")
        if isinstance(selection_metrics, dict) and isinstance(selection_metrics.get("phase_objective"), (int, float)):
            return float(selection_metrics["phase_objective"])
        return 0.0

    def _build_selection_metrics(self, epoch_summary: dict[str, Any]) -> dict[str, Any]:
        val_summary = epoch_summary.get("val")
        metrics = val_summary.get("metrics") if isinstance(val_summary, dict) and isinstance(val_summary.get("metrics"), dict) else {}
        detector = metrics.get("detector", {}) if isinstance(metrics.get("detector"), dict) else {}
        traffic_light = metrics.get("traffic_light", {}) if isinstance(metrics.get("traffic_light"), dict) else {}
        lane = metrics.get("lane", {}) if isinstance(metrics.get("lane"), dict) else {}
        stop_line = metrics.get("stop_line", {}) if isinstance(metrics.get("stop_line"), dict) else {}
        crosswalk = metrics.get("crosswalk", {}) if isinstance(metrics.get("crosswalk"), dict) else {}

        component_scores = {
            "detector": (
                0.70 * _float_tree_value(detector, "map50")
                + 0.30 * _float_tree_value(detector, "f1")
            ),
            "traffic_light": (
                0.60 * _float_tree_value(traffic_light, "mean_f1")
                + 0.40 * _float_tree_value(traffic_light, "combo_accuracy")
            ),
            "lane": (
                0.55 * _float_tree_value(lane, "f1")
                + 0.20 * (1.0 - _clamp01(_float_tree_value(lane, "mean_point_distance") / 40.0))
                + 0.15 * _float_tree_value(lane, "color_accuracy")
                + 0.10 * _float_tree_value(lane, "type_accuracy")
            ),
            "stop_line": (
                0.55 * _float_tree_value(stop_line, "f1")
                + 0.25 * (1.0 - _clamp01(_float_tree_value(stop_line, "mean_point_distance") / 40.0))
                + 0.20 * (1.0 - _clamp01(_float_tree_value(stop_line, "mean_angle_error") / 30.0))
            ),
            "crosswalk": (
                0.45 * _float_tree_value(crosswalk, "f1")
                + 0.35 * _float_tree_value(crosswalk, "mean_polygon_iou")
                + 0.20 * (1.0 - _clamp01(_float_tree_value(crosswalk, "mean_vertex_distance") / 60.0))
            ),
        }
        support = {
            "lane": _lane_family_support(lane),
            "stop_line": _lane_family_support(stop_line),
            "crosswalk": _lane_family_support(crosswalk),
        }
        reliability = {
            "detector": 1.0,
            "traffic_light": 1.0,
            "lane": _lane_family_reliability(lane, ref=_LANE_FAMILY_SUPPORT_REFS["lane"]),
            "stop_line": _lane_family_reliability(stop_line, ref=_LANE_FAMILY_SUPPORT_REFS["stop_line"]),
            "crosswalk": _lane_family_reliability(crosswalk, ref=_LANE_FAMILY_SUPPORT_REFS["crosswalk"]),
        }
        component_weights = dict(_PHASE_OBJECTIVE_WEIGHTS.get(self.phase.stage, {}))
        objective_total = 0.0
        effective_weight_total = 0.0
        components: dict[str, dict[str, Any]] = {}
        for component_name, weight in component_weights.items():
            score = float(component_scores.get(component_name, 0.0))
            raw_weight = float(weight)
            component_reliability = float(reliability.get(component_name, 1.0))
            if component_name in {"lane", "stop_line", "crosswalk"}:
                included = bool(raw_weight > 0.0 and component_reliability > 0.0)
                effective_weight = raw_weight * component_reliability if included else 0.0
            else:
                included = bool(raw_weight > 0.0)
                effective_weight = raw_weight if included else 0.0
            if included:
                objective_total += score * effective_weight
                effective_weight_total += effective_weight
            component_payload = {
                "score": score,
                "weight": raw_weight,
                "effective_weight": float(effective_weight),
                "reliability": component_reliability,
                "included": included,
            }
            if component_name in support:
                component_payload["support"] = int(support[component_name])
            components[component_name] = component_payload
        phase_objective = objective_total / effective_weight_total if effective_weight_total > 0.0 else 0.0
        return {
            "phase_objective": float(phase_objective),
            "policy_version": PHASE_OBJECTIVE_POLICY_VERSION,
            "components": components,
            "support": support,
            "reliability": reliability,
        }

    def annotate_epoch(self, epoch_summary: dict[str, Any]) -> None:
        selection_metrics = epoch_summary.get("selection_metrics")
        if isinstance(selection_metrics, dict) and isinstance(selection_metrics.get("phase_objective"), (int, float)):
            return
        epoch_summary["selection_metrics"] = self._build_selection_metrics(epoch_summary)

    def _phase_state(
        self,
        *,
        epoch: int,
        current_metric_value: float,
        current_phase_objective: float,
    ) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "metric_path": self.selection.metric_path,
            "metric_mode": self.selection.mode,
            "current_metric_value": float(current_metric_value),
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "current_phase_objective": float(current_phase_objective),
            "best_phase_objective": self.best_phase_objective,
            "best_phase_objective_epoch": self.best_phase_objective_epoch,
            "plateau_count": int(self.plateau_count),
            "last_improvement_abs": self.last_improvement_abs,
            "last_improvement_pct": self.last_improvement_pct,
            "selection_metric_path": self.selection.metric_path,
            "selection_mode": self.selection.mode,
            "min_epochs": int(self.phase.min_epochs),
            "max_epochs": int(self.phase.max_epochs),
            "patience": int(self.phase.patience),
            "min_improvement_pct": float(self.phase.min_improvement_pct),
            "min_delta_abs": self.phase.min_delta_abs,
            "improvement_policy": "absolute_delta" if self._uses_absolute_delta() else "relative_pct",
            "policy_version": PHASE_OBJECTIVE_POLICY_VERSION,
            "transition_eligible": bool(epoch >= self.phase.min_epochs),
        }

    def observe_epoch(self, epoch_summary: dict[str, Any]) -> dict[str, Any] | None:
        self.annotate_epoch(epoch_summary)
        epoch = int(epoch_summary["epoch"])
        current_metric = float(self._resolve_summary_path(epoch_summary, self.selection.metric_path))
        current_phase_objective = self._current_phase_objective(epoch_summary)
        if self.best_phase_objective is None or current_phase_objective > self.best_phase_objective:
            self.best_phase_objective = current_phase_objective
            self.best_phase_objective_epoch = epoch
        if self.best_metric_value is None:
            self.best_metric_value = current_metric
            self.best_epoch = epoch
            self.plateau_count = 0
            self.last_improvement_abs = None
            self.last_improvement_pct = None
        else:
            better = self._is_better(current_metric)
            if better:
                improvement_abs = self._absolute_improvement(self.best_metric_value, current_metric)
                improvement_pct = self._relative_improvement(self.best_metric_value, current_metric)
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                self.last_improvement_abs = improvement_abs
                self.last_improvement_pct = improvement_pct
                if epoch >= self.phase.min_epochs:
                    if self._uses_absolute_delta():
                        threshold_met = improvement_abs >= float(self.phase.min_delta_abs or 0.0)
                    else:
                        threshold_met = improvement_pct >= float(self.phase.min_improvement_pct)
                    if threshold_met:
                        self.plateau_count = 0
                    else:
                        self.plateau_count += 1
                else:
                    self.plateau_count = 0
            else:
                self.last_improvement_abs = 0.0
                self.last_improvement_pct = 0.0
                if epoch >= self.phase.min_epochs:
                    self.plateau_count += 1

        phase_state = self._phase_state(
            epoch=epoch,
            current_metric_value=current_metric,
            current_phase_objective=current_phase_objective,
        )
        epoch_summary["phase_transition"] = dict(phase_state)

        stop_state: dict[str, Any] | None = None
        if epoch >= self.phase.max_epochs:
            stop_state = {
                "should_stop": True,
                "reason": "max_epochs_reached",
                "phase_state": phase_state,
            }
        elif epoch >= self.phase.min_epochs and self.plateau_count >= self.phase.patience:
            stop_state = {
                "should_stop": True,
                "reason": "plateau",
                "phase_state": phase_state,
            }
        self.last_stop_state = dict(stop_state) if stop_state is not None else None
        return stop_state

    def replay(self, epoch_summaries: list[dict[str, Any]]) -> None:
        for summary in epoch_summaries:
            self.observe_epoch(summary)


def find_phase_by_stage(
    scenario: MetaTrainScenario,
    *,
    stage: str,
) -> tuple[int, PhaseConfig]:
    for phase_index, phase in enumerate(scenario.phases, start=1):
        if str(phase.stage) == str(stage):
            return phase_index, phase
    raise KeyError(f"phase stage not found in scenario: {stage}")


def is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def existing_dataset_roots(scenario: MetaTrainScenario) -> list[Path]:
    dataset_roots = list(scenario.dataset.roots)
    missing_roots = [str(path) for path in dataset_roots if not path.is_dir()]
    if missing_roots:
        raise SystemExit(f"canonical dataset roots not found: {missing_roots}")
    return dataset_roots


def normalize_selected_phase_indices(
    scenario: MetaTrainScenario,
    *,
    selected_phase_indices: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    total = len(scenario.phases)
    if selected_phase_indices is None:
        return tuple(range(1, total + 1))
    ordered: list[int] = []
    seen: set[int] = set()
    for raw_value in selected_phase_indices:
        phase_index = int(raw_value)
        if phase_index < 1 or phase_index > total:
            raise ValueError(f"selected phase index out of range: {phase_index}")
        if phase_index in seen:
            continue
        ordered.append(phase_index)
        seen.add(phase_index)
    if not ordered:
        raise ValueError("selected phase indices must not be empty")
    return tuple(ordered)


def selected_phase_window(
    scenario: MetaTrainScenario,
    *,
    selected_phase_indices: tuple[int, ...] | list[int] | None,
) -> dict[str, Any] | None:
    selected = normalize_selected_phase_indices(
        scenario,
        selected_phase_indices=selected_phase_indices,
    )
    all_indices = tuple(range(1, len(scenario.phases) + 1))
    if selected == all_indices:
        return None
    start_index = min(selected)
    end_index = max(selected)
    return {
        "selected_phase_indices": list(selected),
        "start_phase_index": int(start_index),
        "start_phase_name": scenario.phases[start_index - 1].name,
        "start_phase_stage": scenario.phases[start_index - 1].stage,
        "end_phase_index": int(end_index),
        "end_phase_name": scenario.phases[end_index - 1].name,
        "end_phase_stage": scenario.phases[end_index - 1].stage,
    }


def apply_selected_phase_window_to_manifest(
    manifest: dict[str, Any],
    *,
    selected_phase_indices: tuple[int, ...],
) -> None:
    phases = manifest.get("phases")
    if not isinstance(phases, list):
        return
    selected = set(int(value) for value in selected_phase_indices)
    for phase_index, phase_entry in enumerate(phases, start=1):
        if not isinstance(phase_entry, dict):
            continue
        if phase_index in selected:
            continue
        if str(phase_entry.get("status") or "") == "completed":
            continue
        phase_entry["status"] = "skipped"
        phase_entry["promotion_reason"] = "window_excluded"
        phase_entry["phase_state"] = None


def phase_vram_stress_train_config(
    scenario: MetaTrainScenario,
    *,
    stage: str,
    batch_size: int,
    stress_iters: int,
    scenario_phase_defaults: Callable[[TrainDefaultsConfig, Any], TrainDefaultsConfig],
) -> tuple[int, PhaseConfig, TrainDefaultsConfig]:
    phase_index, phase = find_phase_by_stage(scenario, stage=stage)
    phase_train_config = scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    phase_train_config = replace(
        phase_train_config,
        batch_size=batch_size,
        train_batches=stress_iters,
        val_batches=0,
        log_every_n_steps=1,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
    )
    return phase_index, phase, phase_train_config


def stage3_stress_train_config(
    scenario: MetaTrainScenario,
    *,
    batch_size: int,
    stress_iters: int,
    scenario_phase_defaults: Callable[[TrainDefaultsConfig, Any], TrainDefaultsConfig],
) -> tuple[int, PhaseConfig, TrainDefaultsConfig]:
    return phase_vram_stress_train_config(
        scenario,
        stage="stage_3_end_to_end_finetune",
        batch_size=batch_size,
        stress_iters=stress_iters,
        scenario_phase_defaults=scenario_phase_defaults,
    )


def run_stage3_probe(
    trainer: Any,
    train_loader: Any,
    *,
    scenario: MetaTrainScenario,
    phase_index: int,
    phase_name: str,
    phase_train_config: TrainDefaultsConfig,
    stress_iters: int,
    is_oom_error: Callable[[RuntimeError], bool],
) -> tuple[str, dict[str, Any] | None, str | None]:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(trainer.device)
    status = "ok"
    train_summary: dict[str, Any] | None = None
    error: str | None = None
    try:
        train_summary = trainer.train_epoch(
            train_loader,
            epoch=1,
            epoch_total=1,
            phase_index=phase_index,
            phase_count=len(scenario.phases),
            phase_name=phase_name,
            max_batches=stress_iters,
            step_log_path=None,
            log_every_n_steps=1,
            profile_window=phase_train_config.profile_window,
            profile_device_sync=phase_train_config.profile_device_sync,
        )
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        status = "oom"
        error = str(exc)
        torch.cuda.empty_cache()
    return status, train_summary, error


def stage3_stress_summary(
    *,
    scenario_path: Path,
    phase_index: int,
    phase: PhaseConfig,
    trainer: Any,
    train_config: TrainDefaultsConfig,
    batch_size: int,
    stress_iters: int,
    duration_sec: float,
    status: str,
    train_summary: dict[str, Any] | None,
    error: str | None,
    json_ready: Callable[[Any], Any],
    cuda_memory_stats: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "mode": "stage3_vram_stress",
        "status": status,
        "scenario_path": str(scenario_path),
        "phase_index": int(phase_index),
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "device": str(trainer.device),
        "backbone_variant": train_config.backbone_variant,
        "batch_size": int(batch_size),
        "stress_iters": int(stress_iters),
        "duration_sec": duration_sec,
        "memory": cuda_memory_stats(trainer.device),
        "train_summary": json_ready(train_summary) if train_summary is not None else None,
        "error": error,
    }
    if status == "oom":
        result["recommendation"] = "reduce batch_size and rerun the stage3 stress probe"
    return result


def phase_vram_stress_summary(
    *,
    scenario_path: Path,
    phase_index: int,
    phase: PhaseConfig,
    trainer: Any,
    train_config: TrainDefaultsConfig,
    batch_size: int,
    stress_iters: int,
    duration_sec: float,
    status: str,
    train_summary: dict[str, Any] | None,
    error: str | None,
    json_ready: Callable[[Any], Any],
    cuda_memory_stats: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    result = stage3_stress_summary(
        scenario_path=scenario_path,
        phase_index=phase_index,
        phase=phase,
        trainer=trainer,
        train_config=train_config,
        batch_size=batch_size,
        stress_iters=stress_iters,
        duration_sec=duration_sec,
        status=status,
        train_summary=train_summary,
        error=error,
        json_ready=json_ready,
        cuda_memory_stats=cuda_memory_stats,
    )
    result["mode"] = "phase_vram_stress"
    if status == "oom":
        result["recommendation"] = "reduce batch_size or choose a lighter phase and rerun the VRAM probe"
    return result


def run_stage3_vram_stress(
    scenario: MetaTrainScenario,
    *,
    scenario_path: Path,
    batch_size: int | None = None,
    stress_iters: int | None = None,
    configure_torch_multiprocessing: Callable[[], None],
    log_meta_train: Callable[[str], None],
    canonical_dataset_cls: Any,
    build_phase_train_loaders: Callable[..., tuple[Any, Any]],
    build_phase_trainer: Callable[[PhaseConfig, TrainDefaultsConfig], Any],
    scenario_phase_defaults: Callable[[TrainDefaultsConfig, Any], TrainDefaultsConfig],
    json_ready: Callable[[Any], Any],
    cuda_memory_stats: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return run_phase_vram_stress(
        scenario,
        scenario_path=scenario_path,
        stage="stage_3_end_to_end_finetune",
        batch_size=batch_size,
        stress_iters=stress_iters,
        configure_torch_multiprocessing=configure_torch_multiprocessing,
        log_meta_train=log_meta_train,
        canonical_dataset_cls=canonical_dataset_cls,
        build_phase_train_loaders=build_phase_train_loaders,
        build_phase_trainer=build_phase_trainer,
        scenario_phase_defaults=scenario_phase_defaults,
        json_ready=json_ready,
        cuda_memory_stats=cuda_memory_stats,
    )


def run_phase_vram_stress(
    scenario: MetaTrainScenario,
    *,
    scenario_path: Path,
    stage: str | None = None,
    batch_size: int | None = None,
    stress_iters: int | None = None,
    configure_torch_multiprocessing: Callable[[], None],
    log_meta_train: Callable[[str], None],
    canonical_dataset_cls: Any,
    build_phase_train_loaders: Callable[..., tuple[Any, Any]],
    build_phase_trainer: Callable[[PhaseConfig, TrainDefaultsConfig], Any],
    scenario_phase_defaults: Callable[[TrainDefaultsConfig, Any], TrainDefaultsConfig],
    json_ready: Callable[[Any], Any],
    cuda_memory_stats: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    resolved_stage = str(stage or "stage_3_end_to_end_finetune")
    configure_torch_multiprocessing()
    resolved_stress_iters = int(stress_iters) if stress_iters is not None else 12
    if resolved_stress_iters <= 0:
        raise ValueError("stress iterations must be > 0")

    dataset_roots = existing_dataset_roots(scenario)
    phase_index, phase = find_phase_by_stage(scenario, stage=resolved_stage)
    phase_defaults = scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    resolved_batch_size = int(batch_size) if batch_size is not None else int(phase_defaults.batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("stress batch size must be > 0")
    phase_index, phase, phase_train_config = phase_vram_stress_train_config(
        scenario,
        stage=resolved_stage,
        batch_size=resolved_batch_size,
        stress_iters=resolved_stress_iters,
        scenario_phase_defaults=scenario_phase_defaults,
    )

    log_meta_train(f"loading scenario for phase stress: {scenario_path}")
    log_meta_train(f"dataset roots: {[str(path) for path in dataset_roots]}")
    log_meta_train("building canonical dataset index for phase stress")
    dataset = canonical_dataset_cls(
        dataset_roots,
        train_augmentation=phase_defaults.train_augmentation,
        train_augmentation_seed=phase_defaults.train_augmentation_seed,
        progress_callback=log_meta_train,
    )
    log_meta_train(
        f"phase stress config: stage={phase.stage}, batch_size={resolved_batch_size}, "
        f"stress_iters={resolved_stress_iters}, device={phase_train_config.device}, "
        f"backbone={phase_train_config.backbone_variant}"
    )
    log_meta_train("phase stress loader override: num_workers=0, persistent_workers=False, prefetch_factor=None")
    train_loader, _ = build_phase_train_loaders(dataset, train_config=phase_train_config, phase=phase)
    trainer = build_phase_trainer(phase, phase_train_config)
    trainer.oom_guard = False
    if trainer.device.type != "cuda":
        raise SystemExit(f"phase VRAM stress requires a CUDA device, got device={trainer.device}")

    run_started_at = time.perf_counter()
    status, train_summary, error = run_stage3_probe(
        trainer,
        train_loader,
        scenario=scenario,
        phase_index=phase_index,
        phase_name=phase.name,
        phase_train_config=phase_train_config,
        stress_iters=resolved_stress_iters,
        is_oom_error=is_oom_error,
    )
    duration_sec = max(0.0, time.perf_counter() - run_started_at)
    return phase_vram_stress_summary(
        scenario_path=scenario_path,
        phase_index=phase_index,
        phase=phase,
        trainer=trainer,
        train_config=phase_train_config,
        batch_size=resolved_batch_size,
        stress_iters=resolved_stress_iters,
        duration_sec=duration_sec,
        status=status,
        train_summary=train_summary,
        error=error,
        json_ready=json_ready,
        cuda_memory_stats=cuda_memory_stats,
    )


def run_meta_train_scenario(
    scenario: MetaTrainScenario,
    *,
    scenario_path: Path,
    configure_torch_multiprocessing: Callable[[], None],
    log_meta_train: Callable[[str], None],
    canonical_dataset_cls: Any,
    resolve_meta_run_dir: Callable[..., Path],
    sample_preview_selection_with_logging: Callable[..., list[dict[str, Any]]],
    load_or_init_meta_manifest: Callable[..., tuple[dict[str, Any], Path]],
    phase_entry_is_completed: Callable[[dict[str, Any], PhaseConfig], bool],
    recover_phase_entry_from_run_dir: Callable[[dict[str, Any], PhaseConfig], dict[str, Any] | None],
    scenario_snapshot_for_run: Callable[..., dict[str, Any]],
    write_meta_manifest: Callable[[Path, dict[str, Any]], None],
    write_meta_summary: Callable[[Path, dict[str, Any]], None],
    resolve_phase_selection: Callable[[Any, PhaseConfig], Any],
    execute_phase: Callable[..., dict[str, Any]],
    phase_entry_is_terminal: Callable[[dict[str, Any], PhaseConfig], bool] | None = None,
    selected_phase_indices: tuple[int, ...] | list[int] | None = None,
    initial_best_checkpoint: Path | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configure_torch_multiprocessing()
    dataset_roots = existing_dataset_roots(scenario)
    selected_indices = normalize_selected_phase_indices(
        scenario,
        selected_phase_indices=selected_phase_indices,
    )
    phase_window = selected_phase_window(
        scenario,
        selected_phase_indices=selected_indices,
    )
    is_terminal = phase_entry_is_terminal or phase_entry_is_completed

    log_meta_train(f"loading scenario: {scenario_path}")
    log_meta_train(f"dataset roots: {[str(path) for path in dataset_roots]}")
    run_dir = resolve_meta_run_dir(scenario, scenario_path=scenario_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_meta_train(f"meta run directory: {run_dir}")
    log_meta_train("building canonical dataset index")
    dataset = canonical_dataset_cls(
        dataset_roots,
        train_augmentation=scenario.train_defaults.train_augmentation,
        train_augmentation_seed=scenario.train_defaults.train_augmentation_seed,
        progress_callback=log_meta_train,
    )
    split_counts = Counter(record.split for record in dataset.records)
    dataset_key_counts = Counter(record.dataset_key for record in dataset.records)
    log_meta_train(f"dataset split counts: {dict(sorted(split_counts.items()))}")
    log_meta_train(f"dataset key counts: {dict(sorted(dataset_key_counts.items()))}")
    log_meta_train("selecting preview samples")
    preview_samples = sample_preview_selection_with_logging(
        dataset,
        scenario.preview,
        progress_callback=log_meta_train,
    )
    log_meta_train(f"prepared {len(preview_samples)} preview samples")
    scenario_snapshot = scenario_snapshot_for_run(scenario, run_dir=run_dir)
    manifest, manifest_path = load_or_init_meta_manifest(
        scenario=scenario,
        scenario_path=scenario_path,
        run_dir=run_dir,
        meta_manifest_version="pv26-meta-train-v1",
        scenario_snapshot=scenario_snapshot,
        selected_phase_window=phase_window,
        lineage=lineage,
    )
    apply_selected_phase_window_to_manifest(
        manifest,
        selected_phase_indices=selected_indices,
    )
    if phase_window is not None:
        manifest["selected_phase_window"] = dict(phase_window)
    if lineage is not None and not isinstance(manifest.get("lineage"), dict):
        manifest["lineage"] = dict(lineage)
    if all(
        is_terminal(entry, phase)
        for entry, phase in zip(manifest["phases"], scenario.phases)
    ):
        raise SystemExit(f"selected phase window already complete: {run_dir}")
    write_meta_manifest(manifest_path, manifest)
    write_meta_summary(run_dir, manifest)

    previous_best_checkpoint: Path | None = initial_best_checkpoint
    for phase_index, phase in enumerate(scenario.phases, start=1):
        phase_entry = manifest["phases"][phase_index - 1]
        if phase_index not in selected_indices:
            if str(phase_entry.get("status") or "") != "completed":
                phase_entry["status"] = "skipped"
            continue
        if phase_entry_is_completed(phase_entry, phase):
            recovered_entry = recover_phase_entry_from_run_dir(phase_entry, phase)
            if recovered_entry is not None:
                phase_entry.update(recovered_entry)
            else:
                phase_entry["status"] = "completed"
            log_meta_train(
                f"skipping completed phase {phase_index}/{len(scenario.phases)}: "
                f"{phase.name} ({phase.stage})"
            )
            if phase_entry.get("best_checkpoint_path"):
                previous_best_checkpoint = Path(phase_entry["best_checkpoint_path"])
            continue

        manifest["status"] = "running"
        manifest["active_phase_index"] = phase_index
        manifest["active_phase_name"] = phase.name
        phase_entry["status"] = "running"
        write_meta_manifest(manifest_path, manifest)
        write_meta_summary(run_dir, manifest)
        log_meta_train(
            f"starting phase {phase_index}/{len(scenario.phases)}: "
            f"{phase.name} ({phase.stage})"
        )
        phase_selection = resolve_phase_selection(scenario.selection, phase)
        log_meta_train(
            f"phase policy: min_epochs={phase.min_epochs}, max_epochs={phase.max_epochs}, "
            f"patience={phase.patience}, {phase_stop_policy_label(phase)}, "
            f"selection={phase_selection.metric_path} ({phase_selection.mode})"
        )

        phase_result = execute_phase(
            scenario=scenario,
            scenario_path=scenario_path,
            dataset=dataset,
            preview_samples=preview_samples,
            phase_index=phase_index,
            phase=phase,
            run_dir=run_dir,
            previous_best_checkpoint=previous_best_checkpoint,
        )
        manifest["phases"][phase_index - 1].update(phase_result)
        previous_best_checkpoint = (
            Path(phase_result["best_checkpoint_path"]) if phase_result["best_checkpoint_path"] is not None else None
        )
        log_meta_train(
            f"completed phase {phase_index}/{len(scenario.phases)}: "
            f"reason={phase_result['promotion_reason']}, "
            f"completed_epochs={phase_result['completed_epochs']}, "
            f"best_checkpoint={phase_result['best_checkpoint_path']}"
        )
        manifest["active_phase_index"] = None
        manifest["active_phase_name"] = None
        write_meta_manifest(manifest_path, manifest)
        write_meta_summary(run_dir, manifest)

    manifest["status"] = "completed"
    manifest["active_phase_index"] = None
    manifest["active_phase_name"] = None
    write_meta_manifest(manifest_path, manifest)
    write_meta_summary(run_dir, manifest)
    return {
        "version": "pv26-meta-train-v1",
        "status": manifest["status"],
        "scenario_path": str(scenario_path),
        "run_dir": str(run_dir),
        "meta_manifest_path": str(manifest_path),
        "summary_path": str(run_dir / "summary.json"),
        "completed_phases": len([phase_entry for phase_entry in manifest["phases"] if phase_entry["status"] == "completed"]),
        "skipped_phases": len([phase_entry for phase_entry in manifest["phases"] if phase_entry["status"] == "skipped"]),
        "selected_phase_window": phase_window,
        "lineage": lineage,
        "phases": manifest["phases"],
        "final_checkpoint_path": previous_best_checkpoint,
    }
