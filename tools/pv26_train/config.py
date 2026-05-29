from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common.config_coercion import (
    coerce_bool as _coerce_bool,
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
    coerce_mapping as _coerce_mapping,
    coerce_str as _coerce_str,
)
from common.paths import resolve_optional_path, resolve_path
from common.user_config import deep_merge_mappings, nested_get, resolve_repo_path, resolve_repo_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "pv26_exhaustive_od_lane_train"
DEFAULT_PRESET_NAME = "default"
BACKBONE_VARIANTS = ("n", "s")
ROADMARK_ARCHITECTURES = (
    "native",
    "v3_stopline_isolated",
    "lane_only_row_classifier",
    "stopline_only_mask_first",
)
LANE_HEAD_MODES = ("seg_first", "row_native")
LOSS_WEIGHT_NAMES = ("det", "tl_attr", "lane", "stop_line", "crosswalk")
MULTITASK_CONFLICT_TASK_NAMES = LOSS_WEIGHT_NAMES
DEFAULT_SAMPLER_RATIOS = {
    "bdd100k": 0.30,
    "aihub_traffic": 0.30,
    "aihub_lane": 0.25,
    "aihub_obstacle": 0.15,
}
PHASE_STAGE_ORDER = (
    "stage_1_frozen_trunk_warmup",
    "stage_2_partial_unfreeze",
    "stage_3_end_to_end_finetune",
    "stage_4_lane_family_finetune",
)


@dataclass(frozen=True)
class EntryConfig:
    preset_name: str = DEFAULT_PRESET_NAME


@dataclass(frozen=True)
class DatasetConfig:
    root: Path = DEFAULT_DATASET_ROOT
    additional_roots: tuple[Path, ...] = ()

    @property
    def roots(self) -> tuple[Path, ...]:
        ordered_roots: list[Path] = []
        seen: set[Path] = set()
        for path in (self.root, *self.additional_roots):
            resolved = Path(path).resolve()
            if resolved in seen:
                continue
            ordered_roots.append(resolved)
            seen.add(resolved)
        return tuple(ordered_roots)


@dataclass(frozen=True)
class RunConfig:
    run_root: Path = DEFAULT_RUN_ROOT
    run_name_prefix: str = "meta_train"
    run_dir: Path | None = None


@dataclass(frozen=True)
class TrainDefaultsConfig:
    device: str = "cuda:0"
    batch_size: int = 64
    train_batches: int = -1
    val_batches: int = -1
    trunk_lr: float = 1e-4
    head_lr: float = 5e-3
    weight_decay: float = 1e-4
    schedule: str = "cosine"
    amp: bool = False
    amp_init_scale: float = 1024.0
    accumulate_steps: int = 1
    grad_clip_norm: float = 5.0
    skip_non_finite_loss: bool = False
    oom_guard: bool = False
    checkpoint_every: int = 1
    num_workers: int = 6
    pin_memory: bool = True
    log_every_n_steps: int = 20
    profile_window: int = 20
    profile_device_sync: bool = True
    step_history_enabled: bool = True
    step_history_every_n_steps: int = 100
    step_history_include_grad_details: bool = False
    pcgrad_diagnostics_enabled: bool = True
    pcgrad_aggregate_every_n_steps: int = 100
    pcgrad_keep_raw_every_n_steps: int = 1000
    encode_train_batches_in_loader: bool = True
    encode_val_batches_in_loader: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    train_augmentation: bool = False
    train_augmentation_seed: int | None = None
    train_aug_stopline_focus_crop_prob: float = 0.0
    train_aug_stopline_focus_crop_scale_min: float = 1.25
    train_aug_stopline_focus_crop_scale_max: float = 1.75
    train_aug_stopline_focus_crop_jitter: float = 0.10
    backbone_variant: str = "s"
    backbone_weights: str | None = None
    roadmark_architecture: str = "native"
    lane_head_mode: str = "seg_first"
    sampler_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SAMPLER_RATIOS))
    task_positive_task: str | None = "multi:lane,stopline,crosswalk"
    task_positive_fraction: float | None = 0.75
    det_conf_threshold: float = 0.25
    det_iou_threshold: float = 0.70
    lane_obj_threshold: float = 0.45
    stop_line_obj_threshold: float = 0.50
    crosswalk_obj_threshold: float = 0.50
    allow_python_nms_fallback: bool = False
    task_mode: str = "roadmark_joint"
    lane_assignment_mode: str = "fixed_slot"
    lane_dynamic_coverage_weight: float = 0.0
    lane_centerline_focal_weight: float = 0.0
    lane_centerline_dice_weight: float = 0.0
    lane_segfirst_centerline_target_mode: str = "soft"
    lane_segfirst_centerline_max_positive_weight: float = 32.0
    lane_segfirst_residual_risk_core_weight: float = 0.0
    lane_segfirst_residual_risk_ring_weight: float = 0.0
    lane_segfirst_residual_risk_ring_margin: float = 0.20
    lane_segfirst_center_offset_aux_weight: float = 0.0
    lane_segfirst_task_conflict_negative_mode: str = "none"
    lane_segfirst_task_conflict_negative_weight: float = 0.0
    lane_segfirst_task_conflict_negative_margin: float = 0.15
    lane_conditional_row_aux_weight: float = 0.0
    lane_conditional_seed_aux_weight: float = 0.0
    lane_conditional_seed_target_mode: str = "centerline_core"
    lane_conditional_objectness_target_mode: str = "binary"
    lane_conditional_row_x_weight: float = 0.05
    lane_conditional_row_enabled: bool = False
    lane_family_shared_adapter_enabled: bool = False
    lane_family_task_adapter_enabled: bool = False
    lane_segfirst_track_mode: str = "component"
    lane_segfirst_max_row_gap: int = 12
    lane_segfirst_max_link_dx: float = 8.0
    lane_segfirst_seed_threshold: float = 0.50
    lane_segfirst_seed_trace_max_seeds: int = 24
    lane_segfirst_center_offset_enabled: bool = False
    lane_segfirst_center_offset_max_shift_px: float = 4.0
    lane_segfirst_center_offset_min_support_score: float = 0.50
    lane_segfirst_loss_weights: dict[str, float] = field(default_factory=dict)
    lane_segfirst_color_class_weights: dict[str, float] = field(default_factory=dict)
    stopline_local_x_aux_weight: float = 0.0
    stopline_selector_aux_weight: float = 1.0
    stopline_selector_target_mode: str = "centerline"
    stopline_geometry_aux_weight: float = 1.0
    stopline_center_target_mode: str = "union"
    stopline_centerline_target_weight: float = 1.0
    stopline_midpoint_aux_weight: float = 0.0
    stopline_haf_aux_weight: float = 0.0
    stopline_axis_distance_aux_weight: float = 0.0
    stopline_endpoint_pair_aux_weight: float = 0.0
    stopline_endpoint_pair_segment_aux_weight: float = 0.0
    stopline_endpoint_pair_verifier_aux_weight: float = 0.0
    stopline_segment_set_aux_weight: float = 0.0
    stopline_segment_verifier_aux_weight: float = 0.0
    stopline_segment_denoise_aux_weight: float = 0.0
    stopline_axis_segment_set_aux_weight: float = 0.0
    stopline_axis_segment_verifier_aux_weight: float = 0.0
    stopline_patch_segment_set_aux_weight: float = 0.0
    stopline_patch_segment_verifier_aux_weight: float = 0.0
    stopline_segment_verifier_target_mode: str = "matched_objectness"
    stopline_segment_verifier_quality_tau_px: float = 24.0
    stop_line_haf_enabled: bool = False
    stop_line_haf_valid_threshold: float = 0.50
    stop_line_haf_min_votes: int = 4
    stop_line_haf_cluster_endpoint_tolerance: float = 3.0
    stop_line_haf_max_endpoint_covariance: float = 9.0
    stop_line_haf_max_segments: int = 3
    stop_line_axis_distance_enabled: bool = False
    stop_line_axis_distance_valid_threshold: float = 0.75
    stop_line_axis_distance_min_votes: int = 3
    stop_line_axis_distance_cluster_endpoint_tolerance: float = 4.0
    stop_line_axis_distance_max_endpoint_covariance: float = 16.0
    stop_line_axis_distance_min_support_score: float = 0.35
    stop_line_axis_distance_max_segments: int = 3
    stop_line_endpoint_pair_enabled: bool = False
    stop_line_endpoint_pair_score_threshold: float = 0.55
    stop_line_endpoint_pair_topk: int = 8
    stop_line_endpoint_pair_max_segments: int = 3
    stop_line_endpoint_pair_segment_enabled: bool = False
    stop_line_endpoint_pair_segment_score_threshold: float = 0.50
    stop_line_endpoint_pair_segment_max_segments: int = 3
    stop_line_endpoint_pair_verifier_score_weight: float = 0.0
    stop_line_segment_set_enabled: bool = False
    stop_line_segment_set_score_threshold: float = 0.50
    stop_line_segment_set_max_segments: int = 3
    stop_line_segment_verifier_score_weight: float = 0.0
    stop_line_axis_segment_set_enabled: bool = False
    stop_line_axis_segment_set_score_threshold: float = 0.50
    stop_line_axis_segment_set_max_segments: int = 3
    stop_line_axis_segment_verifier_score_weight: float = 0.0
    stop_line_patch_segment_set_enabled: bool = False
    stop_line_patch_segment_set_score_threshold: float = 0.50
    stop_line_patch_segment_set_max_segments: int = 3
    stop_line_patch_segment_verifier_score_weight: float = 0.0
    stop_line_projection_comp_enabled: bool = False
    stop_line_projection_comp_proposal_source: str = "max"
    stop_line_projection_comp_min_gap: float = 4.0
    stop_line_projection_comp_topk: int = 50
    stop_line_projection_comp_union_min_score: float = 0.80
    stop_line_projection_comp_single_min_score: float = 0.90
    stop_line_projection_comp_angle_threshold_deg: float = 16.0
    stop_line_projection_comp_offset_threshold_px: float = 48.0
    stop_line_projection_comp_min_cluster_count: int = 2
    stop_line_projection_comp_projection_gap_px: float = 320.0
    stop_line_projection_comp_max_predictions: int = 2
    stop_line_projection_comp_second_min_score: float = 0.0
    stop_line_projection_comp_second_min_fragment_count: int = 5
    stop_line_projection_comp_second_min_length_ratio: float = 0.0
    stop_line_component_gate_source: str = "center"
    crosswalk_polygon_mode: str = "rect"
    distill_enabled: bool = False
    distill_teacher_checkpoint: str | None = None
    distill_task_teacher_checkpoints: dict[str, str] = field(default_factory=dict)
    distill_teacher_mode: str = "cache"
    distill_loss_weights: dict[str, float] = field(default_factory=dict)
    distill_normalize_mode: str = "none"
    distill_ema_decay: float = 0.95
    distill_ema_warmup_steps: int = 4
    distill_ema_eps: float = 1.0e-6
    task_loss_normalize_mode: str = "none"
    task_loss_normalize_tasks: tuple[str, ...] = ("lane", "stop_line", "crosswalk")
    task_loss_ema_decay: float = 0.95
    task_loss_ema_warmup_steps: int = 8
    task_loss_ema_eps: float = 1.0e-6
    task_loss_scale_min: float = 0.25
    task_loss_scale_max: float = 4.0
    multitask_conflict: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "mode": "none",
        "tasks": list(MULTITASK_CONFLICT_TASK_NAMES),
        "param_groups": ["trunk"],
    })


@dataclass(frozen=True)
class SelectionConfig:
    metric_path: str = "selection_metrics.phase_objective"
    mode: str = "max"
    eps: float = 1e-8


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool = True
    split: str = "val"
    dataset_keys: tuple[str, ...] = (
        "aihub_traffic_seoul",
        "aihub_obstacle_seoul",
        "aihub_lane_seoul",
        "bdd100k_det_100k",
    )
    max_samples_per_dataset: int = 1
    write_overlay: bool = True
    epoch_comparison_grid: bool = False
    epoch_comparison_every_n_epochs: int = 1
    epoch_comparison_sample_count: int = 12
    epoch_comparison_columns: int = 3


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    stage: str
    min_epochs: int
    max_epochs: int
    patience: int
    min_improvement_pct: float = 0.0
    min_delta_abs: float | None = None
    selection: SelectionConfig | None = None
    loss_weights: dict[str, float] = field(default_factory=dict)
    freeze_policy: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaTrainScenario:
    dataset: DatasetConfig
    run: RunConfig
    train_defaults: TrainDefaultsConfig
    selection: SelectionConfig
    preview: PreviewConfig
    phases: tuple[PhaseConfig, ...]


def phase(
    name: str,
    stage: str,
    *,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    min_improvement_pct: float = 0.0,
    min_delta_abs: float | None = None,
    selection: SelectionConfig | None = None,
    loss_weights: dict[str, float] | None = None,
    freeze_policy: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> PhaseConfig:
    return PhaseConfig(
        name=name,
        stage=stage,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        patience=patience,
        min_improvement_pct=min_improvement_pct,
        min_delta_abs=min_delta_abs,
        selection=selection,
        loss_weights=dict(loss_weights or {}),
        freeze_policy=freeze_policy,
        overrides=dict(overrides or {}),
    )


def phase_to_mapping(phase_config: PhaseConfig) -> dict[str, Any]:
    return {
        "name": phase_config.name,
        "stage": phase_config.stage,
        "min_epochs": phase_config.min_epochs,
        "max_epochs": phase_config.max_epochs,
        "patience": phase_config.patience,
        "min_improvement_pct": phase_config.min_improvement_pct,
        "min_delta_abs": phase_config.min_delta_abs,
        "selection": asdict(phase_config.selection) if phase_config.selection is not None else None,
        "loss_weights": dict(phase_config.loss_weights),
        "freeze_policy": phase_config.freeze_policy,
        "overrides": dict(phase_config.overrides),
    }


def scenario_to_mapping(scenario: MetaTrainScenario) -> dict[str, Any]:
    train_defaults = asdict(scenario.train_defaults)
    train_defaults["task_loss_normalize_tasks"] = list(scenario.train_defaults.task_loss_normalize_tasks)
    return {
        "dataset": {
            "root": str(scenario.dataset.root),
            "additional_roots": [str(path) for path in scenario.dataset.additional_roots],
        },
        "run": {
            "run_root": str(scenario.run.run_root),
            "run_name_prefix": scenario.run.run_name_prefix,
            "run_dir": str(scenario.run.run_dir) if scenario.run.run_dir is not None else None,
        },
        "train_defaults": train_defaults,
        "selection": asdict(scenario.selection),
        "preview": {
            "enabled": scenario.preview.enabled,
            "split": scenario.preview.split,
            "dataset_keys": list(scenario.preview.dataset_keys),
            "max_samples_per_dataset": scenario.preview.max_samples_per_dataset,
            "write_overlay": scenario.preview.write_overlay,
            "epoch_comparison_grid": scenario.preview.epoch_comparison_grid,
            "epoch_comparison_every_n_epochs": scenario.preview.epoch_comparison_every_n_epochs,
            "epoch_comparison_sample_count": scenario.preview.epoch_comparison_sample_count,
            "epoch_comparison_columns": scenario.preview.epoch_comparison_columns,
        },
        "phases": [phase_to_mapping(phase_config) for phase_config in scenario.phases],
    }


def build_pv26_train_path_overrides(paths_config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    legacy_stress_run_root = nested_get(paths_config, "pv26_train", "stress_run_root")
    if legacy_stress_run_root not in {None, ""}:
        raise ValueError("pv26_train.stress_run_root is no longer supported; use pv26_train.run_root")
    dataset_root = resolve_repo_path(
        nested_get(paths_config, "pv26_train", "dataset_root"),
        repo_root=repo_root,
    )
    additional_roots = resolve_repo_paths(
        nested_get(paths_config, "pv26_train", "additional_roots"),
        repo_root=repo_root,
    )
    run_root = resolve_repo_path(
        nested_get(paths_config, "pv26_train", "run_root"),
        repo_root=repo_root,
    )
    overrides: dict[str, Any] = {}
    if dataset_root is not None or additional_roots:
        overrides["dataset"] = {}
        if dataset_root is not None:
            overrides["dataset"]["root"] = str(dataset_root)
        if additional_roots:
            overrides["dataset"]["additional_roots"] = [str(path) for path in additional_roots]
    if run_root is not None:
        overrides["run"] = {"run_root": str(run_root)}
    return overrides


def apply_user_config_to_preset(
    preset_name: str,
    scenario: MetaTrainScenario,
    *,
    paths_config: dict[str, Any],
    hyperparameters_config: dict[str, Any],
    repo_root: Path,
) -> MetaTrainScenario:
    scenario_mapping = scenario_to_mapping(scenario)
    path_overrides = build_pv26_train_path_overrides(paths_config, repo_root=repo_root)
    hyperparameter_overrides = nested_get(
        hyperparameters_config,
        "pv26_train",
        "presets",
        preset_name,
        default={},
    )
    if hyperparameter_overrides is None:
        hyperparameter_overrides = {}
    if not isinstance(hyperparameter_overrides, dict):
        raise TypeError(f"pv26_train.presets.{preset_name} must be a mapping")
    merged_mapping = deep_merge_mappings(scenario_mapping, path_overrides)
    merged_mapping = deep_merge_mappings(merged_mapping, hyperparameter_overrides)
    return meta_train_scenario_from_mapping(merged_mapping, base_dir=repo_root)

def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name)


def _coerce_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, field_name=field_name)


def _coerce_optional_str(value: Any, *, field_name: str) -> str | None:
    if value in {None, ""}:
        return None
    return _coerce_str(value, field_name=field_name)


def _coerce_path_list(value: Any, *, field_name: str, base_dir: Path) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(resolve_path(item, base_dir=base_dir) for item in value)


def dataset_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> DatasetConfig:
    data = _coerce_mapping(payload, field_name="dataset")
    unknown_keys = sorted(set(data) - {"root", "additional_roots"})
    if unknown_keys:
        raise ValueError(
            "unsupported dataset config keys: "
            f"{unknown_keys}; use dataset.root and dataset.additional_roots"
        )
    additional_roots = _coerce_path_list(
        data.get("additional_roots"),
        field_name="dataset.additional_roots",
        base_dir=base_dir,
    )
    return DatasetConfig(
        root=resolve_path(data.get("root", DEFAULT_DATASET_ROOT), base_dir=base_dir),
        additional_roots=additional_roots,
    )


def run_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> RunConfig:
    data = _coerce_mapping(payload, field_name="run")
    return RunConfig(
        run_root=resolve_path(data.get("run_root", DEFAULT_RUN_ROOT), base_dir=base_dir),
        run_name_prefix=_coerce_str(data.get("run_name_prefix", "meta_train"), field_name="run.run_name_prefix"),
        run_dir=resolve_optional_path(data.get("run_dir"), base_dir=base_dir),
    )


def train_defaults_from_mapping(payload: dict[str, Any]) -> TrainDefaultsConfig:
    data = _coerce_mapping(payload, field_name="train_defaults")
    defaults = TrainDefaultsConfig()
    backbone_variant = _coerce_str(
        data.get("backbone_variant", defaults.backbone_variant),
        field_name="train_defaults.backbone_variant",
    )
    if backbone_variant not in BACKBONE_VARIANTS:
        raise ValueError(
            "train_defaults.backbone_variant must be one of "
            f"{BACKBONE_VARIANTS}, got {backbone_variant!r}"
        )
    roadmark_architecture = _coerce_str(
        data.get("roadmark_architecture", defaults.roadmark_architecture),
        field_name="train_defaults.roadmark_architecture",
    )
    if roadmark_architecture not in ROADMARK_ARCHITECTURES:
        raise ValueError(
            "train_defaults.roadmark_architecture must be one of "
            f"{ROADMARK_ARCHITECTURES}, got {roadmark_architecture!r}"
        )
    lane_head_mode = _coerce_str(
        data.get("lane_head_mode", defaults.lane_head_mode),
        field_name="train_defaults.lane_head_mode",
    )
    if lane_head_mode not in LANE_HEAD_MODES:
        raise ValueError(
            "train_defaults.lane_head_mode must be one of "
            f"{LANE_HEAD_MODES}, got {lane_head_mode!r}"
        )
    sampler_ratios_payload = _coerce_mapping(
        data.get("sampler_ratios", defaults.sampler_ratios),
        field_name="train_defaults.sampler_ratios",
    )
    sampler_ratios = {
        _coerce_str(name, field_name="train_defaults.sampler_ratios.key"): _coerce_float(
            value,
            field_name=f"train_defaults.sampler_ratios[{name!r}]",
        )
        for name, value in sampler_ratios_payload.items()
    }
    multitask_conflict_payload = _coerce_mapping(
        data.get("multitask_conflict", defaults.multitask_conflict),
        field_name="train_defaults.multitask_conflict",
    )
    lane_segfirst_loss_weights_payload = _coerce_mapping(
        data.get("lane_segfirst_loss_weights", defaults.lane_segfirst_loss_weights),
        field_name="train_defaults.lane_segfirst_loss_weights",
    )
    lane_segfirst_color_class_weights_payload = _coerce_mapping(
        data.get("lane_segfirst_color_class_weights", defaults.lane_segfirst_color_class_weights),
        field_name="train_defaults.lane_segfirst_color_class_weights",
    )
    distill_loss_weights_payload = _coerce_mapping(
        data.get("distill_loss_weights", defaults.distill_loss_weights),
        field_name="train_defaults.distill_loss_weights",
    )
    distill_task_teacher_checkpoints_payload = _coerce_mapping(
        data.get("distill_task_teacher_checkpoints", defaults.distill_task_teacher_checkpoints),
        field_name="train_defaults.distill_task_teacher_checkpoints",
    )
    task_loss_normalize_tasks_payload = data.get(
        "task_loss_normalize_tasks",
        defaults.task_loss_normalize_tasks,
    )
    if not isinstance(task_loss_normalize_tasks_payload, (list, tuple)):
        raise TypeError("train_defaults.task_loss_normalize_tasks must be a list")
    multitask_conflict_tasks = multitask_conflict_payload.get(
        "tasks",
        defaults.multitask_conflict.get("tasks", list(MULTITASK_CONFLICT_TASK_NAMES)),
    )
    if not isinstance(multitask_conflict_tasks, (list, tuple)):
        raise TypeError("train_defaults.multitask_conflict.tasks must be a list")
    multitask_conflict_param_groups = multitask_conflict_payload.get(
        "param_groups",
        defaults.multitask_conflict.get("param_groups", ["trunk"]),
    )
    if not isinstance(multitask_conflict_param_groups, (list, tuple)):
        raise TypeError("train_defaults.multitask_conflict.param_groups must be a list")
    multitask_conflict = {
        "enabled": _coerce_bool(
            multitask_conflict_payload.get("enabled", False),
            field_name="train_defaults.multitask_conflict.enabled",
        ),
        "mode": _coerce_str(
            multitask_conflict_payload.get("mode", "none"),
            field_name="train_defaults.multitask_conflict.mode",
        ),
        "tasks": [
            _coerce_str(task, field_name="train_defaults.multitask_conflict.tasks[]")
            for task in multitask_conflict_tasks
        ],
        "param_groups": [
            _coerce_str(group, field_name="train_defaults.multitask_conflict.param_groups[]")
            for group in multitask_conflict_param_groups
        ],
    }
    return TrainDefaultsConfig(
        device=_coerce_str(data.get("device", defaults.device), field_name="train_defaults.device"),
        batch_size=_coerce_int(data.get("batch_size", defaults.batch_size), field_name="train_defaults.batch_size"),
        train_batches=_coerce_int(data.get("train_batches", defaults.train_batches), field_name="train_defaults.train_batches"),
        val_batches=_coerce_int(data.get("val_batches", defaults.val_batches), field_name="train_defaults.val_batches"),
        trunk_lr=_coerce_float(data.get("trunk_lr", defaults.trunk_lr), field_name="train_defaults.trunk_lr"),
        head_lr=_coerce_float(data.get("head_lr", defaults.head_lr), field_name="train_defaults.head_lr"),
        weight_decay=_coerce_float(data.get("weight_decay", defaults.weight_decay), field_name="train_defaults.weight_decay"),
        schedule=_coerce_str(data.get("schedule", defaults.schedule), field_name="train_defaults.schedule"),
        amp=_coerce_bool(data.get("amp", defaults.amp), field_name="train_defaults.amp"),
        amp_init_scale=_coerce_float(
            data.get("amp_init_scale", defaults.amp_init_scale),
            field_name="train_defaults.amp_init_scale",
        ),
        accumulate_steps=_coerce_int(
            data.get("accumulate_steps", defaults.accumulate_steps),
            field_name="train_defaults.accumulate_steps",
        ),
        grad_clip_norm=_coerce_float(
            data.get("grad_clip_norm", defaults.grad_clip_norm),
            field_name="train_defaults.grad_clip_norm",
        ),
        skip_non_finite_loss=_coerce_bool(
            data.get("skip_non_finite_loss", defaults.skip_non_finite_loss),
            field_name="train_defaults.skip_non_finite_loss",
        ),
        oom_guard=_coerce_bool(
            data.get("oom_guard", defaults.oom_guard),
            field_name="train_defaults.oom_guard",
        ),
        checkpoint_every=_coerce_int(
            data.get("checkpoint_every", defaults.checkpoint_every),
            field_name="train_defaults.checkpoint_every",
        ),
        num_workers=_coerce_int(data.get("num_workers", defaults.num_workers), field_name="train_defaults.num_workers"),
        pin_memory=_coerce_bool(data.get("pin_memory", defaults.pin_memory), field_name="train_defaults.pin_memory"),
        log_every_n_steps=_coerce_int(
            data.get("log_every_n_steps", defaults.log_every_n_steps),
            field_name="train_defaults.log_every_n_steps",
        ),
        profile_window=_coerce_int(
            data.get("profile_window", defaults.profile_window),
            field_name="train_defaults.profile_window",
        ),
        profile_device_sync=_coerce_bool(
            data.get("profile_device_sync", defaults.profile_device_sync),
            field_name="train_defaults.profile_device_sync",
        ),
        step_history_enabled=_coerce_bool(
            data.get("step_history_enabled", defaults.step_history_enabled),
            field_name="train_defaults.step_history_enabled",
        ),
        step_history_every_n_steps=_coerce_int(
            data.get("step_history_every_n_steps", defaults.step_history_every_n_steps),
            field_name="train_defaults.step_history_every_n_steps",
        ),
        step_history_include_grad_details=_coerce_bool(
            data.get("step_history_include_grad_details", defaults.step_history_include_grad_details),
            field_name="train_defaults.step_history_include_grad_details",
        ),
        pcgrad_diagnostics_enabled=_coerce_bool(
            data.get("pcgrad_diagnostics_enabled", defaults.pcgrad_diagnostics_enabled),
            field_name="train_defaults.pcgrad_diagnostics_enabled",
        ),
        pcgrad_aggregate_every_n_steps=_coerce_int(
            data.get("pcgrad_aggregate_every_n_steps", defaults.pcgrad_aggregate_every_n_steps),
            field_name="train_defaults.pcgrad_aggregate_every_n_steps",
        ),
        pcgrad_keep_raw_every_n_steps=_coerce_int(
            data.get("pcgrad_keep_raw_every_n_steps", defaults.pcgrad_keep_raw_every_n_steps),
            field_name="train_defaults.pcgrad_keep_raw_every_n_steps",
        ),
        encode_train_batches_in_loader=_coerce_bool(
            data.get("encode_train_batches_in_loader", defaults.encode_train_batches_in_loader),
            field_name="train_defaults.encode_train_batches_in_loader",
        ),
        encode_val_batches_in_loader=_coerce_bool(
            data.get("encode_val_batches_in_loader", defaults.encode_val_batches_in_loader),
            field_name="train_defaults.encode_val_batches_in_loader",
        ),
        persistent_workers=_coerce_bool(
            data.get("persistent_workers", defaults.persistent_workers),
            field_name="train_defaults.persistent_workers",
        ),
        prefetch_factor=_coerce_optional_int(
            data.get("prefetch_factor", defaults.prefetch_factor),
            field_name="train_defaults.prefetch_factor",
        ),
        train_augmentation=_coerce_bool(
            data.get("train_augmentation", defaults.train_augmentation),
            field_name="train_defaults.train_augmentation",
        ),
        train_augmentation_seed=_coerce_optional_int(
            data.get("train_augmentation_seed", defaults.train_augmentation_seed),
            field_name="train_defaults.train_augmentation_seed",
        ),
        train_aug_stopline_focus_crop_prob=_coerce_float(
            data.get("train_aug_stopline_focus_crop_prob", defaults.train_aug_stopline_focus_crop_prob),
            field_name="train_defaults.train_aug_stopline_focus_crop_prob",
        ),
        train_aug_stopline_focus_crop_scale_min=_coerce_float(
            data.get("train_aug_stopline_focus_crop_scale_min", defaults.train_aug_stopline_focus_crop_scale_min),
            field_name="train_defaults.train_aug_stopline_focus_crop_scale_min",
        ),
        train_aug_stopline_focus_crop_scale_max=_coerce_float(
            data.get("train_aug_stopline_focus_crop_scale_max", defaults.train_aug_stopline_focus_crop_scale_max),
            field_name="train_defaults.train_aug_stopline_focus_crop_scale_max",
        ),
        train_aug_stopline_focus_crop_jitter=_coerce_float(
            data.get("train_aug_stopline_focus_crop_jitter", defaults.train_aug_stopline_focus_crop_jitter),
            field_name="train_defaults.train_aug_stopline_focus_crop_jitter",
        ),
        backbone_variant=backbone_variant,
        backbone_weights=_coerce_optional_str(
            data.get("backbone_weights", defaults.backbone_weights),
            field_name="train_defaults.backbone_weights",
        ),
        roadmark_architecture=roadmark_architecture,
        lane_head_mode=lane_head_mode,
        sampler_ratios=sampler_ratios,
        task_positive_task=_coerce_optional_str(
            data.get("task_positive_task", defaults.task_positive_task),
            field_name="train_defaults.task_positive_task",
        ),
        task_positive_fraction=_coerce_optional_float(
            data.get("task_positive_fraction", defaults.task_positive_fraction),
            field_name="train_defaults.task_positive_fraction",
        ),
        det_conf_threshold=_coerce_float(
            data.get("det_conf_threshold", defaults.det_conf_threshold),
            field_name="train_defaults.det_conf_threshold",
        ),
        det_iou_threshold=_coerce_float(
            data.get("det_iou_threshold", defaults.det_iou_threshold),
            field_name="train_defaults.det_iou_threshold",
        ),
        lane_obj_threshold=_coerce_float(
            data.get("lane_obj_threshold", defaults.lane_obj_threshold),
            field_name="train_defaults.lane_obj_threshold",
        ),
        stop_line_obj_threshold=_coerce_float(
            data.get("stop_line_obj_threshold", defaults.stop_line_obj_threshold),
            field_name="train_defaults.stop_line_obj_threshold",
        ),
        crosswalk_obj_threshold=_coerce_float(
            data.get("crosswalk_obj_threshold", defaults.crosswalk_obj_threshold),
            field_name="train_defaults.crosswalk_obj_threshold",
        ),
        allow_python_nms_fallback=_coerce_bool(
            data.get("allow_python_nms_fallback", defaults.allow_python_nms_fallback),
            field_name="train_defaults.allow_python_nms_fallback",
        ),
        task_mode=_coerce_str(
            data.get("task_mode", defaults.task_mode),
            field_name="train_defaults.task_mode",
        ),
        lane_assignment_mode=_coerce_str(
            data.get("lane_assignment_mode", defaults.lane_assignment_mode),
            field_name="train_defaults.lane_assignment_mode",
        ),
        lane_dynamic_coverage_weight=_coerce_float(
            data.get("lane_dynamic_coverage_weight", defaults.lane_dynamic_coverage_weight),
            field_name="train_defaults.lane_dynamic_coverage_weight",
        ),
        lane_centerline_focal_weight=_coerce_float(
            data.get("lane_centerline_focal_weight", defaults.lane_centerline_focal_weight),
            field_name="train_defaults.lane_centerline_focal_weight",
        ),
        lane_centerline_dice_weight=_coerce_float(
            data.get("lane_centerline_dice_weight", defaults.lane_centerline_dice_weight),
            field_name="train_defaults.lane_centerline_dice_weight",
        ),
        lane_segfirst_centerline_target_mode=_coerce_str(
            data.get("lane_segfirst_centerline_target_mode", defaults.lane_segfirst_centerline_target_mode),
            field_name="train_defaults.lane_segfirst_centerline_target_mode",
        ),
        lane_segfirst_centerline_max_positive_weight=_coerce_float(
            data.get(
                "lane_segfirst_centerline_max_positive_weight",
                defaults.lane_segfirst_centerline_max_positive_weight,
            ),
            field_name="train_defaults.lane_segfirst_centerline_max_positive_weight",
        ),
        lane_segfirst_residual_risk_core_weight=_coerce_float(
            data.get(
                "lane_segfirst_residual_risk_core_weight",
                defaults.lane_segfirst_residual_risk_core_weight,
            ),
            field_name="train_defaults.lane_segfirst_residual_risk_core_weight",
        ),
        lane_segfirst_residual_risk_ring_weight=_coerce_float(
            data.get(
                "lane_segfirst_residual_risk_ring_weight",
                defaults.lane_segfirst_residual_risk_ring_weight,
            ),
            field_name="train_defaults.lane_segfirst_residual_risk_ring_weight",
        ),
        lane_segfirst_residual_risk_ring_margin=_coerce_float(
            data.get(
                "lane_segfirst_residual_risk_ring_margin",
                defaults.lane_segfirst_residual_risk_ring_margin,
            ),
            field_name="train_defaults.lane_segfirst_residual_risk_ring_margin",
        ),
        lane_segfirst_task_conflict_negative_mode=_coerce_str(
            data.get(
                "lane_segfirst_task_conflict_negative_mode",
                defaults.lane_segfirst_task_conflict_negative_mode,
            ),
            field_name="train_defaults.lane_segfirst_task_conflict_negative_mode",
        ),
        lane_segfirst_task_conflict_negative_weight=_coerce_float(
            data.get(
                "lane_segfirst_task_conflict_negative_weight",
                defaults.lane_segfirst_task_conflict_negative_weight,
            ),
            field_name="train_defaults.lane_segfirst_task_conflict_negative_weight",
        ),
        lane_segfirst_task_conflict_negative_margin=_coerce_float(
            data.get(
                "lane_segfirst_task_conflict_negative_margin",
                defaults.lane_segfirst_task_conflict_negative_margin,
            ),
            field_name="train_defaults.lane_segfirst_task_conflict_negative_margin",
        ),
        lane_segfirst_center_offset_aux_weight=_coerce_float(
            data.get(
                "lane_segfirst_center_offset_aux_weight",
                defaults.lane_segfirst_center_offset_aux_weight,
            ),
            field_name="train_defaults.lane_segfirst_center_offset_aux_weight",
        ),
        lane_conditional_row_aux_weight=_coerce_float(
            data.get("lane_conditional_row_aux_weight", defaults.lane_conditional_row_aux_weight),
            field_name="train_defaults.lane_conditional_row_aux_weight",
        ),
        lane_conditional_seed_aux_weight=_coerce_float(
            data.get("lane_conditional_seed_aux_weight", defaults.lane_conditional_seed_aux_weight),
            field_name="train_defaults.lane_conditional_seed_aux_weight",
        ),
        lane_conditional_seed_target_mode=_coerce_str(
            data.get("lane_conditional_seed_target_mode", defaults.lane_conditional_seed_target_mode),
            field_name="train_defaults.lane_conditional_seed_target_mode",
        ),
        lane_conditional_objectness_target_mode=_coerce_str(
            data.get(
                "lane_conditional_objectness_target_mode",
                defaults.lane_conditional_objectness_target_mode,
            ),
            field_name="train_defaults.lane_conditional_objectness_target_mode",
        ),
        lane_conditional_row_x_weight=_coerce_float(
            data.get("lane_conditional_row_x_weight", defaults.lane_conditional_row_x_weight),
            field_name="train_defaults.lane_conditional_row_x_weight",
        ),
        lane_conditional_row_enabled=_coerce_bool(
            data.get("lane_conditional_row_enabled", defaults.lane_conditional_row_enabled),
            field_name="train_defaults.lane_conditional_row_enabled",
        ),
        lane_family_shared_adapter_enabled=_coerce_bool(
            data.get("lane_family_shared_adapter_enabled", defaults.lane_family_shared_adapter_enabled),
            field_name="train_defaults.lane_family_shared_adapter_enabled",
        ),
        lane_family_task_adapter_enabled=_coerce_bool(
            data.get("lane_family_task_adapter_enabled", defaults.lane_family_task_adapter_enabled),
            field_name="train_defaults.lane_family_task_adapter_enabled",
        ),
        lane_segfirst_track_mode=_coerce_str(
            data.get("lane_segfirst_track_mode", defaults.lane_segfirst_track_mode),
            field_name="train_defaults.lane_segfirst_track_mode",
        ),
        lane_segfirst_max_row_gap=_coerce_int(
            data.get("lane_segfirst_max_row_gap", defaults.lane_segfirst_max_row_gap),
            field_name="train_defaults.lane_segfirst_max_row_gap",
        ),
        lane_segfirst_max_link_dx=_coerce_float(
            data.get("lane_segfirst_max_link_dx", defaults.lane_segfirst_max_link_dx),
            field_name="train_defaults.lane_segfirst_max_link_dx",
        ),
        lane_segfirst_seed_threshold=_coerce_float(
            data.get("lane_segfirst_seed_threshold", defaults.lane_segfirst_seed_threshold),
            field_name="train_defaults.lane_segfirst_seed_threshold",
        ),
        lane_segfirst_seed_trace_max_seeds=_coerce_int(
            data.get("lane_segfirst_seed_trace_max_seeds", defaults.lane_segfirst_seed_trace_max_seeds),
            field_name="train_defaults.lane_segfirst_seed_trace_max_seeds",
        ),
        lane_segfirst_center_offset_enabled=_coerce_bool(
            data.get(
                "lane_segfirst_center_offset_enabled",
                defaults.lane_segfirst_center_offset_enabled,
            ),
            field_name="train_defaults.lane_segfirst_center_offset_enabled",
        ),
        lane_segfirst_center_offset_max_shift_px=_coerce_float(
            data.get(
                "lane_segfirst_center_offset_max_shift_px",
                defaults.lane_segfirst_center_offset_max_shift_px,
            ),
            field_name="train_defaults.lane_segfirst_center_offset_max_shift_px",
        ),
        lane_segfirst_center_offset_min_support_score=_coerce_float(
            data.get(
                "lane_segfirst_center_offset_min_support_score",
                defaults.lane_segfirst_center_offset_min_support_score,
            ),
            field_name="train_defaults.lane_segfirst_center_offset_min_support_score",
        ),
        lane_segfirst_loss_weights={
            _coerce_str(name, field_name="train_defaults.lane_segfirst_loss_weights.key"): _coerce_float(
                value,
                field_name=f"train_defaults.lane_segfirst_loss_weights.{name}",
            )
            for name, value in lane_segfirst_loss_weights_payload.items()
        },
        lane_segfirst_color_class_weights={
            _coerce_str(name, field_name="train_defaults.lane_segfirst_color_class_weights.key"): _coerce_float(
                value,
                field_name=f"train_defaults.lane_segfirst_color_class_weights.{name}",
            )
            for name, value in lane_segfirst_color_class_weights_payload.items()
        },
        stopline_local_x_aux_weight=_coerce_float(
            data.get("stopline_local_x_aux_weight", defaults.stopline_local_x_aux_weight),
            field_name="train_defaults.stopline_local_x_aux_weight",
        ),
        stopline_selector_aux_weight=_coerce_float(
            data.get("stopline_selector_aux_weight", defaults.stopline_selector_aux_weight),
            field_name="train_defaults.stopline_selector_aux_weight",
        ),
        stopline_selector_target_mode=_coerce_str(
            data.get("stopline_selector_target_mode", defaults.stopline_selector_target_mode),
            field_name="train_defaults.stopline_selector_target_mode",
        ),
        stopline_geometry_aux_weight=_coerce_float(
            data.get("stopline_geometry_aux_weight", defaults.stopline_geometry_aux_weight),
            field_name="train_defaults.stopline_geometry_aux_weight",
        ),
        stopline_center_target_mode=_coerce_str(
            data.get("stopline_center_target_mode", defaults.stopline_center_target_mode),
            field_name="train_defaults.stopline_center_target_mode",
        ),
        stopline_centerline_target_weight=_coerce_float(
            data.get("stopline_centerline_target_weight", defaults.stopline_centerline_target_weight),
            field_name="train_defaults.stopline_centerline_target_weight",
        ),
        stopline_midpoint_aux_weight=_coerce_float(
            data.get("stopline_midpoint_aux_weight", defaults.stopline_midpoint_aux_weight),
            field_name="train_defaults.stopline_midpoint_aux_weight",
        ),
        stopline_haf_aux_weight=_coerce_float(
            data.get("stopline_haf_aux_weight", defaults.stopline_haf_aux_weight),
            field_name="train_defaults.stopline_haf_aux_weight",
        ),
        stopline_axis_distance_aux_weight=_coerce_float(
            data.get("stopline_axis_distance_aux_weight", defaults.stopline_axis_distance_aux_weight),
            field_name="train_defaults.stopline_axis_distance_aux_weight",
        ),
        stopline_endpoint_pair_aux_weight=_coerce_float(
            data.get("stopline_endpoint_pair_aux_weight", defaults.stopline_endpoint_pair_aux_weight),
            field_name="train_defaults.stopline_endpoint_pair_aux_weight",
        ),
        stopline_endpoint_pair_segment_aux_weight=_coerce_float(
            data.get(
                "stopline_endpoint_pair_segment_aux_weight",
                defaults.stopline_endpoint_pair_segment_aux_weight,
            ),
            field_name="train_defaults.stopline_endpoint_pair_segment_aux_weight",
        ),
        stopline_endpoint_pair_verifier_aux_weight=_coerce_float(
            data.get(
                "stopline_endpoint_pair_verifier_aux_weight",
                defaults.stopline_endpoint_pair_verifier_aux_weight,
            ),
            field_name="train_defaults.stopline_endpoint_pair_verifier_aux_weight",
        ),
        stopline_segment_set_aux_weight=_coerce_float(
            data.get("stopline_segment_set_aux_weight", defaults.stopline_segment_set_aux_weight),
            field_name="train_defaults.stopline_segment_set_aux_weight",
        ),
        stopline_segment_verifier_aux_weight=_coerce_float(
            data.get("stopline_segment_verifier_aux_weight", defaults.stopline_segment_verifier_aux_weight),
            field_name="train_defaults.stopline_segment_verifier_aux_weight",
        ),
        stopline_segment_denoise_aux_weight=_coerce_float(
            data.get("stopline_segment_denoise_aux_weight", defaults.stopline_segment_denoise_aux_weight),
            field_name="train_defaults.stopline_segment_denoise_aux_weight",
        ),
        stopline_axis_segment_set_aux_weight=_coerce_float(
            data.get("stopline_axis_segment_set_aux_weight", defaults.stopline_axis_segment_set_aux_weight),
            field_name="train_defaults.stopline_axis_segment_set_aux_weight",
        ),
        stopline_axis_segment_verifier_aux_weight=_coerce_float(
            data.get(
                "stopline_axis_segment_verifier_aux_weight",
                defaults.stopline_axis_segment_verifier_aux_weight,
            ),
            field_name="train_defaults.stopline_axis_segment_verifier_aux_weight",
        ),
        stopline_patch_segment_set_aux_weight=_coerce_float(
            data.get("stopline_patch_segment_set_aux_weight", defaults.stopline_patch_segment_set_aux_weight),
            field_name="train_defaults.stopline_patch_segment_set_aux_weight",
        ),
        stopline_patch_segment_verifier_aux_weight=_coerce_float(
            data.get(
                "stopline_patch_segment_verifier_aux_weight",
                defaults.stopline_patch_segment_verifier_aux_weight,
            ),
            field_name="train_defaults.stopline_patch_segment_verifier_aux_weight",
        ),
        stopline_segment_verifier_target_mode=_coerce_str(
            data.get("stopline_segment_verifier_target_mode", defaults.stopline_segment_verifier_target_mode),
            field_name="train_defaults.stopline_segment_verifier_target_mode",
        ),
        stopline_segment_verifier_quality_tau_px=_coerce_float(
            data.get("stopline_segment_verifier_quality_tau_px", defaults.stopline_segment_verifier_quality_tau_px),
            field_name="train_defaults.stopline_segment_verifier_quality_tau_px",
        ),
        stop_line_haf_enabled=_coerce_bool(
            data.get("stop_line_haf_enabled", defaults.stop_line_haf_enabled),
            field_name="train_defaults.stop_line_haf_enabled",
        ),
        stop_line_haf_valid_threshold=_coerce_float(
            data.get("stop_line_haf_valid_threshold", defaults.stop_line_haf_valid_threshold),
            field_name="train_defaults.stop_line_haf_valid_threshold",
        ),
        stop_line_haf_min_votes=_coerce_int(
            data.get("stop_line_haf_min_votes", defaults.stop_line_haf_min_votes),
            field_name="train_defaults.stop_line_haf_min_votes",
        ),
        stop_line_haf_cluster_endpoint_tolerance=_coerce_float(
            data.get(
                "stop_line_haf_cluster_endpoint_tolerance",
                defaults.stop_line_haf_cluster_endpoint_tolerance,
            ),
            field_name="train_defaults.stop_line_haf_cluster_endpoint_tolerance",
        ),
        stop_line_haf_max_endpoint_covariance=_coerce_float(
            data.get("stop_line_haf_max_endpoint_covariance", defaults.stop_line_haf_max_endpoint_covariance),
            field_name="train_defaults.stop_line_haf_max_endpoint_covariance",
        ),
        stop_line_haf_max_segments=_coerce_int(
            data.get("stop_line_haf_max_segments", defaults.stop_line_haf_max_segments),
            field_name="train_defaults.stop_line_haf_max_segments",
        ),
        stop_line_axis_distance_enabled=_coerce_bool(
            data.get("stop_line_axis_distance_enabled", defaults.stop_line_axis_distance_enabled),
            field_name="train_defaults.stop_line_axis_distance_enabled",
        ),
        stop_line_axis_distance_valid_threshold=_coerce_float(
            data.get(
                "stop_line_axis_distance_valid_threshold",
                defaults.stop_line_axis_distance_valid_threshold,
            ),
            field_name="train_defaults.stop_line_axis_distance_valid_threshold",
        ),
        stop_line_axis_distance_min_votes=_coerce_int(
            data.get("stop_line_axis_distance_min_votes", defaults.stop_line_axis_distance_min_votes),
            field_name="train_defaults.stop_line_axis_distance_min_votes",
        ),
        stop_line_axis_distance_cluster_endpoint_tolerance=_coerce_float(
            data.get(
                "stop_line_axis_distance_cluster_endpoint_tolerance",
                defaults.stop_line_axis_distance_cluster_endpoint_tolerance,
            ),
            field_name="train_defaults.stop_line_axis_distance_cluster_endpoint_tolerance",
        ),
        stop_line_axis_distance_max_endpoint_covariance=_coerce_float(
            data.get(
                "stop_line_axis_distance_max_endpoint_covariance",
                defaults.stop_line_axis_distance_max_endpoint_covariance,
            ),
            field_name="train_defaults.stop_line_axis_distance_max_endpoint_covariance",
        ),
        stop_line_axis_distance_min_support_score=_coerce_float(
            data.get(
                "stop_line_axis_distance_min_support_score",
                defaults.stop_line_axis_distance_min_support_score,
            ),
            field_name="train_defaults.stop_line_axis_distance_min_support_score",
        ),
        stop_line_axis_distance_max_segments=_coerce_int(
            data.get("stop_line_axis_distance_max_segments", defaults.stop_line_axis_distance_max_segments),
            field_name="train_defaults.stop_line_axis_distance_max_segments",
        ),
        stop_line_endpoint_pair_enabled=_coerce_bool(
            data.get("stop_line_endpoint_pair_enabled", defaults.stop_line_endpoint_pair_enabled),
            field_name="train_defaults.stop_line_endpoint_pair_enabled",
        ),
        stop_line_endpoint_pair_score_threshold=_coerce_float(
            data.get(
                "stop_line_endpoint_pair_score_threshold",
                defaults.stop_line_endpoint_pair_score_threshold,
            ),
            field_name="train_defaults.stop_line_endpoint_pair_score_threshold",
        ),
        stop_line_endpoint_pair_topk=_coerce_int(
            data.get("stop_line_endpoint_pair_topk", defaults.stop_line_endpoint_pair_topk),
            field_name="train_defaults.stop_line_endpoint_pair_topk",
        ),
        stop_line_endpoint_pair_max_segments=_coerce_int(
            data.get("stop_line_endpoint_pair_max_segments", defaults.stop_line_endpoint_pair_max_segments),
            field_name="train_defaults.stop_line_endpoint_pair_max_segments",
        ),
        stop_line_endpoint_pair_segment_enabled=_coerce_bool(
            data.get(
                "stop_line_endpoint_pair_segment_enabled",
                defaults.stop_line_endpoint_pair_segment_enabled,
            ),
            field_name="train_defaults.stop_line_endpoint_pair_segment_enabled",
        ),
        stop_line_endpoint_pair_segment_score_threshold=_coerce_float(
            data.get(
                "stop_line_endpoint_pair_segment_score_threshold",
                defaults.stop_line_endpoint_pair_segment_score_threshold,
            ),
            field_name="train_defaults.stop_line_endpoint_pair_segment_score_threshold",
        ),
        stop_line_endpoint_pair_segment_max_segments=_coerce_int(
            data.get(
                "stop_line_endpoint_pair_segment_max_segments",
                defaults.stop_line_endpoint_pair_segment_max_segments,
            ),
            field_name="train_defaults.stop_line_endpoint_pair_segment_max_segments",
        ),
        stop_line_endpoint_pair_verifier_score_weight=_coerce_float(
            data.get(
                "stop_line_endpoint_pair_verifier_score_weight",
                defaults.stop_line_endpoint_pair_verifier_score_weight,
            ),
            field_name="train_defaults.stop_line_endpoint_pair_verifier_score_weight",
        ),
        stop_line_segment_set_enabled=_coerce_bool(
            data.get("stop_line_segment_set_enabled", defaults.stop_line_segment_set_enabled),
            field_name="train_defaults.stop_line_segment_set_enabled",
        ),
        stop_line_segment_set_score_threshold=_coerce_float(
            data.get("stop_line_segment_set_score_threshold", defaults.stop_line_segment_set_score_threshold),
            field_name="train_defaults.stop_line_segment_set_score_threshold",
        ),
        stop_line_segment_set_max_segments=_coerce_int(
            data.get("stop_line_segment_set_max_segments", defaults.stop_line_segment_set_max_segments),
            field_name="train_defaults.stop_line_segment_set_max_segments",
        ),
        stop_line_segment_verifier_score_weight=_coerce_float(
            data.get(
                "stop_line_segment_verifier_score_weight",
                defaults.stop_line_segment_verifier_score_weight,
            ),
            field_name="train_defaults.stop_line_segment_verifier_score_weight",
        ),
        stop_line_axis_segment_set_enabled=_coerce_bool(
            data.get("stop_line_axis_segment_set_enabled", defaults.stop_line_axis_segment_set_enabled),
            field_name="train_defaults.stop_line_axis_segment_set_enabled",
        ),
        stop_line_axis_segment_set_score_threshold=_coerce_float(
            data.get(
                "stop_line_axis_segment_set_score_threshold",
                defaults.stop_line_axis_segment_set_score_threshold,
            ),
            field_name="train_defaults.stop_line_axis_segment_set_score_threshold",
        ),
        stop_line_axis_segment_set_max_segments=_coerce_int(
            data.get("stop_line_axis_segment_set_max_segments", defaults.stop_line_axis_segment_set_max_segments),
            field_name="train_defaults.stop_line_axis_segment_set_max_segments",
        ),
        stop_line_axis_segment_verifier_score_weight=_coerce_float(
            data.get(
                "stop_line_axis_segment_verifier_score_weight",
                defaults.stop_line_axis_segment_verifier_score_weight,
            ),
            field_name="train_defaults.stop_line_axis_segment_verifier_score_weight",
        ),
        stop_line_patch_segment_set_enabled=_coerce_bool(
            data.get("stop_line_patch_segment_set_enabled", defaults.stop_line_patch_segment_set_enabled),
            field_name="train_defaults.stop_line_patch_segment_set_enabled",
        ),
        stop_line_patch_segment_set_score_threshold=_coerce_float(
            data.get(
                "stop_line_patch_segment_set_score_threshold",
                defaults.stop_line_patch_segment_set_score_threshold,
            ),
            field_name="train_defaults.stop_line_patch_segment_set_score_threshold",
        ),
        stop_line_patch_segment_set_max_segments=_coerce_int(
            data.get("stop_line_patch_segment_set_max_segments", defaults.stop_line_patch_segment_set_max_segments),
            field_name="train_defaults.stop_line_patch_segment_set_max_segments",
        ),
        stop_line_patch_segment_verifier_score_weight=_coerce_float(
            data.get(
                "stop_line_patch_segment_verifier_score_weight",
                defaults.stop_line_patch_segment_verifier_score_weight,
            ),
            field_name="train_defaults.stop_line_patch_segment_verifier_score_weight",
        ),
        stop_line_projection_comp_enabled=_coerce_bool(
            data.get("stop_line_projection_comp_enabled", defaults.stop_line_projection_comp_enabled),
            field_name="train_defaults.stop_line_projection_comp_enabled",
        ),
        stop_line_projection_comp_proposal_source=_coerce_str(
            data.get(
                "stop_line_projection_comp_proposal_source",
                defaults.stop_line_projection_comp_proposal_source,
            ),
            field_name="train_defaults.stop_line_projection_comp_proposal_source",
        ),
        stop_line_projection_comp_min_gap=_coerce_float(
            data.get("stop_line_projection_comp_min_gap", defaults.stop_line_projection_comp_min_gap),
            field_name="train_defaults.stop_line_projection_comp_min_gap",
        ),
        stop_line_projection_comp_topk=_coerce_int(
            data.get("stop_line_projection_comp_topk", defaults.stop_line_projection_comp_topk),
            field_name="train_defaults.stop_line_projection_comp_topk",
        ),
        stop_line_projection_comp_union_min_score=_coerce_float(
            data.get(
                "stop_line_projection_comp_union_min_score",
                defaults.stop_line_projection_comp_union_min_score,
            ),
            field_name="train_defaults.stop_line_projection_comp_union_min_score",
        ),
        stop_line_projection_comp_single_min_score=_coerce_float(
            data.get(
                "stop_line_projection_comp_single_min_score",
                defaults.stop_line_projection_comp_single_min_score,
            ),
            field_name="train_defaults.stop_line_projection_comp_single_min_score",
        ),
        stop_line_projection_comp_angle_threshold_deg=_coerce_float(
            data.get(
                "stop_line_projection_comp_angle_threshold_deg",
                defaults.stop_line_projection_comp_angle_threshold_deg,
            ),
            field_name="train_defaults.stop_line_projection_comp_angle_threshold_deg",
        ),
        stop_line_projection_comp_offset_threshold_px=_coerce_float(
            data.get(
                "stop_line_projection_comp_offset_threshold_px",
                defaults.stop_line_projection_comp_offset_threshold_px,
            ),
            field_name="train_defaults.stop_line_projection_comp_offset_threshold_px",
        ),
        stop_line_projection_comp_min_cluster_count=_coerce_int(
            data.get(
                "stop_line_projection_comp_min_cluster_count",
                defaults.stop_line_projection_comp_min_cluster_count,
            ),
            field_name="train_defaults.stop_line_projection_comp_min_cluster_count",
        ),
        stop_line_projection_comp_projection_gap_px=_coerce_float(
            data.get(
                "stop_line_projection_comp_projection_gap_px",
                defaults.stop_line_projection_comp_projection_gap_px,
            ),
            field_name="train_defaults.stop_line_projection_comp_projection_gap_px",
        ),
        stop_line_projection_comp_max_predictions=_coerce_int(
            data.get("stop_line_projection_comp_max_predictions", defaults.stop_line_projection_comp_max_predictions),
            field_name="train_defaults.stop_line_projection_comp_max_predictions",
        ),
        stop_line_projection_comp_second_min_score=_coerce_float(
            data.get(
                "stop_line_projection_comp_second_min_score",
                defaults.stop_line_projection_comp_second_min_score,
            ),
            field_name="train_defaults.stop_line_projection_comp_second_min_score",
        ),
        stop_line_projection_comp_second_min_fragment_count=_coerce_int(
            data.get(
                "stop_line_projection_comp_second_min_fragment_count",
                defaults.stop_line_projection_comp_second_min_fragment_count,
            ),
            field_name="train_defaults.stop_line_projection_comp_second_min_fragment_count",
        ),
        stop_line_projection_comp_second_min_length_ratio=_coerce_float(
            data.get(
                "stop_line_projection_comp_second_min_length_ratio",
                defaults.stop_line_projection_comp_second_min_length_ratio,
            ),
            field_name="train_defaults.stop_line_projection_comp_second_min_length_ratio",
        ),
        stop_line_component_gate_source=_coerce_str(
            data.get("stop_line_component_gate_source", defaults.stop_line_component_gate_source),
            field_name="train_defaults.stop_line_component_gate_source",
        ),
        crosswalk_polygon_mode=_coerce_str(
            data.get("crosswalk_polygon_mode", defaults.crosswalk_polygon_mode),
            field_name="train_defaults.crosswalk_polygon_mode",
        ),
        distill_enabled=_coerce_bool(
            data.get("distill_enabled", defaults.distill_enabled),
            field_name="train_defaults.distill_enabled",
        ),
        distill_teacher_checkpoint=_coerce_optional_str(
            data.get("distill_teacher_checkpoint", defaults.distill_teacher_checkpoint),
            field_name="train_defaults.distill_teacher_checkpoint",
        ),
        distill_task_teacher_checkpoints={
            _coerce_str(
                name,
                field_name="train_defaults.distill_task_teacher_checkpoints.key",
            ): _coerce_str(
                value,
                field_name=f"train_defaults.distill_task_teacher_checkpoints.{name}",
            )
            for name, value in distill_task_teacher_checkpoints_payload.items()
        },
        distill_teacher_mode=_coerce_str(
            data.get("distill_teacher_mode", defaults.distill_teacher_mode),
            field_name="train_defaults.distill_teacher_mode",
        ),
        distill_loss_weights={
            _coerce_str(name, field_name="train_defaults.distill_loss_weights.key"): _coerce_float(
                value,
                field_name=f"train_defaults.distill_loss_weights.{name}",
            )
            for name, value in distill_loss_weights_payload.items()
        },
        distill_normalize_mode=_coerce_str(
            data.get("distill_normalize_mode", defaults.distill_normalize_mode),
            field_name="train_defaults.distill_normalize_mode",
        ),
        distill_ema_decay=_coerce_float(
            data.get("distill_ema_decay", defaults.distill_ema_decay),
            field_name="train_defaults.distill_ema_decay",
        ),
        distill_ema_warmup_steps=_coerce_int(
            data.get("distill_ema_warmup_steps", defaults.distill_ema_warmup_steps),
            field_name="train_defaults.distill_ema_warmup_steps",
        ),
        distill_ema_eps=_coerce_float(
            data.get("distill_ema_eps", defaults.distill_ema_eps),
            field_name="train_defaults.distill_ema_eps",
        ),
        task_loss_normalize_mode=_coerce_str(
            data.get("task_loss_normalize_mode", defaults.task_loss_normalize_mode),
            field_name="train_defaults.task_loss_normalize_mode",
        ),
        task_loss_normalize_tasks=tuple(
            _coerce_str(task, field_name="train_defaults.task_loss_normalize_tasks[]")
            for task in task_loss_normalize_tasks_payload
        ),
        task_loss_ema_decay=_coerce_float(
            data.get("task_loss_ema_decay", defaults.task_loss_ema_decay),
            field_name="train_defaults.task_loss_ema_decay",
        ),
        task_loss_ema_warmup_steps=_coerce_int(
            data.get("task_loss_ema_warmup_steps", defaults.task_loss_ema_warmup_steps),
            field_name="train_defaults.task_loss_ema_warmup_steps",
        ),
        task_loss_ema_eps=_coerce_float(
            data.get("task_loss_ema_eps", defaults.task_loss_ema_eps),
            field_name="train_defaults.task_loss_ema_eps",
        ),
        task_loss_scale_min=_coerce_float(
            data.get("task_loss_scale_min", defaults.task_loss_scale_min),
            field_name="train_defaults.task_loss_scale_min",
        ),
        task_loss_scale_max=_coerce_float(
            data.get("task_loss_scale_max", defaults.task_loss_scale_max),
            field_name="train_defaults.task_loss_scale_max",
        ),
        multitask_conflict=multitask_conflict,
    )


def selection_config_from_mapping(payload: dict[str, Any]) -> SelectionConfig:
    data = _coerce_mapping(payload, field_name="selection")
    return SelectionConfig(
        metric_path=_coerce_str(data.get("metric_path", "selection_metrics.phase_objective"), field_name="selection.metric_path"),
        mode=_coerce_str(data.get("mode", "max"), field_name="selection.mode"),
        eps=_coerce_float(data.get("eps", 1e-8), field_name="selection.eps"),
    )


def preview_config_from_mapping(payload: dict[str, Any]) -> PreviewConfig:
    data = _coerce_mapping(payload, field_name="preview")
    dataset_keys = data.get("dataset_keys", PreviewConfig().dataset_keys)
    if not isinstance(dataset_keys, (list, tuple)):
        raise TypeError("preview.dataset_keys must be a list")
    return PreviewConfig(
        enabled=_coerce_bool(data.get("enabled", True), field_name="preview.enabled"),
        split=_coerce_str(data.get("split", "val"), field_name="preview.split"),
        dataset_keys=tuple(_coerce_str(value, field_name="preview.dataset_keys[]") for value in dataset_keys),
        max_samples_per_dataset=_coerce_int(
            data.get("max_samples_per_dataset", 1),
            field_name="preview.max_samples_per_dataset",
        ),
        write_overlay=_coerce_bool(data.get("write_overlay", True), field_name="preview.write_overlay"),
        epoch_comparison_grid=_coerce_bool(
            data.get("epoch_comparison_grid", False),
            field_name="preview.epoch_comparison_grid",
        ),
        epoch_comparison_every_n_epochs=_coerce_int(
            data.get("epoch_comparison_every_n_epochs", 1),
            field_name="preview.epoch_comparison_every_n_epochs",
        ),
        epoch_comparison_sample_count=_coerce_int(
            data.get("epoch_comparison_sample_count", 12),
            field_name="preview.epoch_comparison_sample_count",
        ),
        epoch_comparison_columns=_coerce_int(
            data.get("epoch_comparison_columns", 3),
            field_name="preview.epoch_comparison_columns",
        ),
    )


def phase_config_from_mapping(payload: dict[str, Any], *, index: int) -> PhaseConfig:
    data = _coerce_mapping(payload, field_name=f"phases[{index}]")
    overrides = _coerce_mapping(data.get("overrides"), field_name=f"phases[{index}].overrides")
    selection_payload = data.get("selection")
    loss_weights_payload = _coerce_mapping(
        data.get("loss_weights"),
        field_name=f"phases[{index}].loss_weights",
    )
    return PhaseConfig(
        name=_coerce_str(data.get("name", f"phase_{index + 1}"), field_name=f"phases[{index}].name"),
        stage=_coerce_str(data.get("stage"), field_name=f"phases[{index}].stage"),
        min_epochs=_coerce_int(data.get("min_epochs"), field_name=f"phases[{index}].min_epochs"),
        max_epochs=_coerce_int(data.get("max_epochs"), field_name=f"phases[{index}].max_epochs"),
        patience=_coerce_int(data.get("patience"), field_name=f"phases[{index}].patience"),
        min_improvement_pct=_coerce_float(
            data.get("min_improvement_pct", 0.0),
            field_name=f"phases[{index}].min_improvement_pct",
        ),
        min_delta_abs=_coerce_float(
            data.get("min_delta_abs"),
            field_name=f"phases[{index}].min_delta_abs",
        ) if data.get("min_delta_abs") is not None else None,
        selection=selection_config_from_mapping(selection_payload) if selection_payload is not None else None,
        loss_weights={
            _coerce_str(name, field_name=f"phases[{index}].loss_weights.key"): _coerce_float(
                value,
                field_name=f"phases[{index}].loss_weights[{name!r}]",
            )
            for name, value in loss_weights_payload.items()
        },
        freeze_policy=_coerce_optional_str(
            data.get("freeze_policy"),
            field_name=f"phases[{index}].freeze_policy",
        ),
        overrides=overrides,
    )


def meta_train_scenario_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> MetaTrainScenario:
    phases_payload = payload.get("phases", ())
    if not isinstance(phases_payload, (list, tuple)):
        raise TypeError("phases must be a list")
    return MetaTrainScenario(
        dataset=dataset_config_from_mapping(payload.get("dataset", {}), base_dir=base_dir),
        run=run_config_from_mapping(payload.get("run", {}), base_dir=base_dir),
        train_defaults=train_defaults_from_mapping(payload.get("train_defaults", {})),
        selection=selection_config_from_mapping(payload.get("selection", {})),
        preview=preview_config_from_mapping(payload.get("preview", {})),
        phases=tuple(
            phase_config_from_mapping(phase_payload, index=index)
            for index, phase_payload in enumerate(phases_payload)
        ),
    )


def resolve_train_batch_limit(value: int) -> int | None:
    if int(value) <= 0:
        return None
    return int(value)


def resolve_val_batch_limit(value: int) -> int | None:
    if int(value) < 0:
        return None
    return int(value)


def scenario_phase_defaults(
    defaults: TrainDefaultsConfig,
    overrides: dict[str, Any],
) -> TrainDefaultsConfig:
    allowed_keys = set(TrainDefaultsConfig.__dataclass_fields__.keys())
    unknown_keys = sorted(set(overrides) - allowed_keys)
    if unknown_keys:
        raise KeyError(f"unsupported phase override keys: {unknown_keys}")
    merged = asdict(defaults)
    merged.update(overrides)
    return train_defaults_from_mapping(merged)


def resolve_phase_selection(default_selection: SelectionConfig, phase: PhaseConfig) -> SelectionConfig:
    return phase.selection if phase.selection is not None else default_selection


def validate_meta_train_scenario(
    scenario: MetaTrainScenario,
    *,
    phase_stage_order: tuple[str, ...] = PHASE_STAGE_ORDER,
) -> None:
    if len(scenario.phases) != len(phase_stage_order):
        raise ValueError(f"meta_train requires exactly {len(phase_stage_order)} phases")
    for index, (phase_config, expected_stage) in enumerate(
        zip(scenario.phases, phase_stage_order),
        start=1,
    ):
        if phase_config.stage != expected_stage:
            raise ValueError(f"phase {index} must use stage {expected_stage!r}, got {phase_config.stage!r}")
        if phase_config.min_epochs <= 0:
            raise ValueError(f"phase {index} min_epochs must be > 0")
        if phase_config.max_epochs < phase_config.min_epochs:
            raise ValueError(f"phase {index} max_epochs must be >= min_epochs")
        if phase_config.patience <= 0:
            raise ValueError(f"phase {index} patience must be > 0")
        if phase_config.min_improvement_pct < 0.0:
            raise ValueError(f"phase {index} min_improvement_pct must be >= 0")
        if phase_config.min_delta_abs is not None and float(phase_config.min_delta_abs) < 0.0:
            raise ValueError(f"phase {index} min_delta_abs must be >= 0")
        unknown_loss_weight_names = sorted(set(phase_config.loss_weights) - set(LOSS_WEIGHT_NAMES))
        if unknown_loss_weight_names:
            raise ValueError(
                f"phase {index} uses unsupported loss weight names: {unknown_loss_weight_names}"
            )
        for name, value in phase_config.loss_weights.items():
            if float(value) < 0.0:
                raise ValueError(f"phase {index} loss weight {name!r} must be >= 0")
        phase_train = scenario_phase_defaults(scenario.train_defaults, phase_config.overrides)
        phase_selection = resolve_phase_selection(scenario.selection, phase_config)
        if phase_selection.mode not in {"min", "max"}:
            raise ValueError(f"phase {index} selection.mode must be 'min' or 'max'")
        if phase_selection.eps <= 0.0:
            raise ValueError(f"phase {index} selection.eps must be > 0")
        if not any(float(value) > 0.0 for value in phase_train.sampler_ratios.values()):
            raise ValueError(f"phase {index} sampler_ratios must contain at least one positive value")
        if phase_train.task_positive_fraction is not None and not 0.0 <= float(phase_train.task_positive_fraction) <= 1.0:
            raise ValueError(f"phase {index} task_positive_fraction must be between 0 and 1")
        if float(phase_train.amp_init_scale) <= 0.0:
            raise ValueError(f"phase {index} amp_init_scale must be > 0")
        multitask_conflict = dict(phase_train.multitask_conflict)
        conflict_mode = str(multitask_conflict.get("mode", "none"))
        if conflict_mode not in {"none", "pcgrad_style"}:
            raise ValueError(f"phase {index} multitask_conflict.mode must be one of: none, pcgrad_style")
        conflict_tasks = multitask_conflict.get("tasks", ())
        if not isinstance(conflict_tasks, (list, tuple)):
            raise TypeError(f"phase {index} multitask_conflict.tasks must be a list")
        unknown_conflict_tasks = sorted(set(str(task) for task in conflict_tasks) - set(MULTITASK_CONFLICT_TASK_NAMES))
        if unknown_conflict_tasks:
            raise ValueError(
                f"phase {index} multitask_conflict uses unsupported task names: {unknown_conflict_tasks}"
            )
        conflict_param_groups = multitask_conflict.get("param_groups", ("trunk",))
        if not isinstance(conflict_param_groups, (list, tuple)):
            raise TypeError(f"phase {index} multitask_conflict.param_groups must be a list")
        unknown_conflict_param_groups = sorted(
            set(str(group) for group in conflict_param_groups) - {"trunk", "heads", "lane_family_adapters"}
        )
        if unknown_conflict_param_groups:
            raise ValueError(
                f"phase {index} multitask_conflict uses unsupported param groups: {unknown_conflict_param_groups}"
            )
        if phase_train.distill_teacher_mode != "cache":
            raise ValueError(f"phase {index} distill_teacher_mode must be 'cache'")
        if phase_train.distill_normalize_mode not in {"none", "ema"}:
            raise ValueError(f"phase {index} distill_normalize_mode must be one of: none, ema")
        if phase_train.task_loss_normalize_mode not in {"none", "ema"}:
            raise ValueError(f"phase {index} task_loss_normalize_mode must be one of: none, ema")
        unknown_task_loss_normalize_tasks = sorted(
            set(str(task) for task in phase_train.task_loss_normalize_tasks)
            - {"lane", "stop_line", "crosswalk"}
        )
        if unknown_task_loss_normalize_tasks:
            raise ValueError(
                f"phase {index} task_loss_normalize_tasks uses unsupported task names: "
                f"{unknown_task_loss_normalize_tasks}"
            )
        if float(phase_train.task_loss_ema_decay) < 0.0 or float(phase_train.task_loss_ema_decay) >= 1.0:
            raise ValueError(f"phase {index} task_loss_ema_decay must be in [0, 1)")
        if int(phase_train.task_loss_ema_warmup_steps) < 0:
            raise ValueError(f"phase {index} task_loss_ema_warmup_steps must be >= 0")
        if float(phase_train.task_loss_ema_eps) <= 0.0:
            raise ValueError(f"phase {index} task_loss_ema_eps must be > 0")
        if float(phase_train.task_loss_scale_min) <= 0.0:
            raise ValueError(f"phase {index} task_loss_scale_min must be > 0")
        if float(phase_train.task_loss_scale_max) < float(phase_train.task_loss_scale_min):
            raise ValueError(f"phase {index} task_loss_scale_max must be >= task_loss_scale_min")
        if phase_train.distill_enabled and not phase_train.distill_teacher_checkpoint:
            raise ValueError(f"phase {index} distill_enabled requires distill_teacher_checkpoint")
        unknown_distill_teacher_tasks = sorted(
            set(phase_train.distill_task_teacher_checkpoints) - {"lane", "stop_line", "crosswalk"}
        )
        if unknown_distill_teacher_tasks:
            raise ValueError(
                f"phase {index} distill_task_teacher_checkpoints uses unsupported task names: "
                f"{unknown_distill_teacher_tasks}"
            )
        unknown_distill_tasks = sorted(set(phase_train.distill_loss_weights) - {"lane", "stop_line", "crosswalk"})
        if unknown_distill_tasks:
            raise ValueError(
                f"phase {index} distill_loss_weights uses unsupported task names: {unknown_distill_tasks}"
            )
        for task_name, value in phase_train.distill_loss_weights.items():
            if float(value) < 0.0:
                raise ValueError(f"phase {index} distill_loss_weights {task_name!r} must be >= 0")
        selection_requires_val = (
            phase_selection.metric_path.startswith("val.")
            or phase_selection.metric_path.startswith("selection_metrics.")
        )
        if selection_requires_val and resolve_val_batch_limit(phase_train.val_batches) == 0:
            raise ValueError(
                f"phase {index} disables validation but selection.metric_path="
                f"{phase_selection.metric_path!r} requires val"
            )
    if scenario.selection.mode not in {"min", "max"}:
        raise ValueError("selection.mode must be 'min' or 'max'")
    if scenario.selection.eps <= 0.0:
        raise ValueError("selection.eps must be > 0")
    if scenario.preview.max_samples_per_dataset <= 0:
        raise ValueError("preview.max_samples_per_dataset must be > 0")
    if scenario.preview.epoch_comparison_grid:
        if scenario.preview.epoch_comparison_every_n_epochs <= 0:
            raise ValueError("preview.epoch_comparison_every_n_epochs must be > 0")
        if scenario.preview.epoch_comparison_sample_count <= 0:
            raise ValueError("preview.epoch_comparison_sample_count must be > 0")
        if scenario.preview.epoch_comparison_columns <= 0:
            raise ValueError("preview.epoch_comparison_columns must be > 0")
