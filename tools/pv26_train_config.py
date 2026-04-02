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
from common.user_config import nested_get, resolve_repo_path, resolve_repo_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "pv26_exhaustive_od_lane_train"
DEFAULT_PRESET_NAME = "default"
BACKBONE_VARIANTS = ("n", "s")
LOSS_WEIGHT_NAMES = ("det", "tl_attr", "lane", "stop_line", "crosswalk")
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
    amp: bool = True
    accumulate_steps: int = 1
    grad_clip_norm: float = 5.0
    val_every: int = 1
    checkpoint_every: int = 1
    num_workers: int = 6
    pin_memory: bool = True
    log_every_n_steps: int = 20
    profile_window: int = 20
    profile_device_sync: bool = True
    encode_train_batches_in_loader: bool = True
    encode_val_batches_in_loader: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    backbone_variant: str = "s"
    backbone_weights: str | None = None
    sampler_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SAMPLER_RATIOS))
    det_conf_threshold: float = 0.25
    det_iou_threshold: float = 0.70
    lane_obj_threshold: float = 0.50
    stop_line_obj_threshold: float = 0.50
    crosswalk_obj_threshold: float = 0.50


@dataclass(frozen=True)
class SelectionConfig:
    metric_path: str = "val.losses.total.mean"
    mode: str = "min"
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


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    stage: str
    min_epochs: int
    max_epochs: int
    patience: int
    min_improvement_pct: float
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
    min_improvement_pct: float,
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
        selection=selection,
        loss_weights=dict(loss_weights or {}),
        freeze_policy=freeze_policy,
        overrides=dict(overrides or {}),
    )


def deep_merge_mappings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def phase_to_mapping(phase_config: PhaseConfig) -> dict[str, Any]:
    return {
        "name": phase_config.name,
        "stage": phase_config.stage,
        "min_epochs": phase_config.min_epochs,
        "max_epochs": phase_config.max_epochs,
        "patience": phase_config.patience,
        "min_improvement_pct": phase_config.min_improvement_pct,
        "selection": asdict(phase_config.selection) if phase_config.selection is not None else None,
        "loss_weights": dict(phase_config.loss_weights),
        "freeze_policy": phase_config.freeze_policy,
        "overrides": dict(phase_config.overrides),
    }


def scenario_to_mapping(scenario: MetaTrainScenario) -> dict[str, Any]:
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
        "train_defaults": asdict(scenario.train_defaults),
        "selection": asdict(scenario.selection),
        "preview": {
            "enabled": scenario.preview.enabled,
            "split": scenario.preview.split,
            "dataset_keys": list(scenario.preview.dataset_keys),
            "max_samples_per_dataset": scenario.preview.max_samples_per_dataset,
            "write_overlay": scenario.preview.write_overlay,
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


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value in {None, ""}:
        return None
    return _resolve_path(value, base_dir=base_dir)


def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name)


def _coerce_optional_str(value: Any, *, field_name: str) -> str | None:
    if value in {None, ""}:
        return None
    return _coerce_str(value, field_name=field_name)


def _coerce_path_list(value: Any, *, field_name: str, base_dir: Path) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(_resolve_path(item, base_dir=base_dir) for item in value)


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
        root=_resolve_path(data.get("root", DEFAULT_DATASET_ROOT), base_dir=base_dir),
        additional_roots=additional_roots,
    )


def run_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> RunConfig:
    data = _coerce_mapping(payload, field_name="run")
    return RunConfig(
        run_root=_resolve_path(data.get("run_root", DEFAULT_RUN_ROOT), base_dir=base_dir),
        run_name_prefix=_coerce_str(data.get("run_name_prefix", "meta_train"), field_name="run.run_name_prefix"),
        run_dir=_resolve_optional_path(data.get("run_dir"), base_dir=base_dir),
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
        accumulate_steps=_coerce_int(
            data.get("accumulate_steps", defaults.accumulate_steps),
            field_name="train_defaults.accumulate_steps",
        ),
        grad_clip_norm=_coerce_float(
            data.get("grad_clip_norm", defaults.grad_clip_norm),
            field_name="train_defaults.grad_clip_norm",
        ),
        val_every=_coerce_int(data.get("val_every", defaults.val_every), field_name="train_defaults.val_every"),
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
        backbone_variant=backbone_variant,
        backbone_weights=_coerce_optional_str(
            data.get("backbone_weights", defaults.backbone_weights),
            field_name="train_defaults.backbone_weights",
        ),
        sampler_ratios=sampler_ratios,
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
    )


def selection_config_from_mapping(payload: dict[str, Any]) -> SelectionConfig:
    data = _coerce_mapping(payload, field_name="selection")
    return SelectionConfig(
        metric_path=_coerce_str(data.get("metric_path", "val.losses.total.mean"), field_name="selection.metric_path"),
        mode=_coerce_str(data.get("mode", "min"), field_name="selection.mode"),
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
            data.get("min_improvement_pct"),
            field_name=f"phases[{index}].min_improvement_pct",
        ),
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
        if phase_selection.metric_path.startswith("val.") and resolve_val_batch_limit(phase_train.val_batches) == 0:
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
