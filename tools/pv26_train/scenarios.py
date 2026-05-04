from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .config import (
    DatasetConfig,
    MetaTrainScenario,
    PreviewConfig,
    RunConfig,
    SelectionConfig,
    TrainDefaultsConfig,
    apply_user_config_to_preset,
    meta_train_scenario_from_mapping,
    phase,
)


def build_meta_train_presets(
    *,
    repo_root: Path,
    default_dataset_root: Path,
    default_run_root: Path,
    load_user_paths_config: Callable[[], dict[str, Any]],
    load_user_hyperparameters_config: Callable[[], dict[str, Any]],
    nested_get: Callable[..., Any],
) -> dict[str, MetaTrainScenario]:
    paths_config = load_user_paths_config()
    hyperparameters_config = load_user_hyperparameters_config()
    preset_overrides = nested_get(hyperparameters_config, "pv26_train", "presets", default={})
    if preset_overrides is None:
        preset_overrides = {}
    if not isinstance(preset_overrides, dict):
        raise TypeError("pv26_train.presets must be a mapping")
    supported_preset_names = {
        "default",
        "pv26_unified_roadmark_native_a100",
        "pv26_unified_roadmark_segfirst_a100",
    }
    unsupported_preset_overrides = sorted(set(preset_overrides) - supported_preset_names)
    if unsupported_preset_overrides:
        raise KeyError(
            "unsupported PV26 meta-train preset overrides: "
            f"{unsupported_preset_overrides}; supported presets are {sorted(supported_preset_names)}"
        )

    preview_dataset_keys = (
        "pv26_exhaustive_bdd100k_det_100k",
        "pv26_exhaustive_aihub_traffic_seoul",
        "pv26_exhaustive_aihub_obstacle_seoul",
        "aihub_lane_seoul",
    )

    default_dataset = DatasetConfig(root=default_dataset_root)
    default_run = RunConfig(
        run_root=default_run_root,
        run_name_prefix="exhaustive_od_lane",
    )
    default_train_defaults = TrainDefaultsConfig(
        device="cuda:0",
        batch_size=40,
        train_batches=-1,
        val_batches=-1,
        trunk_lr=1e-4,
        head_lr=5e-3,
        weight_decay=1e-4,
        schedule="cosine",
        amp=False,
        amp_init_scale=1024.0,
        accumulate_steps=1,
        grad_clip_norm=5.0,
        checkpoint_every=10,
        num_workers=6,
        pin_memory=True,
        log_every_n_steps=20,
        profile_window=20,
        profile_device_sync=True,
        encode_train_batches_in_loader=True,
        encode_val_batches_in_loader=True,
        persistent_workers=True,
        prefetch_factor=2,
        backbone_variant="s",
        task_positive_task="multi:lane,stopline,crosswalk",
        task_positive_fraction=0.75,
    )
    default_selection = SelectionConfig(
        metric_path="selection_metrics.phase_objective",
        mode="max",
        eps=1e-8,
    )
    default_preview = PreviewConfig(
        enabled=True,
        split="val",
        dataset_keys=preview_dataset_keys,
        max_samples_per_dataset=1,
        write_overlay=True,
    )
    default_phases = (
        phase(
            "head_warmup",
            "stage_1_frozen_trunk_warmup",
            min_epochs=4,
            max_epochs=12,
            patience=2,
            min_improvement_pct=2.0,
            min_delta_abs=0.005,
            overrides={
                "trunk_lr": 5e-5,
                "head_lr": 3e-3,
            },
        ),
        phase(
            "partial_unfreeze",
            "stage_2_partial_unfreeze",
            min_epochs=6,
            max_epochs=18,
            patience=2,
            min_improvement_pct=0.5,
            min_delta_abs=0.004,
            overrides={
                "trunk_lr": 3e-5,
                "head_lr": 8e-4,
            },
        ),
        phase(
            "end_to_end_finetune",
            "stage_3_end_to_end_finetune",
            min_epochs=8,
            max_epochs=24,
            patience=3,
            min_improvement_pct=0.25,
            min_delta_abs=0.003,
            overrides={
                "trunk_lr": 1e-5,
                "head_lr": 4e-4,
            },
        ),
        phase(
            "lane_family_finetune",
            "stage_4_lane_family_finetune",
            min_epochs=4,
            max_epochs=12,
            patience=3,
            min_improvement_pct=0.25,
            min_delta_abs=0.003,
            loss_weights={
                "det": 0.0,
                "tl_attr": 0.0,
                "lane": 1.5,
                "stop_line": 1.25,
                "crosswalk": 1.0,
            },
            freeze_policy="lane_family_heads_only",
            overrides={
                "trunk_lr": 0.0,
                "head_lr": 2e-4,
                "task_positive_task": "multi:lane,stopline,crosswalk",
                "task_positive_fraction": 1.0,
                "sampler_ratios": {
                    "bdd100k": 0.0,
                    "aihub_traffic": 0.0,
                    "aihub_lane": 1.0,
                    "aihub_obstacle": 0.0,
                },
            },
        ),
    )

    a100_train_defaults = replace(
        default_train_defaults,
        batch_size=64,
        train_batches=-1,
        val_batches=-1,
        amp=True,
        num_workers=12,
        persistent_workers=True,
        prefetch_factor=4,
        checkpoint_every=0,
        lane_dynamic_coverage_weight=0.1,
    )
    segfirst_a100_train_defaults = replace(
        a100_train_defaults,
        batch_size=64,
        train_batches=512,
        val_batches=256,
        amp=False,
        train_augmentation=True,
        train_augmentation_seed=26,
        task_positive_task="multi:lane,stopline,crosswalk",
        task_positive_fraction=0.5,
        sampler_ratios={
            "bdd100k": 0.35,
            "aihub_traffic": 0.35,
            "aihub_lane": 0.0,
            "aihub_obstacle": 0.30,
        },
    )
    a100_run = replace(default_run, run_name_prefix="pv26_unified_roadmark_native_a100")
    segfirst_a100_run = replace(default_run, run_name_prefix="pv26_unified_roadmark_segfirst_a100")
    a100_preview = replace(
        default_preview,
        max_samples_per_dataset=3,
        epoch_comparison_grid=True,
        epoch_comparison_every_n_epochs=1,
        epoch_comparison_sample_count=12,
        epoch_comparison_columns=3,
    )
    a100_phases = tuple(
        replace(phase_config, max_epochs=max(phase_config.max_epochs, 25), patience=max(phase_config.patience, 5))
        for phase_config in default_phases
    )
    segfirst_a100_phases = (
        phase(
            "head_warmup",
            "stage_1_frozen_trunk_warmup",
            min_epochs=2,
            max_epochs=2,
            patience=2,
            min_delta_abs=0.003,
            overrides={
                "trunk_lr": 5e-5,
                "head_lr": 3e-3,
            },
        ),
        phase(
            "partial_unfreeze",
            "stage_2_partial_unfreeze",
            min_epochs=3,
            max_epochs=3,
            patience=2,
            min_delta_abs=0.003,
            overrides={
                "trunk_lr": 3e-5,
                "head_lr": 8e-4,
            },
        ),
        phase(
            "end_to_end_finetune",
            "stage_3_end_to_end_finetune",
            min_epochs=12,
            max_epochs=20,
            patience=5,
            min_delta_abs=0.002,
            overrides={
                "trunk_lr": 1e-5,
                "head_lr": 4e-4,
            },
        ),
        phase(
            "lane_family_finetune",
            "stage_4_lane_family_finetune",
            min_epochs=2,
            max_epochs=3,
            patience=2,
            min_delta_abs=0.002,
            loss_weights={
                "det": 0.0,
                "tl_attr": 0.0,
                "lane": 1.5,
                "stop_line": 1.25,
                "crosswalk": 1.0,
            },
            freeze_policy="lane_family_heads_only",
            overrides={
                "trunk_lr": 0.0,
                "head_lr": 2e-4,
                "task_positive_task": "multi:lane,stopline,crosswalk",
                "task_positive_fraction": 1.0,
                "sampler_ratios": {
                    "bdd100k": 0.0,
                    "aihub_traffic": 0.0,
                    "aihub_lane": 1.0,
                    "aihub_obstacle": 0.0,
                },
            },
        ),
    )
    presets = {
        "default": MetaTrainScenario(
            dataset=default_dataset,
            run=default_run,
            train_defaults=default_train_defaults,
            selection=default_selection,
            preview=default_preview,
            phases=default_phases,
        ),
        "pv26_unified_roadmark_native_a100": MetaTrainScenario(
            dataset=default_dataset,
            run=a100_run,
            train_defaults=a100_train_defaults,
            selection=default_selection,
            preview=a100_preview,
            phases=a100_phases,
        ),
        "pv26_unified_roadmark_segfirst_a100": MetaTrainScenario(
            dataset=default_dataset,
            run=segfirst_a100_run,
            train_defaults=segfirst_a100_train_defaults,
            selection=default_selection,
            preview=a100_preview,
            phases=segfirst_a100_phases,
        ),
    }
    return {
        preset_name: apply_user_config_to_preset(
            preset_name,
            scenario,
            paths_config=paths_config,
            hyperparameters_config=hyperparameters_config,
            repo_root=repo_root,
        )
        for preset_name, scenario in presets.items()
    }


def preset_key(preset_name: str | Path) -> str:
    return Path(preset_name).name


def default_scenario_path(
    preset_name: str | Path,
    *,
    preset_path_root: Path,
) -> Path:
    return preset_path_root / preset_key(preset_name)


def resolve_scenario_path(
    value: str | Path | None,
    *,
    fallback: Path,
    repo_root: Path,
) -> Path:
    if value in {None, ""}:
        return fallback
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = (repo_root / resolved).resolve()
    return resolved


def scenario_snapshot_for_run(
    scenario: MetaTrainScenario,
    *,
    run_dir: Path,
    scenario_to_mapping: Callable[[MetaTrainScenario], dict[str, Any]],
) -> dict[str, Any]:
    snapshot = scenario_to_mapping(scenario)
    snapshot["run"]["run_dir"] = str(run_dir)
    return snapshot


def load_meta_resume_manifest(
    run_dir: Path,
    *,
    read_json: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = run_dir / "meta_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"resume run is missing meta_manifest.json: {run_dir}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest.get("phases"), list):
        raise SystemExit(f"resume run has invalid phases payload: {manifest_path}")
    return manifest


def manifest_phase_signature(phases: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    signature: list[tuple[str, str]] = []
    for index, entry in enumerate(phases, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"resume manifest phase entry must be an object: index={index}")
        signature.append((str(entry.get("name") or ""), str(entry.get("stage") or "")))
    return tuple(signature)


def scenario_phase_signature(scenario: MetaTrainScenario) -> tuple[tuple[str, str], ...]:
    return tuple((str(phase_config.name), str(phase_config.stage)) for phase_config in scenario.phases)


def load_meta_train_scenario(
    preset_name: str | Path,
    *,
    repo_root: Path,
    default_dataset_root: Path,
    default_run_root: Path,
    load_user_paths_config: Callable[[], dict[str, Any]],
    load_user_hyperparameters_config: Callable[[], dict[str, Any]],
    nested_get: Callable[..., Any],
    validate_meta_train_scenario: Callable[[MetaTrainScenario], Any],
) -> MetaTrainScenario:
    preset_name_key = preset_key(preset_name)
    presets = build_meta_train_presets(
        repo_root=repo_root,
        default_dataset_root=default_dataset_root,
        default_run_root=default_run_root,
        load_user_paths_config=load_user_paths_config,
        load_user_hyperparameters_config=load_user_hyperparameters_config,
        nested_get=nested_get,
    )
    if preset_name_key not in presets:
        raise KeyError(f"unsupported PV26 meta-train preset: {preset_name_key}")
    scenario = presets[preset_name_key]
    validate_meta_train_scenario(scenario)
    return scenario


def load_resume_scenario_from_snapshot(
    scenario_snapshot: dict[str, Any],
    *,
    run_dir: Path,
    repo_root: Path,
    validate_meta_train_scenario: Callable[[MetaTrainScenario], Any],
) -> MetaTrainScenario:
    snapshot_mapping = dict(scenario_snapshot)
    run_mapping = dict(snapshot_mapping.get("run") or {})
    run_mapping["run_dir"] = str(run_dir)
    snapshot_mapping["run"] = run_mapping
    scenario = meta_train_scenario_from_mapping(snapshot_mapping, base_dir=repo_root)
    validate_meta_train_scenario(scenario)
    return scenario


def resume_scenario_path(
    manifest: dict[str, Any],
    *,
    preset_name: str,
    repo_root: Path,
    preset_path_root: Path,
) -> Path:
    return resolve_scenario_path(
        manifest.get("scenario_path"),
        fallback=default_scenario_path(preset_name, preset_path_root=preset_path_root),
        repo_root=repo_root,
    )


def load_legacy_resume_scenario(
    manifest: dict[str, Any],
    *,
    preset_name: str,
    run_dir: Path,
    repo_root: Path,
    preset_path_root: Path,
    load_meta_train_scenario: Callable[[str], MetaTrainScenario],
    scenario_to_mapping: Callable[[MetaTrainScenario], dict[str, Any]],
    validate_meta_train_scenario: Callable[[MetaTrainScenario], Any],
) -> tuple[MetaTrainScenario, Path]:
    scenario = load_meta_train_scenario(preset_name)
    current_mapping = scenario_to_mapping(scenario)
    expected_sections = ("dataset", "train_defaults", "selection", "preview")
    mismatched_sections = [
        key
        for key in expected_sections
        if manifest.get(key) != current_mapping.get(key)
    ]
    if manifest_phase_signature(manifest["phases"]) != scenario_phase_signature(scenario):
        mismatched_sections.append("phases")
    if mismatched_sections:
        raise SystemExit(
            "legacy resume run is incompatible with the current preset; "
            f"mismatched sections: {sorted(set(mismatched_sections))}"
        )

    scenario_mapping = current_mapping
    scenario_mapping["run"]["run_dir"] = str(run_dir)
    resumed_scenario = meta_train_scenario_from_mapping(scenario_mapping, base_dir=repo_root)
    validate_meta_train_scenario(resumed_scenario)
    scenario_path = resolve_scenario_path(
        manifest.get("scenario_path"),
        fallback=default_scenario_path(preset_name, preset_path_root=preset_path_root),
        repo_root=repo_root,
    )
    return resumed_scenario, scenario_path


def load_meta_train_resume_scenario(
    run_dir: str | Path,
    *,
    preset_name: str,
    repo_root: Path,
    preset_path_root: Path,
    load_meta_train_scenario: Callable[[str], MetaTrainScenario],
    read_json: Callable[[Path], dict[str, Any]],
    scenario_to_mapping: Callable[[MetaTrainScenario], dict[str, Any]],
    validate_meta_train_scenario: Callable[[MetaTrainScenario], Any],
) -> tuple[MetaTrainScenario, Path]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise SystemExit(f"resume run directory does not exist: {resolved_run_dir}")
    manifest = load_meta_resume_manifest(resolved_run_dir, read_json=read_json)
    if str(manifest.get("status") or "").strip() == "completed":
        raise SystemExit(f"exact resume only supports incomplete runs: {resolved_run_dir}")

    scenario_path = resume_scenario_path(
        manifest,
        preset_name=preset_name,
        repo_root=repo_root,
        preset_path_root=preset_path_root,
    )
    scenario_snapshot = manifest.get("scenario_snapshot")
    if isinstance(scenario_snapshot, dict):
        scenario = load_resume_scenario_from_snapshot(
            scenario_snapshot,
            run_dir=resolved_run_dir,
            repo_root=repo_root,
            validate_meta_train_scenario=validate_meta_train_scenario,
        )
        return scenario, scenario_path

    return load_legacy_resume_scenario(
        manifest,
        preset_name=preset_name,
        run_dir=resolved_run_dir,
        repo_root=repo_root,
        preset_path_root=preset_path_root,
        load_meta_train_scenario=load_meta_train_scenario,
        scenario_to_mapping=scenario_to_mapping,
        validate_meta_train_scenario=validate_meta_train_scenario,
    )


def load_meta_train_resume_context(
    run_dir: str | Path,
    *,
    read_json: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    manifest = load_meta_resume_manifest(resolved_run_dir, read_json=read_json)
    return {
        "selected_phase_window": manifest.get("selected_phase_window"),
        "lineage": manifest.get("lineage"),
    }


def stage_index_bounds(
    scenario: MetaTrainScenario,
    *,
    start_stage: str,
    end_stage: str,
) -> tuple[int, int]:
    start_index: int | None = None
    end_index: int | None = None
    for index, phase_config in enumerate(scenario.phases, start=1):
        if str(phase_config.stage) == str(start_stage):
            start_index = index
        if str(phase_config.stage) == str(end_stage):
            end_index = index
    if start_index is None:
        raise SystemExit(f"unknown start stage: {start_stage}")
    if end_index is None:
        raise SystemExit(f"unknown end stage: {end_stage}")
    if end_index < start_index:
        raise SystemExit(
            f"invalid stage window: end_stage={end_stage} comes before start_stage={start_stage}"
        )
    return start_index, end_index


def resolve_phase_best_checkpoint(
    run_dir: Path,
    phase_entry: dict[str, Any],
    *,
    phase_index: int,
) -> Path | None:
    checkpoint_value = phase_entry.get("best_checkpoint_path")
    if checkpoint_value not in {None, ""}:
        checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
        if checkpoint_path.is_file():
            return checkpoint_path
    fallback = run_dir / f"phase_{phase_index}" / "checkpoints" / "best.pt"
    if fallback.is_file():
        return fallback.resolve()
    return None


def resolve_derived_seed_checkpoint(
    source_run_dir: str | Path,
    *,
    start_phase_index: int,
    read_json: Callable[[Path], dict[str, Any]],
) -> tuple[Path | None, str | None]:
    resolved_run_dir = Path(source_run_dir).expanduser().resolve()
    manifest = load_meta_resume_manifest(resolved_run_dir, read_json=read_json)
    phases = manifest.get("phases")
    if not isinstance(phases, list):
        raise SystemExit(f"source run has invalid phases payload: {resolved_run_dir}")
    if start_phase_index <= len(phases):
        same_phase_checkpoint = resolve_phase_best_checkpoint(
            resolved_run_dir,
            phases[start_phase_index - 1],
            phase_index=start_phase_index,
        )
        if same_phase_checkpoint is not None:
            return same_phase_checkpoint, f"phase_{start_phase_index} best.pt"
    if start_phase_index <= 1:
        return None, None
    for phase_index in range(start_phase_index - 1, 0, -1):
        checkpoint_path = resolve_phase_best_checkpoint(
            resolved_run_dir,
            phases[phase_index - 1],
            phase_index=phase_index,
        )
        if checkpoint_path is not None:
            return checkpoint_path, f"phase_{phase_index} best.pt -> phase_{start_phase_index}"
    raise SystemExit(
        "source run does not have a usable seed checkpoint for "
        f"start phase {start_phase_index}: {resolved_run_dir}"
    )


def build_derived_run_name_prefix(
    scenario: MetaTrainScenario,
    *,
    source_run_dir: Path,
    start_stage: str,
    end_stage: str,
) -> str:
    stage_suffix = f"{start_stage}_to_{end_stage}" if start_stage != end_stage else start_stage
    return (
        f"{scenario.run.run_name_prefix}_derived_{source_run_dir.name}_{stage_suffix}"
    )


def load_meta_train_derived_scenario(
    source_run_dir: str | Path,
    *,
    preset_name: str,
    start_stage: str,
    end_stage: str,
    repo_root: Path,
    preset_path_root: Path,
    load_meta_train_scenario: Callable[[str], MetaTrainScenario],
    read_json: Callable[[Path], dict[str, Any]],
    validate_meta_train_scenario: Callable[[MetaTrainScenario], Any],
) -> tuple[MetaTrainScenario, Path, dict[str, Any]]:
    scenario = load_meta_train_scenario(preset_name)
    validate_meta_train_scenario(scenario)
    resolved_source_run_dir = Path(source_run_dir).expanduser().resolve()
    if not resolved_source_run_dir.is_dir():
        raise SystemExit(f"derived source run directory does not exist: {resolved_source_run_dir}")
    start_phase_index, end_phase_index = stage_index_bounds(
        scenario,
        start_stage=start_stage,
        end_stage=end_stage,
    )
    seed_checkpoint_path, seed_description = resolve_derived_seed_checkpoint(
        resolved_source_run_dir,
        start_phase_index=start_phase_index,
        read_json=read_json,
    )
    manifest = load_meta_resume_manifest(resolved_source_run_dir, read_json=read_json)
    lineage = {
        "mode": "derived_run",
        "source_run_dir": str(resolved_source_run_dir),
        "source_run_name": resolved_source_run_dir.name,
        "source_run_status": str(manifest.get("status") or "unknown"),
        "seed_checkpoint_path": str(seed_checkpoint_path) if seed_checkpoint_path is not None else None,
        "seed_checkpoint_source": seed_description,
        "start_stage": start_stage,
        "end_stage": end_stage,
    }
    derived_run = replace(
        scenario.run,
        run_dir=None,
        run_name_prefix=build_derived_run_name_prefix(
            scenario,
            source_run_dir=resolved_source_run_dir,
            start_stage=start_stage,
            end_stage=end_stage,
        ),
    )
    derived_scenario = replace(scenario, run=derived_run)
    scenario_path = default_scenario_path(preset_name, preset_path_root=preset_path_root)
    return derived_scenario, scenario_path, {
        "selected_phase_indices": tuple(range(start_phase_index, end_phase_index + 1)),
        "initial_best_checkpoint": seed_checkpoint_path,
        "lineage": lineage,
    }


__all__ = [
    "build_derived_run_name_prefix",
    "build_meta_train_presets",
    "default_scenario_path",
    "load_legacy_resume_scenario",
    "load_meta_resume_manifest",
    "load_meta_train_derived_scenario",
    "load_meta_train_resume_context",
    "load_meta_train_resume_scenario",
    "load_meta_train_scenario",
    "load_resume_scenario_from_snapshot",
    "manifest_phase_signature",
    "preset_key",
    "resolve_derived_seed_checkpoint",
    "resolve_phase_best_checkpoint",
    "resolve_scenario_path",
    "resume_scenario_path",
    "scenario_phase_signature",
    "scenario_snapshot_for_run",
    "stage_index_bounds",
]
