from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import site
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(REPO_ROOT))

from common.config_coercion import (
    coerce_bool as _coerce_bool,
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
    coerce_mapping as _coerce_mapping,
    coerce_str as _coerce_str,
)
from common.overlay import render_overlay
from common.user_config import (
    load_user_hyperparameters_config,
    load_user_paths_config,
    nested_get,
    resolve_repo_path,
    resolve_repo_paths,
)
from model.engine.evaluator import PV26Evaluator
from model.data import (
    PV26CanonicalDataset,
    build_pv26_eval_dataloader,
    build_pv26_train_dataloader,
    collate_pv26_samples,
)
from model.engine.trainer import PV26Trainer, build_pv26_optimizer, build_pv26_scheduler
from model.engine.trainer import _resolve_summary_path
from model.net import PV26Heads
from model.net import build_yolo26n_trunk
from tools.pv26_train_artifacts import (
    json_ready as _json_ready,
    load_or_init_meta_manifest as _load_or_init_meta_manifest,
    phase_entry_is_completed as _phase_entry_is_completed,
    read_json as _read_json,
    read_jsonl as _read_jsonl,
    recover_phase_entry_from_run_dir as _recover_phase_entry_from_run_dir,
    resolve_meta_run_dir as _resolve_meta_run_dir,
    safe_name as _safe_name,
    write_json as _write_json,
    write_meta_manifest as _write_meta_manifest,
    write_meta_summary as _write_meta_summary,
)


DEFAULT_DATASET_ROOT = REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "pv26_exhaustive_od_lane_train"
PRESET_PATH_ROOT = REPO_ROOT / "presets" / "pv26_meta_train"
DEFAULT_PRESET_NAME = "default"
HEAD_CHANNELS = (64, 128, 256)
META_MANIFEST_VERSION = "pv26-meta-train-v1"
PHASE_STAGE_ORDER = (
    "stage_1_frozen_trunk_warmup",
    "stage_2_partial_unfreeze",
    "stage_3_end_to_end_finetune",
)
# IDE에서 아래 검색어로 조절 지점을 바로 찾을 수 있다.
# ===== USER CONFIG =====
# ===== HYPERPARAMETERS =====
# ===== PHASE HYPERPARAMETERS =====


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
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaTrainScenario:
    dataset: DatasetConfig
    run: RunConfig
    train_defaults: TrainDefaultsConfig
    selection: SelectionConfig
    preview: PreviewConfig
    phases: tuple[PhaseConfig, ...]


def _phase(
    name: str,
    stage: str,
    *,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    min_improvement_pct: float,
    overrides: dict[str, Any] | None = None,
) -> PhaseConfig:
    return PhaseConfig(
        name=name,
        stage=stage,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        patience=patience,
        min_improvement_pct=min_improvement_pct,
        overrides=dict(overrides or {}),
    )


def _deep_merge_mappings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def _phase_to_mapping(phase: PhaseConfig) -> dict[str, Any]:
    return {
        "name": phase.name,
        "stage": phase.stage,
        "min_epochs": phase.min_epochs,
        "max_epochs": phase.max_epochs,
        "patience": phase.patience,
        "min_improvement_pct": phase.min_improvement_pct,
        "overrides": dict(phase.overrides),
    }


def _scenario_to_mapping(scenario: MetaTrainScenario) -> dict[str, Any]:
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
        "phases": [_phase_to_mapping(phase) for phase in scenario.phases],
    }


def _build_pv26_train_path_overrides(paths_config: dict[str, Any]) -> dict[str, Any]:
    legacy_stress_run_root = nested_get(paths_config, "pv26_train", "stress_run_root")
    if legacy_stress_run_root not in {None, ""}:
        raise ValueError("pv26_train.stress_run_root is no longer supported; use pv26_train.run_root")
    dataset_root = resolve_repo_path(
        nested_get(paths_config, "pv26_train", "dataset_root"),
        repo_root=REPO_ROOT,
    )
    additional_roots = resolve_repo_paths(
        nested_get(paths_config, "pv26_train", "additional_roots"),
        repo_root=REPO_ROOT,
    )
    run_root = resolve_repo_path(
        nested_get(paths_config, "pv26_train", "run_root"),
        repo_root=REPO_ROOT,
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


def _meta_train_scenario_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> MetaTrainScenario:
    phases_payload = payload.get("phases", ())
    if not isinstance(phases_payload, (list, tuple)):
        raise TypeError("phases must be a list")
    return MetaTrainScenario(
        dataset=_dataset_config_from_mapping(payload.get("dataset", {}), base_dir=base_dir),
        run=_run_config_from_mapping(payload.get("run", {}), base_dir=base_dir),
        train_defaults=_train_defaults_from_mapping(payload.get("train_defaults", {})),
        selection=_selection_config_from_mapping(payload.get("selection", {})),
        preview=_preview_config_from_mapping(payload.get("preview", {})),
        phases=tuple(
            _phase_config_from_mapping(phase_payload, index=index)
            for index, phase_payload in enumerate(phases_payload)
        ),
    )


def _apply_user_config_to_preset(
    preset_name: str,
    scenario: MetaTrainScenario,
    *,
    paths_config: dict[str, Any],
    hyperparameters_config: dict[str, Any],
) -> MetaTrainScenario:
    scenario_mapping = _scenario_to_mapping(scenario)
    path_overrides = _build_pv26_train_path_overrides(paths_config)
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
    merged_mapping = _deep_merge_mappings(scenario_mapping, path_overrides)
    merged_mapping = _deep_merge_mappings(merged_mapping, hyperparameter_overrides)
    return _meta_train_scenario_from_mapping(merged_mapping, base_dir=REPO_ROOT)


def _build_meta_train_presets() -> dict[str, MetaTrainScenario]:
    paths_config = load_user_paths_config()
    hyperparameters_config = load_user_hyperparameters_config()
    preset_overrides = nested_get(hyperparameters_config, "pv26_train", "presets", default={})
    if preset_overrides is None:
        preset_overrides = {}
    if not isinstance(preset_overrides, dict):
        raise TypeError("pv26_train.presets must be a mapping")
    unsupported_preset_overrides = sorted(set(preset_overrides) - {"default"})
    if unsupported_preset_overrides:
        raise KeyError(
            "unsupported PV26 meta-train preset overrides: "
            f"{unsupported_preset_overrides}; only 'default' is supported"
        )

    # ===== USER CONFIG: PV26 PREVIEW DATASETS =====
    preview_dataset_keys = (
        "pv26_exhaustive_bdd100k_det_100k",  # BDD 계열 preview 샘플
        "pv26_exhaustive_aihub_traffic_seoul",  # AIHUB traffic preview 샘플
        "pv26_exhaustive_aihub_obstacle_seoul",  # AIHUB obstacle preview 샘플
        "aihub_lane_seoul",  # lane 계열 preview 샘플
    )

    # ===== USER CONFIG: PV26 META TRAIN / DEFAULT =====
    default_dataset = DatasetConfig(root=DEFAULT_DATASET_ROOT)  # 기본 PV26 학습 dataset 루트
    default_run = RunConfig(
        run_root=DEFAULT_RUN_ROOT,
        run_name_prefix="exhaustive_od_lane",  # run 디렉터리 prefix
    )

    # ===== HYPERPARAMETERS: PV26 META TRAIN / DEFAULT =====
    default_train_defaults = TrainDefaultsConfig(
        device="cuda:0",  # 학습 장치
        batch_size=40,  # 기본 train/eval batch 크기
        train_batches=-1,  # -1이면 전체 train 배치 사용
        val_batches=-1,  # -1이면 전체 val 배치 사용
        trunk_lr=1e-4,  # backbone learning rate
        head_lr=5e-3,  # head learning rate
        weight_decay=1e-4,  # optimizer weight decay
        schedule="cosine",  # LR schedule 종류
        amp=True,  # mixed precision 사용 여부
        accumulate_steps=1,  # gradient accumulation step
        grad_clip_norm=5.0,  # gradient clipping norm
        val_every=1,  # 몇 epoch마다 validation할지
        checkpoint_every=3,  # 몇 epoch마다 체크포인트를 남길지
        num_workers=6,  # dataloader worker 수
        pin_memory=True,  # host->GPU 전송 최적화
        log_every_n_steps=20,  # step 로그 간격
        profile_window=20,  # timing 평균 창 길이
        profile_device_sync=True,  # timing 측정 전 device sync 여부
        encode_train_batches_in_loader=True,  # train loader에서 미리 target encode 수행 여부
        encode_val_batches_in_loader=True,  # val loader에서 미리 target encode 수행 여부
        persistent_workers=True,  # epoch 사이 worker 유지 여부
        prefetch_factor=2,  # worker별 prefetch 배치 수
    )
    default_selection = SelectionConfig(
        metric_path="val.losses.total.mean",  # phase 승급/종료 판단 metric
        mode="min",  # metric이 낮을수록 좋은지 여부
        eps=1e-8,  # improvement 계산용 안정화 상수
    )
    default_preview = PreviewConfig(
        enabled=True,  # 학습 요약용 preview 생성 여부
        split="val",  # preview 샘플을 뽑을 split
        dataset_keys=preview_dataset_keys,  # preview 대상 dataset key
        max_samples_per_dataset=1,  # dataset key별 preview 샘플 수
        write_overlay=True,  # overlay 이미지 저장 여부
    )

    # ===== PHASE HYPERPARAMETERS: PV26 META TRAIN / DEFAULT =====
    default_phases = (
        _phase(
            "head_warmup",
            "stage_1_frozen_trunk_warmup",
            min_epochs=2,  # 최소 epoch
            max_epochs=4,  # 최대 epoch
            patience=2,  # plateau 허용 횟수
            min_improvement_pct=2.0,  # 승급 유지에 필요한 최소 개선율(%)
            overrides={
                "trunk_lr": 5e-5,  # stage 1 backbone LR
                "head_lr": 3e-3,  # stage 1 head LR
            },
        ),
        _phase(
            "partial_unfreeze",
            "stage_2_partial_unfreeze",
            min_epochs=3,
            max_epochs=6,
            patience=2,
            min_improvement_pct=1.0,
            overrides={
                "trunk_lr": 3e-5,  # stage 2 backbone LR
                "head_lr": 8e-4,  # stage 2 head LR
            },
        ),
        _phase(
            "end_to_end_finetune",
            "stage_3_end_to_end_finetune",
            min_epochs=4,
            max_epochs=10,
            patience=2,
            min_improvement_pct=0.5,
            overrides={
                "trunk_lr": 1e-5,  # stage 3 backbone LR
                "head_lr": 4e-4,  # stage 3 head LR
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
    }
    return {
        preset_name: _apply_user_config_to_preset(
            preset_name,
            scenario,
            paths_config=paths_config,
            hyperparameters_config=hyperparameters_config,
        )
        for preset_name, scenario in presets.items()
    }


# Default CLI preset; override with `--preset`.
ENTRY_CONFIG = EntryConfig()


def _log_meta_train(message: str) -> None:
    print(f"[meta_train] {message}", flush=True)


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


def _coerce_path_list(value: Any, *, field_name: str, base_dir: Path) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(_resolve_path(item, base_dir=base_dir) for item in value)


def _dataset_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> DatasetConfig:
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


def _run_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> RunConfig:
    data = _coerce_mapping(payload, field_name="run")
    return RunConfig(
        run_root=_resolve_path(data.get("run_root", DEFAULT_RUN_ROOT), base_dir=base_dir),
        run_name_prefix=_coerce_str(data.get("run_name_prefix", "meta_train"), field_name="run.run_name_prefix"),
        run_dir=_resolve_optional_path(data.get("run_dir"), base_dir=base_dir),
    )


def _train_defaults_from_mapping(payload: dict[str, Any]) -> TrainDefaultsConfig:
    data = _coerce_mapping(payload, field_name="train_defaults")
    defaults = TrainDefaultsConfig()
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
        accumulate_steps=_coerce_int(data.get("accumulate_steps", defaults.accumulate_steps), field_name="train_defaults.accumulate_steps"),
        grad_clip_norm=_coerce_float(data.get("grad_clip_norm", defaults.grad_clip_norm), field_name="train_defaults.grad_clip_norm"),
        val_every=_coerce_int(data.get("val_every", defaults.val_every), field_name="train_defaults.val_every"),
        checkpoint_every=_coerce_int(data.get("checkpoint_every", defaults.checkpoint_every), field_name="train_defaults.checkpoint_every"),
        num_workers=_coerce_int(data.get("num_workers", defaults.num_workers), field_name="train_defaults.num_workers"),
        pin_memory=_coerce_bool(data.get("pin_memory", defaults.pin_memory), field_name="train_defaults.pin_memory"),
        log_every_n_steps=_coerce_int(data.get("log_every_n_steps", defaults.log_every_n_steps), field_name="train_defaults.log_every_n_steps"),
        profile_window=_coerce_int(data.get("profile_window", defaults.profile_window), field_name="train_defaults.profile_window"),
        profile_device_sync=_coerce_bool(data.get("profile_device_sync", defaults.profile_device_sync), field_name="train_defaults.profile_device_sync"),
        encode_train_batches_in_loader=_coerce_bool(
            data.get("encode_train_batches_in_loader", defaults.encode_train_batches_in_loader),
            field_name="train_defaults.encode_train_batches_in_loader",
        ),
        encode_val_batches_in_loader=_coerce_bool(
            data.get("encode_val_batches_in_loader", defaults.encode_val_batches_in_loader),
            field_name="train_defaults.encode_val_batches_in_loader",
        ),
        persistent_workers=_coerce_bool(data.get("persistent_workers", defaults.persistent_workers), field_name="train_defaults.persistent_workers"),
        prefetch_factor=_coerce_optional_int(data.get("prefetch_factor", defaults.prefetch_factor), field_name="train_defaults.prefetch_factor"),
    )


def _selection_config_from_mapping(payload: dict[str, Any]) -> SelectionConfig:
    data = _coerce_mapping(payload, field_name="selection")
    return SelectionConfig(
        metric_path=_coerce_str(data.get("metric_path", "val.losses.total.mean"), field_name="selection.metric_path"),
        mode=_coerce_str(data.get("mode", "min"), field_name="selection.mode"),
        eps=_coerce_float(data.get("eps", 1e-8), field_name="selection.eps"),
    )


def _preview_config_from_mapping(payload: dict[str, Any]) -> PreviewConfig:
    data = _coerce_mapping(payload, field_name="preview")
    dataset_keys = data.get("dataset_keys", PreviewConfig().dataset_keys)
    if not isinstance(dataset_keys, (list, tuple)):
        raise TypeError("preview.dataset_keys must be a list")
    return PreviewConfig(
        enabled=_coerce_bool(data.get("enabled", True), field_name="preview.enabled"),
        split=_coerce_str(data.get("split", "val"), field_name="preview.split"),
        dataset_keys=tuple(_coerce_str(value, field_name="preview.dataset_keys[]") for value in dataset_keys),
        max_samples_per_dataset=_coerce_int(data.get("max_samples_per_dataset", 1), field_name="preview.max_samples_per_dataset"),
        write_overlay=_coerce_bool(data.get("write_overlay", True), field_name="preview.write_overlay"),
    )


def _phase_config_from_mapping(payload: dict[str, Any], *, index: int) -> PhaseConfig:
    data = _coerce_mapping(payload, field_name=f"phases[{index}]")
    overrides = _coerce_mapping(data.get("overrides"), field_name=f"phases[{index}].overrides")
    return PhaseConfig(
        name=_coerce_str(data.get("name", f"phase_{index + 1}"), field_name=f"phases[{index}].name"),
        stage=_coerce_str(data.get("stage"), field_name=f"phases[{index}].stage"),
        min_epochs=_coerce_int(data.get("min_epochs"), field_name=f"phases[{index}].min_epochs"),
        max_epochs=_coerce_int(data.get("max_epochs"), field_name=f"phases[{index}].max_epochs"),
        patience=_coerce_int(data.get("patience"), field_name=f"phases[{index}].patience"),
        min_improvement_pct=_coerce_float(data.get("min_improvement_pct"), field_name=f"phases[{index}].min_improvement_pct"),
        overrides=overrides,
    )


def _resolve_train_batch_limit(value: int) -> int | None:
    if int(value) <= 0:
        return None
    return int(value)


def _resolve_val_batch_limit(value: int) -> int | None:
    if int(value) < 0:
        return None
    return int(value)


def _scenario_phase_defaults(defaults: TrainDefaultsConfig, overrides: dict[str, Any]) -> TrainDefaultsConfig:
    allowed_keys = set(TrainDefaultsConfig.__dataclass_fields__.keys())
    unknown_keys = sorted(set(overrides) - allowed_keys)
    if unknown_keys:
        raise KeyError(f"unsupported phase override keys: {unknown_keys}")
    merged = asdict(defaults)
    merged.update(overrides)
    return _train_defaults_from_mapping(merged)


def _validate_meta_train_scenario(scenario: MetaTrainScenario) -> None:
    if len(scenario.phases) != len(PHASE_STAGE_ORDER):
        raise ValueError(f"meta_train requires exactly {len(PHASE_STAGE_ORDER)} phases")
    for index, (phase, expected_stage) in enumerate(zip(scenario.phases, PHASE_STAGE_ORDER), start=1):
        if phase.stage != expected_stage:
            raise ValueError(f"phase {index} must use stage {expected_stage!r}, got {phase.stage!r}")
        if phase.min_epochs <= 0:
            raise ValueError(f"phase {index} min_epochs must be > 0")
        if phase.max_epochs < phase.min_epochs:
            raise ValueError(f"phase {index} max_epochs must be >= min_epochs")
        if phase.patience <= 0:
            raise ValueError(f"phase {index} patience must be > 0")
        if phase.min_improvement_pct < 0.0:
            raise ValueError(f"phase {index} min_improvement_pct must be >= 0")
        phase_train = _scenario_phase_defaults(scenario.train_defaults, phase.overrides)
        if scenario.selection.metric_path.startswith("val.") and _resolve_val_batch_limit(phase_train.val_batches) == 0:
            raise ValueError(
                f"phase {index} disables validation but selection.metric_path={scenario.selection.metric_path!r} requires val"
            )
    if scenario.selection.mode not in {"min", "max"}:
        raise ValueError("selection.mode must be 'min' or 'max'")
    if scenario.selection.eps <= 0.0:
        raise ValueError("selection.eps must be > 0")
    if scenario.preview.max_samples_per_dataset <= 0:
        raise ValueError("preview.max_samples_per_dataset must be > 0")


def load_meta_train_scenario(preset_name: str | Path) -> MetaTrainScenario:
    preset_key = Path(preset_name).name
    presets = _build_meta_train_presets()
    if preset_key not in presets:
        raise KeyError(f"unsupported PV26 meta-train preset: {preset_key}")
    scenario = presets[preset_key]
    _validate_meta_train_scenario(scenario)
    return scenario


class PhaseTransitionController:
    def __init__(
        self,
        *,
        phase: PhaseConfig,
        selection: SelectionConfig,
    ) -> None:
        self.phase = phase
        self.selection = selection
        self.best_metric_value: float | None = None
        self.best_epoch: int | None = None
        self.plateau_count = 0
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

    def _phase_state(self, *, epoch: int, current_metric_value: float) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "current_metric_value": float(current_metric_value),
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "plateau_count": int(self.plateau_count),
            "last_improvement_pct": self.last_improvement_pct,
            "min_epochs": int(self.phase.min_epochs),
            "max_epochs": int(self.phase.max_epochs),
            "patience": int(self.phase.patience),
            "min_improvement_pct": float(self.phase.min_improvement_pct),
            "transition_eligible": bool(epoch >= self.phase.min_epochs),
        }

    def observe_epoch(self, epoch_summary: dict[str, Any]) -> dict[str, Any] | None:
        epoch = int(epoch_summary["epoch"])
        current_metric = float(_resolve_summary_path(epoch_summary, self.selection.metric_path))
        if self.best_metric_value is None:
            self.best_metric_value = current_metric
            self.best_epoch = epoch
            self.plateau_count = 0
            self.last_improvement_pct = None
        else:
            better = self._is_better(current_metric)
            if better:
                improvement_pct = self._relative_improvement(self.best_metric_value, current_metric)
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                self.last_improvement_pct = improvement_pct
                if epoch >= self.phase.min_epochs:
                    if improvement_pct >= float(self.phase.min_improvement_pct):
                        self.plateau_count = 0
                    else:
                        self.plateau_count += 1
                else:
                    self.plateau_count = 0
            else:
                self.last_improvement_pct = 0.0
                if epoch >= self.phase.min_epochs:
                    self.plateau_count += 1

        phase_state = self._phase_state(epoch=epoch, current_metric_value=current_metric)
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


def _configure_torch_multiprocessing() -> None:
    try:
        import torch

        if torch.multiprocessing.get_sharing_strategy() != "file_system":
            torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def _build_phase_train_loaders(
    dataset: PV26CanonicalDataset,
    *,
    train_config: TrainDefaultsConfig,
) -> tuple[Any, Any]:
    train_batches = _resolve_train_batch_limit(train_config.train_batches)
    val_batches = _resolve_val_batch_limit(train_config.val_batches)
    train_loader = build_pv26_train_dataloader(
        dataset,
        batch_size=train_config.batch_size,
        num_batches=train_batches,
        split="train",
        seed=26,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        encode_batches=train_config.encode_train_batches_in_loader,
        persistent_workers=train_config.persistent_workers,
        prefetch_factor=train_config.prefetch_factor,
    )
    val_loader = None
    if val_batches != 0:
        val_loader = build_pv26_eval_dataloader(
            dataset,
            batch_size=train_config.batch_size,
            num_batches=val_batches,
            split="val",
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            encode_batches=train_config.encode_val_batches_in_loader,
            persistent_workers=train_config.persistent_workers,
            prefetch_factor=train_config.prefetch_factor,
        )
    return train_loader, val_loader


def _build_phase_trainer(phase: PhaseConfig, train_config: TrainDefaultsConfig) -> PV26Trainer:
    adapter = build_yolo26n_trunk()
    heads = PV26Heads(in_channels=HEAD_CHANNELS)
    optimizer = build_pv26_optimizer(
        adapter,
        heads,
        trunk_lr=train_config.trunk_lr,
        head_lr=train_config.head_lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = build_pv26_scheduler(
        optimizer,
        epochs=phase.max_epochs,
        schedule=train_config.schedule,
    )
    return PV26Trainer(
        adapter,
        heads,
        stage=phase.stage,
        device=train_config.device,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=train_config.amp,
        accumulate_steps=train_config.accumulate_steps,
        grad_clip_norm=train_config.grad_clip_norm,
    )


def _sample_preview_selection(dataset: PV26CanonicalDataset, preview: PreviewConfig) -> list[dict[str, Any]]:
    if not preview.enabled:
        return []
    selected: list[dict[str, Any]] = []
    counts = {dataset_key: 0 for dataset_key in preview.dataset_keys}
    selected_indices: list[int] = []
    for index, record in enumerate(dataset.records):
        dataset_key = str(record.dataset_key)
        split = str(record.split)
        if split != preview.split or dataset_key not in counts:
            continue
        if counts[dataset_key] >= preview.max_samples_per_dataset:
            continue
        selected_indices.append(index)
        counts[dataset_key] += 1
        if all(count >= preview.max_samples_per_dataset for count in counts.values()):
            break
    missing = [dataset_key for dataset_key, count in counts.items() if count < preview.max_samples_per_dataset]
    if missing:
        raise RuntimeError(f"missing preview samples for dataset keys: {missing}")
    for index in selected_indices:
        selected.append(dataset[index])
    return selected


def _prediction_to_overlay_scene(prediction: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample["meta"]
    scene: dict[str, Any] = {
        "source": {"image_path": str(meta["image_path"])},
        "detections": [],
        "traffic_lights": [],
        "traffic_signs": [],
        "lanes": [],
        "stop_lines": [],
        "crosswalks": [],
        "debug_rectangles": [],
    }
    for detection in prediction.get("detections", []):
        item = {
            "bbox": [float(value) for value in detection.get("box_xyxy", [])],
            "class_name": str(detection.get("class_name") or "unknown"),
        }
        class_name = item["class_name"]
        if class_name == "traffic_light":
            scene["traffic_lights"].append(item)
        elif class_name == "sign":
            scene["traffic_signs"].append(item)
        else:
            scene["detections"].append(item)
    for lane in prediction.get("lanes", []):
        scene["lanes"].append(
            {
                "class_name": lane.get("class_name"),
                "points": [[float(x), float(y)] for x, y in lane.get("points_xy", [])],
            }
        )
    for stop_line in prediction.get("stop_lines", []):
        scene["stop_lines"].append(
            {"points": [[float(x), float(y)] for x, y in stop_line.get("points_xy", [])]}
        )
    for crosswalk in prediction.get("crosswalks", []):
        scene["crosswalks"].append(
            {"points": [[float(x), float(y)] for x, y in crosswalk.get("points_xy", [])]}
        )
    return scene


def _build_preview_evaluator(
    phase: PhaseConfig,
    train_config: TrainDefaultsConfig,
    checkpoint_path: Path,
) -> PV26Evaluator:
    trainer = _build_phase_trainer(phase, train_config)
    trainer.load_model_weights(checkpoint_path, map_location=train_config.device)
    return trainer.build_evaluator()


def _generate_phase_preview_bundle(
    *,
    phase: PhaseConfig,
    train_config: TrainDefaultsConfig,
    checkpoint_path: Path,
    preview_kind: str,
    preview_dir: Path,
    preview_samples: list[dict[str, Any]],
    preview_config: PreviewConfig,
) -> dict[str, Any]:
    if not preview_config.enabled:
        return {"enabled": False}
    evaluator = _build_preview_evaluator(phase, train_config, checkpoint_path)
    output_dir = preview_dir / preview_kind
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_entries: list[dict[str, Any]] = []
    for sample in preview_samples:
        sample_meta = sample["meta"]
        sample_id = _safe_name(str(sample_meta["sample_id"]))
        dataset_key = _safe_name(str(sample_meta["dataset_key"]))
        stem = f"{dataset_key}__{sample_id}"
        batch = collate_pv26_samples([sample])
        prediction = evaluator.predict_batch(batch)[0]
        prediction_path = output_dir / f"{stem}.json"
        prediction_payload = {
            "phase_name": phase.name,
            "phase_stage": phase.stage,
            "preview_kind": preview_kind,
            "sample_meta": _json_ready(sample_meta),
            "prediction": _json_ready(prediction),
        }
        _write_json(prediction_path, prediction_payload)
        overlay_path = output_dir / f"{stem}.png"
        overlay_error = None
        if preview_config.write_overlay:
            try:
                render_overlay(_prediction_to_overlay_scene(prediction, sample), overlay_path)
            except Exception as exc:  # pragma: no cover - depends on local ImageMagick availability.
                overlay_error = str(exc)
        preview_entries.append(
            {
                "sample_id": str(sample_meta["sample_id"]),
                "dataset_key": str(sample_meta["dataset_key"]),
                "prediction_path": str(prediction_path),
                "overlay_path": str(overlay_path) if preview_config.write_overlay and overlay_error is None else None,
                "overlay_error": overlay_error,
            }
        )
    index_payload = {
        "phase_name": phase.name,
        "phase_stage": phase.stage,
        "preview_kind": preview_kind,
        "entries": preview_entries,
    }
    index_path = output_dir / "index.json"
    _write_json(index_path, index_payload)
    return {
        "enabled": True,
        "index_path": str(index_path),
        "output_dir": str(output_dir),
        "entries": preview_entries,
    }


def _phase_manifest_extra(
    *,
    scenario_path: Path,
    phase_index: int,
    phase: PhaseConfig,
    train_config: TrainDefaultsConfig,
    scenario: MetaTrainScenario,
) -> dict[str, Any]:
    return {
        "entry_script": "tools/run_pv26_train.py",
        "scenario_path": str(scenario_path),
        "dataset_config": _json_ready(asdict(scenario.dataset)),
        "selection": _json_ready(asdict(scenario.selection)),
        "preview": _json_ready(asdict(scenario.preview)),
        "phase": {
            "index": int(phase_index),
            "name": phase.name,
            "stage": phase.stage,
            "min_epochs": int(phase.min_epochs),
            "max_epochs": int(phase.max_epochs),
            "patience": int(phase.patience),
            "min_improvement_pct": float(phase.min_improvement_pct),
        },
        "phase_train_config": _json_ready(asdict(train_config)),
        "head_channels": list(HEAD_CHANNELS),
    }


def _execute_phase(
    *,
    scenario: MetaTrainScenario,
    scenario_path: Path,
    dataset: PV26CanonicalDataset,
    preview_samples: list[dict[str, Any]],
    phase_index: int,
    phase: PhaseConfig,
    run_dir: Path,
    previous_best_checkpoint: Path | None,
) -> dict[str, Any]:
    phase_run_dir = run_dir / f"phase_{phase_index}"
    phase_train_config = _scenario_phase_defaults(scenario.train_defaults, phase.overrides)
    _log_meta_train(f"building loaders for phase_{phase_index} at {phase_run_dir}")
    train_loader, val_loader = _build_phase_train_loaders(dataset, train_config=phase_train_config)
    controller = PhaseTransitionController(phase=phase, selection=scenario.selection)
    controller.replay(_read_jsonl(phase_run_dir / "history" / "epochs.jsonl"))

    trainer = _build_phase_trainer(phase, phase_train_config)
    last_checkpoint_path = phase_run_dir / "checkpoints" / "last.pt"
    if previous_best_checkpoint is not None and not last_checkpoint_path.is_file():
        _log_meta_train(
            f"loading weights-only handoff for phase_{phase_index} from {previous_best_checkpoint}"
        )
        trainer.load_model_weights(previous_best_checkpoint, map_location=phase_train_config.device)
    elif last_checkpoint_path.is_file():
        _log_meta_train(f"auto-resume checkpoint found for phase_{phase_index}: {last_checkpoint_path}")

    phase_summary = trainer.fit(
        train_loader,
        epochs=phase.max_epochs,
        phase_index=phase_index,
        phase_count=len(scenario.phases),
        phase_name=phase.name,
        val_loader=val_loader,
        run_dir=phase_run_dir,
        val_every=phase_train_config.val_every,
        checkpoint_every=phase_train_config.checkpoint_every,
        max_train_batches=_resolve_train_batch_limit(phase_train_config.train_batches),
        max_val_batches=_resolve_val_batch_limit(phase_train_config.val_batches),
        best_metric=scenario.selection.metric_path,
        best_mode=scenario.selection.mode,
        auto_resume=True,
        enable_tensorboard=True,
        early_exit_callback=controller.observe_epoch,
        log_every_n_steps=phase_train_config.log_every_n_steps,
        profile_window=phase_train_config.profile_window,
        profile_device_sync=phase_train_config.profile_device_sync,
        run_manifest_extra=_phase_manifest_extra(
            scenario_path=scenario_path,
            phase_index=phase_index,
            phase=phase,
            train_config=phase_train_config,
            scenario=scenario,
        ),
    )

    phase_preview_dir = run_dir / "preview" / f"phase_{phase_index}"
    preview_payload = {
        "best": None,
        "last": None,
    }
    best_checkpoint = Path(phase_summary["checkpoint_paths"]["best"]) if phase_summary["checkpoint_paths"]["best"] else None
    last_checkpoint = Path(phase_summary["checkpoint_paths"]["last"]) if phase_summary["checkpoint_paths"]["last"] else None
    if best_checkpoint is not None and best_checkpoint.is_file():
        _log_meta_train(f"writing best preview bundle for phase_{phase_index}: {best_checkpoint}")
        preview_payload["best"] = _generate_phase_preview_bundle(
            phase=phase,
            train_config=phase_train_config,
            checkpoint_path=best_checkpoint,
            preview_kind="best",
            preview_dir=phase_preview_dir,
            preview_samples=preview_samples,
            preview_config=scenario.preview,
        )
    if last_checkpoint is not None and last_checkpoint.is_file():
        _log_meta_train(f"writing last preview bundle for phase_{phase_index}: {last_checkpoint}")
        preview_payload["last"] = _generate_phase_preview_bundle(
            phase=phase,
            train_config=phase_train_config,
            checkpoint_path=last_checkpoint,
            preview_kind="last",
            preview_dir=phase_preview_dir,
            preview_samples=preview_samples,
            preview_config=scenario.preview,
        )

    early_exit = phase_summary.get("early_exit", {})
    return {
        "index": int(phase_index),
        "name": phase.name,
        "stage": phase.stage,
        "status": "completed",
        "run_dir": str(phase_run_dir),
        "summary_path": str(phase_run_dir / "summary.json"),
        "run_manifest_path": str(phase_run_dir / "run_manifest.json"),
        "best_checkpoint_path": str(best_checkpoint) if best_checkpoint is not None else None,
        "last_checkpoint_path": str(last_checkpoint) if last_checkpoint is not None else None,
        "completed_epochs": int(phase_summary["completed_epochs"]),
        "best_metric_value": phase_summary.get("best_metric_value"),
        "best_epoch": phase_summary.get("best_epoch"),
        "promotion_reason": early_exit.get("reason", "completed"),
        "phase_state": early_exit.get("phase_state"),
        "preview": _json_ready(preview_payload),
        "phase_train_config": _json_ready(asdict(phase_train_config)),
        "run_summary": _json_ready(phase_summary),
    }


def run_meta_train_scenario(
    scenario: MetaTrainScenario,
    *,
    scenario_path: Path,
) -> dict[str, Any]:
    _configure_torch_multiprocessing()
    dataset_roots = list(scenario.dataset.roots)
    missing_roots = [str(path) for path in dataset_roots if not path.is_dir()]
    if missing_roots:
        raise SystemExit(f"canonical dataset roots not found: {missing_roots}")

    _log_meta_train(f"loading scenario: {scenario_path}")
    _log_meta_train(f"dataset roots: {[str(path) for path in dataset_roots]}")
    run_dir = _resolve_meta_run_dir(scenario, scenario_path=scenario_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_meta_train(f"meta run directory: {run_dir}")
    _log_meta_train("building canonical dataset index")
    dataset = PV26CanonicalDataset(
        dataset_roots,
        progress_callback=_log_meta_train,
    )
    split_counts = Counter(record.split for record in dataset.records)
    dataset_key_counts = Counter(record.dataset_key for record in dataset.records)
    _log_meta_train(f"dataset split counts: {dict(sorted(split_counts.items()))}")
    _log_meta_train(f"dataset key counts: {dict(sorted(dataset_key_counts.items()))}")
    _log_meta_train("selecting preview samples")
    preview_samples = _sample_preview_selection(dataset, scenario.preview)
    _log_meta_train(f"prepared {len(preview_samples)} preview samples")
    manifest, manifest_path = _load_or_init_meta_manifest(
        scenario=scenario,
        scenario_path=scenario_path,
        run_dir=run_dir,
        meta_manifest_version=META_MANIFEST_VERSION,
    )

    previous_best_checkpoint: Path | None = None
    for phase_index, phase in enumerate(scenario.phases, start=1):
        phase_entry = manifest["phases"][phase_index - 1]
        if _phase_entry_is_completed(phase_entry, phase):
            recovered_entry = _recover_phase_entry_from_run_dir(phase_entry, phase)
            if recovered_entry is not None:
                phase_entry.update(recovered_entry)
            else:
                phase_entry["status"] = "completed"
            _log_meta_train(
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
        _write_meta_manifest(manifest_path, manifest)
        _write_meta_summary(run_dir, manifest)
        _log_meta_train(
            f"starting phase {phase_index}/{len(scenario.phases)}: "
            f"{phase.name} ({phase.stage})"
        )
        _log_meta_train(
            f"phase policy: min_epochs={phase.min_epochs}, max_epochs={phase.max_epochs}, "
            f"patience={phase.patience}, min_improvement_pct={phase.min_improvement_pct}"
        )

        phase_result = _execute_phase(
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
        _log_meta_train(
            f"completed phase {phase_index}/{len(scenario.phases)}: "
            f"reason={phase_result['promotion_reason']}, "
            f"completed_epochs={phase_result['completed_epochs']}, "
            f"best_checkpoint={phase_result['best_checkpoint_path']}"
        )
        manifest["active_phase_index"] = None
        manifest["active_phase_name"] = None
        _write_meta_manifest(manifest_path, manifest)
        _write_meta_summary(run_dir, manifest)

    manifest["status"] = "completed"
    manifest["active_phase_index"] = None
    manifest["active_phase_name"] = None
    _write_meta_manifest(manifest_path, manifest)
    _write_meta_summary(run_dir, manifest)
    return {
        "version": META_MANIFEST_VERSION,
        "status": manifest["status"],
        "scenario_path": str(scenario_path),
        "run_dir": str(run_dir),
        "meta_manifest_path": str(manifest_path),
        "summary_path": str(run_dir / "summary.json"),
        "completed_phases": len([phase for phase in manifest["phases"] if phase["status"] == "completed"]),
        "phases": manifest["phases"],
        "final_checkpoint_path": previous_best_checkpoint,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    preset_names = tuple(sorted(_build_meta_train_presets().keys()))
    parser = argparse.ArgumentParser(description="Run the PV26 meta-train scenario.")
    parser.add_argument(
        "--preset",
        choices=preset_names,
        default=ENTRY_CONFIG.preset_name,
        help="PV26 meta-train preset name.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    preset_name = str(args.preset)
    scenario = load_meta_train_scenario(preset_name)
    scenario_path = PRESET_PATH_ROOT / preset_name
    summary = run_meta_train_scenario(scenario, scenario_path=scenario_path)
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
