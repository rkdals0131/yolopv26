from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.io import read_yaml
from common.pv26_schema import (
    AIHUB_OBSTACLE_DATASET_KEY,
    AIHUB_TRAFFIC_DATASET_KEY,
    OD_CLASSES,
)
from common.user_config import (
    load_user_hyperparameters_config,
    load_user_paths_config,
    nested_get,
    resolve_repo_path,
)
from tools.od_bootstrap.source.types import (
    SourcePrepConfig,
    SourceRoots,
)
from tools.od_bootstrap.build.sweep_types import (
    BootstrapSweepScenario,
    ClassPolicy,
    ImageListConfig,
    MaterializationConfig,
    RunConfig as SweepRunConfig,
    TeacherConfig,
)
from tools.od_bootstrap.teacher.calibration_types import (
    CalibrationDatasetConfig,
    CalibrationRunConfig,
    CalibrationScenario,
    CalibrationSearchConfig,
    CalibrationTeacherConfig,
    HardNegativeConfig,
)
from tools.od_bootstrap.teacher.eval_types import (
    CheckpointEvalDatasetConfig,
    CheckpointEvalModelConfig,
    CheckpointEvalParams,
    CheckpointEvalRunConfig,
    CheckpointEvalScenario,
)
from tools.od_bootstrap.teacher.policy import class_policy_from_dict
from tools.od_bootstrap.teacher.train_types import (
    TeacherDatasetConfig,
    TeacherModelConfig,
    TeacherRunConfig,
    TeacherTrainParams,
    TeacherTrainScenario,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
# IDE에서 아래 검색어로 조절 지점을 바로 찾을 수 있다.
# ===== USER CONFIG =====
# ===== HYPERPARAMETERS =====


@dataclass(frozen=True)
class FinalDatasetPreset:
    exhaustive_od_root: Path
    aihub_canonical_root: Path
    output_root: Path
    copy_images: bool = False


@dataclass(frozen=True)
class TeacherDatasetPreset:
    canonical_root: Path
    output_root: Path
    copy_images: bool = False
    workers: int = 8
    log_every: int = 500
    debug_vis_count: int = 20
    debug_vis_seed: int = 26


@dataclass(frozen=True)
class SourceDebugVisPreset:
    image_list_manifest_path: Path
    canonical_root: Path
    debug_vis_count: int = 20
    debug_vis_seed: int = 26


def _coerce_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean")


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float")
    return float(value)


def _coerce_str(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _coerce_str_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_coerce_str(value, field_name=field_name),)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(_coerce_str(item, field_name=f"{field_name}[]") for item in value)


def _coerce_float_tuple(value: Any, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(_coerce_float(item, field_name=f"{field_name}[]") for item in value)


def _coerce_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(_coerce_int(item, field_name=f"{field_name}[]") for item in value)


def _config_section(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    field_name = ".".join(keys) if keys else "config"
    return _coerce_mapping(nested_get(payload, *keys, default={}), field_name=field_name)


def _load_od_bootstrap_paths_config() -> dict[str, Any]:
    return _config_section(load_user_paths_config(), "od_bootstrap")


def _load_od_bootstrap_hyperparameters_config() -> dict[str, Any]:
    return _config_section(load_user_hyperparameters_config(), "od_bootstrap")


def _path_from_config(config: dict[str, Any], *keys: str, default: Path) -> Path:
    resolved = resolve_repo_path(nested_get(config, *keys), repo_root=REPO_ROOT)
    return (resolved or default).resolve()


def _optional_path_from_config(config: dict[str, Any], *keys: str) -> Path | None:
    return resolve_repo_path(nested_get(config, *keys), repo_root=REPO_ROOT)


def _class_policy_defaults_from_config(hyperparameters_config: dict[str, Any]) -> dict[str, ClassPolicy]:
    defaults = {
        "vehicle": ClassPolicy(0.25, 0.55, 4),
        "bike": ClassPolicy(0.25, 0.55, 4),
        "pedestrian": ClassPolicy(0.25, 0.55, 4),
        "traffic_light": ClassPolicy(0.30, 0.50, 4),
        "sign": ClassPolicy(0.25, 0.50, 4),
        "traffic_cone": ClassPolicy(0.25, 0.55, 4),
        "obstacle": ClassPolicy(0.25, 0.55, 4),
    }
    payload = _config_section(hyperparameters_config, "exhaustive_od", "class_policy_defaults")
    merged = dict(defaults)
    for class_name, raw_policy in payload.items():
        merged[str(class_name)] = class_policy_from_dict(
            raw_policy,
            default_policy=merged.get(str(class_name)),
        )
    missing = sorted(set(OD_CLASSES) - set(merged))
    if missing:
        raise ValueError(f"missing exhaustive class_policy defaults for: {missing}")
    return merged


def _effective_class_policy(
    *,
    hyperparameters_config: dict[str, Any],
    class_policy_path: Path,
) -> dict[str, ClassPolicy]:
    defaults = _class_policy_defaults_from_config(hyperparameters_config)
    if not class_policy_path.is_file():
        return defaults
    payload = read_yaml(class_policy_path)
    merged = dict(defaults)
    for class_name, raw_policy in payload.items():
        merged[str(class_name)] = class_policy_from_dict(
            raw_policy,
            default_policy=merged.get(str(class_name)),
        )
    return merged


def build_default_source_preset(*, output_root: Path | None = None) -> SourcePrepConfig:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()

    # ===== USER CONFIG: SOURCE PATHS =====
    bdd_root = _path_from_config(
        paths_config,
        "raw_sources",
        "bdd_root",
        default=REPO_ROOT / "seg_dataset" / "BDD100K",
    )  # BDD100K 원본 루트
    bdd_images_root = _path_from_config(
        paths_config,
        "raw_sources",
        "bdd_images_root",
        default=bdd_root / "bdd100k_images_100k" / "100k",
    )  # BDD image split 루트
    bdd_labels_root = _path_from_config(
        paths_config,
        "raw_sources",
        "bdd_labels_root",
        default=bdd_root / "bdd100k_labels" / "100k",
    )  # BDD detection label split 루트
    aihub_root = _path_from_config(
        paths_config,
        "raw_sources",
        "aihub_root",
        default=REPO_ROOT / "seg_dataset" / "AIHUB",
    )  # AIHUB 원본 통합 루트
    source_output_root = (
        Path(output_root).resolve()
        if output_root is not None
        else _path_from_config(
            paths_config,
            "outputs",
            "bootstrap_root",
            default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap",
        )
    )  # canonical 출력 루트

    # ===== HYPERPARAMETERS: SOURCE PREP =====
    source_prep = _config_section(hyperparameters_config, "source_prep")
    workers = _coerce_int(source_prep.get("workers", 4), field_name="od_bootstrap.source_prep.workers")
    force_reprocess = _coerce_bool(
        source_prep.get("force_reprocess", False),
        field_name="od_bootstrap.source_prep.force_reprocess",
    )
    write_source_readmes = _coerce_bool(
        source_prep.get("write_source_readmes", False),
        field_name="od_bootstrap.source_prep.write_source_readmes",
    )
    debug_vis_count = _coerce_int(
        source_prep.get("debug_vis_count", 50),
        field_name="od_bootstrap.source_prep.debug_vis_count",
    )
    debug_vis_seed = _coerce_int(
        source_prep.get("debug_vis_seed", 42),
        field_name="od_bootstrap.source_prep.debug_vis_seed",
    )

    roots = SourceRoots(
        bdd_root=bdd_root,
        bdd_images_root=bdd_images_root,
        bdd_labels_root=bdd_labels_root,
        aihub_root=aihub_root,
        aihub_lane_root=_optional_path_from_config(paths_config, "raw_sources", "aihub_lane_root"),
        aihub_obstacle_root=_optional_path_from_config(paths_config, "raw_sources", "aihub_obstacle_root"),
        aihub_traffic_root=_optional_path_from_config(paths_config, "raw_sources", "aihub_traffic_root"),
        aihub_docs_root=_optional_path_from_config(paths_config, "raw_sources", "aihub_docs_root"),
    )
    return SourcePrepConfig(
        roots=roots,
        output_root=source_output_root.resolve(),
        workers=workers,
        force_reprocess=force_reprocess,
        write_source_readmes=write_source_readmes,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
    )


def build_teacher_dataset_preset(*, output_root: Path | None = None) -> TeacherDatasetPreset:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()

    # ===== USER CONFIG: TEACHER DATASET PATHS =====
    canonical_root = _path_from_config(
        paths_config,
        "outputs",
        "bootstrap_root",
        default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap",
    )  # canonical bootstrap 입력 루트
    teacher_dataset_root = (
        Path(output_root).resolve()
        if output_root is not None
        else _path_from_config(
            paths_config,
            "outputs",
            "teacher_dataset_root",
            default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets",
        )
    )  # teacher dataset 출력 루트

    # ===== HYPERPARAMETERS: TEACHER DATASET BUILD =====
    teacher_dataset = _config_section(hyperparameters_config, "teacher_dataset")
    copy_images = _coerce_bool(
        teacher_dataset.get("copy_images", False),
        field_name="od_bootstrap.teacher_dataset.copy_images",
    )
    workers = _coerce_int(
        teacher_dataset.get("workers", 8),
        field_name="od_bootstrap.teacher_dataset.workers",
    )
    log_every = _coerce_int(
        teacher_dataset.get("log_every", 500),
        field_name="od_bootstrap.teacher_dataset.log_every",
    )
    debug_vis_count = _coerce_int(
        teacher_dataset.get("debug_vis_count", 20),
        field_name="od_bootstrap.teacher_dataset.debug_vis_count",
    )
    debug_vis_seed = _coerce_int(
        teacher_dataset.get("debug_vis_seed", 26),
        field_name="od_bootstrap.teacher_dataset.debug_vis_seed",
    )

    return TeacherDatasetPreset(
        canonical_root=canonical_root,
        output_root=teacher_dataset_root,
        copy_images=copy_images,
        workers=workers,
        log_every=log_every,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
    )


def build_final_dataset_preset(*, output_root: Path | None = None) -> FinalDatasetPreset:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()
    bootstrap_root = _path_from_config(
        paths_config,
        "outputs",
        "bootstrap_root",
        default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap",
    )

    # ===== USER CONFIG: FINAL DATASET PATHS =====
    exhaustive_od_root = _path_from_config(
        paths_config,
        "outputs",
        "exhaustive_od_root",
        default=bootstrap_root / "exhaustive_od",
    )
    aihub_canonical_root = (bootstrap_root / "canonical" / "aihub_standardized").resolve()
    final_output_root = (
        Path(output_root).resolve()
        if output_root is not None
        else _path_from_config(
            paths_config,
            "outputs",
            "final_dataset_root",
            default=REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset",
        )
    )  # PV26 최종 학습 dataset 출력 루트

    # ===== HYPERPARAMETERS: FINAL DATASET BUILD =====
    final_dataset = _config_section(hyperparameters_config, "final_dataset")
    copy_images = _coerce_bool(
        final_dataset.get("copy_images", False),
        field_name="od_bootstrap.final_dataset.copy_images",
    )

    return FinalDatasetPreset(
        exhaustive_od_root=exhaustive_od_root,
        aihub_canonical_root=aihub_canonical_root,
        output_root=final_output_root,
        copy_images=copy_images,
    )


def build_teacher_train_preset(teacher_name: str) -> TeacherTrainScenario:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()
    teacher_common = _config_section(hyperparameters_config, "teacher_train", "common")
    teacher_specific = _config_section(hyperparameters_config, "teacher_train", teacher_name)

    # ===== USER CONFIG: TEACHER TASK SPLIT =====
    default_class_names = {
        "mobility": ("vehicle", "bike", "pedestrian"),
        "signal": ("traffic_light", "sign"),
        "obstacle": ("traffic_cone", "obstacle"),
    }
    default_model_size = {
        "mobility": "s",
        "signal": "s",
        "obstacle": "m",
    }
    default_weights = {
        "mobility": "yolo26s.pt",
        "signal": "yolo26s.pt",
        "obstacle": "yolo26m.pt",
    }
    default_epochs = {
        "mobility": 200,
        "signal": 200,
        "obstacle": 100,
    }
    default_batch = {
        "mobility": 20,
        "signal": 20,
        "obstacle": 12,
    }
    if teacher_name not in default_class_names:
        raise KeyError(f"unknown teacher preset: {teacher_name}")

    run_output_root = _path_from_config(
        paths_config,
        "runs",
        "train_root",
        default=REPO_ROOT / "runs" / "od_bootstrap" / "train",
    )
    teacher_dataset_root = _path_from_config(
        paths_config,
        "outputs",
        "teacher_dataset_root",
        default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets",
    )
    dataset_root = (teacher_dataset_root / teacher_name).resolve()

    class_names = _coerce_str_tuple(
        teacher_specific.get("classes", default_class_names[teacher_name]),
        field_name=f"od_bootstrap.teacher_train.{teacher_name}.classes",
    )
    model_size = _coerce_str(
        teacher_specific.get("model_size", default_model_size[teacher_name]),
        field_name=f"od_bootstrap.teacher_train.{teacher_name}.model_size",
    )
    weights = _coerce_str(
        teacher_specific.get("weights", default_weights[teacher_name]),
        field_name=f"od_bootstrap.teacher_train.{teacher_name}.weights",
    )
    epochs = _coerce_int(
        teacher_specific.get("epochs", default_epochs[teacher_name]),
        field_name=f"od_bootstrap.teacher_train.{teacher_name}.epochs",
    )
    batch = _coerce_int(
        teacher_specific.get("batch", default_batch[teacher_name]),
        field_name=f"od_bootstrap.teacher_train.{teacher_name}.batch",
    )

    # ===== HYPERPARAMETERS: TEACHER TRAIN =====
    return TeacherTrainScenario(
        teacher_name=teacher_name,
        run=TeacherRunConfig(output_root=run_output_root, exist_ok=True),
        dataset=TeacherDatasetConfig(root=dataset_root),
        model=TeacherModelConfig(
            model_size=model_size,
            weights=weights,
            class_names=class_names,
        ),
        train=TeacherTrainParams(
            epochs=epochs,
            imgsz=_coerce_int(teacher_common.get("imgsz", 640), field_name="od_bootstrap.teacher_train.common.imgsz"),
            batch=batch,
            device=_coerce_str(teacher_common.get("device", "cuda:0"), field_name="od_bootstrap.teacher_train.common.device"),
            workers=_coerce_int(teacher_common.get("workers", 8), field_name="od_bootstrap.teacher_train.common.workers"),
            pin_memory=_coerce_bool(
                teacher_common.get("pin_memory", True),
                field_name="od_bootstrap.teacher_train.common.pin_memory",
            ),
            persistent_workers=_coerce_bool(
                teacher_common.get("persistent_workers", True),
                field_name="od_bootstrap.teacher_train.common.persistent_workers",
            ),
            prefetch_factor=_coerce_int(
                teacher_common.get("prefetch_factor", 4),
                field_name="od_bootstrap.teacher_train.common.prefetch_factor",
            ),
            patience=_coerce_int(
                teacher_common.get("patience", 50),
                field_name="od_bootstrap.teacher_train.common.patience",
            ),
            cache=_coerce_bool(teacher_common.get("cache", False), field_name="od_bootstrap.teacher_train.common.cache"),
            amp=_coerce_bool(teacher_common.get("amp", True), field_name="od_bootstrap.teacher_train.common.amp"),
            optimizer=_coerce_str(
                teacher_common.get("optimizer", "auto"),
                field_name="od_bootstrap.teacher_train.common.optimizer",
            ),
            seed=_coerce_int(teacher_common.get("seed", 0), field_name="od_bootstrap.teacher_train.common.seed"),
            resume=_coerce_bool(teacher_common.get("resume", False), field_name="od_bootstrap.teacher_train.common.resume"),
            val=_coerce_bool(teacher_common.get("val", True), field_name="od_bootstrap.teacher_train.common.val"),
            save_period=_coerce_int(
                teacher_common.get("save_period", 10),
                field_name="od_bootstrap.teacher_train.common.save_period",
            ),
            log_every_n_steps=_coerce_int(
                teacher_common.get("log_every_n_steps", 20),
                field_name="od_bootstrap.teacher_train.common.log_every_n_steps",
            ),
            profile_window=_coerce_int(
                teacher_common.get("profile_window", 20),
                field_name="od_bootstrap.teacher_train.common.profile_window",
            ),
            profile_device_sync=_coerce_bool(
                teacher_common.get("profile_device_sync", True),
                field_name="od_bootstrap.teacher_train.common.profile_device_sync",
            ),
        ),
    )


def build_teacher_eval_preset(teacher_name: str) -> CheckpointEvalScenario:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()
    eval_common = _config_section(hyperparameters_config, "teacher_eval", "common")
    eval_specific = _config_section(hyperparameters_config, "teacher_eval", teacher_name)

    # ===== USER CONFIG: TEACHER EVAL TASK SPLIT =====
    default_class_names = {
        "mobility": ("vehicle", "bike", "pedestrian"),
        "signal": ("traffic_light", "sign"),
        "obstacle": ("traffic_cone", "obstacle"),
    }
    if teacher_name not in default_class_names:
        raise KeyError(f"unknown teacher preset: {teacher_name}")

    eval_output_root = _path_from_config(
        paths_config,
        "runs",
        "eval_root",
        default=REPO_ROOT / "runs" / "od_bootstrap" / "eval",
    )
    train_output_root = _path_from_config(
        paths_config,
        "runs",
        "train_root",
        default=REPO_ROOT / "runs" / "od_bootstrap" / "train",
    )
    teacher_dataset_root = _path_from_config(
        paths_config,
        "outputs",
        "teacher_dataset_root",
        default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets",
    )
    dataset_root = (teacher_dataset_root / teacher_name).resolve()
    checkpoint_path = (train_output_root / teacher_name / "weights" / "best.pt").resolve()

    class_names = _coerce_str_tuple(
        eval_specific.get("classes", default_class_names[teacher_name]),
        field_name=f"od_bootstrap.teacher_eval.{teacher_name}.classes",
    )

    # ===== HYPERPARAMETERS: TEACHER EVAL =====
    return CheckpointEvalScenario(
        teacher_name=teacher_name,
        run=CheckpointEvalRunConfig(output_root=eval_output_root, exist_ok=True),
        dataset=CheckpointEvalDatasetConfig(
            root=dataset_root,
            split=_coerce_str(eval_specific.get("split", "val"), field_name=f"od_bootstrap.teacher_eval.{teacher_name}.split"),
            sample_limit=_coerce_int(
                eval_common.get("sample_limit", 8),
                field_name="od_bootstrap.teacher_eval.common.sample_limit",
            ),
        ),
        model=CheckpointEvalModelConfig(
            checkpoint_path=checkpoint_path,
            class_names=class_names,
        ),
        eval=CheckpointEvalParams(
            imgsz=_coerce_int(eval_common.get("imgsz", 640), field_name="od_bootstrap.teacher_eval.common.imgsz"),
            batch=_coerce_int(eval_common.get("batch", 1), field_name="od_bootstrap.teacher_eval.common.batch"),
            device=_coerce_str(eval_common.get("device", "cuda:0"), field_name="od_bootstrap.teacher_eval.common.device"),
            conf=_coerce_float(eval_common.get("conf", 0.25), field_name="od_bootstrap.teacher_eval.common.conf"),
            iou=_coerce_float(eval_common.get("iou", 0.7), field_name="od_bootstrap.teacher_eval.common.iou"),
            predict=_coerce_bool(eval_common.get("predict", True), field_name="od_bootstrap.teacher_eval.common.predict"),
            val=_coerce_bool(eval_common.get("val", True), field_name="od_bootstrap.teacher_eval.common.val"),
            save_conf=_coerce_bool(
                eval_common.get("save_conf", False),
                field_name="od_bootstrap.teacher_eval.common.save_conf",
            ),
            verbose=_coerce_bool(
                eval_common.get("verbose", False),
                field_name="od_bootstrap.teacher_eval.common.verbose",
            ),
        ),
    )


def build_calibration_preset() -> CalibrationScenario:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()
    calibration_run = _config_section(hyperparameters_config, "calibration", "run")
    calibration_search = _config_section(hyperparameters_config, "calibration", "search")
    calibration_hard_negative = _config_section(hyperparameters_config, "calibration", "hard_negative")
    calibration_teachers = _config_section(hyperparameters_config, "calibration", "teachers")

    teacher_defaults = {
        "mobility": {
            "model_version": "mobility_yolov26s_bootstrap_v1",
            "source_dataset_key": "bdd100k_det_100k",
            "image_dir": "images",
            "label_dir": "labels",
            "split": "val",
            "classes": ("vehicle", "bike", "pedestrian"),
        },
        "signal": {
            "model_version": "signal_yolov26s_bootstrap_v1",
            "source_dataset_key": AIHUB_TRAFFIC_DATASET_KEY,
            "image_dir": "images",
            "label_dir": "labels",
            "split": "val",
            "classes": ("traffic_light", "sign"),
        },
        "obstacle": {
            "model_version": "obstacle_yolov26m_bootstrap_v1",
            "source_dataset_key": AIHUB_OBSTACLE_DATASET_KEY,
            "image_dir": "images",
            "label_dir": "labels",
            "split": "val",
            "classes": ("traffic_cone", "obstacle"),
        },
    }
    calibration_output_root = _path_from_config(
        paths_config,
        "runs",
        "calibration_root",
        default=REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default",
    )
    train_output_root = _path_from_config(
        paths_config,
        "runs",
        "train_root",
        default=REPO_ROOT / "runs" / "od_bootstrap" / "train",
    )
    teacher_dataset_root = _path_from_config(
        paths_config,
        "outputs",
        "teacher_dataset_root",
        default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets",
    )

    teachers: list[CalibrationTeacherConfig] = []
    for teacher_name, defaults in teacher_defaults.items():
        teacher_config = dict(defaults)
        teacher_config.update(_config_section(calibration_teachers, teacher_name))
        teachers.append(
            CalibrationTeacherConfig(
                name=teacher_name,
                checkpoint_path=(train_output_root / teacher_name / "weights" / "best.pt").resolve(),
                model_version=_coerce_str(
                    teacher_config.get("model_version", defaults["model_version"]),
                    field_name=f"od_bootstrap.calibration.teachers.{teacher_name}.model_version",
                ),
                dataset=CalibrationDatasetConfig(
                    root=(teacher_dataset_root / teacher_name).resolve(),
                    source_dataset_key=_coerce_str(
                        teacher_config.get("source_dataset_key", defaults["source_dataset_key"]),
                        field_name=f"od_bootstrap.calibration.teachers.{teacher_name}.source_dataset_key",
                    ),
                    image_dir=_coerce_str(
                        teacher_config.get("image_dir", defaults["image_dir"]),
                        field_name=f"od_bootstrap.calibration.teachers.{teacher_name}.image_dir",
                    ),
                    label_dir=_coerce_str(
                        teacher_config.get("label_dir", defaults["label_dir"]),
                        field_name=f"od_bootstrap.calibration.teachers.{teacher_name}.label_dir",
                    ),
                    split=_coerce_str(
                        teacher_config.get("split", defaults["split"]),
                        field_name=f"od_bootstrap.calibration.teachers.{teacher_name}.split",
                    ),
                ),
                classes=_coerce_str_tuple(
                    teacher_config.get("classes", defaults["classes"]),
                    field_name=f"od_bootstrap.calibration.teachers.{teacher_name}.classes",
                ),
            )
        )

    # ===== USER CONFIG: CALIBRATION INPUTS =====
    policy_template_path = _optional_path_from_config(paths_config, "calibration", "policy_template_path")

    # ===== HYPERPARAMETERS: CALIBRATION SEARCH =====
    return CalibrationScenario(
        run=CalibrationRunConfig(
            output_root=calibration_output_root,
            device=_coerce_str(calibration_run.get("device", "cuda:0"), field_name="od_bootstrap.calibration.run.device"),
            imgsz=_coerce_int(calibration_run.get("imgsz", 640), field_name="od_bootstrap.calibration.run.imgsz"),
            batch_size=_coerce_int(
                calibration_run.get("batch_size", 8),
                field_name="od_bootstrap.calibration.run.batch_size",
            ),
            predict_conf=_coerce_float(
                calibration_run.get("predict_conf", 0.001),
                field_name="od_bootstrap.calibration.run.predict_conf",
            ),
            predict_iou=_coerce_float(
                calibration_run.get("predict_iou", 0.99),
                field_name="od_bootstrap.calibration.run.predict_iou",
            ),
        ),
        search=CalibrationSearchConfig(
            match_iou=_coerce_float(
                calibration_search.get("match_iou", 0.5),
                field_name="od_bootstrap.calibration.search.match_iou",
            ),
            min_precision=_coerce_float(
                calibration_search.get("min_precision", 0.90),
                field_name="od_bootstrap.calibration.search.min_precision",
            ),
            min_precision_by_class={
                str(class_name): _coerce_float(
                    value,
                    field_name=f"od_bootstrap.calibration.search.min_precision_by_class.{class_name}",
                )
                for class_name, value in _config_section(calibration_search, "min_precision_by_class").items()
            },
            score_thresholds=_coerce_float_tuple(
                calibration_search.get(
                    "score_thresholds",
                    (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
                ),
                field_name="od_bootstrap.calibration.search.score_thresholds",
            ),
            nms_iou_thresholds=_coerce_float_tuple(
                calibration_search.get("nms_iou_thresholds", (0.40, 0.45, 0.50, 0.55)),
                field_name="od_bootstrap.calibration.search.nms_iou_thresholds",
            ),
            min_box_sizes=_coerce_int_tuple(
                calibration_search.get("min_box_sizes", (4, 6, 8, 10, 12)),
                field_name="od_bootstrap.calibration.search.min_box_sizes",
            ),
        ),
        teachers=tuple(teachers),
        policy_template_path=policy_template_path,
        policy_template=None,
        hard_negative=HardNegativeConfig(
            manifest_path=(calibration_output_root / "hard_negative_manifest.json").resolve(),
            top_k_per_class=_coerce_int(
                calibration_hard_negative.get("top_k_per_class", 25),
                field_name="od_bootstrap.calibration.hard_negative.top_k_per_class",
            ),
            focus_classes=_coerce_str_tuple(
                calibration_hard_negative.get("focus_classes", ("traffic_cone", "obstacle")),
                field_name="od_bootstrap.calibration.hard_negative.focus_classes",
            ),
        ),
    )


def build_sweep_preset() -> BootstrapSweepScenario:
    paths_config = _load_od_bootstrap_paths_config()
    hyperparameters_config = _load_od_bootstrap_hyperparameters_config()
    exhaustive_run = _config_section(hyperparameters_config, "exhaustive_od", "run")
    exhaustive_materialization = _config_section(hyperparameters_config, "exhaustive_od", "materialization")
    exhaustive_teachers = _config_section(hyperparameters_config, "exhaustive_od", "teachers")

    teacher_defaults = {
        "mobility": {
            "base_model": "yolov26s",
            "model_version": "mobility_yolov26s_bootstrap_v1",
            "classes": ("vehicle", "bike", "pedestrian"),
        },
        "signal": {
            "base_model": "yolov26s",
            "model_version": "signal_yolov26s_bootstrap_v1",
            "classes": ("traffic_light", "sign"),
        },
        "obstacle": {
            "base_model": "yolov26m",
            "model_version": "obstacle_yolov26m_bootstrap_v1",
            "classes": ("traffic_cone", "obstacle"),
        },
    }
    bootstrap_root = _path_from_config(
        paths_config,
        "outputs",
        "bootstrap_root",
        default=REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap",
    )
    train_output_root = _path_from_config(
        paths_config,
        "runs",
        "train_root",
        default=REPO_ROOT / "runs" / "od_bootstrap" / "train",
    )
    exhaustive_run_output_root = _path_from_config(
        paths_config,
        "runs",
        "exhaustive_run_root",
        default=REPO_ROOT / "runs" / "od_bootstrap",
    )
    bootstrap_image_list_manifest = (bootstrap_root / "meta" / "bootstrap_image_list.jsonl").resolve()
    exhaustive_output_root = _path_from_config(
        paths_config,
        "outputs",
        "exhaustive_od_root",
        default=bootstrap_root / "exhaustive_od",
    )
    class_policy_path = (
        _path_from_config(
            paths_config,
            "runs",
            "calibration_root",
            default=REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default",
        )
        / "class_policy.yaml"
    ).resolve()

    teachers: list[TeacherConfig] = []
    for teacher_name, defaults in teacher_defaults.items():
        teacher_config = dict(defaults)
        teacher_config.update(_config_section(exhaustive_teachers, teacher_name))
        teachers.append(
            TeacherConfig(
                name=teacher_name,
                base_model=_coerce_str(
                    teacher_config.get("base_model", defaults["base_model"]),
                    field_name=f"od_bootstrap.exhaustive_od.teachers.{teacher_name}.base_model",
                ),
                checkpoint_path=(train_output_root / teacher_name / "weights" / "best.pt").resolve(),
                model_version=_coerce_str(
                    teacher_config.get("model_version", defaults["model_version"]),
                    field_name=f"od_bootstrap.exhaustive_od.teachers.{teacher_name}.model_version",
                ),
                classes=_coerce_str_tuple(
                    teacher_config.get("classes", defaults["classes"]),
                    field_name=f"od_bootstrap.exhaustive_od.teachers.{teacher_name}.classes",
                ),
            )
        )

    # ===== USER CONFIG: EXHAUSTIVE OD INPUTS =====

    # ===== HYPERPARAMETERS: EXHAUSTIVE OD BUILD =====
    return BootstrapSweepScenario(
        run=SweepRunConfig(
            output_root=exhaustive_run_output_root,
            execution_mode=_coerce_str(
                exhaustive_run.get("execution_mode", "model-centric"),
                field_name="od_bootstrap.exhaustive_od.run.execution_mode",
            ),
            device=_coerce_str(
                exhaustive_run.get("device", "cuda:0"),
                field_name="od_bootstrap.exhaustive_od.run.device",
            ),
            imgsz=_coerce_int(
                exhaustive_run.get("imgsz", 640),
                field_name="od_bootstrap.exhaustive_od.run.imgsz",
            ),
            batch_size=_coerce_int(
                exhaustive_run.get("batch_size", 8),
                field_name="od_bootstrap.exhaustive_od.run.batch_size",
            ),
            predict_conf=_coerce_float(
                exhaustive_run.get("predict_conf", 0.001),
                field_name="od_bootstrap.exhaustive_od.run.predict_conf",
            ),
            predict_iou=_coerce_float(
                exhaustive_run.get("predict_iou", 0.99),
                field_name="od_bootstrap.exhaustive_od.run.predict_iou",
            ),
        ),
        image_list=ImageListConfig(manifest_path=bootstrap_image_list_manifest),
        materialization=MaterializationConfig(
            output_root=exhaustive_output_root,
            copy_images=_coerce_bool(
                exhaustive_materialization.get("copy_images", False),
                field_name="od_bootstrap.exhaustive_od.materialization.copy_images",
            ),
        ),
        teachers=tuple(teachers),
        class_policy_path=class_policy_path,
        class_policy=_effective_class_policy(
            hyperparameters_config=hyperparameters_config,
            class_policy_path=class_policy_path,
        ),
    )


__all__ = [
    "FinalDatasetPreset",
    "SourceDebugVisPreset",
    "TeacherDatasetPreset",
    "build_calibration_preset",
    "build_default_source_preset",
    "build_final_dataset_preset",
    "build_sweep_preset",
    "build_teacher_dataset_preset",
    "build_teacher_eval_preset",
    "build_teacher_train_preset",
]
