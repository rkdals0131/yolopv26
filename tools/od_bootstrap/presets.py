from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.pv26_schema import (
    AIHUB_OBSTACLE_DATASET_KEY,
    AIHUB_TRAFFIC_DATASET_KEY,
)
from tools.od_bootstrap.data.source_common import (
    SourcePrepConfig,
    SourceRoots,
)
from tools.od_bootstrap.data.sweep_types import (
    BootstrapSweepScenario,
    ClassPolicy,
    ImageListConfig,
    MaterializationConfig,
    RunConfig as SweepRunConfig,
    TeacherConfig,
)
from tools.od_bootstrap.teacher.eval_types import (
    CheckpointEvalDatasetConfig,
    CheckpointEvalModelConfig,
    CheckpointEvalParams,
    CheckpointEvalRunConfig,
    CheckpointEvalScenario,
)
from tools.od_bootstrap.teacher.calibration_types import (
    CalibrationDatasetConfig,
    CalibrationRunConfig,
    CalibrationScenario,
    CalibrationSearchConfig,
    CalibrationTeacherConfig,
    HardNegativeConfig,
)
from tools.od_bootstrap.teacher.train_types import (
    TeacherDatasetConfig,
    TeacherModelConfig,
    TeacherRunConfig,
    TeacherTrainParams,
    TeacherTrainScenario,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def build_default_source_preset(*, output_root: Path | None = None) -> SourcePrepConfig:
    roots = SourceRoots(
        bdd_root=REPO_ROOT / "seg_dataset" / "BDD100K",
        bdd_images_root=REPO_ROOT / "seg_dataset" / "BDD100K" / "bdd100k_images_100k" / "100k",
        bdd_labels_root=REPO_ROOT / "seg_dataset" / "BDD100K" / "bdd100k_labels_100k" / "100k",
        aihub_root=REPO_ROOT / "seg_dataset" / "AIHUB",
        aihub_lane_root=None,
        aihub_obstacle_root=None,
        aihub_traffic_root=None,
        aihub_docs_root=None,
    )
    return SourcePrepConfig(
        roots=roots,
        output_root=(output_root or (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap")).resolve(),
        workers=1,
        force_reprocess=False,
        write_source_readmes=False,
        debug_vis_count=20,
        debug_vis_seed=26,
    )


def build_teacher_dataset_preset(*, output_root: Path | None = None) -> TeacherDatasetPreset:
    return TeacherDatasetPreset(
        canonical_root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap").resolve(),
        output_root=(output_root or (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets")).resolve(),
        copy_images=False,
        workers=8,
        log_every=500,
        debug_vis_count=20,
        debug_vis_seed=26,
    )


def build_final_dataset_preset(*, output_root: Path | None = None) -> FinalDatasetPreset:
    return FinalDatasetPreset(
        exhaustive_od_root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "exhaustive_od").resolve(),
        aihub_canonical_root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "canonical" / "aihub_standardized").resolve(),
        output_root=(output_root or (REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset")).resolve(),
        copy_images=False,
    )


def build_teacher_train_preset(teacher_name: str) -> TeacherTrainScenario:
    class_names_by_teacher = {
        "mobility": ("vehicle", "bike", "pedestrian"),
        "signal": ("traffic_light", "sign"),
        "obstacle": ("traffic_cone", "obstacle"),
    }
    model_size_by_teacher = {
        "mobility": "s",
        "signal": "s",
        "obstacle": "m",
    }
    weights_by_teacher = {
        "mobility": "yolo26s.pt",
        "signal": "yolo26s.pt",
        "obstacle": "yolo26m.pt",
    }
    batch_by_teacher = {
        "mobility": 20,
        "signal": 20,
        "obstacle": 12,
    }
    epochs_by_teacher = {
        "mobility": 200,
        "signal": 200,
        "obstacle": 100,
    }
    if teacher_name not in class_names_by_teacher:
        raise KeyError(f"unknown teacher preset: {teacher_name}")
    return TeacherTrainScenario(
        teacher_name=teacher_name,
        run=TeacherRunConfig(output_root=(REPO_ROOT / "runs" / "od_bootstrap" / "train").resolve(), exist_ok=True),
        dataset=TeacherDatasetConfig(
            root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / teacher_name).resolve(),
        ),
        model=TeacherModelConfig(
            model_size=model_size_by_teacher[teacher_name],
            weights=weights_by_teacher[teacher_name],
            class_names=class_names_by_teacher[teacher_name],
        ),
        train=TeacherTrainParams(
            epochs=epochs_by_teacher[teacher_name],
            imgsz=640,
            batch=batch_by_teacher[teacher_name],
            device="cuda:0",
            workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            patience=50,
            cache=False,
            amp=True,
            optimizer="auto",
            seed=0,
            resume=False,
            val=True,
            save_period=10,
            log_every_n_steps=20,
            profile_window=20,
            profile_device_sync=True,
        ),
    )


def build_teacher_eval_preset(teacher_name: str) -> CheckpointEvalScenario:
    class_names_by_teacher = {
        "mobility": ("vehicle", "bike", "pedestrian"),
        "signal": ("traffic_light", "sign"),
        "obstacle": ("traffic_cone", "obstacle"),
    }
    model_size_by_teacher = {
        "mobility": "n",
        "signal": "n",
        "obstacle": "n",
    }
    if teacher_name not in class_names_by_teacher:
        raise KeyError(f"unknown teacher preset: {teacher_name}")
    return CheckpointEvalScenario(
        teacher_name=teacher_name,
        run=CheckpointEvalRunConfig(output_root=(REPO_ROOT / "runs" / "od_bootstrap" / "eval").resolve(), exist_ok=True),
        dataset=CheckpointEvalDatasetConfig(
            root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / teacher_name).resolve(),
            split="val",
            sample_limit=8,
        ),
        model=CheckpointEvalModelConfig(
            checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / teacher_name / "weights" / "best.pt").resolve(),
            class_names=class_names_by_teacher[teacher_name],
            model_size=model_size_by_teacher[teacher_name],
        ),
        eval=CheckpointEvalParams(
            imgsz=640,
            batch=1,
            device="cuda:0",
            conf=0.25,
            iou=0.7,
            predict=True,
            val=True,
            save_conf=False,
            verbose=False,
        ),
    )


def build_calibration_preset() -> CalibrationScenario:
    return CalibrationScenario(
        run=CalibrationRunConfig(
            output_root=(REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default").resolve(),
            device="cuda:0",
            imgsz=640,
            batch_size=8,
            predict_conf=0.001,
            predict_iou=0.99,
        ),
        search=CalibrationSearchConfig(
            match_iou=0.5,
            min_precision=0.90,
            min_precision_by_class={"traffic_light": 0.75},
            score_thresholds=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
            nms_iou_thresholds=(0.40, 0.45, 0.50, 0.55),
            min_box_sizes=(4, 6, 8, 10, 12),
        ),
        teachers=(
            CalibrationTeacherConfig(
                name="mobility",
                checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / "mobility" / "weights" / "best.pt").resolve(),
                model_version="mobility_yolov26s_bootstrap_v1",
                dataset=CalibrationDatasetConfig(
                    root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / "mobility").resolve(),
                    source_dataset_key="bdd100k_det_100k",
                    image_dir="images",
                    label_dir="labels",
                    split="val",
                ),
                classes=("vehicle", "bike", "pedestrian"),
            ),
            CalibrationTeacherConfig(
                name="signal",
                checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / "signal" / "weights" / "best.pt").resolve(),
                model_version="signal_yolov26s_bootstrap_v1",
                dataset=CalibrationDatasetConfig(
                    root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / "signal").resolve(),
                    source_dataset_key=AIHUB_TRAFFIC_DATASET_KEY,
                    image_dir="images",
                    label_dir="labels",
                    split="val",
                ),
                classes=("traffic_light", "sign"),
            ),
            CalibrationTeacherConfig(
                name="obstacle",
                checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / "obstacle" / "weights" / "best.pt").resolve(),
                model_version="obstacle_yolov26m_bootstrap_v1",
                dataset=CalibrationDatasetConfig(
                    root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / "obstacle").resolve(),
                    source_dataset_key=AIHUB_OBSTACLE_DATASET_KEY,
                    image_dir="images",
                    label_dir="labels",
                    split="val",
                ),
                classes=("traffic_cone", "obstacle"),
            ),
        ),
        policy_template_path=(REPO_ROOT / "tools" / "od_bootstrap" / "config" / "sweep" / "class_policy.template.yaml").resolve(),
        policy_template=None,
        hard_negative=HardNegativeConfig(
            manifest_path=(REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default" / "hard_negative_manifest.json").resolve(),
            top_k_per_class=25,
            focus_classes=("traffic_cone", "obstacle"),
        ),
    )


def build_sweep_preset() -> BootstrapSweepScenario:
    return BootstrapSweepScenario(
        run=SweepRunConfig(
            output_root=(REPO_ROOT / "runs" / "od_bootstrap").resolve(),
            execution_mode="model-centric",
            device="cuda:0",
            imgsz=640,
            batch_size=8,
            predict_conf=0.001,
            predict_iou=0.99,
        ),
        image_list=ImageListConfig(
            manifest_path=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "meta" / "bootstrap_image_list.jsonl").resolve()
        ),
        materialization=MaterializationConfig(
            output_root=(REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "exhaustive_od").resolve(),
            copy_images=False,
        ),
        teachers=(
            TeacherConfig(
                name="mobility",
                base_model="yolov26s",
                checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / "mobility" / "weights" / "best.pt").resolve(),
                model_version="mobility_yolov26s_bootstrap_v1",
                classes=("vehicle", "bike", "pedestrian"),
            ),
            TeacherConfig(
                name="signal",
                base_model="yolov26s",
                checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / "signal" / "weights" / "best.pt").resolve(),
                model_version="signal_yolov26s_bootstrap_v1",
                classes=("traffic_light", "sign"),
            ),
            TeacherConfig(
                name="obstacle",
                base_model="yolov26m",
                checkpoint_path=(REPO_ROOT / "runs" / "od_bootstrap" / "train" / "obstacle" / "weights" / "best.pt").resolve(),
                model_version="obstacle_yolov26m_bootstrap_v1",
                classes=("traffic_cone", "obstacle"),
            ),
        ),
        class_policy_path=(REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default" / "class_policy.yaml").resolve(),
        class_policy={
            "vehicle": ClassPolicy(0.25, 0.55, 4),
            "bike": ClassPolicy(0.25, 0.55, 4),
            "pedestrian": ClassPolicy(0.25, 0.55, 4),
            "traffic_light": ClassPolicy(0.30, 0.50, 4),
            "sign": ClassPolicy(0.25, 0.50, 4),
            "traffic_cone": ClassPolicy(0.25, 0.55, 4),
            "obstacle": ClassPolicy(0.25, 0.55, 4),
        },
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
