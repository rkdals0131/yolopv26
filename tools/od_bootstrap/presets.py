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


def build_default_source_preset(*, output_root: Path | None = None) -> SourcePrepConfig:
    # ===== USER CONFIG: SOURCE PATHS =====
    bdd_root = REPO_ROOT / "seg_dataset" / "BDD100K"  # BDD100K 원본 루트
    bdd_images_root = bdd_root / "bdd100k_images_100k" / "100k"  # BDD image split 루트
    bdd_labels_root = bdd_root / "bdd100k_labels" / "100k"  # BDD detection label split 루트
    aihub_root = REPO_ROOT / "seg_dataset" / "AIHUB"  # AIHUB 원본 통합 루트
    source_output_root = output_root or (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap")  # canonical 출력 루트

    # ===== HYPERPARAMETERS: SOURCE PREP =====
    workers = 1  # 전처리 병렬 worker 수
    force_reprocess = False  # 기존 산출물이 있어도 다시 만들지 여부
    write_source_readmes = False  # 원본 dataset README를 다시 생성할지 여부
    debug_vis_count = 20  # 샘플 debug vis 생성 개수
    debug_vis_seed = 26  # debug vis 샘플링 시드

    roots = SourceRoots(
        bdd_root=bdd_root,
        bdd_images_root=bdd_images_root,
        bdd_labels_root=bdd_labels_root,
        aihub_root=aihub_root,
        aihub_lane_root=None,
        aihub_obstacle_root=None,
        aihub_traffic_root=None,
        aihub_docs_root=None,
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
    # ===== USER CONFIG: TEACHER DATASET PATHS =====
    canonical_root = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap").resolve()  # canonical bootstrap 입력 루트
    teacher_dataset_root = (
        output_root or (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets")
    ).resolve()  # teacher dataset 출력 루트

    # ===== HYPERPARAMETERS: TEACHER DATASET BUILD =====
    copy_images = False  # 이미지를 복사할지, 링크/참조 중심으로 둘지
    workers = 8  # teacher dataset materialize worker 수
    log_every = 500  # 진행 로그 간격
    debug_vis_count = 20  # teacher dataset debug vis 개수
    debug_vis_seed = 26  # teacher dataset debug vis 시드

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
    # ===== USER CONFIG: FINAL DATASET PATHS =====
    exhaustive_od_root = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "exhaustive_od").resolve()
    aihub_canonical_root = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "canonical" / "aihub_standardized").resolve()
    final_output_root = (
        output_root or (REPO_ROOT / "seg_dataset" / "pv26_exhaustive_od_lane_dataset")
    ).resolve()  # PV26 최종 학습 dataset 출력 루트

    # ===== HYPERPARAMETERS: FINAL DATASET BUILD =====
    copy_images = False  # 최종 dataset 이미지 복사 여부

    return FinalDatasetPreset(
        exhaustive_od_root=exhaustive_od_root,
        aihub_canonical_root=aihub_canonical_root,
        output_root=final_output_root,
        copy_images=copy_images,
    )


def build_teacher_train_preset(teacher_name: str) -> TeacherTrainScenario:
    # ===== USER CONFIG: TEACHER TASK SPLIT =====
    class_names_by_teacher = {
        "mobility": ("vehicle", "bike", "pedestrian"),  # 이동체 teacher가 담당할 클래스
        "signal": ("traffic_light", "sign"),  # 신호등/표지판 teacher 클래스
        "obstacle": ("traffic_cone", "obstacle"),  # 장애물 teacher 클래스
    }
    model_size_by_teacher = {
        "mobility": "s",  # mobility는 속도 우선으로 small 백본 사용
        "signal": "s",  # signal도 small 백본 사용
        "obstacle": "m",  # obstacle은 난도가 높아 medium 백본 사용
    }
    weights_by_teacher = {
        "mobility": "yolo26s.pt",  # mobility teacher 초기 가중치
        "signal": "yolo26s.pt",  # signal teacher 초기 가중치
        "obstacle": "yolo26m.pt",  # obstacle teacher 초기 가중치
    }

    # ===== HYPERPARAMETERS: TEACHER TRAIN =====
    batch_by_teacher = {
        "mobility": 20,  # teacher별 train batch 크기
        "signal": 20,
        "obstacle": 12,
    }
    epochs_by_teacher = {
        "mobility": 200,  # teacher별 총 epoch
        "signal": 200,
        "obstacle": 100,
    }
    if teacher_name not in class_names_by_teacher:
        raise KeyError(f"unknown teacher preset: {teacher_name}")

    run_output_root = (REPO_ROOT / "runs" / "od_bootstrap" / "train").resolve()  # teacher checkpoint 저장 루트
    dataset_root = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / teacher_name).resolve()

    return TeacherTrainScenario(
        teacher_name=teacher_name,
        run=TeacherRunConfig(output_root=run_output_root, exist_ok=True),
        dataset=TeacherDatasetConfig(
            root=dataset_root,
        ),
        model=TeacherModelConfig(
            model_size=model_size_by_teacher[teacher_name],
            weights=weights_by_teacher[teacher_name],
            class_names=class_names_by_teacher[teacher_name],
        ),
        train=TeacherTrainParams(
            epochs=epochs_by_teacher[teacher_name],  # 총 학습 epoch
            imgsz=640,  # 입력 해상도
            batch=batch_by_teacher[teacher_name],  # teacher별 batch 크기
            device="cuda:0",  # 학습 장치
            workers=8,  # dataloader worker 수
            pin_memory=True,  # host->GPU 전송 최적화
            persistent_workers=True,  # epoch 사이 worker 재사용
            prefetch_factor=4,  # worker별 prefetch 배치 수
            patience=50,  # early stopping patience
            cache=False,  # ultralytics dataset cache 사용 여부
            amp=True,  # mixed precision 사용 여부
            optimizer="auto",  # ultralytics optimizer 선택 방식
            seed=0,  # 재현성 시드
            resume=False,  # 이전 teacher run 이어서 학습 여부
            val=True,  # epoch 검증 수행 여부
            save_period=10,  # 체크포인트 저장 주기
            log_every_n_steps=20,  # step 로그 간격
            profile_window=20,  # profiling 평균 창 길이
            profile_device_sync=True,  # timing 측정 전 device sync 여부
        ),
    )


def build_teacher_eval_preset(teacher_name: str) -> CheckpointEvalScenario:
    # ===== USER CONFIG: TEACHER EVAL TASK SPLIT =====
    class_names_by_teacher = {
        "mobility": ("vehicle", "bike", "pedestrian"),
        "signal": ("traffic_light", "sign"),
        "obstacle": ("traffic_cone", "obstacle"),
    }

    # ===== HYPERPARAMETERS: TEACHER EVAL =====
    model_size_by_teacher = {
        "mobility": "n",  # eval 리포트용 model-size 표시값
        "signal": "n",
        "obstacle": "n",
    }
    if teacher_name not in class_names_by_teacher:
        raise KeyError(f"unknown teacher preset: {teacher_name}")

    eval_output_root = (REPO_ROOT / "runs" / "od_bootstrap" / "eval").resolve()
    dataset_root = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / teacher_name).resolve()
    checkpoint_path = (REPO_ROOT / "runs" / "od_bootstrap" / "train" / teacher_name / "weights" / "best.pt").resolve()

    return CheckpointEvalScenario(
        teacher_name=teacher_name,
        run=CheckpointEvalRunConfig(output_root=eval_output_root, exist_ok=True),
        dataset=CheckpointEvalDatasetConfig(
            root=dataset_root,
            split="val",  # 평가에 사용할 split
            sample_limit=8,  # 빠른 체크용 샘플 제한
        ),
        model=CheckpointEvalModelConfig(
            checkpoint_path=checkpoint_path,
            class_names=class_names_by_teacher[teacher_name],
            model_size=model_size_by_teacher[teacher_name],
        ),
        eval=CheckpointEvalParams(
            imgsz=640,  # 평가 입력 해상도
            batch=1,  # eval batch 크기
            device="cuda:0",  # 평가 장치
            conf=0.25,  # 예측 confidence threshold
            iou=0.7,  # NMS IoU threshold
            predict=True,  # 예측 결과 저장/요약 수행 여부
            val=True,  # 정식 validation 수행 여부
            save_conf=False,  # confidence를 결과 파일에 남길지 여부
            verbose=False,  # ultralytics verbose 출력 여부
        ),
    )


def build_calibration_preset() -> CalibrationScenario:
    # ===== USER CONFIG: CALIBRATION INPUTS =====
    calibration_output_root = (REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default").resolve()

    # ===== HYPERPARAMETERS: CALIBRATION SEARCH =====
    return CalibrationScenario(
        run=CalibrationRunConfig(
            output_root=calibration_output_root,
            device="cuda:0",  # calibration 추론 장치
            imgsz=640,  # calibration 입력 해상도
            batch_size=8,  # calibration 추론 batch 크기
            predict_conf=0.001,  # calibration용 낮은 confidence threshold
            predict_iou=0.99,  # calibration용 느슨한 NMS IoU threshold
        ),
        search=CalibrationSearchConfig(
            match_iou=0.5,  # GT와 teacher 예측을 match할 IoU
            min_precision=0.90,  # 기본 최소 precision 목표
            min_precision_by_class={"traffic_light": 0.75},  # 클래스별 예외 precision 목표
            score_thresholds=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),  # score 탐색 후보
            nms_iou_thresholds=(0.40, 0.45, 0.50, 0.55),  # NMS IoU 탐색 후보
            min_box_sizes=(4, 6, 8, 10, 12),  # 최소 box 크기 탐색 후보
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
        policy_template_path=None,
        policy_template=None,
        hard_negative=HardNegativeConfig(
            manifest_path=(calibration_output_root / "hard_negative_manifest.json").resolve(),
            top_k_per_class=25,  # 클래스별 hard negative 최대 샘플 수
            focus_classes=("traffic_cone", "obstacle"),  # hard negative 집중 클래스
        ),
    )


def build_sweep_preset() -> BootstrapSweepScenario:
    # ===== USER CONFIG: EXHAUSTIVE OD INPUTS =====
    exhaustive_run_output_root = (REPO_ROOT / "runs" / "od_bootstrap").resolve()
    bootstrap_image_list_manifest = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "meta" / "bootstrap_image_list.jsonl").resolve()
    exhaustive_output_root = (REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "exhaustive_od").resolve()
    class_policy_path = (REPO_ROOT / "runs" / "od_bootstrap" / "calibration" / "default" / "class_policy.yaml").resolve()

    # ===== HYPERPARAMETERS: EXHAUSTIVE OD BUILD =====
    return BootstrapSweepScenario(
        run=SweepRunConfig(
            output_root=exhaustive_run_output_root,
            execution_mode="model-centric",  # teacher inference 실행 방식
            device="cuda:0",  # exhaustive OD 생성 추론 장치
            imgsz=640,  # teacher 입력 해상도
            batch_size=8,  # teacher inference batch 크기
            predict_conf=0.001,  # teacher 예측 수집 confidence threshold
            predict_iou=0.99,  # teacher 예측 수집 NMS IoU threshold
        ),
        image_list=ImageListConfig(
            manifest_path=bootstrap_image_list_manifest
        ),
        materialization=MaterializationConfig(
            output_root=exhaustive_output_root,
            copy_images=False,  # exhaustive dataset 이미지 복사 여부
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
        class_policy_path=class_policy_path,
        class_policy={
            "vehicle": ClassPolicy(0.25, 0.55, 4),  # score, nms_iou, min_box_size
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
