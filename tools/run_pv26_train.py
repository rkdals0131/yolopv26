from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import asdict
from pathlib import Path
import site
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(REPO_ROOT))

from common.overlay import render_overlay
from common.user_config import (
    load_user_hyperparameters_config,
    load_user_paths_config,
    nested_get,
)
from model.engine.evaluator import PV26Evaluator
from model.engine.postprocess import PV26PostprocessConfig
from model.data import (
    PV26CanonicalDataset,
    build_pv26_eval_dataloader,
    build_pv26_train_dataloader,
    collate_pv26_samples,
)
from model.engine.trainer import PV26Trainer, build_pv26_scheduler
from model.engine.trainer import _resolve_summary_path
from model.net import PV26Heads
from model.net import build_yolo26n_trunk
try:
    from model.net import build_yolo26_trunk
except ImportError:  # pragma: no cover - compatibility while trunk API is being generalized.
    build_yolo26_trunk = None
try:
    from model.net import infer_pyramid_channels
except ImportError:  # pragma: no cover - compatibility while trunk API is being generalized.
    infer_pyramid_channels = None
try:
    from model.net import resolve_yolo26_weights
except ImportError:  # pragma: no cover - compatibility while trunk API is being generalized.
    resolve_yolo26_weights = None
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
from tools.pv26_train_config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_PRESET_NAME,
    DEFAULT_RUN_ROOT,
    PHASE_STAGE_ORDER,
    DatasetConfig,
    EntryConfig,
    MetaTrainScenario,
    PhaseConfig,
    PreviewConfig,
    RunConfig,
    SelectionConfig,
    TrainDefaultsConfig,
    apply_user_config_to_preset as _apply_user_config_to_preset,
    build_pv26_train_path_overrides as _build_pv26_train_path_overrides,
    deep_merge_mappings as _deep_merge_mappings,
    meta_train_scenario_from_mapping as _meta_train_scenario_from_mapping,
    phase as _phase,
    phase_config_from_mapping as _phase_config_from_mapping,
    phase_to_mapping as _phase_to_mapping,
    preview_config_from_mapping as _preview_config_from_mapping,
    resolve_train_batch_limit as _resolve_train_batch_limit,
    resolve_val_batch_limit as _resolve_val_batch_limit,
    run_config_from_mapping as _run_config_from_mapping,
    scenario_phase_defaults as _scenario_phase_defaults,
    scenario_to_mapping as _scenario_to_mapping,
    selection_config_from_mapping as _selection_config_from_mapping,
    train_defaults_from_mapping as _train_defaults_from_mapping,
    validate_meta_train_scenario as _validate_meta_train_scenario,
    dataset_config_from_mapping as _dataset_config_from_mapping,
    resolve_phase_selection as _resolve_phase_selection,
)


PRESET_PATH_ROOT = REPO_ROOT / "presets" / "pv26_meta_train"
BACKBONE_HEAD_CHANNELS = {
    "n": (64, 128, 256),
    "s": (128, 256, 512),
}
META_MANIFEST_VERSION = "pv26-meta-train-v1"
# IDE에서 아래 검색어로 조절 지점을 바로 찾을 수 있다.
# ===== USER CONFIG =====
# ===== HYPERPARAMETERS =====
# ===== PHASE HYPERPARAMETERS =====


def _resolve_backbone_weights(train_config: TrainDefaultsConfig) -> str:
    if resolve_yolo26_weights is not None:
        return resolve_yolo26_weights(
            variant=train_config.backbone_variant,
            weights=train_config.backbone_weights,
        )
    if train_config.backbone_weights:
        return str(train_config.backbone_weights)
    return f"yolo26{train_config.backbone_variant}.pt"


def _build_backbone_adapter(train_config: TrainDefaultsConfig) -> Any:
    weights = _resolve_backbone_weights(train_config)
    if build_yolo26_trunk is not None:
        return build_yolo26_trunk(
            variant=train_config.backbone_variant,
            weights=weights,
        )
    return build_yolo26n_trunk(weights=weights)


def _resolve_head_channels(adapter: Any, train_config: TrainDefaultsConfig) -> tuple[int, int, int]:
    if infer_pyramid_channels is not None:
        channels = tuple(int(value) for value in infer_pyramid_channels(adapter))
        if len(channels) == 3:
            return channels
    return _configured_head_channels(train_config)


def _configured_head_channels(train_config: TrainDefaultsConfig) -> tuple[int, int, int]:
    try:
        return BACKBONE_HEAD_CHANNELS[str(train_config.backbone_variant)]
    except KeyError as exc:
        raise KeyError(
            "unsupported backbone variant for fallback head-channel resolution: "
            f"{train_config.backbone_variant!r}"
        ) from exc


def _build_postprocess_config(train_config: TrainDefaultsConfig) -> PV26PostprocessConfig:
    return PV26PostprocessConfig(
        det_conf_threshold=float(train_config.det_conf_threshold),
        det_iou_threshold=float(train_config.det_iou_threshold),
        lane_obj_threshold=float(train_config.lane_obj_threshold),
        stop_line_obj_threshold=float(train_config.stop_line_obj_threshold),
        crosswalk_obj_threshold=float(train_config.crosswalk_obj_threshold),
    )


def _wrap_evaluator_postprocess_config(
    evaluator: PV26Evaluator,
    *,
    postprocess_config: PV26PostprocessConfig,
) -> PV26Evaluator:
    original_evaluate_batch = evaluator.evaluate_batch
    original_predict_batch = evaluator.predict_batch

    def evaluate_batch_with_default(
        self: PV26Evaluator,
        batch: dict[str, Any],
        *,
        include_predictions: bool = False,
        compute_loss: bool = True,
        config: PV26PostprocessConfig | None = None,
    ) -> dict[str, Any]:
        return original_evaluate_batch(
            batch,
            include_predictions=include_predictions,
            compute_loss=compute_loss,
            config=config or postprocess_config,
        )

    def predict_batch_with_default(
        self: PV26Evaluator,
        batch: dict[str, Any],
        *,
        config: PV26PostprocessConfig | None = None,
    ) -> list[dict[str, Any]]:
        return original_predict_batch(batch, config=config or postprocess_config)

    evaluator.evaluate_batch = MethodType(evaluate_batch_with_default, evaluator)
    evaluator.predict_batch = MethodType(predict_batch_with_default, evaluator)
    setattr(evaluator, "postprocess_config", postprocess_config)
    return evaluator


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
        checkpoint_every=10,  # 몇 epoch마다 체크포인트를 남길지
        num_workers=6,  # dataloader worker 수
        pin_memory=True,  # host->GPU 전송 최적화
        log_every_n_steps=20,  # step 로그 간격
        profile_window=20,  # timing 평균 창 길이
        profile_device_sync=True,  # timing 측정 전 device sync 여부
        encode_train_batches_in_loader=True,  # train loader에서 미리 target encode 수행 여부
        encode_val_batches_in_loader=True,  # val loader에서 미리 target encode 수행 여부
        persistent_workers=True,  # epoch 사이 worker 유지 여부
        prefetch_factor=2,  # worker별 prefetch 배치 수
        backbone_variant="s",  # 기본 YOLO26 backbone scale
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
            min_epochs=4,  # 최소 epoch
            max_epochs=12,  # 최대 epoch
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
            min_epochs=6,
            max_epochs=18,
            patience=2,
            min_improvement_pct=0.5,
            overrides={
                "trunk_lr": 3e-5,  # stage 2 backbone LR
                "head_lr": 8e-4,  # stage 2 head LR
            },
        ),
        _phase(
            "end_to_end_finetune",
            "stage_3_end_to_end_finetune",
            min_epochs=8,
            max_epochs=24,
            patience=3,
            min_improvement_pct=0.25,
            overrides={
                "trunk_lr": 1e-5,  # stage 3 backbone LR
                "head_lr": 4e-4,  # stage 3 head LR
            },
        ),
        _phase(
            "lane_family_finetune",
            "stage_4_lane_family_finetune",
            min_epochs=4,
            max_epochs=12,
            patience=3,
            min_improvement_pct=0.25,
            selection=SelectionConfig(
                metric_path="val.metrics.lane_family.mean_f1",
                mode="max",
                eps=1e-8,
            ),
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
    }
    return {
        preset_name: _apply_user_config_to_preset(
            preset_name,
            scenario,
            paths_config=paths_config,
            hyperparameters_config=hyperparameters_config,
            repo_root=REPO_ROOT,
        )
        for preset_name, scenario in presets.items()
    }


# Default CLI preset; override with `--preset`.
ENTRY_CONFIG = EntryConfig()


def _log_meta_train(message: str) -> None:
    print(f"[meta_train] {message}", flush=True)


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
            "metric_path": self.selection.metric_path,
            "metric_mode": self.selection.mode,
            "current_metric_value": float(current_metric_value),
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "plateau_count": int(self.plateau_count),
            "last_improvement_pct": self.last_improvement_pct,
            "selection_metric_path": self.selection.metric_path,
            "selection_mode": self.selection.mode,
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

        # Large encoded CPU batches can exhaust process file descriptors when
        # PyTorch shares storages via duplicated FDs.
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
        ratios=train_config.sampler_ratios,
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
    adapter = _build_backbone_adapter(train_config)
    head_channels = _resolve_head_channels(adapter, train_config)
    heads = PV26Heads(in_channels=head_channels)
    trainer = PV26Trainer(
        adapter,
        heads,
        stage=phase.stage,
        device=train_config.device,
        loss_weights=phase.loss_weights or None,
        freeze_policy=phase.freeze_policy,
        trunk_lr=train_config.trunk_lr,
        head_lr=train_config.head_lr,
        weight_decay=train_config.weight_decay,
        amp=train_config.amp,
        accumulate_steps=train_config.accumulate_steps,
        grad_clip_norm=train_config.grad_clip_norm,
    )
    trainer.scheduler = build_pv26_scheduler(
        trainer.optimizer,
        epochs=phase.max_epochs,
        schedule=train_config.schedule,
    )
    postprocess_config = _build_postprocess_config(train_config)
    original_build_evaluator = trainer.build_evaluator

    def build_evaluator_with_postprocess(self: PV26Trainer):
        evaluator = original_build_evaluator()
        return _wrap_evaluator_postprocess_config(
            evaluator,
            postprocess_config=postprocess_config,
        )

    trainer.build_evaluator = MethodType(build_evaluator_with_postprocess, trainer)
    setattr(trainer, "postprocess_config", postprocess_config)
    return trainer


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
    phase_selection = _resolve_phase_selection(scenario.selection, phase)
    backbone_weights = _resolve_backbone_weights(train_config)
    head_channels = _configured_head_channels(train_config)
    postprocess_config = _build_postprocess_config(train_config)
    return {
        "entry_script": "tools/run_pv26_train.py",
        "scenario_path": str(scenario_path),
        "dataset_config": _json_ready(asdict(scenario.dataset)),
        "selection": _json_ready(asdict(scenario.selection)),
        "phase_selection": _json_ready(asdict(phase_selection)),
        "preview": _json_ready(asdict(scenario.preview)),
        "phase": {
            "index": int(phase_index),
            "name": phase.name,
            "stage": phase.stage,
            "min_epochs": int(phase.min_epochs),
            "max_epochs": int(phase.max_epochs),
            "patience": int(phase.patience),
            "min_improvement_pct": float(phase.min_improvement_pct),
            "selection": _json_ready(asdict(phase_selection)),
            "loss_weights": _json_ready(phase.loss_weights),
            "freeze_policy": phase.freeze_policy,
        },
        "phase_train_config": _json_ready(asdict(train_config)),
        "backbone": {
            "variant": train_config.backbone_variant,
            "weights": backbone_weights,
        },
        "postprocess": _json_ready(asdict(postprocess_config)),
        "head_channels": list(head_channels),
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
    phase_selection = _resolve_phase_selection(scenario.selection, phase)
    _log_meta_train(f"building loaders for phase_{phase_index} at {phase_run_dir}")
    train_loader, val_loader = _build_phase_train_loaders(dataset, train_config=phase_train_config)
    controller = PhaseTransitionController(phase=phase, selection=phase_selection)
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
        best_metric=phase_selection.metric_path,
        best_mode=phase_selection.mode,
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
        "selection": _json_ready(asdict(phase_selection)),
        "backbone": {
            "variant": phase_train_config.backbone_variant,
            "weights": _resolve_backbone_weights(phase_train_config),
        },
        "postprocess": _json_ready(asdict(_build_postprocess_config(phase_train_config))),
        "head_channels": list(_configured_head_channels(phase_train_config)),
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
            f"patience={phase.patience}, min_improvement_pct={phase.min_improvement_pct}, "
            f"selection={_resolve_phase_selection(scenario.selection, phase).metric_path} "
            f"({_resolve_phase_selection(scenario.selection, phase).mode})"
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
