from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import os
from pathlib import Path
import time
from types import MethodType
from typing import Any

from common.overlay import render_overlay
from model.data import (
    PV26CanonicalDataset,
    build_pv26_eval_dataloader,
    build_pv26_train_dataloader,
    collate_pv26_samples,
)
from model.engine.evaluator import PV26Evaluator
from model.engine.postprocess import PV26PostprocessConfig
from model.engine.trainer import PV26Trainer, build_pv26_scheduler
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
    read_jsonl as _read_jsonl,
    recover_phase_entry_from_run_dir as _recover_phase_entry_from_run_dir,
    resolve_meta_run_dir as _resolve_meta_run_dir,
    safe_name as _safe_name,
    write_json as _write_json,
    write_meta_manifest as _write_meta_manifest,
    write_meta_summary as _write_meta_summary,
)
from tools.pv26_train_config import (
    MetaTrainScenario,
    PhaseConfig,
    PreviewConfig,
    TrainDefaultsConfig,
    resolve_phase_selection as _resolve_phase_selection,
    resolve_train_batch_limit as _resolve_train_batch_limit,
    resolve_val_batch_limit as _resolve_val_batch_limit,
    scenario_phase_defaults as _scenario_phase_defaults,
)

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


def _log_meta_train(message: str) -> None:
    print(f"[pv26-meta-train] {message}", flush=True)


def _facade_module():
    from tools import run_pv26_train as facade

    return facade


def _scenario_snapshot_for_run(
    scenario: MetaTrainScenario,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    return _facade_module()._scenario_snapshot_for_run(scenario, run_dir=run_dir)


def PhaseTransitionController(*args, **kwargs):
    return _facade_module().PhaseTransitionController(*args, **kwargs)

def _facade_module()._configure_torch_multiprocessing() -> None:
    try:
        import torch

        torch_module_path = getattr(torch, "__file__", None)
        if torch_module_path:
            torch_lib_dir = Path(torch_module_path).resolve().parent / "lib"
            if torch_lib_dir.is_dir():
                current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                ld_library_entries = [entry for entry in current_ld_library_path.split(os.pathsep) if entry]
                torch_lib_dir_str = str(torch_lib_dir)
                if torch_lib_dir_str not in ld_library_entries:
                    os.environ["LD_LIBRARY_PATH"] = (
                        os.pathsep.join((torch_lib_dir_str, *ld_library_entries))
                        if ld_library_entries
                        else torch_lib_dir_str
                    )

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
    return _sample_preview_selection_with_logging(dataset, preview, progress_callback=None)


def _sample_preview_selection_with_logging(
    dataset: PV26CanonicalDataset,
    preview: PreviewConfig,
    *,
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
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
    if missing and progress_callback is not None:
        available = {dataset_key: count for dataset_key, count in counts.items() if count > 0}
        if available:
            progress_callback(
                "preview selection fallback: "
                f"missing keys={missing}, available_counts={available}, split={preview.split}"
            )
        else:
            progress_callback(
                "preview selection skipped: "
                f"no samples found for requested keys={list(preview.dataset_keys)} on split={preview.split}"
            )
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
    train_loader, val_loader = _facade_module()._build_phase_train_loaders(dataset, train_config=phase_train_config)
    controller = PhaseTransitionController(phase=phase, selection=phase_selection)
    controller.replay(_read_jsonl(phase_run_dir / "history" / "epochs.jsonl"))

    trainer = _facade_module()._build_phase_trainer(phase, phase_train_config)
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


def _find_phase_by_stage(
    scenario: MetaTrainScenario,
    *,
    stage: str,
) -> tuple[int, PhaseConfig]:
    for phase_index, phase in enumerate(scenario.phases, start=1):
        if str(phase.stage) == str(stage):
            return phase_index, phase
    raise KeyError(f"phase stage not found in scenario: {stage}")


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _cuda_memory_stats(device: Any) -> dict[str, Any]:
    import torch

    if getattr(device, "type", None) != "cuda":
        return {
            "device": str(device),
            "current_allocated_bytes": None,
            "current_allocated_gib": None,
            "peak_allocated_bytes": None,
            "peak_allocated_gib": None,
            "current_reserved_bytes": None,
            "current_reserved_gib": None,
            "peak_reserved_bytes": None,
            "peak_reserved_gib": None,
        }
    torch.cuda.synchronize(device)
    current_allocated = int(torch.cuda.memory_allocated(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    current_reserved = int(torch.cuda.memory_reserved(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    def _to_gib(value: int) -> float:
        return float(value) / float(1024**3)

    return {
        "device": str(device),
        "current_allocated_bytes": current_allocated,
        "current_allocated_gib": _to_gib(current_allocated),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_gib": _to_gib(peak_allocated),
        "current_reserved_bytes": current_reserved,
        "current_reserved_gib": _to_gib(current_reserved),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": _to_gib(peak_reserved),
    }


def _existing_dataset_roots(scenario: MetaTrainScenario) -> list[Path]:
    dataset_roots = list(scenario.dataset.roots)
    missing_roots = [str(path) for path in dataset_roots if not path.is_dir()]
    if missing_roots:
        raise SystemExit(f"canonical dataset roots not found: {missing_roots}")
    return dataset_roots


def _stage3_stress_train_config(
    scenario: MetaTrainScenario,
    *,
    batch_size: int,
    stress_iters: int,
) -> tuple[int, PhaseConfig, TrainDefaultsConfig]:
    phase_index, phase = _find_phase_by_stage(scenario, stage="stage_3_end_to_end_finetune")
    phase_train_config = _scenario_phase_defaults(scenario.train_defaults, phase.overrides)
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


def _run_stage3_probe(
    trainer: PV26Trainer,
    train_loader: Any,
    *,
    scenario: MetaTrainScenario,
    phase_index: int,
    phase_name: str,
    phase_train_config: TrainDefaultsConfig,
    stress_iters: int,
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
        if not _is_oom_error(exc):
            raise
        status = "oom"
        error = str(exc)
        torch.cuda.empty_cache()
    return status, train_summary, error


def _stage3_stress_summary(
    *,
    scenario_path: Path,
    phase_index: int,
    phase: PhaseConfig,
    trainer: PV26Trainer,
    train_config: TrainDefaultsConfig,
    batch_size: int,
    stress_iters: int,
    duration_sec: float,
    status: str,
    train_summary: dict[str, Any] | None,
    error: str | None,
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
        "memory": _facade_module()._cuda_memory_stats(trainer.device),
        "train_summary": _json_ready(train_summary) if train_summary is not None else None,
        "error": error,
    }
    if status == "oom":
        result["recommendation"] = "reduce batch_size and rerun the stage3 stress probe"
    return result


def run_stage3_vram_stress(
    scenario: MetaTrainScenario,
    *,
    scenario_path: Path,
    batch_size: int | None = None,
    stress_iters: int | None = None,
) -> dict[str, Any]:
    _facade_module()._configure_torch_multiprocessing()
    resolved_batch_size = int(batch_size) if batch_size is not None else int(scenario.train_defaults.batch_size)
    resolved_stress_iters = int(stress_iters) if stress_iters is not None else 12
    if resolved_batch_size <= 0:
        raise ValueError("stress batch size must be > 0")
    if resolved_stress_iters <= 0:
        raise ValueError("stress iterations must be > 0")

    dataset_roots = _existing_dataset_roots(scenario)
    phase_index, phase, phase_train_config = _stage3_stress_train_config(
        scenario,
        batch_size=resolved_batch_size,
        stress_iters=resolved_stress_iters,
    )

    _log_meta_train(f"loading scenario for stage3 stress: {scenario_path}")
    _log_meta_train(f"dataset roots: {[str(path) for path in dataset_roots]}")
    _log_meta_train("building canonical dataset index for stage3 stress")
    dataset = _facade_module().PV26CanonicalDataset(
        dataset_roots,
        progress_callback=_log_meta_train,
    )
    _log_meta_train(
        f"stage3 stress config: batch_size={resolved_batch_size}, stress_iters={resolved_stress_iters}, "
        f"device={phase_train_config.device}, backbone={phase_train_config.backbone_variant}"
    )
    _log_meta_train("stage3 stress loader override: num_workers=0, persistent_workers=False, prefetch_factor=None")
    train_loader, _ = _facade_module()._build_phase_train_loaders(dataset, train_config=phase_train_config)
    trainer = _facade_module()._build_phase_trainer(phase, phase_train_config)
    trainer.oom_guard = False
    if trainer.device.type != "cuda":
        raise SystemExit(
            f"stage3 VRAM stress requires a CUDA device, got device={trainer.device}"
        )

    run_started_at = time.perf_counter()
    status, train_summary, error = _run_stage3_probe(
        trainer,
        train_loader,
        scenario=scenario,
        phase_index=phase_index,
        phase_name=phase.name,
        phase_train_config=phase_train_config,
        stress_iters=resolved_stress_iters,
    )
    duration_sec = max(0.0, time.perf_counter() - run_started_at)
    return _stage3_stress_summary(
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
    )


def run_meta_train_scenario(
    scenario: MetaTrainScenario,
    *,
    scenario_path: Path,
) -> dict[str, Any]:
    _facade_module()._configure_torch_multiprocessing()
    dataset_roots = _existing_dataset_roots(scenario)

    _log_meta_train(f"loading scenario: {scenario_path}")
    _log_meta_train(f"dataset roots: {[str(path) for path in dataset_roots]}")
    run_dir = _resolve_meta_run_dir(scenario, scenario_path=scenario_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_meta_train(f"meta run directory: {run_dir}")
    _log_meta_train("building canonical dataset index")
    dataset = _facade_module().PV26CanonicalDataset(
        dataset_roots,
        progress_callback=_log_meta_train,
    )
    split_counts = Counter(record.split for record in dataset.records)
    dataset_key_counts = Counter(record.dataset_key for record in dataset.records)
    _log_meta_train(f"dataset split counts: {dict(sorted(split_counts.items()))}")
    _log_meta_train(f"dataset key counts: {dict(sorted(dataset_key_counts.items()))}")
    _log_meta_train("selecting preview samples")
    preview_samples = _facade_module()._sample_preview_selection_with_logging(
        dataset,
        scenario.preview,
        progress_callback=_log_meta_train,
    )
    _log_meta_train(f"prepared {len(preview_samples)} preview samples")
    scenario_snapshot = _scenario_snapshot_for_run(scenario, run_dir=run_dir)
    manifest, manifest_path = _load_or_init_meta_manifest(
        scenario=scenario,
        scenario_path=scenario_path,
        run_dir=run_dir,
        meta_manifest_version=META_MANIFEST_VERSION,
        scenario_snapshot=scenario_snapshot,
    )
    if all(
        _phase_entry_is_completed(entry, phase)
        for entry, phase in zip(manifest["phases"], scenario.phases)
    ):
        raise SystemExit(f"exact resume only supports incomplete runs: {run_dir}")

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

        phase_result = _facade_module()._execute_phase(
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
