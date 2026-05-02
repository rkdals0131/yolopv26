from __future__ import annotations

from typing import Any

import torch

from ..data.target_encoder import encode_pv26_batch
from common.task_mode import LANE_FAMILY_TASK_MODE
from .batch import augment_lane_family_metrics, move_batch_to_device, raw_batch_for_metrics
from ..net.trunk import forward_pyramid_features
from .metrics import PV26MetricConfig, summarize_pv26_metrics
from .postprocess import PV26PostprocessConfig, postprocess_pv26_batch
from .loss import PV26MultiTaskLoss


def _weighted_loss_summary(losses: dict[str, torch.Tensor], criterion: torch.nn.Module) -> dict[str, float]:
    export_config = getattr(criterion, "export_config", None)
    if not callable(export_config):
        return {}
    raw_config = export_config()
    raw_weights = raw_config.get("loss_weights")
    if not isinstance(raw_weights, dict):
        return {}
    output: dict[str, float] = {}
    for name, weight in raw_weights.items():
        if name == "total":
            continue
        value = losses.get(name)
        if isinstance(value, torch.Tensor):
            output[name] = float(value.detach().cpu()) * float(weight)
    return output


def _summarize_counts(encoded: dict[str, Any]) -> dict[str, int]:
    masks = encoded["mask"]
    return {
        "det_gt": int(encoded["det_gt"]["valid_mask"].sum().item()),
        "tl_attr_gt": int(encoded["tl_attr_gt_mask"].sum().item()),
        "lane_raw_rows": int(masks.get("lane_raw_count", masks["lane_valid"].sum()).sum().item()),
        "lane_input_valid_rows": int(masks.get("lane_input_valid_count", masks["lane_valid"].sum()).sum().item()),
        "lane_supervised_rows": int(masks.get("lane_supervised_count", masks["lane_valid"].sum()).sum().item()),
        "lane_rows": int(masks["lane_valid"].sum().item()),
        "stop_line_rows": int(masks["stop_line_valid"].sum().item()),
        "crosswalk_rows": int(masks["crosswalk_valid"].sum().item()),
    }


class PV26Evaluator:
    def __init__(
        self,
        adapter: Any,
        heads: torch.nn.Module,
        *,
        stage: str = "stage_1_frozen_trunk_warmup",
        device: str | torch.device = "cpu",
        criterion: torch.nn.Module | None = None,
    ) -> None:
        self.adapter = adapter
        self.heads = heads
        self.device = torch.device(device)
        self.stage = stage
        self.adapter.raw_model.to(self.device)
        self.heads.to(self.device)
        self.criterion = (criterion or PV26MultiTaskLoss(stage=stage)).to(self.device)

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        task_mode = str(getattr(self.criterion, "task_mode", LANE_FAMILY_TASK_MODE))
        include_segfirst = str(getattr(self.heads, "lane_head_mode", "row_native")) == "seg_first"
        if "det_gt" in batch:
            encoded = batch
            raw_batch = encoded.get("_raw_batch") if isinstance(encoded.get("_raw_batch"), dict) else None
            has_segfirst = isinstance(encoded.get("roadmark_v2"), dict) and "lane_seg_centerline_core" in encoded["roadmark_v2"]
            if include_segfirst and not has_segfirst and raw_batch is not None:
                encoded = encode_pv26_batch(
                    {"image": encoded["image"], **raw_batch},
                    task_mode=task_mode,
                    include_lane_segfirst_targets=True,
                )
                encoded["_raw_batch"] = raw_batch
        else:
            encoded = encode_pv26_batch(
                batch,
                task_mode=task_mode,
                include_lane_segfirst_targets=include_segfirst,
            )
        raw_batch = encoded.get("_raw_batch") if isinstance(encoded.get("_raw_batch"), dict) else None
        encoded_payload = {key: value for key, value in encoded.items() if key != "_raw_batch"}
        moved = move_batch_to_device(encoded_payload, self.device)
        if raw_batch is not None:
            moved["_raw_batch"] = raw_batch
        return moved

    def forward_encoded_batch(self, encoded: dict[str, Any]) -> dict[str, Any]:
        features = forward_pyramid_features(self.adapter, encoded["image"])
        return self.heads(features, encoded=encoded) if getattr(self.heads, "supports_encoded_context", False) else self.heads(features)

    def _run_batch(
        self,
        batch: dict[str, Any],
        *,
        include_predictions: bool = False,
        compute_loss: bool = True,
        config: PV26PostprocessConfig | None = None,
    ) -> dict[str, Any]:
        self.adapter.raw_model.eval()
        self.heads.eval()
        raw_batch = raw_batch_for_metrics(batch)
        encoded = self.prepare_batch(batch)
        with torch.no_grad():
            predictions = self.forward_encoded_batch(encoded)
            losses = {}
            if compute_loss:
                losses = self.criterion(predictions, encoded)
            postprocessed = (
                postprocess_pv26_batch(predictions, encoded["meta"], config=config)
                if include_predictions or raw_batch is not None
                else None
            )
        summary = {
            "stage": self.stage,
            "batch_size": int(encoded["image"].shape[0]),
            "losses": {
                name: float(value.detach().cpu())
                for name, value in losses.items()
            },
            "counts": _summarize_counts(encoded),
            "prediction_shapes": {
                name: list(value.shape) for name, value in predictions.items() if isinstance(value, torch.Tensor)
            },
            "metrics": summarize_pv26_metrics(postprocessed, raw_batch, config=PV26MetricConfig())
            if raw_batch is not None and postprocessed is not None
            else {},
        }
        weighted_losses = _weighted_loss_summary(losses, self.criterion)
        if weighted_losses:
            summary["losses_weighted"] = weighted_losses
        if summary["metrics"]:
            summary["metrics"] = augment_lane_family_metrics(summary["metrics"])
        if include_predictions:
            summary["predictions"] = postprocessed or []
        return summary

    def evaluate_batch(
        self,
        batch: dict[str, Any],
        *,
        include_predictions: bool = False,
        compute_loss: bool = True,
        config: PV26PostprocessConfig | None = None,
    ) -> dict[str, Any]:
        return self._run_batch(
            batch,
            include_predictions=include_predictions,
            compute_loss=compute_loss,
            config=config,
        )

    def predict_batch(
        self,
        batch: dict[str, Any],
        *,
        config: PV26PostprocessConfig | None = None,
    ) -> list[dict[str, Any]]:
        return list(
            self._run_batch(
                batch,
                include_predictions=True,
                compute_loss=False,
                config=config,
            )["predictions"]
        )


__all__ = ["PV26Evaluator"]
