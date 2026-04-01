from __future__ import annotations

from typing import Any

import torch

from ..encoding import encode_pv26_batch
from ..loss import PV26MultiTaskLoss
from ..trunk import forward_pyramid_features
from .metrics import PV26MetricConfig, summarize_pv26_metrics
from .postprocess import PV26PostprocessConfig, postprocess_pv26_batch


def _move_to_device(item: Any, device: torch.device) -> Any:
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return {key: _move_to_device(value, device) for key, value in item.items()}
    if isinstance(item, list):
        return [_move_to_device(value, device) for value in item]
    if isinstance(item, tuple):
        return tuple(_move_to_device(value, device) for value in item)
    return item


def _raw_batch_for_metrics(batch: dict[str, Any]) -> dict[str, Any] | None:
    raw_batch = batch.get("_raw_batch")
    if isinstance(raw_batch, dict):
        return raw_batch
    if "det_targets" in batch:
        return batch
    return None


def _summarize_counts(encoded: dict[str, Any]) -> dict[str, int]:
    return {
        "det_gt": int(encoded["det_gt"]["valid_mask"].sum().item()),
        "tl_attr_gt": int(encoded["tl_attr_gt_mask"].sum().item()),
        "lane_rows": int(encoded["mask"]["lane_valid"].sum().item()),
        "stop_line_rows": int(encoded["mask"]["stop_line_valid"].sum().item()),
        "crosswalk_rows": int(encoded["mask"]["crosswalk_valid"].sum().item()),
    }


class PV26Evaluator:
    def __init__(
        self,
        adapter: Any,
        heads: torch.nn.Module,
        *,
        stage: str = "stage_0_smoke",
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
        encoded = batch if "det_gt" in batch else encode_pv26_batch(batch)
        raw_batch = encoded.get("_raw_batch") if isinstance(encoded.get("_raw_batch"), dict) else None
        encoded_payload = {key: value for key, value in encoded.items() if key != "_raw_batch"}
        moved = _move_to_device(encoded_payload, self.device)
        if raw_batch is not None:
            moved["_raw_batch"] = raw_batch
        return moved

    def forward_encoded_batch(self, encoded: dict[str, Any]) -> dict[str, Any]:
        features = forward_pyramid_features(self.adapter, encoded["image"])
        return self.heads(features)

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
        raw_batch = _raw_batch_for_metrics(batch)
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
