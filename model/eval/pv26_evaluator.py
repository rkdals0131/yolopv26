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
        return _move_to_device(encoded, self.device)

    def forward_encoded_batch(self, encoded: dict[str, Any]) -> dict[str, Any]:
        features = forward_pyramid_features(self.adapter, encoded["image"])
        return self.heads(features)

    def evaluate_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.adapter.raw_model.eval()
        self.heads.eval()
        raw_batch = batch if "det_targets" in batch else None
        encoded = self.prepare_batch(batch)
        with torch.no_grad():
            predictions = self.forward_encoded_batch(encoded)
            losses = self.criterion(predictions, encoded)
            metric_predictions = postprocess_pv26_batch(predictions, encoded["meta"]) if raw_batch is not None else None
        return {
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
            "metrics": summarize_pv26_metrics(metric_predictions, raw_batch, config=PV26MetricConfig())
            if raw_batch is not None and metric_predictions is not None
            else {},
        }

    def predict_batch(
        self,
        batch: dict[str, Any],
        *,
        config: PV26PostprocessConfig | None = None,
    ) -> list[dict[str, Any]]:
        self.adapter.raw_model.eval()
        self.heads.eval()
        encoded = self.prepare_batch(batch)
        with torch.no_grad():
            predictions = self.forward_encoded_batch(encoded)
        return postprocess_pv26_batch(predictions, encoded["meta"], config=config)


__all__ = ["PV26Evaluator"]
