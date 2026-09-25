"""Official YOLO26 detection loss with partial-label road-mark supervision."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from model.net.pv26 import PV26FocusedModel


class PV26FocusedLoss(nn.Module):
    def __init__(
        self,
        model: PV26FocusedModel,
        *,
        det_weight: float = 1.0,
        roadmark_weight: float = 1.0,
        detector_loss_schedule: str = "restart",
        roadmark_dice: str = "linear",
    ) -> None:
        super().__init__()
        if detector_loss_schedule not in {"restart", "mature"}:
            raise ValueError("detector_loss_schedule must be restart or mature")
        if roadmark_dice not in {"linear", "squared"}:
            raise ValueError("roadmark_dice must be linear or squared")
        # Keep a non-registered reference: the criterion must not duplicate model
        # parameters in its own state_dict or optimizer parameter groups.
        self.__dict__["_model"] = model
        self.det_weight = float(det_weight)
        self.roadmark_weight = float(roadmark_weight)
        self.detector_loss_schedule = detector_loss_schedule
        self.roadmark_dice = roadmark_dice
        self._official = None
        self._official_device = None
        self._detector_progress = (0, 1)

    def set_progress(self, completed_steps: int, total_steps: int) -> None:
        """Set official E2E loss decay within a detector/joint training run."""
        if completed_steps < 0 or total_steps <= 0:
            raise ValueError("detection progress requires non-negative completed_steps and positive total_steps")
        if self.__dict__["_model"].train_stage == "roadmark":
            return
        self._detector_progress = (int(completed_steps), int(total_steps))
        self._apply_progress()

    def _apply_progress(self) -> None:
        if self._official is None:
            return
        completed, total = self._detector_progress
        official = self._official
        official.updates = completed
        if self.detector_loss_schedule == "mature":
            official.o2m = official.final_o2m
        else:
            fraction_remaining = max(1.0 - completed / total, 0.0)
            official.o2m = fraction_remaining * (official.o2m_copy - official.final_o2m) + official.final_o2m
        official.o2o = official.total - official.o2m

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.float().sum() * 0.0

    @staticmethod
    def _select_detection(raw: dict[str, Any], selected: torch.Tensor) -> dict[str, Any]:
        picked: dict[str, Any] = {}
        for branch_name in ("one2many", "one2one"):
            branch = raw[branch_name]
            picked[branch_name] = {
                "boxes": branch["boxes"].index_select(0, selected).float(),
                "scores": branch["scores"].index_select(0, selected).float(),
                # Ultralytics only uses feature shapes/device to create anchors.
                "feats": [feature.index_select(0, selected).detach().float() for feature in branch["feats"]],
            }
        return picked

    def _detection_loss(self, raw: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        labeled = batch["det_labeled"].to(device=raw["one2many"]["boxes"].device, dtype=torch.bool)
        selected = labeled.nonzero(as_tuple=False).flatten()
        count = selected.numel()
        if count == 0:
            return self._zero(raw["one2many"]["scores"]), selected.new_tensor(0)

        batch_idx = batch["batch_idx"].view(-1).to(device=selected.device, dtype=torch.long)
        classes = batch["cls"].to(device=selected.device).view(-1, 1).float()
        boxes = batch["bboxes"].to(device=selected.device).float()
        model = self.__dict__["_model"]
        device = selected.device
        if self._official is None or self._official_device != device:
            self._official = model.detector.init_criterion()
            self._official_device = device
            self._apply_progress()
        image_losses = []
        with torch.autocast(device_type=device.type, enabled=False):
            for image_index in selected:
                image_selection = image_index.view(1)
                keep = batch_idx == image_index
                det_batch = {
                    "batch_idx": torch.zeros_like(batch_idx[keep]),
                    "cls": classes[keep],
                    "bboxes": boxes[keep],
                }
                prediction = self._select_detection(raw, image_selection)
                official_loss, _ = self._official(prediction, det_batch)
                image_losses.append(official_loss.float().sum())
        # The official loss and assignment operate on each image. Averaging those
        # image losses gives one objective independent of physical microbatching.
        return torch.stack(image_losses).mean(), selected.new_tensor(count)

    def _roadmark_loss(
        self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = logits.float()
        target = target.to(device=logits.device, dtype=torch.float32)
        valid = valid.to(device=logits.device, dtype=torch.bool)
        if logits.shape != target.shape or logits.shape != valid.shape:
            raise ValueError("roadmark logits, target and valid must have the same [B,3,H/4,W/4] shape")
        valid_count = valid.sum()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            bce = F.binary_cross_entropy_with_logits(logits[valid], target[valid], reduction="sum")
            bce = bce / valid_count.clamp_min(1)
            probabilities = torch.where(valid, logits.sigmoid(), 0.0)
            truth = torch.where(valid, target, 0.0)
            overlap = 2.0 * (probabilities * truth).sum(dim=(-2, -1))
            probability_sum = probabilities.sum(dim=(-2, -1))
            truth_sum = truth.sum(dim=(-2, -1))
            has_positive = truth_sum > 0
            dice_count = has_positive.sum()
            if self.roadmark_dice == "squared":
                probability_sum = probabilities.square().sum(dim=(-2, -1))
                truth_sum = truth.square().sum(dim=(-2, -1))
            dice_by_class = 1.0 - (overlap + 1.0) / (probability_sum + truth_sum + 1.0)
            dice = torch.where(has_positive, dice_by_class, 0.0).sum() / dice_count.clamp_min(1)
            # Keep nonfinite model output visible to the trainer's skipped-update
            # handling, even if it occurs under an unlabelled/padding pixel.
            nonfinite_marker = logits.sum() * 0.0
            bce = bce + nonfinite_marker
            dice = dice + nonfinite_marker
        return bce, dice, valid_count.detach(), dice_count

    def forward(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        model = self.__dict__["_model"]
        raw = outputs["det"]
        roadmark_logits = outputs["roadmark_logits"]
        if model.train_stage == "roadmark":
            det = self._zero(roadmark_logits)
            det_count = torch.zeros((), device=roadmark_logits.device, dtype=torch.long)
        else:
            if not isinstance(raw, dict) or "one2many" not in raw or "one2one" not in raw:
                raise ValueError("PV26FocusedLoss requires raw official detection output; call forward_for_loss in eval mode")
            det, det_count = self._detection_loss(raw, batch)
        if model.train_stage == "detector":
            roadmark_bce = self._zero(raw["one2many"]["scores"])
            roadmark_dice = roadmark_bce
            valid_count = torch.zeros((), device=roadmark_bce.device, dtype=torch.long)
            dice_count = valid_count
        else:
            roadmark_bce, roadmark_dice, valid_count, dice_count = self._roadmark_loss(
                roadmark_logits, batch["roadmark_target"], batch["roadmark_valid"]
            )
        roadmark = roadmark_bce + roadmark_dice
        return {
            "total": self.det_weight * det + self.roadmark_weight * roadmark,
            "det": det,
            "roadmark": roadmark,
            "roadmark_bce": roadmark_bce,
            "roadmark_dice": roadmark_dice,
            "det_count": det_count,
            "roadmark_valid_count": valid_count,
            "roadmark_dice_count": dice_count,
        }
