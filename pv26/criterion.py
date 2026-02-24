from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .multitask_model import PV26MultiHeadOutput
from .torch_dataset import Pv26Sample


@dataclass(frozen=True)
class PV26LossBreakdown:
    total: Tensor
    od: Tensor
    da: Tensor
    rm: Tensor


class PV26Criterion(nn.Module):
    """
    Multi-task criterion for PV26 step-3 training bootstrap.

    Loss contract:
      - OD: minimal dense YOLO-style loss (box + obj + class)
      - DA: BCE-with-logits on {0,1} with ignore=255 masking
      - RM: per-channel binary focal + dice with ignore=255 masking
      - Weights: w_od=1, w_da=1, w_rm=2 (defaults)
    """

    def __init__(
        self,
        *,
        num_det_classes: int,
        w_od: float = 1.0,
        w_da: float = 1.0,
        w_rm: float = 2.0,
        focal_gamma: float = 2.0,
        dice_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_det_classes = int(num_det_classes)
        self.w_od = float(w_od)
        self.w_da = float(w_da)
        self.w_rm = float(w_rm)
        self.focal_gamma = float(focal_gamma)
        self.dice_eps = float(dice_eps)

    def forward(self, preds: PV26MultiHeadOutput, batch: Any) -> Dict[str, Tensor]:
        t = self._normalize_batch(batch=batch, device=preds.det.device)

        od = self._od_loss(
            det_logits=preds.det,
            det_yolo=t["det_yolo"],
            has_det=t["has_det"],
            det_label_scope=t["det_label_scope"],
        )
        da = self._da_loss(da_logits=preds.da, da_mask=t["da_mask"], has_da=t["has_da"])
        rm = self._rm_loss(
            rm_logits=preds.rm,
            rm_mask=t["rm_mask"],
            has_rm=t["has_rm"],
        )

        total = self.w_od * od + self.w_da * da + self.w_rm * rm
        return {"total": total, "od": od, "da": da, "rm": rm}

    def _normalize_batch(self, batch: Any, device: torch.device) -> Dict[str, Any]:
        if isinstance(batch, Sequence) and batch and isinstance(batch[0], Pv26Sample):
            samples = list(batch)
            return {
                "det_yolo": [s.det_yolo.to(device=device, dtype=torch.float32) for s in samples],
                "da_mask": torch.stack([s.da_mask for s in samples], dim=0).to(device=device),
                "rm_mask": torch.stack([s.rm_mask for s in samples], dim=0).to(device=device),
                "has_det": torch.tensor([s.has_det for s in samples], dtype=torch.long, device=device),
                "has_da": torch.tensor([s.has_da for s in samples], dtype=torch.long, device=device),
                "has_rm": torch.tensor(
                    [
                        [s.has_rm_lane_marker, s.has_rm_road_marker_non_lane, s.has_rm_stop_line]
                        for s in samples
                    ],
                    dtype=torch.long,
                    device=device,
                ),
                "det_label_scope": [s.det_label_scope for s in samples],
            }

        if not isinstance(batch, Mapping):
            raise TypeError("batch must be Sequence[Pv26Sample] or Mapping[str, Any]")

        det_yolo = batch["det_yolo"]
        if isinstance(det_yolo, Tensor):
            det_list = [det_yolo[i] for i in range(det_yolo.shape[0])]
        else:
            det_list = list(det_yolo)

        det_list = [d.to(device=device, dtype=torch.float32) for d in det_list]
        det_list = [d[(d[:, 3] > 0) & (d[:, 4] > 0)] if d.numel() else d for d in det_list]

        has_rm = torch.stack(
            [
                batch["has_rm_lane_marker"],
                batch["has_rm_road_marker_non_lane"],
                batch["has_rm_stop_line"],
            ],
            dim=1,
        ).to(device=device, dtype=torch.long)

        scopes = [str(s) for s in list(batch["det_label_scope"])]

        return {
            "det_yolo": det_list,
            "da_mask": batch["da_mask"].to(device=device),
            "rm_mask": batch["rm_mask"].to(device=device),
            "has_det": batch["has_det"].to(device=device, dtype=torch.long),
            "has_da": batch["has_da"].to(device=device, dtype=torch.long),
            "has_rm": has_rm,
            "det_label_scope": scopes,
        }

    def _od_loss(
        self,
        *,
        det_logits: Tensor,
        det_yolo: Sequence[Tensor],
        has_det: Tensor,
        det_label_scope: Sequence[str],
    ) -> Tensor:
        bsz, out_ch, grid_h, grid_w = det_logits.shape
        num_classes = out_ch - 5
        if num_classes != self.num_det_classes:
            raise ValueError(f"det channel mismatch: expected classes={self.num_det_classes}, got={num_classes}")
        if len(det_yolo) != bsz or len(det_label_scope) != bsz:
            raise ValueError("batch size mismatch between det logits and labels")

        pred_box = torch.sigmoid(det_logits[:, 0:4])
        pred_obj = det_logits[:, 4]
        pred_cls = det_logits[:, 5:]

        zero = det_logits.new_zeros(())
        obj_losses: List[Tensor] = []
        box_losses: List[Tensor] = []
        cls_losses: List[Tensor] = []

        for b in range(bsz):
            scope = str(det_label_scope[b])
            if scope not in {"full", "subset", "none"}:
                raise ValueError(f"invalid det_label_scope: {scope}")
            if int(has_det[b].item()) == 0 or scope == "none":
                continue

            gt = det_yolo[b]
            if gt.ndim != 2 or gt.shape[-1] != 5:
                raise ValueError("det_yolo per sample must be [N,5]")

            obj_target = torch.zeros((grid_h, grid_w), dtype=det_logits.dtype, device=det_logits.device)
            pos_mask = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=det_logits.device)
            box_target = torch.zeros((grid_h, grid_w, 4), dtype=det_logits.dtype, device=det_logits.device)
            cls_target = torch.full((grid_h, grid_w), -1, dtype=torch.long, device=det_logits.device)

            for i in range(gt.shape[0]):
                cls_idx = int(gt[i, 0].item())
                if cls_idx < 0 or cls_idx >= self.num_det_classes:
                    continue
                cx = float(gt[i, 1].clamp(0.0, 1.0).item())
                cy = float(gt[i, 2].clamp(0.0, 1.0).item())
                bw = float(gt[i, 3].clamp(0.0, 1.0).item())
                bh = float(gt[i, 4].clamp(0.0, 1.0).item())

                gx = min(int(cx * grid_w), grid_w - 1)
                gy = min(int(cy * grid_h), grid_h - 1)
                pos_mask[gy, gx] = True
                obj_target[gy, gx] = 1.0
                box_target[gy, gx] = torch.tensor([cx, cy, bw, bh], dtype=det_logits.dtype, device=det_logits.device)
                cls_target[gy, gx] = cls_idx

            if scope == "full":
                obj_losses.append(F.binary_cross_entropy_with_logits(pred_obj[b], obj_target, reduction="mean"))
            elif pos_mask.any():
                obj_losses.append(
                    F.binary_cross_entropy_with_logits(pred_obj[b][pos_mask], obj_target[pos_mask], reduction="mean")
                )

            if pos_mask.any():
                box_losses.append(F.smooth_l1_loss(pred_box[b].permute(1, 2, 0)[pos_mask], box_target[pos_mask], reduction="mean"))
                cls_losses.append(F.cross_entropy(pred_cls[b].permute(1, 2, 0)[pos_mask], cls_target[pos_mask], reduction="mean"))

        total = zero
        if obj_losses:
            total = total + torch.stack(obj_losses).mean()
        if box_losses:
            total = total + torch.stack(box_losses).mean()
        if cls_losses:
            total = total + torch.stack(cls_losses).mean()
        return total

    def _da_loss(self, *, da_logits: Tensor, da_mask: Tensor, has_da: Tensor) -> Tensor:
        bsz = da_logits.shape[0]
        if da_mask.shape[0] != bsz:
            raise ValueError("da batch size mismatch")
        losses: List[Tensor] = []
        for b in range(bsz):
            if int(has_da[b].item()) == 0:
                continue
            tgt = da_mask[b]
            valid = tgt != 255
            if not bool(valid.any()):
                continue
            losses.append(
                F.binary_cross_entropy_with_logits(
                    da_logits[b, 0][valid],
                    tgt[valid].to(dtype=da_logits.dtype),
                    reduction="mean",
                )
            )
        if not losses:
            return da_logits.new_zeros(())
        return torch.stack(losses).mean()

    def _rm_loss(self, *, rm_logits: Tensor, rm_mask: Tensor, has_rm: Tensor) -> Tensor:
        bsz, ch, _, _ = rm_logits.shape
        if ch != 3:
            raise ValueError("rm logits must have 3 channels")
        if rm_mask.shape[0] != bsz or rm_mask.shape[1] != 3:
            raise ValueError("rm batch shape mismatch")

        losses: List[Tensor] = []
        for b in range(bsz):
            for c in range(3):
                if int(has_rm[b, c].item()) == 0:
                    continue
                tgt = rm_mask[b, c]
                valid = tgt != 255
                if not bool(valid.any()):
                    continue

                logit = rm_logits[b, c][valid]
                target = tgt[valid].to(dtype=rm_logits.dtype)

                bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
                prob = torch.sigmoid(logit)
                pt = prob * target + (1.0 - prob) * (1.0 - target)
                focal = ((1.0 - pt).pow(self.focal_gamma) * bce).mean()

                inter = (prob * target).sum()
                dice = 1.0 - (2.0 * inter + self.dice_eps) / (prob.sum() + target.sum() + self.dice_eps)
                losses.append(focal + dice)

        if not losses:
            return rm_logits.new_zeros(())
        return torch.stack(losses).mean()
