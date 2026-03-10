from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.outputs import PV26MultiHeadOutput
from ..dataset.loading.sample_types import Pv26Sample
from ..training.prepared_batch import PV26PreparedBatch


@dataclass(frozen=True)
class PV26LossBreakdown:
    total: Tensor
    od: Tensor
    da: Tensor
    rm: Tensor
    rm_lane_subclass: Tensor


def _pv26_da_loss_impl(*, da_logits: Tensor, da_mask: Tensor, has_da: Tensor) -> Tensor:
    bsz = da_logits.shape[0]
    if da_mask.shape[0] != bsz:
        raise ValueError("da batch size mismatch")
    logits = da_logits[:, 0]  # [B,H,W]
    valid = (da_mask != 255) & has_da.view(-1, 1, 1).bool()
    valid_count = valid.to(dtype=da_logits.dtype).sum(dim=(1, 2))  # [B]
    target = torch.where(valid, da_mask, torch.zeros_like(da_mask)).to(dtype=da_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per_sample = (bce * valid.to(dtype=bce.dtype)).sum(dim=(1, 2)) / valid_count.clamp_min(1.0)
    keep_f = valid_count.gt(0).to(dtype=per_sample.dtype)
    return (per_sample * keep_f).sum() / keep_f.sum().clamp_min(1.0)


def _pv26_rm_loss_impl(
    *,
    rm_logits: Tensor,
    rm_mask: Tensor,
    has_rm: Tensor,
    focal_gamma: float,
    dice_eps: float,
) -> Tensor:
    bsz, ch, _, _ = rm_logits.shape
    if ch != 3:
        raise ValueError("rm logits must have 3 channels")
    if rm_mask.shape[0] != bsz or rm_mask.shape[1] != 3:
        raise ValueError("rm batch shape mismatch")

    supervised = has_rm.view(bsz, ch, 1, 1).bool()
    valid = (rm_mask != 255) & supervised
    valid_count = valid.to(dtype=rm_logits.dtype).sum(dim=(2, 3))  # [B,C]
    target = torch.where(valid, rm_mask, torch.zeros_like(rm_mask)).to(dtype=rm_logits.dtype)
    valid_f = valid.to(dtype=rm_logits.dtype)

    bce = F.binary_cross_entropy_with_logits(rm_logits, target, reduction="none")
    prob = torch.sigmoid(rm_logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    focal_num = (((1.0 - pt).pow(focal_gamma) * bce) * valid_f).sum(dim=(2, 3))
    focal = focal_num / valid_count.clamp_min(1.0)

    inter = (prob * target * valid_f).sum(dim=(2, 3))
    prob_sum = (prob * valid_f).sum(dim=(2, 3))
    target_sum = (target * valid_f).sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + dice_eps) / (prob_sum + target_sum + dice_eps)

    per_channel = focal + dice
    keep_f = valid_count.gt(0).to(dtype=per_channel.dtype)
    return (per_channel * keep_f).sum() / keep_f.sum().clamp_min(1.0)


class _PV26SegLossBlock(nn.Module):
    def __init__(self, *, focal_gamma: float, dice_eps: float):
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.dice_eps = float(dice_eps)

    def forward(
        self,
        da_logits: Tensor,
        da_mask: Tensor,
        has_da: Tensor,
        rm_logits: Tensor,
        rm_mask: Tensor,
        has_rm: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            _pv26_da_loss_impl(da_logits=da_logits, da_mask=da_mask, has_da=has_da),
            _pv26_rm_loss_impl(
                rm_logits=rm_logits,
                rm_mask=rm_mask,
                has_rm=has_rm,
                focal_gamma=self.focal_gamma,
                dice_eps=self.dice_eps,
            ),
        )


class PV26Criterion(nn.Module):
    """
    Multi-task criterion for PV26 step-3 training bootstrap.

    Loss contract:
      - OD: minimal dense YOLO-style loss (box + obj + class)
      - DA: BCE-with-logits on {0,1} with ignore=255 masking
      - RM: per-channel binary focal + dice with ignore=255 masking
      - Lane-subclass: cross-entropy on {1..4} positive pixels only (ignore=255 and background=0 are masked out)
      - Weights: w_od=1, w_da=1, w_rm=2, w_rm_lane_subclass=1 (defaults)
    """

    def __init__(
        self,
        *,
        num_det_classes: int,
        od_loss_impl: str = "dense",
        rm_lane_subclass_loss_impl: str = "dense_masked",
        det_loss_adapter: Optional[Callable[..., Tensor]] = None,
        w_od: float = 1.0,
        w_da: float = 1.0,
        w_rm: float = 2.0,
        w_rm_lane_subclass: float = 1.0,
        focal_gamma: float = 2.0,
        dice_eps: float = 1e-6,
        num_lane_subclasses: int = 4,
    ):
        super().__init__()
        self.num_det_classes = int(num_det_classes)
        self.od_loss_impl = str(od_loss_impl).strip().lower()
        self.rm_lane_subclass_loss_impl = str(rm_lane_subclass_loss_impl).strip().lower()
        self.w_od = float(w_od)
        self.w_da = float(w_da)
        self.w_rm = float(w_rm)
        self.w_rm_lane_subclass = float(w_rm_lane_subclass)
        self.focal_gamma = float(focal_gamma)
        self.dice_eps = float(dice_eps)
        self.num_lane_subclasses = int(num_lane_subclasses)
        self._seg_loss_block = _PV26SegLossBlock(focal_gamma=self.focal_gamma, dice_eps=self.dice_eps)
        self._seg_loss_block_impl: Any = self._seg_loss_block
        self.seg_loss_compile_enabled = False
        self.det_loss_adapter = det_loss_adapter

        if self.od_loss_impl not in {"dense", "ultralytics_e2e"}:
            raise ValueError(f"invalid od_loss_impl: {self.od_loss_impl}")
        if self.rm_lane_subclass_loss_impl not in {"dense_masked", "sparse_pos"}:
            raise ValueError(f"invalid rm_lane_subclass_loss_impl: {self.rm_lane_subclass_loss_impl}")
        if self.od_loss_impl == "ultralytics_e2e" and self.det_loss_adapter is None:
            raise ValueError("det_loss_adapter must be provided when od_loss_impl='ultralytics_e2e'")

    def disable_compile_seg_loss(self) -> None:
        self._seg_loss_block_impl = self._seg_loss_block
        self.seg_loss_compile_enabled = False

    def enable_compile_seg_loss(self, *, compile_mode: str, compile_fullgraph: bool) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile unavailable on this torch build")
        self._seg_loss_block_impl = torch.compile(
            self._seg_loss_block,
            mode=str(compile_mode),
            fullgraph=bool(compile_fullgraph),
        )
        self.seg_loss_compile_enabled = True

    def forward(self, preds: PV26MultiHeadOutput, batch: Any) -> Dict[str, Tensor]:
        t = self._normalize_batch(batch=batch, device=preds.da.device)

        if self.od_loss_impl == "dense":
            od = self._od_loss(
                det_logits=preds.det,
                det_yolo=t.det_yolo,
                has_det=t.has_det,
                det_label_scope=t.det_label_scope,
            )
        else:
            od = self._od_loss_ultralytics(
                det_out=preds.det,
                det_yolo=t.det_yolo,
                has_det=t.has_det,
                det_label_scope=t.det_label_scope,
                det_scope_code=t.det_scope_code,
                det_tgt_batch_idx=t.det_tgt_batch_idx,
                det_tgt_cls=t.det_tgt_cls,
                det_tgt_bboxes=t.det_tgt_bboxes,
            )
        da, rm = self._seg_loss(
            da_logits=preds.da,
            da_mask=t.da_mask,
            has_da=t.has_da,
            rm_logits=preds.rm,
            rm_mask=t.rm_mask,
            has_rm=t.has_rm,
        )
        rm_lane_subclass = self._rm_lane_subclass_loss(
            rm_lane_subclass_logits=preds.rm_lane_subclass,
            rm_lane_subclass_mask=t.rm_lane_subclass_mask,
            has_rm_lane_subclass=t.has_rm_lane_subclass,
        )

        total = self.w_od * od + self.w_da * da + self.w_rm * rm + self.w_rm_lane_subclass * rm_lane_subclass
        return {"total": total, "od": od, "da": da, "rm": rm, "rm_lane_subclass": rm_lane_subclass}

    def _od_loss_ultralytics(
        self,
        *,
        det_out: Any,
        det_yolo: Sequence[Tensor],
        has_det: Tensor,
        det_label_scope: Sequence[str],
        det_scope_code: Optional[Tensor] = None,
        det_tgt_batch_idx: Optional[Tensor] = None,
        det_tgt_cls: Optional[Tensor] = None,
        det_tgt_bboxes: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Ultralytics YOLO26 detection loss (E2E loss wrapper over v8DetectionLoss).

        Notes:
        - This path supports the coarse 7-class manifest contract where detection labels are either
          full supervision or absent (`none`).
        """
        if self.det_loss_adapter is None:
            raise RuntimeError("det loss adapter is not initialized")
        return self.det_loss_adapter(
            det_out=det_out,
            det_yolo=det_yolo,
            has_det=has_det,
            det_label_scope=det_label_scope,
            det_scope_code=det_scope_code,
            det_tgt_batch_idx=det_tgt_batch_idx,
            det_tgt_cls=det_tgt_cls,
            det_tgt_bboxes=det_tgt_bboxes,
        )

    def _normalize_batch(self, batch: Any, device: torch.device) -> PV26PreparedBatch:
        if isinstance(batch, PV26PreparedBatch):
            return batch.to_device(device=device)

        if isinstance(batch, Sequence) and batch and isinstance(batch[0], Pv26Sample):
            raise TypeError(
                "Sequence[Pv26Sample] batches are no longer supported in PV26Criterion; "
                "prepare the batch first with PV26PreparedBatch.from_samples(..., seg_output_stride=...)"
            )

        if not isinstance(batch, Mapping):
            raise TypeError("batch must be PV26PreparedBatch or Mapping[str, Any]")
        return PV26PreparedBatch.from_mapping(batch, device=device)

    def _seg_loss(
        self,
        *,
        da_logits: Tensor,
        da_mask: Tensor,
        has_da: Tensor,
        rm_logits: Tensor,
        rm_mask: Tensor,
        has_rm: Tensor,
    ) -> tuple[Tensor, Tensor]:
        da, rm = self._seg_loss_block_impl(
            da_logits,
            da_mask,
            has_da,
            rm_logits,
            rm_mask,
            has_rm,
        )
        return da, rm

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
            scope = str(det_label_scope[b]).strip().lower()
            if scope not in {"full", "none"}:
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

            obj_losses.append(F.binary_cross_entropy_with_logits(pred_obj[b], obj_target, reduction="mean"))

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
        return _pv26_da_loss_impl(da_logits=da_logits, da_mask=da_mask, has_da=has_da)

    def _rm_loss(self, *, rm_logits: Tensor, rm_mask: Tensor, has_rm: Tensor) -> Tensor:
        return _pv26_rm_loss_impl(
            rm_logits=rm_logits,
            rm_mask=rm_mask,
            has_rm=has_rm,
            focal_gamma=self.focal_gamma,
            dice_eps=self.dice_eps,
        )

    def _rm_lane_subclass_loss(
        self,
        *,
        rm_lane_subclass_logits: Tensor,
        rm_lane_subclass_mask: Tensor,
        has_rm_lane_subclass: Tensor,
    ) -> Tensor:
        bsz, ch, h, w = rm_lane_subclass_logits.shape
        expected_ch = int(self.num_lane_subclasses + 1)  # + background
        if ch != expected_ch:
            raise ValueError(
                f"rm lane-subclass logits must have {expected_ch} channels "
                f"(bg + {self.num_lane_subclasses} subclasses), got {ch}"
            )
        if rm_lane_subclass_mask.shape != (bsz, h, w):
            raise ValueError(
                "rm lane-subclass batch shape mismatch: "
                f"logits={(bsz, ch, h, w)} mask={tuple(rm_lane_subclass_mask.shape)}"
            )

        supervised = has_rm_lane_subclass.view(bsz, 1, 1).bool()
        # Lane subclasses are very sparse; training over all background pixels tends to dominate.
        # We supervise subclass CE only on positive pixels (1..K), masking out background(0) and ignore(255).
        valid = (rm_lane_subclass_mask != 255) & (rm_lane_subclass_mask != 0) & supervised
        valid_count = valid.to(dtype=rm_lane_subclass_logits.dtype).sum(dim=(1, 2))
        keep_f = valid_count.gt(0).to(dtype=rm_lane_subclass_logits.dtype)
        if not bool(keep_f.any()):
            return torch.zeros((), dtype=rm_lane_subclass_logits.dtype, device=rm_lane_subclass_logits.device)

        if self.rm_lane_subclass_loss_impl == "dense_masked":
            per_sample = self._rm_lane_subclass_loss_dense_masked(
                rm_lane_subclass_logits=rm_lane_subclass_logits,
                rm_lane_subclass_mask=rm_lane_subclass_mask,
                valid=valid,
                valid_count=valid_count,
            )
        else:
            per_sample = self._rm_lane_subclass_loss_sparse_pos(
                rm_lane_subclass_logits=rm_lane_subclass_logits,
                rm_lane_subclass_mask=rm_lane_subclass_mask,
                valid=valid,
                valid_count=valid_count,
            )
        return (per_sample * keep_f).sum() / keep_f.sum().clamp_min(1.0)

    def _rm_lane_subclass_loss_dense_masked(
        self,
        *,
        rm_lane_subclass_logits: Tensor,
        rm_lane_subclass_mask: Tensor,
        valid: Tensor,
        valid_count: Tensor,
    ) -> Tensor:
        target = torch.where(valid, rm_lane_subclass_mask, torch.zeros_like(rm_lane_subclass_mask)).to(dtype=torch.long)
        ce = F.cross_entropy(rm_lane_subclass_logits, target, reduction="none")
        return (ce * valid.to(dtype=ce.dtype)).sum(dim=(1, 2)) / valid_count.clamp_min(1.0)

    def _rm_lane_subclass_loss_sparse_pos(
        self,
        *,
        rm_lane_subclass_logits: Tensor,
        rm_lane_subclass_mask: Tensor,
        valid: Tensor,
        valid_count: Tensor,
    ) -> Tensor:
        bsz = int(rm_lane_subclass_logits.shape[0])
        logits_flat = rm_lane_subclass_logits.permute(0, 2, 3, 1)[valid]
        target_flat = rm_lane_subclass_mask[valid].to(dtype=torch.long)
        ce_flat = F.cross_entropy(logits_flat, target_flat, reduction="none")
        sample_idx = torch.nonzero(valid, as_tuple=False)[:, 0]
        per_sample_sum = torch.zeros((bsz,), dtype=ce_flat.dtype, device=ce_flat.device)
        per_sample_sum.index_add_(0, sample_idx, ce_flat)
        return per_sample_sum / valid_count.to(dtype=ce_flat.dtype).clamp_min(1.0)
