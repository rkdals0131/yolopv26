from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    rm_lane_subclass: Tensor


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
        ultra_det_model: Optional[nn.Module] = None,
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

        self._ultra_det_loss = None
        if self.od_loss_impl not in {"dense", "ultralytics_e2e"}:
            raise ValueError(f"invalid od_loss_impl: {self.od_loss_impl}")
        if self.rm_lane_subclass_loss_impl not in {"dense_masked", "sparse_pos"}:
            raise ValueError(f"invalid rm_lane_subclass_loss_impl: {self.rm_lane_subclass_loss_impl}")
        if self.od_loss_impl == "ultralytics_e2e":
            if ultra_det_model is None:
                raise ValueError("ultra_det_model must be provided when od_loss_impl='ultralytics_e2e'")
            # Lazy import to avoid ultralytics side-effects unless requested.
            import os

            os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
            from ultralytics.utils.loss import E2ELoss  # type: ignore
            from ultralytics.utils import DEFAULT_CFG  # type: ignore

            # Ultralytics loss expects model.args to exist and provide hyp (box/cls/dfl/epochs...).
            if not hasattr(ultra_det_model, "args"):
                ultra_det_model.args = DEFAULT_CFG  # type: ignore[attr-defined]
            self._ultra_det_loss = E2ELoss(ultra_det_model)

    def forward(self, preds: PV26MultiHeadOutput, batch: Any) -> Dict[str, Tensor]:
        if isinstance(batch, Mapping) and bool(batch.get("_pv26_prepared", False)):
            t = dict(batch)
        else:
            t = self._normalize_batch(batch=batch, device=preds.da.device)

        if "has_rm_lane_subclass" not in t:
            if isinstance(t.get("has_rm"), Tensor):
                bsz = int(t["has_rm"].shape[0])
                t["has_rm_lane_subclass"] = torch.zeros((bsz,), dtype=torch.long, device=preds.da.device)
            else:
                raise ValueError("missing has_rm_lane_subclass in normalized batch")
        if "rm_lane_subclass_mask" not in t:
            if isinstance(t.get("rm_mask"), Tensor):
                bsz, _ch, h, w = t["rm_mask"].shape
                t["rm_lane_subclass_mask"] = torch.full(
                    (bsz, h, w),
                    255,
                    dtype=torch.uint8,
                    device=preds.da.device,
                )
            else:
                raise ValueError("missing rm_lane_subclass_mask in normalized batch")

        if self.od_loss_impl == "dense":
            od = self._od_loss(
                det_logits=preds.det,
                det_yolo=t["det_yolo"],
                has_det=t["has_det"],
                det_label_scope=t["det_label_scope"],
            )
        else:
            od = self._od_loss_ultralytics(
                det_out=preds.det,
                det_yolo=t.get("det_yolo", ()),
                has_det=t["has_det"],
                det_label_scope=t.get("det_label_scope", ()),
                det_scope_code=t.get("det_scope_code", None),
                det_tgt_batch_idx=t.get("det_tgt_batch_idx", None),
                det_tgt_cls=t.get("det_tgt_cls", None),
                det_tgt_bboxes=t.get("det_tgt_bboxes", None),
            )
        da = self._da_loss(da_logits=preds.da, da_mask=t["da_mask"], has_da=t["has_da"])
        rm = self._rm_loss(
            rm_logits=preds.rm,
            rm_mask=t["rm_mask"],
            has_rm=t["has_rm"],
        )
        rm_lane_subclass = self._rm_lane_subclass_loss(
            rm_lane_subclass_logits=preds.rm_lane_subclass,
            rm_lane_subclass_mask=t["rm_lane_subclass_mask"],
            has_rm_lane_subclass=t["has_rm_lane_subclass"],
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
        - This path currently supports only per-sample gating for `has_det` and `det_label_scope=none` by filtering the
          batch. `det_label_scope=subset` is conservatively excluded from OD loss until class-aware negative masking is
          implemented.
        """
        if self._ultra_det_loss is None:
            raise RuntimeError("ultralytics det loss is not initialized")

        # Parse predictions dict (E2ELoss can parse tuple internally, but we need filtering).
        preds = det_out[1] if isinstance(det_out, tuple) else det_out
        if not isinstance(preds, Mapping) or "one2many" not in preds or "one2one" not in preds:
            raise TypeError("det_out must be an ultralytics Detect output (dict or (y, preds))")

        bsz = int(has_det.shape[0])
        keep_idx: Tensor
        if det_scope_code is not None:
            if det_scope_code.shape[0] != bsz:
                raise ValueError("det_scope_code length mismatch")
            scope_code = det_scope_code.to(device=has_det.device, dtype=torch.long)
            keep_mask = has_det.to(dtype=torch.long) != 0
            keep_mask = keep_mask & (scope_code == 0)
            keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        else:
            if len(det_label_scope) != bsz:
                raise ValueError("det_label_scope length mismatch")
            has_det_cpu = has_det.to(device="cpu", dtype=torch.long).tolist()
            keep: List[int] = []
            for i in range(bsz):
                scope = str(det_label_scope[i]).strip().lower()
                if scope not in {"full", "subset", "none"}:
                    raise ValueError(f"invalid det_label_scope: {scope}")
                if int(has_det_cpu[i]) == 0 or scope != "full":
                    continue
                keep.append(i)
            keep_idx = torch.as_tensor(keep, dtype=torch.long, device=has_det.device)

        if int(keep_idx.numel()) == 0:
            # No supervised detection samples in this batch.
            return torch.zeros((), dtype=torch.float32, device=has_det.device)

        device = preds["one2many"]["boxes"].device
        idx = keep_idx.to(device=device, dtype=torch.long)
        full_batch_kept = int(idx.shape[0]) == bsz

        if det_tgt_batch_idx is not None and det_tgt_cls is not None and det_tgt_bboxes is not None:
            if full_batch_kept:
                preds_sel = preds
                batch_idx_t = det_tgt_batch_idx.to(device=device, dtype=torch.float32)
                cls_t = det_tgt_cls.to(device=device, dtype=torch.float32)
                bboxes_t = det_tgt_bboxes.to(device=device, dtype=torch.float32)
            else:
                def _index_head(h: Mapping[str, Any]) -> Dict[str, Any]:
                    return {
                        "boxes": h["boxes"].index_select(0, idx),
                        "scores": h["scores"].index_select(0, idx),
                        "feats": [f.index_select(0, idx) for f in h["feats"]],
                    }

                preds_sel = {"one2many": _index_head(preds["one2many"]), "one2one": _index_head(preds["one2one"])}

                old_to_new = torch.full((bsz,), -1, device=device, dtype=torch.long)
                old_to_new[idx] = torch.arange(idx.shape[0], device=device, dtype=torch.long)

                src_old = det_tgt_batch_idx.to(device=device, dtype=torch.long)
                new_idx = old_to_new[src_old]
                m = new_idx.ge(0)
                det_tgt_cls_dev = det_tgt_cls.to(device=device, dtype=torch.float32)
                det_tgt_bboxes_dev = det_tgt_bboxes.to(device=device, dtype=torch.float32)
                batch_idx_t = new_idx.masked_select(m).to(dtype=torch.float32)
                cls_t = det_tgt_cls_dev.masked_select(m)
                bboxes_t = det_tgt_bboxes_dev[m]
        else:
            def _index_head(h: Mapping[str, Any]) -> Dict[str, Any]:
                return {
                    "boxes": h["boxes"].index_select(0, idx),
                    "scores": h["scores"].index_select(0, idx),
                    "feats": [f.index_select(0, idx) for f in h["feats"]],
                }

            preds_sel = {"one2many": _index_head(preds["one2many"]), "one2one": _index_head(preds["one2one"])}

            if len(det_yolo) != bsz:
                raise ValueError("det_yolo length mismatch")
            batch_idx_list: List[Tensor] = []
            cls_list: List[Tensor] = []
            box_list: List[Tensor] = []
            for new_i, old_i in enumerate(idx.tolist()):
                gt = det_yolo[int(old_i)]
                if gt.numel() == 0:
                    continue
                # expected gt: [N,5] (cls,cx,cy,w,h) normalized.
                batch_idx_list.append(torch.full((gt.shape[0],), float(new_i), device=device, dtype=torch.float32))
                cls_list.append(gt[:, 0].to(device=device, dtype=torch.float32))
                box_list.append(gt[:, 1:5].to(device=device, dtype=torch.float32))

            if batch_idx_list:
                batch_idx_t = torch.cat(batch_idx_list, dim=0)
                cls_t = torch.cat(cls_list, dim=0)
                bboxes_t = torch.cat(box_list, dim=0)
            else:
                batch_idx_t = torch.zeros((0,), device=device, dtype=torch.float32)
                cls_t = torch.zeros((0,), device=device, dtype=torch.float32)
                bboxes_t = torch.zeros((0, 4), device=device, dtype=torch.float32)

        det_batch = {"batch_idx": batch_idx_t, "cls": cls_t, "bboxes": bboxes_t}

        loss_total, _loss_items = self._ultra_det_loss(preds_sel, det_batch)
        # Ultralytics losses are returned as a vector (box/cls/dfl) scaled by batch size.
        # Convert to mean-per-image scalar for PV26 weighting.
        loss_mean = loss_total / float(int(idx.shape[0]))
        if loss_mean.ndim != 0:
            loss_mean = loss_mean.sum()
        return loss_mean.to(dtype=torch.float32)

    def _normalize_batch(self, batch: Any, device: torch.device) -> Dict[str, Any]:
        if isinstance(batch, Sequence) and batch and isinstance(batch[0], Pv26Sample):
            samples = list(batch)
            return {
                "det_yolo": [s.det_yolo.to(device=device, dtype=torch.float32) for s in samples],
                "da_mask": torch.stack([s.da_mask for s in samples], dim=0).to(device=device),
                "rm_mask": torch.stack([s.rm_mask for s in samples], dim=0).to(device=device),
                "rm_lane_subclass_mask": torch.stack([s.rm_lane_subclass_mask for s in samples], dim=0).to(device=device),
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
                "has_rm_lane_subclass": torch.tensor(
                    [s.has_rm_lane_subclass for s in samples],
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
        bsz = int(has_rm.shape[0])
        if "has_rm_lane_subclass" in batch:
            has_rm_lane_subclass = batch["has_rm_lane_subclass"].to(device=device, dtype=torch.long)
        else:
            has_rm_lane_subclass = torch.zeros((bsz,), dtype=torch.long, device=device)

        if "rm_lane_subclass_mask" in batch:
            rm_lane_subclass_mask = batch["rm_lane_subclass_mask"].to(device=device)
        else:
            _bsz, _ch, h, w = batch["rm_mask"].shape
            rm_lane_subclass_mask = torch.full((int(_bsz), int(h), int(w)), 255, dtype=torch.uint8, device=device)

        scopes = [str(s) for s in list(batch["det_label_scope"])]

        return {
            "det_yolo": det_list,
            "da_mask": batch["da_mask"].to(device=device),
            "rm_mask": batch["rm_mask"].to(device=device),
            "rm_lane_subclass_mask": rm_lane_subclass_mask,
            "has_det": batch["has_det"].to(device=device, dtype=torch.long),
            "has_da": batch["has_da"].to(device=device, dtype=torch.long),
            "has_rm": has_rm,
            "has_rm_lane_subclass": has_rm_lane_subclass,
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
        # Vectorized DA loss:
        # per-sample mean BCE over valid pixels, then mean over supervised samples.
        logits = da_logits[:, 0]  # [B,H,W]
        valid = (da_mask != 255) & has_da.view(-1, 1, 1).bool()
        valid_count = valid.to(dtype=da_logits.dtype).sum(dim=(1, 2))  # [B]
        target = torch.where(valid, da_mask, torch.zeros_like(da_mask)).to(dtype=da_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        per_sample = (bce * valid.to(dtype=bce.dtype)).sum(dim=(1, 2)) / valid_count.clamp_min(1.0)
        keep_f = valid_count.gt(0).to(dtype=per_sample.dtype)
        return (per_sample * keep_f).sum() / keep_f.sum().clamp_min(1.0)

    def _rm_loss(self, *, rm_logits: Tensor, rm_mask: Tensor, has_rm: Tensor) -> Tensor:
        bsz, ch, _, _ = rm_logits.shape
        if ch != 3:
            raise ValueError("rm logits must have 3 channels")
        if rm_mask.shape[0] != bsz or rm_mask.shape[1] != 3:
            raise ValueError("rm batch shape mismatch")
        # Vectorized RM loss:
        # per (B,C) channel focal+dice over valid pixels, then mean over supervised channels.
        supervised = has_rm.view(bsz, ch, 1, 1).bool()
        valid = (rm_mask != 255) & supervised
        valid_count = valid.to(dtype=rm_logits.dtype).sum(dim=(2, 3))  # [B,C]
        target = torch.where(valid, rm_mask, torch.zeros_like(rm_mask)).to(dtype=rm_logits.dtype)
        valid_f = valid.to(dtype=rm_logits.dtype)

        bce = F.binary_cross_entropy_with_logits(rm_logits, target, reduction="none")
        prob = torch.sigmoid(rm_logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        focal_num = (((1.0 - pt).pow(self.focal_gamma) * bce) * valid_f).sum(dim=(2, 3))
        focal = focal_num / valid_count.clamp_min(1.0)

        inter = (prob * target * valid_f).sum(dim=(2, 3))
        prob_sum = (prob * valid_f).sum(dim=(2, 3))
        target_sum = (target * valid_f).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + self.dice_eps) / (prob_sum + target_sum + self.dice_eps)

        per_channel = focal + dice
        keep_f = valid_count.gt(0).to(dtype=per_channel.dtype)
        return (per_channel * keep_f).sum() / keep_f.sum().clamp_min(1.0)

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
