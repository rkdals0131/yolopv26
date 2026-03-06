from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


class UltralyticsE2EDetLossAdapter:
    """
    Backend adapter for Ultralytics YOLO26 E2E detection loss.

    This adapter encapsulates:
    - Ultralytics `E2ELoss` construction
    - filtering of subset/none detection supervision
    - prediction schema handling
    - full-batch fast path for pre-flattened detection targets
    """

    def __init__(self, ultra_det_model: nn.Module):
        import os

        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
        from ultralytics.utils.loss import E2ELoss  # type: ignore
        from ultralytics.utils import DEFAULT_CFG  # type: ignore

        if not hasattr(ultra_det_model, "args"):
            ultra_det_model.args = DEFAULT_CFG  # type: ignore[attr-defined]
        self._ultra_det_loss = E2ELoss(ultra_det_model)

    def __call__(
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
                preds_sel = {
                    "one2many": self._index_head(preds["one2many"], idx),
                    "one2one": self._index_head(preds["one2one"], idx),
                }

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
            preds_sel = {
                "one2many": self._index_head(preds["one2many"], idx),
                "one2one": self._index_head(preds["one2one"], idx),
            }

            if len(det_yolo) != bsz:
                raise ValueError("det_yolo length mismatch")
            batch_idx_list: List[Tensor] = []
            cls_list: List[Tensor] = []
            box_list: List[Tensor] = []
            for new_i, old_i in enumerate(idx.tolist()):
                gt = det_yolo[int(old_i)]
                if gt.numel() == 0:
                    continue
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
        loss_mean = loss_total / float(int(idx.shape[0]))
        if loss_mean.ndim != 0:
            loss_mean = loss_mean.sum()
        return loss_mean.to(dtype=torch.float32)

    @staticmethod
    def _index_head(h: Mapping[str, Any], idx: Tensor) -> Dict[str, Any]:
        return {
            "boxes": h["boxes"].index_select(0, idx),
            "scores": h["scores"].index_select(0, idx),
            "feats": [f.index_select(0, idx) for f in h["feats"]],
        }
