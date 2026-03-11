from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from ..dataset.loading.sample_types import Pv26Sample
from ..model.contracts import validate_seg_output_stride

_SCOPE_TO_CODE = {"full": 0, "none": 1}


def _downsample_binary_mask_u8(mask: Tensor, *, seg_output_stride: int) -> Tensor:
    stride = validate_seg_output_stride(seg_output_stride)
    if stride == 1:
        return mask

    if mask.ndim == 3:
        bsz, h, w = mask.shape
        if (h % stride) != 0 or (w % stride) != 0:
            raise ValueError(f"mask spatial shape must be divisible by seg_output_stride: got {(h, w)} stride={stride}")
        blocks = mask.view(bsz, h // stride, stride, w // stride, stride)
        any_pos = blocks.eq(1).any(dim=(2, 4))
        any_valid = blocks.ne(255).any(dim=(2, 4))
        out = torch.full((bsz, h // stride, w // stride), 255, dtype=torch.uint8, device=mask.device)
        out = torch.where(any_valid & ~any_pos, torch.zeros_like(out), out)
        out = torch.where(any_pos, torch.ones_like(out), out)
        return out

    if mask.ndim == 4:
        bsz, ch, h, w = mask.shape
        if (h % stride) != 0 or (w % stride) != 0:
            raise ValueError(f"mask spatial shape must be divisible by seg_output_stride: got {(h, w)} stride={stride}")
        blocks = mask.view(bsz, ch, h // stride, stride, w // stride, stride)
        any_pos = blocks.eq(1).any(dim=(3, 5))
        any_valid = blocks.ne(255).any(dim=(3, 5))
        out = torch.full((bsz, ch, h // stride, w // stride), 255, dtype=torch.uint8, device=mask.device)
        out = torch.where(any_valid & ~any_pos, torch.zeros_like(out), out)
        out = torch.where(any_pos, torch.ones_like(out), out)
        return out

    raise ValueError(f"binary mask ndim must be 3 or 4, got {mask.ndim}")


def _downsample_lane_subclass_mask_u8(
    mask: Tensor,
    *,
    seg_output_stride: int,
    num_lane_subclasses: int = 4,
) -> Tensor:
    stride = validate_seg_output_stride(seg_output_stride)
    if stride == 1:
        return mask
    if mask.ndim != 3:
        raise ValueError(f"lane-subclass mask ndim must be 3, got {mask.ndim}")

    bsz, h, w = mask.shape
    if (h % stride) != 0 or (w % stride) != 0:
        raise ValueError(f"mask spatial shape must be divisible by seg_output_stride: got {(h, w)} stride={stride}")

    blocks = mask.view(bsz, h // stride, stride, w // stride, stride)
    blocks = blocks.permute(0, 1, 3, 2, 4).reshape(bsz, h // stride, w // stride, stride * stride)

    class_counts = torch.stack(
        [(blocks == int(cls_id)).sum(dim=-1) for cls_id in range(1, int(num_lane_subclasses) + 1)],
        dim=-1,
    )
    max_count, max_idx = class_counts.max(dim=-1)
    has_pos = max_count.gt(0)
    has_bg = blocks.eq(0).any(dim=-1)

    out = torch.full((bsz, h // stride, w // stride), 255, dtype=torch.uint8, device=mask.device)
    out = torch.where(has_bg, torch.zeros_like(out), out)
    out = torch.where(has_pos, (max_idx + 1).to(dtype=torch.uint8), out)
    return out


@dataclass(frozen=True)
class PV26PreparedBatch:
    det_yolo: tuple[Tensor, ...]
    det_label_scope: tuple[str, ...]
    has_det: Tensor
    has_da: Tensor
    has_rm: Tensor
    has_rm_lane_subclass: Tensor
    da_mask: Tensor
    rm_mask: Tensor
    rm_lane_subclass_mask: Tensor
    da_mask_fullres: Optional[Tensor] = None
    rm_mask_fullres: Optional[Tensor] = None
    rm_lane_subclass_mask_fullres: Optional[Tensor] = None
    det_scope_code: Optional[Tensor] = None
    det_tgt_batch_idx: Optional[Tensor] = None
    det_tgt_cls: Optional[Tensor] = None
    det_tgt_bboxes: Optional[Tensor] = None
    sample_id: tuple[str, ...] = ()

    @staticmethod
    def _build_flat_det_targets(det_yolo_list: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        det_tgt_batch_idx_parts: List[Tensor] = []
        det_tgt_cls_parts: List[Tensor] = []
        det_tgt_box_parts: List[Tensor] = []
        for i, gt in enumerate(det_yolo_list):
            if gt.numel() == 0:
                continue
            if gt.ndim != 2 or gt.shape[-1] != 5:
                raise ValueError("det_yolo per sample must be [N,5]")
            det_tgt_batch_idx_parts.append(torch.full((gt.shape[0],), int(i), dtype=torch.long))
            det_tgt_cls_parts.append(gt[:, 0].to(dtype=torch.float32))
            det_tgt_box_parts.append(gt[:, 1:5].to(dtype=torch.float32))

        if det_tgt_batch_idx_parts:
            det_tgt_batch_idx = torch.cat(det_tgt_batch_idx_parts, dim=0)
            det_tgt_cls = torch.cat(det_tgt_cls_parts, dim=0)
            det_tgt_bboxes = torch.cat(det_tgt_box_parts, dim=0)
        else:
            det_tgt_batch_idx = torch.zeros((0,), dtype=torch.long)
            det_tgt_cls = torch.zeros((0,), dtype=torch.float32)
            det_tgt_bboxes = torch.zeros((0, 4), dtype=torch.float32)
        return det_tgt_batch_idx, det_tgt_cls, det_tgt_bboxes

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Pv26Sample],
        *,
        include_sample_id: bool = False,
        include_fullres_masks: bool = False,
        seg_output_stride: int = 1,
    ) -> "PV26PreparedBatch":
        stride = validate_seg_output_stride(seg_output_stride)
        det_yolo = tuple(s.det_yolo.to(dtype=torch.float32) for s in samples)
        det_label_scope = tuple(str(s.det_label_scope).strip().lower() for s in samples)
        for scope in det_label_scope:
            if scope not in _SCOPE_TO_CODE:
                raise ValueError(f"invalid det_label_scope in batch: {scope}")

        det_scope_code = torch.tensor([int(_SCOPE_TO_CODE[s]) for s in det_label_scope], dtype=torch.long)
        det_tgt_batch_idx, det_tgt_cls, det_tgt_bboxes = cls._build_flat_det_targets(det_yolo)
        sample_id = tuple(str(s.sample_id) for s in samples) if include_sample_id else ()
        da_mask = torch.stack([s.da_mask for s in samples], dim=0)
        rm_mask = torch.stack([s.rm_mask for s in samples], dim=0)
        rm_lane_subclass_mask = torch.stack([s.rm_lane_subclass_mask for s in samples], dim=0)
        da_mask_fullres = da_mask if include_fullres_masks else None
        rm_mask_fullres = rm_mask if include_fullres_masks else None
        rm_lane_subclass_mask_fullres = rm_lane_subclass_mask if include_fullres_masks else None
        if stride != 1:
            da_mask = _downsample_binary_mask_u8(da_mask, seg_output_stride=stride)
            rm_mask = _downsample_binary_mask_u8(rm_mask, seg_output_stride=stride)
            rm_lane_subclass_mask = _downsample_lane_subclass_mask_u8(
                rm_lane_subclass_mask,
                seg_output_stride=stride,
            )
        return cls(
            det_yolo=det_yolo,
            det_label_scope=det_label_scope,
            has_det=torch.tensor([s.has_det for s in samples], dtype=torch.long),
            has_da=torch.tensor([s.has_da for s in samples], dtype=torch.long),
            has_rm=torch.tensor(
                [[s.has_rm_lane_marker, s.has_rm_road_marker_non_lane, s.has_rm_stop_line] for s in samples],
                dtype=torch.long,
            ),
            has_rm_lane_subclass=torch.tensor([s.has_rm_lane_subclass for s in samples], dtype=torch.long),
            da_mask=da_mask,
            rm_mask=rm_mask,
            rm_lane_subclass_mask=rm_lane_subclass_mask,
            da_mask_fullres=da_mask_fullres,
            rm_mask_fullres=rm_mask_fullres,
            rm_lane_subclass_mask_fullres=rm_lane_subclass_mask_fullres,
            det_scope_code=det_scope_code,
            det_tgt_batch_idx=det_tgt_batch_idx,
            det_tgt_cls=det_tgt_cls,
            det_tgt_bboxes=det_tgt_bboxes,
            sample_id=sample_id,
        )

    @classmethod
    def from_mapping(
        cls,
        batch: Mapping[str, Any],
        *,
        device: torch.device,
        seg_output_stride: int = 1,
    ) -> "PV26PreparedBatch":
        stride = validate_seg_output_stride(seg_output_stride)
        det_yolo_raw = batch["det_yolo"]
        if isinstance(det_yolo_raw, Tensor):
            det_list = [det_yolo_raw[i] for i in range(det_yolo_raw.shape[0])]
        else:
            det_list = list(det_yolo_raw)
        det_yolo = tuple(
            d.to(device=device, dtype=torch.float32)[(d[:, 3] > 0) & (d[:, 4] > 0)] if d.numel() else d.to(device=device, dtype=torch.float32)
            for d in det_list
        )
        bsz = len(det_yolo)

        if "has_rm" in batch:
            has_rm = batch["has_rm"].to(device=device, dtype=torch.long)
        else:
            has_rm = torch.stack(
                [
                    batch["has_rm_lane_marker"],
                    batch["has_rm_road_marker_non_lane"],
                    batch["has_rm_stop_line"],
                ],
                dim=1,
            ).to(device=device, dtype=torch.long)

        det_label_scope = tuple(str(s).strip().lower() for s in list(batch["det_label_scope"]))
        for scope in det_label_scope:
            if scope not in _SCOPE_TO_CODE:
                raise ValueError(f"invalid det_label_scope: {scope}")

        det_scope_code = batch.get("det_scope_code", None)
        if det_scope_code is None:
            det_scope_code_t = torch.tensor([int(_SCOPE_TO_CODE[s]) for s in det_label_scope], dtype=torch.long, device=device)
        else:
            det_scope_code_t = det_scope_code.to(device=device, dtype=torch.long)

        da_mask = batch["da_mask"].to(device=device)
        rm_mask = batch["rm_mask"].to(device=device)
        rm_lane_subclass_mask = batch.get(
            "rm_lane_subclass_mask",
            torch.full((bsz, int(batch["rm_mask"].shape[2]), int(batch["rm_mask"].shape[3])), 255, dtype=torch.uint8),
        ).to(device=device)
        da_mask_fullres = batch.get("da_mask_fullres", None)
        rm_mask_fullres = batch.get("rm_mask_fullres", None)
        rm_lane_subclass_mask_fullres = batch.get("rm_lane_subclass_mask_fullres", None)
        if da_mask_fullres is not None:
            da_mask_fullres = da_mask_fullres.to(device=device)
        if rm_mask_fullres is not None:
            rm_mask_fullres = rm_mask_fullres.to(device=device)
        if rm_lane_subclass_mask_fullres is not None:
            rm_lane_subclass_mask_fullres = rm_lane_subclass_mask_fullres.to(device=device)
        if stride != 1:
            da_mask = _downsample_binary_mask_u8(da_mask, seg_output_stride=stride)
            rm_mask = _downsample_binary_mask_u8(rm_mask, seg_output_stride=stride)
            rm_lane_subclass_mask = _downsample_lane_subclass_mask_u8(
                rm_lane_subclass_mask,
                seg_output_stride=stride,
            )

        return cls(
            det_yolo=det_yolo,
            det_label_scope=det_label_scope,
            has_det=batch["has_det"].to(device=device, dtype=torch.long),
            has_da=batch["has_da"].to(device=device, dtype=torch.long),
            has_rm=has_rm,
            has_rm_lane_subclass=batch.get(
                "has_rm_lane_subclass",
                torch.zeros((bsz,), dtype=torch.long),
            ).to(device=device, dtype=torch.long),
            da_mask=da_mask,
            rm_mask=rm_mask,
            rm_lane_subclass_mask=rm_lane_subclass_mask,
            da_mask_fullres=da_mask_fullres,
            rm_mask_fullres=rm_mask_fullres,
            rm_lane_subclass_mask_fullres=rm_lane_subclass_mask_fullres,
            det_scope_code=det_scope_code_t,
            det_tgt_batch_idx=(
                batch["det_tgt_batch_idx"].to(device=device, dtype=torch.long)
                if "det_tgt_batch_idx" in batch
                else None
            ),
            det_tgt_cls=(
                batch["det_tgt_cls"].to(device=device, dtype=torch.float32)
                if "det_tgt_cls" in batch
                else None
            ),
            det_tgt_bboxes=(
                batch["det_tgt_bboxes"].to(device=device, dtype=torch.float32)
                if "det_tgt_bboxes" in batch
                else None
            ),
            sample_id=tuple(str(s) for s in list(batch.get("sample_id", ()))),
        )

    def to_device(self, *, device: torch.device) -> "PV26PreparedBatch":
        return replace(
            self,
            det_yolo=tuple(t.to(device=device, dtype=torch.float32, non_blocking=True) for t in self.det_yolo),
            has_det=self.has_det.to(device=device, dtype=torch.long, non_blocking=True),
            has_da=self.has_da.to(device=device, dtype=torch.long, non_blocking=True),
            has_rm=self.has_rm.to(device=device, dtype=torch.long, non_blocking=True),
            has_rm_lane_subclass=self.has_rm_lane_subclass.to(device=device, dtype=torch.long, non_blocking=True),
            da_mask=self.da_mask.to(device=device, non_blocking=True),
            rm_mask=self.rm_mask.to(device=device, non_blocking=True),
            rm_lane_subclass_mask=self.rm_lane_subclass_mask.to(device=device, non_blocking=True),
            da_mask_fullres=None
            if self.da_mask_fullres is None
            else self.da_mask_fullres.to(device=device, non_blocking=True),
            rm_mask_fullres=None
            if self.rm_mask_fullres is None
            else self.rm_mask_fullres.to(device=device, non_blocking=True),
            rm_lane_subclass_mask_fullres=None
            if self.rm_lane_subclass_mask_fullres is None
            else self.rm_lane_subclass_mask_fullres.to(device=device, non_blocking=True),
            det_scope_code=None if self.det_scope_code is None else self.det_scope_code.to(device=device, dtype=torch.long, non_blocking=True),
            det_tgt_batch_idx=None
            if self.det_tgt_batch_idx is None
            else self.det_tgt_batch_idx.to(device=device, dtype=torch.long, non_blocking=True),
            det_tgt_cls=None
            if self.det_tgt_cls is None
            else self.det_tgt_cls.to(device=device, dtype=torch.float32, non_blocking=True),
            det_tgt_bboxes=None
            if self.det_tgt_bboxes is None
            else self.det_tgt_bboxes.to(device=device, dtype=torch.float32, non_blocking=True),
        )

    def pin_memory(self) -> "PV26PreparedBatch":
        def _pin(t: Tensor) -> Tensor:
            try:
                return t.pin_memory()
            except RuntimeError:
                return t

        return replace(
            self,
            det_yolo=tuple(_pin(t) for t in self.det_yolo),
            has_det=_pin(self.has_det),
            has_da=_pin(self.has_da),
            has_rm=_pin(self.has_rm),
            has_rm_lane_subclass=_pin(self.has_rm_lane_subclass),
            da_mask=_pin(self.da_mask),
            rm_mask=_pin(self.rm_mask),
            rm_lane_subclass_mask=_pin(self.rm_lane_subclass_mask),
            da_mask_fullres=None if self.da_mask_fullres is None else _pin(self.da_mask_fullres),
            rm_mask_fullres=None if self.rm_mask_fullres is None else _pin(self.rm_mask_fullres),
            rm_lane_subclass_mask_fullres=None
            if self.rm_lane_subclass_mask_fullres is None
            else _pin(self.rm_lane_subclass_mask_fullres),
            det_scope_code=None if self.det_scope_code is None else _pin(self.det_scope_code),
            det_tgt_batch_idx=None if self.det_tgt_batch_idx is None else _pin(self.det_tgt_batch_idx),
            det_tgt_cls=None if self.det_tgt_cls is None else _pin(self.det_tgt_cls),
            det_tgt_bboxes=None if self.det_tgt_bboxes is None else _pin(self.det_tgt_bboxes),
        )
