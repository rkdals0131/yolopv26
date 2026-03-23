"""BDD100K-native validation for YOLOPv2 TorchScript weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ..dataset.masks import IGNORE_VALUE, convert_bdd_drivable_id_to_da_mask_u8, make_all_ignore_mask
from ..dataset.sources.bdd import bdd_record_to_rm_masks_with_lane_subclass
from .common_validation import (
    BinaryMetricSummary,
    accumulate_binary_confusion,
    binary_metric_summary_from_confusion,
    binary_metric_summary_to_dict,
)
from .detection import compute_map50
from .yolopv2_validation import decode_yolopv2_predictions, nms_yolopv2_predictions


BDD_NATIVE_DET_CLASS_NAMES: tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)
BDD_NATIVE_DET_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(BDD_NATIVE_DET_CLASS_NAMES)}
BDD_NATIVE_DET_ALIASES: dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "light": "traffic light",
    "traffic sign": "traffic sign",
    "sign": "traffic sign",
}

# COCO 80-class indices used by YOLO-style exports.
COCO_TO_BDD_NATIVE_DET: dict[int, int] = {
    0: BDD_NATIVE_DET_NAME_TO_ID["pedestrian"],  # person
    1: BDD_NATIVE_DET_NAME_TO_ID["bicycle"],  # bicycle
    2: BDD_NATIVE_DET_NAME_TO_ID["car"],  # car
    3: BDD_NATIVE_DET_NAME_TO_ID["motorcycle"],  # motorcycle
    5: BDD_NATIVE_DET_NAME_TO_ID["bus"],  # bus
    6: BDD_NATIVE_DET_NAME_TO_ID["train"],  # train
    7: BDD_NATIVE_DET_NAME_TO_ID["truck"],  # truck
    9: BDD_NATIVE_DET_NAME_TO_ID["traffic light"],  # traffic light
    11: BDD_NATIVE_DET_NAME_TO_ID["traffic sign"],  # stop sign -> partial traffic sign coverage
}


@dataclass(frozen=True)
class BddYoloPv2Sample:
    sample_id: str
    image: Tensor
    det_boxes: Tensor
    da_mask: Tensor
    has_da: int
    lane_mask: Tensor
    has_lane: int


@dataclass(frozen=True)
class BddYoloPv2EvalBatch:
    images: Tensor
    sample_ids: list[str]
    det_boxes: list[Tensor]
    da_mask: Tensor
    has_da: Tensor
    lane_mask: Tensor
    has_lane: Tensor


@dataclass(frozen=True)
class BddYoloPv2ValidationSummary:
    weights_path: str
    bdd_root: str
    num_samples: int
    num_batches: int
    input_height: int
    input_width: int
    det_map50: Optional[float]
    det_eval_images: int
    det_gt_boxes: int
    det_predictions: int
    det_ap_by_class: dict[str, Optional[float]]
    da: BinaryMetricSummary
    lane: BinaryMetricSummary
    detection_status: str


def _letterbox_params(in_w: int, in_h: int, out_w: int, out_h: int) -> tuple[float, int, int, int, int]:
    scale = min(float(out_w) / float(in_w), float(out_h) / float(in_h))
    new_w = int(round(in_w * scale))
    new_h = int(round(in_h * scale))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def _letterbox_image(image: Image.Image, *, out_w: int, out_h: int, pad_value: int = 114) -> np.ndarray:
    in_w, in_h = image.size
    _, new_w, new_h, pad_x, pad_y = _letterbox_params(in_w, in_h, out_w, out_h)
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), color=(pad_value, pad_value, pad_value))
    canvas.paste(resized, (pad_x, pad_y))
    return np.asarray(canvas, dtype=np.uint8)


def _letterbox_mask(mask_u8: np.ndarray, *, out_w: int, out_h: int, pad_value: int = IGNORE_VALUE) -> np.ndarray:
    in_h, in_w = mask_u8.shape
    _, new_w, new_h, pad_x, pad_y = _letterbox_params(in_w, in_h, out_w, out_h)
    resized = Image.fromarray(mask_u8, mode="L").resize((new_w, new_h), resample=Image.NEAREST)
    canvas = Image.new("L", (out_w, out_h), color=int(pad_value))
    canvas.paste(resized, (pad_x, pad_y))
    return np.asarray(canvas, dtype=np.uint8)


def _letterbox_boxes_xyxy(boxes_xyxy: Tensor, *, in_w: int, in_h: int, out_w: int, out_h: int) -> Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4).to(dtype=torch.float32)
    scale, _new_w, _new_h, pad_x, pad_y = _letterbox_params(in_w, in_h, out_w, out_h)
    boxes_xyxy = boxes_xyxy.to(dtype=torch.float32).clone()
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2] * float(scale) + float(pad_x)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2] * float(scale) + float(pad_y)
    boxes_xyxy[:, 0::2].clamp_(0.0, float(out_w))
    boxes_xyxy[:, 1::2].clamp_(0.0, float(out_h))
    return boxes_xyxy


def _bdd_record_objects(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    frames = record.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, dict):
            objects = first.get("objects")
            if isinstance(objects, list):
                return [obj for obj in objects if isinstance(obj, dict)]
    labels = record.get("labels")
    if isinstance(labels, list):
        return [obj for obj in labels if isinstance(obj, dict)]
    return []


def _normalize_bdd_native_det_category(category: object) -> Optional[int]:
    if not isinstance(category, str):
        return None
    normalized = BDD_NATIVE_DET_ALIASES.get(category.strip().lower())
    if normalized is None:
        return None
    return BDD_NATIVE_DET_NAME_TO_ID[normalized]


def _bdd_native_det_boxes_from_record(record: Mapping[str, object], *, width: int, height: int) -> Tensor:
    rows: list[list[float]] = []
    for obj in _bdd_record_objects(record):
        cls_id = _normalize_bdd_native_det_category(obj.get("category"))
        if cls_id is None:
            continue
        box2d = obj.get("box2d")
        if not isinstance(box2d, dict):
            continue
        try:
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
        except Exception:
            continue
        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append([float(cls_id), x1, y1, x2, y2])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


class BddYoloPv2ValDataset(Dataset[BddYoloPv2Sample]):
    def __init__(self, *, bdd_root: Path, out_width: int = 960, out_height: int = 544):
        self.bdd_root = Path(bdd_root)
        self.out_width = int(out_width)
        self.out_height = int(out_height)
        self.images_root = self.bdd_root / "bdd100k_images_100k" / "100k" / "val"
        self.labels_root = self.bdd_root / "bdd100k_labels" / "100k" / "val"
        self.drivable_root = self.bdd_root / "bdd100k_drivable_maps" / "labels" / "val"
        if not self.images_root.exists():
            raise FileNotFoundError(f"missing BDD image root: {self.images_root}")
        if not self.labels_root.exists():
            raise FileNotFoundError(f"missing BDD labels root: {self.labels_root}")

        self.samples: list[tuple[Path, Path, Optional[Path]]] = []
        for image_path in sorted(self.images_root.glob("*.jpg")):
            stem = image_path.stem
            label_path = self.labels_root / f"{stem}.json"
            if not label_path.exists():
                continue
            drivable_path = self.drivable_root / f"{stem}_drivable_id.png"
            self.samples.append((image_path, label_path, drivable_path if drivable_path.exists() else None))
        if not self.samples:
            raise RuntimeError("no BDD val samples found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BddYoloPv2Sample:
        image_path, label_path, drivable_path = self.samples[int(idx)]
        record = __import__("json").loads(label_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as image_pil:
            image = image_pil.convert("RGB")
            width, height = image.size
            image_lb = _letterbox_image(image, out_w=self.out_width, out_h=self.out_height).copy()

        if drivable_path is not None:
            drivable_id = np.asarray(Image.open(drivable_path), dtype=np.uint8)
            da_mask = convert_bdd_drivable_id_to_da_mask_u8(drivable_id)
            has_da = 1
        else:
            da_mask = make_all_ignore_mask(height, width)
            has_da = 0
        da_mask_lb = _letterbox_mask(da_mask, out_w=self.out_width, out_h=self.out_height, pad_value=IGNORE_VALUE).copy()

        lane_mask, _road_mask, _stop_mask, _lane_sub, has_lane, _has_road, _has_stop, _has_lane_sub = (
            bdd_record_to_rm_masks_with_lane_subclass(
                record,
                width=width,
                height=height,
            )
        )
        lane_mask_lb = _letterbox_mask(lane_mask, out_w=self.out_width, out_h=self.out_height, pad_value=IGNORE_VALUE).copy()
        det_boxes = _bdd_native_det_boxes_from_record(record, width=width, height=height)
        if det_boxes.numel() > 0:
            det_boxes_xyxy = _letterbox_boxes_xyxy(
                det_boxes[:, 1:5],
                in_w=width,
                in_h=height,
                out_w=self.out_width,
                out_h=self.out_height,
            )
            det_boxes = torch.cat([det_boxes[:, 0:1], det_boxes_xyxy], dim=1)

        return BddYoloPv2Sample(
            sample_id=image_path.stem,
            image=torch.from_numpy(image_lb).permute(2, 0, 1).contiguous(),
            det_boxes=det_boxes,
            da_mask=torch.from_numpy(da_mask_lb).to(torch.uint8),
            has_da=int(has_da),
            lane_mask=torch.from_numpy(lane_mask_lb).to(torch.uint8),
            has_lane=int(has_lane),
        )


def collate_bdd_yolopv2_eval(samples: Sequence[BddYoloPv2Sample]) -> BddYoloPv2EvalBatch:
    return BddYoloPv2EvalBatch(
        images=torch.stack([sample.image for sample in samples], dim=0),
        sample_ids=[sample.sample_id for sample in samples],
        det_boxes=[sample.det_boxes for sample in samples],
        da_mask=torch.stack([sample.da_mask for sample in samples], dim=0),
        has_da=torch.tensor([sample.has_da for sample in samples], dtype=torch.long),
        lane_mask=torch.stack([sample.lane_mask for sample in samples], dim=0),
        has_lane=torch.tensor([sample.has_lane for sample in samples], dtype=torch.long),
    )


def _prepare_images_for_inference(images: Tensor, *, device: torch.device) -> Tensor:
    images = images.to(device=device, non_blocking=True)
    if images.dtype != torch.float32:
        images = images.to(dtype=torch.float32)
        images.mul_(1.0 / 255.0)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def _resize_if_needed(logits: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
    if tuple(logits.shape[-2:]) == tuple(out_hw):
        return logits
    return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


def validate_bdd_yolopv2(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    weights_path: Path,
    bdd_root: Path,
    max_batches: int = 0,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> BddYoloPv2ValidationSummary:
    num_det_classes = len(BDD_NATIVE_DET_CLASS_NAMES)
    preds_by_class: Dict[int, list[tuple[float, str, Tensor]]] = {c: [] for c in range(num_det_classes)}
    gt_by_img_class: Dict[tuple[str, int], Tensor] = {}
    da_stats = {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    lane_stats = {"supervised_samples": 0, "valid_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    det_eval_images = 0
    det_gt_boxes = 0
    det_predictions = 0
    num_batches = 0
    num_samples = 0
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, int(max_batches))
    input_height = 0
    input_width = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if not isinstance(batch, BddYoloPv2EvalBatch):
                raise TypeError(f"unexpected batch type: {type(batch)!r}")

            images = _prepare_images_for_inference(batch.images, device=device)
            input_height = int(images.shape[-2])
            input_width = int(images.shape[-1])
            out = model(images)
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError("unexpected YOLOPv2 TorchScript output: expected ((pred,anchors), seg, ll)")

            det_out, da_logits, lane_logits = out
            if not isinstance(det_out, tuple) or len(det_out) != 2:
                raise RuntimeError("unexpected YOLOPv2 detection output: expected (pred_layers, anchor_grid)")
            raw_preds, anchor_grid = det_out
            decoded = decode_yolopv2_predictions(
                raw_preds,
                anchor_grid,
                input_h=input_height,
                input_w=input_width,
            )
            da_logits = _resize_if_needed(da_logits, out_hw=(batch.da_mask.shape[-2], batch.da_mask.shape[-1]))
            lane_logits = _resize_if_needed(lane_logits, out_hw=(batch.lane_mask.shape[-2], batch.lane_mask.shape[-1]))

            if int(da_logits.shape[1]) == 2:
                pred_da = da_logits[:, 1] > da_logits[:, 0]
            else:
                pred_da = torch.sigmoid(da_logits[:, 0]) > 0.5

            lane_prob = lane_logits[:, 0]
            if float(lane_prob.min().item()) < 0.0 or float(lane_prob.max().item()) > 1.0:
                lane_prob = torch.sigmoid(lane_prob)
            pred_lane = lane_prob > 0.5

            da_target = batch.da_mask.to(device=device)
            lane_target = batch.lane_mask.to(device=device)
            accumulate_binary_confusion(
                da_stats,
                pred_mask=pred_da,
                target_mask=da_target,
                valid_mask=(da_target != 255) & batch.has_da.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )
            accumulate_binary_confusion(
                lane_stats,
                pred_mask=pred_lane,
                target_mask=lane_target,
                valid_mask=(lane_target != 255) & batch.has_lane.to(device=device, dtype=torch.bool).view(-1, 1, 1),
            )

            for sample_idx, sample_id in enumerate(batch.sample_ids):
                num_samples += 1
                det_eval_images += 1

                gt = batch.det_boxes[sample_idx]
                if gt.numel() > 0:
                    gt_classes = gt[:, 0].to(dtype=torch.long)
                    gt_boxes = gt[:, 1:5].to(dtype=torch.float32)
                    det_gt_boxes += int(gt_boxes.shape[0])
                    for cls_id in range(num_det_classes):
                        match = gt_classes == cls_id
                        if bool(match.any()):
                            gt_by_img_class[(sample_id, cls_id)] = gt_boxes[match].detach().cpu()

                sample_preds = decoded[sample_idx]
                boxes_xyxy, scores, classes = nms_yolopv2_predictions(
                    sample_preds,
                    conf_thres=0.01,
                    iou_thres=0.45,
                    max_det=300,
                )
                for pred_idx in range(int(scores.shape[0])):
                    bdd_cls = COCO_TO_BDD_NATIVE_DET.get(int(classes[pred_idx].item()))
                    if bdd_cls is None:
                        continue
                    det_predictions += 1
                    preds_by_class[bdd_cls].append(
                        (
                            float(scores[pred_idx].item()),
                            sample_id,
                            boxes_xyxy[pred_idx].detach().cpu(),
                        )
                    )

            num_batches += 1
            if progress_hook is not None:
                progress_hook(num_batches, total_batches)

    det_map50, det_ap_by_class_idx = compute_map50(
        preds_by_class=preds_by_class,
        gt_by_img_class=gt_by_img_class,
        num_classes=num_det_classes,
        iou_thres=0.5,
    )
    det_ap_by_class = {
        name: det_ap_by_class_idx[idx] for idx, name in enumerate(BDD_NATIVE_DET_CLASS_NAMES)
    }

    return BddYoloPv2ValidationSummary(
        weights_path=str(Path(weights_path)),
        bdd_root=str(Path(bdd_root)),
        num_samples=int(num_samples),
        num_batches=int(num_batches),
        input_height=int(input_height),
        input_width=int(input_width),
        det_map50=det_map50,
        det_eval_images=int(det_eval_images),
        det_gt_boxes=int(det_gt_boxes),
        det_predictions=int(det_predictions),
        det_ap_by_class=det_ap_by_class,
        da=binary_metric_summary_from_confusion(
            supervised_samples=da_stats["supervised_samples"],
            valid_pixels=da_stats["valid_pixels"],
            true_positive=da_stats["tp"],
            false_positive=da_stats["fp"],
            false_negative=da_stats["fn"],
            true_negative=da_stats["tn"],
        ),
        lane=binary_metric_summary_from_confusion(
            supervised_samples=lane_stats["supervised_samples"],
            valid_pixels=lane_stats["valid_pixels"],
            true_positive=lane_stats["tp"],
            false_positive=lane_stats["fp"],
            false_negative=lane_stats["fn"],
            true_negative=lane_stats["tn"],
        ),
        detection_status="best_effort(coco80_to_bdd10_remap; rider_unmapped; traffic_sign_partial_via_stop_sign)",
    )


def validation_summary_to_dict(summary: BddYoloPv2ValidationSummary) -> dict[str, object]:
    return {
        "model_type": "yolopv2_bdd_raw",
        "weights_path": summary.weights_path,
        "bdd_root": summary.bdd_root,
        "num_samples": int(summary.num_samples),
        "num_batches": int(summary.num_batches),
        "input_height": int(summary.input_height),
        "input_width": int(summary.input_width),
        "det_map50": summary.det_map50,
        "det_eval_images": int(summary.det_eval_images),
        "det_gt_boxes": int(summary.det_gt_boxes),
        "det_predictions": int(summary.det_predictions),
        "det_ap_by_class": {name: ap for name, ap in summary.det_ap_by_class.items()},
        "da": binary_metric_summary_to_dict(summary.da),
        "lane": binary_metric_summary_to_dict(summary.lane),
        "detection_status": summary.detection_status,
    }
