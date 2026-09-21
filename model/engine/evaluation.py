"""Validation on labeled images for the two-signal, three-roadmark model."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import islice
from typing import Any

import torch
from torchvision.ops import box_iou

from common.schema import ROADMARK_CLASSES, SIGNAL_CLASSES
from model.engine.geometry_metrics import match_roadmark_lines
from model.engine.postprocess import decode_roadmark_points


def _scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


@torch.inference_mode()
def evaluate_focused(model, criterion, loader, *, device: torch.device,
                     precision: str = "bf16", max_batches: int = 32,
                     confidence: float = 0.25, iou_threshold: float = 0.5,
                     geometry_tolerance_px: float = 8.0) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    det_counts = [[0, 0, 0] for _ in SIGNAL_CLASSES]
    road_counts = [[0, 0, 0] for _ in ROADMARK_CLASSES]
    geometry = match_roadmark_lines([], [], tolerance_px=geometry_tolerance_px)
    loss_sums = {"det": 0.0, "roadmark_bce": 0.0, "roadmark_dice": 0.0}
    counts = {name: 0.0 for name in loss_sums}
    samples = 0
    batches = iter(loader)
    try:
        for cpu_batch in islice(batches, max_batches):
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in cpu_batch.items()}
            amp = (torch.autocast(device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
                   if precision != "fp32" else nullcontext())
            with amp:
                output = model.forward_for_loss(batch["image"])
            losses = criterion(output, batch)
            for name, count_key in (("det", "det_count"), ("roadmark_bce", "roadmark_valid_count"),
                                    ("roadmark_dice", "roadmark_dice_count")):
                amount = float(losses[count_key].item())
                value = float(losses[name].item())
                if not torch.isfinite(losses[name]).item():
                    raise FloatingPointError(f"nonfinite validation {name}")
                loss_sums[name] += value * amount
                counts[name] += amount
            samples += len(batch["image"])
            if output["det"] is not None:
                decoded = model.decode_raw_detection(output["det"]).float()
                height, width = batch["image"].shape[-2:]
                for image_index in batch["det_labeled"].nonzero().flatten().tolist():
                    selected = batch["batch_idx"] == image_index
                    boxes = batch["bboxes"][selected].float() * decoded.new_tensor([width, height, width, height])
                    gt = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
                    gt_classes = batch["cls"][selected].view(-1).long()
                    predictions = decoded[image_index]
                    predictions = predictions[predictions[:, 4] >= confidence]
                    for class_id in range(len(SIGNAL_CLASSES)):
                        truth = gt[gt_classes == class_id]
                        predicted = predictions[predictions[:, 5].long() == class_id]
                        predicted = predicted[predicted[:, 4].argsort(descending=True)]
                        overlaps = box_iou(predicted[:, :4], truth)
                        used: set[int] = set()
                        matched = 0
                        for row in overlaps:
                            for index in row.argsort(descending=True).tolist():
                                if float(row[index]) < iou_threshold:
                                    break
                                if index not in used:
                                    used.add(index)
                                    matched += 1
                                    break
                        det_counts[class_id][0] += matched
                        det_counts[class_id][1] += len(predicted) - matched
                        det_counts[class_id][2] += len(truth) - matched
            if output["roadmark_logits"] is not None:
                predicted = output["roadmark_logits"].float().sigmoid() >= 0.5
                truth = batch["roadmark_target"] >= 0.5
                valid = batch["roadmark_valid"].bool()
                for class_id in range(len(ROADMARK_CLASSES)):
                    p, t, v = predicted[:, class_id], truth[:, class_id], valid[:, class_id]
                    road_counts[class_id][0] += int((p & t & v).sum())
                    road_counts[class_id][1] += int((p & ~t & v).sum())
                    road_counts[class_id][2] += int((~p & t & v).sum())
                lines = decode_roadmark_points(output["roadmark_logits"], batch["meta"])
                for image_index, predicted_lines in enumerate(lines):
                    supervised = valid[image_index].flatten(1).any(dim=1).tolist()
                    if not any(supervised):
                        continue
                    predicted_lines = [line for line in predicted_lines if supervised[line["class_id"]]]
                    ground_truth = [line for line in batch["meta"][image_index]["roadmark_gt"] if supervised[line["class_id"]]]
                    result = match_roadmark_lines(predicted_lines, ground_truth,
                                                  tolerance_px=geometry_tolerance_px)
                    for name, values in result.items():
                        for key, value in values.items():
                            geometry[name][key] += value
    finally:
        model.train(was_training)
        shutdown = getattr(batches, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        if getattr(loader, "_iterator", None) is batches:
            loader._iterator = None
    detection = {name: _scores(*values) for name, values in zip(SIGNAL_CLASSES, det_counts)}
    roadmark = {name: _scores(*values) for name, values in zip(ROADMARK_CLASSES, road_counts)}
    det_total = _scores(*(sum(values[index] for values in det_counts) for index in range(3)))
    road_total = _scores(*(sum(values[index] for values in road_counts) for index in range(3)))
    line_total = _scores(*(sum(int(values[key]) for values in geometry.values()) for key in ("tp", "fp", "fn")))
    for values in geometry.values():
        values.update(_scores(int(values["tp"]), int(values["fp"]), int(values["fn"])))
        count = values["matched_count"]
        values["mean_distance_px"] = values["matched_distance_sum"] / count if count else None
        values["mean_gt_coverage"] = values["matched_gt_coverage_sum"] / count if count else None
        values["mean_pred_coverage"] = values["matched_pred_coverage_sum"] / count if count else None
        if "matched_angle_count" in values:
            count = values["matched_angle_count"]
            values["mean_angle_error_deg"] = values["matched_angle_error_sum_deg"] / count if count else None
    return {"samples": samples,
            "loss": {name: loss_sums[name] / max(counts[name], 1) for name in loss_sums},
            "signal_detection": detection, "signal_detection_total": det_total,
            "roadmark_pixels": roadmark, "roadmark_pixels_total": road_total,
            "roadmark_lines": geometry, "roadmark_lines_total": line_total,
            "geometry_tolerance_px": geometry_tolerance_px,
            "selection_metric": line_total["f1"] if model.train_stage == "roadmark" else det_total["f1"]}
