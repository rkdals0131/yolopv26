from __future__ import annotations

from typing import Any, Iterable, Sequence


def box_size(box: Sequence[float]) -> tuple[float, float]:
    return max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1]))


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter_area = inter_w * inter_h
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def nms_rows(
    rows: Iterable[dict[str, Any]],
    *,
    iou_threshold: float,
    box_key: str = "xyxy",
    score_key: str = "confidence",
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    sorted_rows = sorted(rows, key=lambda item: float(item[score_key]), reverse=True)
    for row in sorted_rows:
        candidate_box = [float(value) for value in row[box_key]]
        if any(iou(candidate_box, [float(value) for value in kept_row[box_key]]) >= iou_threshold for kept_row in kept):
            continue
        kept.append(row)
    return kept


def greedy_match_boxes(
    predictions: Iterable[dict[str, Any]],
    ground_truth_boxes: Iterable[Sequence[float]],
    *,
    match_iou: float,
    box_key: str = "xyxy",
    score_key: str = "confidence",
) -> tuple[int, int, int]:
    gt_rows = [[float(value) for value in box] for box in ground_truth_boxes]
    matched = [False] * len(gt_rows)
    tp = 0
    fp = 0
    for row in sorted(predictions, key=lambda item: float(item[score_key]), reverse=True):
        candidate_box = [float(value) for value in row[box_key]]
        best_index = -1
        best_iou = 0.0
        for index, gt_box in enumerate(gt_rows):
            if matched[index]:
                continue
            overlap = iou(candidate_box, gt_box)
            if overlap >= match_iou and overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index >= 0:
            matched[best_index] = True
            tp += 1
            continue
        fp += 1
    fn = len(gt_rows) - tp
    return tp, fp, fn
