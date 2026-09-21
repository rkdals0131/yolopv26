from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from common.boxes import box_size, iou, nms_rows

from ..build.sweep_types import ClassPolicy, TeacherPredictionRow


def class_policy_to_dict(policy: ClassPolicy) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score_threshold": float(policy.score_threshold),
        "nms_iou_threshold": float(policy.nms_iou_threshold),
        "min_box_size": int(policy.min_box_size),
    }
    if policy.allowed_source_datasets:
        payload["allowed_source_datasets"] = list(policy.allowed_source_datasets)
    if policy.suppress_with_classes:
        payload["suppress_with_classes"] = list(policy.suppress_with_classes)
    if policy.cross_class_iou_threshold is not None:
        payload["cross_class_iou_threshold"] = float(policy.cross_class_iou_threshold)
    if policy.center_x_range is not None:
        payload["center_x_range"] = [float(value) for value in policy.center_x_range]
    if policy.center_y_range is not None:
        payload["center_y_range"] = [float(value) for value in policy.center_y_range]
    if policy.aspect_ratio_range is not None:
        payload["aspect_ratio_range"] = [float(value) for value in policy.aspect_ratio_range]
    if policy.area_ratio_range is not None:
        payload["area_ratio_range"] = [float(value) for value in policy.area_ratio_range]
    return payload


def _optional_float_pair(value: Any, *, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field_name} must be a 2-item list")
    return (float(value[0]), float(value[1]))


def class_policy_from_dict(
    payload: dict[str, Any],
    *,
    default_policy: ClassPolicy | None = None,
) -> ClassPolicy:
    if not isinstance(payload, dict):
        raise TypeError("class policy payload must be a mapping")

    def _required(name: str) -> Any:
        if name in payload:
            return payload[name]
        if default_policy is not None:
            return getattr(default_policy, name)
        raise KeyError(f"class policy missing required field: {name}")

    def _optional_tuple(name: str, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
        if name in payload:
            raw_value = payload[name]
        elif default_policy is not None:
            raw_value = getattr(default_policy, name)
        else:
            raw_value = default
        if raw_value is None:
            return ()
        if isinstance(raw_value, str):
            return (str(raw_value),)
        if not isinstance(raw_value, (list, tuple)):
            raise TypeError(f"{name} must be a list")
        return tuple(str(item) for item in raw_value)

    return ClassPolicy(
        score_threshold=float(_required("score_threshold")),
        nms_iou_threshold=float(_required("nms_iou_threshold")),
        min_box_size=int(_required("min_box_size")),
        allowed_source_datasets=_optional_tuple("allowed_source_datasets"),
        suppress_with_classes=_optional_tuple("suppress_with_classes"),
        cross_class_iou_threshold=(
            float(payload["cross_class_iou_threshold"])
            if payload.get("cross_class_iou_threshold") is not None
            else (
                float(default_policy.cross_class_iou_threshold)
                if default_policy is not None and default_policy.cross_class_iou_threshold is not None
                else None
            )
        ),
        center_x_range=_optional_float_pair(
            payload.get("center_x_range", default_policy.center_x_range if default_policy is not None else None),
            field_name="center_x_range",
        ),
        center_y_range=_optional_float_pair(
            payload.get("center_y_range", default_policy.center_y_range if default_policy is not None else None),
            field_name="center_y_range",
        ),
        aspect_ratio_range=_optional_float_pair(
            payload.get("aspect_ratio_range", default_policy.aspect_ratio_range if default_policy is not None else None),
            field_name="aspect_ratio_range",
        ),
        area_ratio_range=_optional_float_pair(
            payload.get("area_ratio_range", default_policy.area_ratio_range if default_policy is not None else None),
            field_name="area_ratio_range",
        ),
    )


def _range_contains(bounds: tuple[float, float] | None, value: float) -> bool:
    if bounds is None:
        return True
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def _resolve_cross_class_iou_threshold(
    policy: ClassPolicy,
    other_policy: ClassPolicy | None = None,
) -> float:
    if policy.cross_class_iou_threshold is not None:
        return float(policy.cross_class_iou_threshold)
    if other_policy is not None and other_policy.cross_class_iou_threshold is not None:
        return float(other_policy.cross_class_iou_threshold)
    if other_policy is not None:
        return min(float(policy.nms_iou_threshold), float(other_policy.nms_iou_threshold))
    return float(policy.nms_iou_threshold)


def row_passes_geometry_priors(
    *,
    row: TeacherPredictionRow,
    policy: ClassPolicy,
    image_width: int,
    image_height: int,
) -> bool:
    box = [float(value) for value in row["xyxy"]]
    width_px, height_px = box_size(box)
    if min(width_px, height_px) < int(policy.min_box_size):
        return False
    if image_width <= 0 or image_height <= 0:
        return True
    center_x = ((box[0] + box[2]) * 0.5) / float(image_width)
    center_y = ((box[1] + box[3]) * 0.5) / float(image_height)
    area_ratio = (width_px * height_px) / float(image_width * image_height)
    aspect_ratio = width_px / height_px if height_px > 0.0 else float("inf")
    if not _range_contains(policy.center_x_range, center_x):
        return False
    if not _range_contains(policy.center_y_range, center_y):
        return False
    if not _range_contains(policy.aspect_ratio_range, aspect_ratio):
        return False
    if not _range_contains(policy.area_ratio_range, area_ratio):
        return False
    return True


def row_passes_policy(
    *,
    row: TeacherPredictionRow,
    policy: ClassPolicy,
    dataset_key: str,
    image_width: int,
    image_height: int,
) -> bool:
    if float(row["confidence"]) < float(policy.score_threshold):
        return False
    if policy.allowed_source_datasets and dataset_key not in policy.allowed_source_datasets:
        return False
    return row_passes_geometry_priors(
        row=row,
        policy=policy,
        image_width=image_width,
        image_height=image_height,
    )


def apply_policy_to_predictions(
    *,
    rows: Iterable[TeacherPredictionRow],
    class_policy: dict[str, ClassPolicy],
    dataset_key: str,
    image_width: int,
    image_height: int,
    raw_boxes_by_class: dict[str, list[list[float]]] | None = None,
) -> list[TeacherPredictionRow]:
    raw_boxes = raw_boxes_by_class or {}
    filtered_predictions: list[TeacherPredictionRow] = []
    for row in rows:
        class_name = str(row["class_name"])
        policy = class_policy[class_name]
        if not row_passes_policy(
            row=row,
            policy=policy,
            dataset_key=dataset_key,
            image_width=image_width,
            image_height=image_height,
        ):
            continue
        box = [float(value) for value in row["xyxy"]]
        if any(iou(box, raw_box) >= float(policy.nms_iou_threshold) for raw_box in raw_boxes.get(class_name, [])):
            continue
        cross_threshold = _resolve_cross_class_iou_threshold(policy)
        if any(
            iou(box, raw_box) >= cross_threshold
            for other_class in policy.suppress_with_classes
            for raw_box in raw_boxes.get(other_class, [])
        ):
            continue
        filtered_predictions.append(row)

    predictions_by_class: dict[str, list[TeacherPredictionRow]] = defaultdict(list)
    for row in filtered_predictions:
        predictions_by_class[str(row["class_name"])].append(row)
    per_class_kept: list[TeacherPredictionRow] = []
    for class_name, class_rows in predictions_by_class.items():
        per_class_kept.extend(nms_rows(class_rows, iou_threshold=float(class_policy[class_name].nms_iou_threshold)))

    accepted: list[TeacherPredictionRow] = []
    for row in sorted(per_class_kept, key=lambda item: float(item["confidence"]), reverse=True):
        class_name = str(row["class_name"])
        policy = class_policy[class_name]
        candidate_box = [float(value) for value in row["xyxy"]]
        rejected = False
        for accepted_row in accepted:
            accepted_class = str(accepted_row["class_name"])
            accepted_policy = class_policy[accepted_class]
            if accepted_class not in policy.suppress_with_classes and class_name not in accepted_policy.suppress_with_classes:
                continue
            if (
                iou(candidate_box, [float(value) for value in accepted_row["xyxy"]])
                < _resolve_cross_class_iou_threshold(policy, accepted_policy)
            ):
                continue
            rejected = True
            break
        if not rejected:
            accepted.append(row)
    return accepted


__all__ = [
    "apply_policy_to_predictions",
    "class_policy_from_dict",
    "class_policy_to_dict",
    "row_passes_geometry_priors",
    "row_passes_policy",
]
