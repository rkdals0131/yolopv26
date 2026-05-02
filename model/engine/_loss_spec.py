from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.pv26_schema import LANE_CLASSES, LANE_TYPES, OD_CLASSES, TL_BITS

SPEC_VERSION = "pv26-loss-v10"
LossSpec = dict[str, Any]


@dataclass(frozen=True)
class StageSpec:
    name: str
    objective: str
    freeze: str
    loss_weights: dict[str, float]


def _build_training_stages() -> list[StageSpec]:
    return [
        StageSpec(
            name="stage_1_frozen_trunk_warmup",
            objective="새 head와 loss를 먼저 안정화",
            freeze="backbone_and_neck",
            loss_weights={
                "det": 1.0,
                "tl_attr": 0.5,
                "lane": 2.0,
                "stop_line": 1.5,
                "crosswalk": 0.0,
            },
        ),
        StageSpec(
            name="stage_2_partial_unfreeze",
            objective="상위 표현을 다시 맞추면서 lane/geometry 쪽 수렴 가속",
            freeze="lower_backbone_only",
            loss_weights={
                "det": 1.0,
                "tl_attr": 0.5,
                "lane": 1.5,
                "stop_line": 1.0,
                "crosswalk": 0.5,
            },
        ),
        StageSpec(
            name="stage_3_end_to_end_finetune",
            objective="전체 task joint fine-tuning",
            freeze="none",
            loss_weights={
                "det": 1.0,
                "tl_attr": 0.5,
                "lane": 1.0,
                "stop_line": 1.0,
                "crosswalk": 0.75,
            },
        ),
        StageSpec(
            name="stage_4_lane_family_finetune",
            objective="lane family late fine-tuning with detector/TL frozen",
            freeze="trunk_and_detector_tl_heads_frozen",
            loss_weights={
                "det": 0.0,
                "tl_attr": 0.0,
                "lane": 1.5,
                "stop_line": 1.25,
                "crosswalk": 1.0,
            },
        ),
    ]


def _build_model_contract() -> dict[str, Any]:
    return {
        "od_classes": list(OD_CLASSES),
        "tl_bits": list(TL_BITS),
        "lane_classes": list(LANE_CLASSES),
        "lane_types": list(LANE_TYPES),
    }


def _build_heads() -> dict[str, Any]:
    return {
        "det": {
            "shape": "B x Q_det x (4 bbox + 1 obj + 7 cls)",
            "notes": [
                "OD taxonomy is fixed to 7 classes.",
                "traffic_light remains a generic detector class.",
                "Q_det means detector prediction slot count, not GT row count.",
            ],
        },
        "tl_attr": {
            "shape": "B x Q_det x 4",
            "bits": list(TL_BITS),
            "notes": [
                "Attached only to traffic_light predictions/matches.",
                "Training reads the same matched GT index produced by detector assignment.",
                "Uses independent sigmoid logits, not softmax.",
            ],
        },
        "lane": {
            "shape": "B x 24 x 38",
            "query_count": 24,
            "target_encoding": {
                "objectness": 1,
                "color_logits": 3,
                "type_logits": 2,
                "anchor_rows": 16,
                "x_coordinates": 16,
                "visibility_logits": 16,
            },
        },
        "stop_line": {
            "shape": "B x 8 x 9",
            "query_count": 8,
            "target_encoding": {
                "objectness": 1,
                "polyline_points": 4,
                "point_coordinates": 8,
            },
        },
        "crosswalk": {
            "shape": "B x 8 x 33",
            "query_count": 8,
            "target_encoding": {
                "objectness": 1,
                "sequence_points": 16,
                "point_coordinates": 32,
            },
        },
    }


def _build_sample_contract() -> dict[str, Any]:
    return {
        "naming": {
            "N_gt_det": "number of GT detection rows in one sample",
            "Q_det": "number of detector prediction slots in one image",
        },
        "image": {
            "dtype": "float32",
            "shape": [3, 608, 800],
            "range": [0.0, 1.0],
            "color_order": "RGB",
        },
        "det_targets": {
            "boxes_xyxy": "float32[N_gt_det, 4] in network pixel space",
            "classes": "int64[N_gt_det]",
        },
        "tl_attr_targets": {
            "bits": "float32[N_gt_det, 4] aligned 1:1 with det_targets",
            "is_traffic_light": "bool[N_gt_det]",
            "collapse_reason": "list[str] aligned 1:1 with det_targets",
        },
        "lane_targets": {
            "lanes": "list[{points_xy: float32[P,2], color: int, lane_type: int, visibility: float32[P]}]",
            "stop_lines": "list[{points_xy: float32[P,2]}]",
            "crosswalks": "list[{points_xy: float32[P,2]}]",
        },
        "source_mask": {
            "det": "bool",
            "tl_attr": "bool",
            "lane": "bool",
            "stop_line": "bool",
            "crosswalk": "bool",
        },
        "valid_mask": {
            "det": "bool[N_gt_det]",
            "tl_attr": "bool[N_gt_det]",
            "lane": "bool[N_lane]",
            "stop_line": "bool[N_stop]",
            "crosswalk": "bool[N_cross]",
        },
        "meta": {
            "sample_id": "str",
            "dataset_key": "str",
            "split": "str",
            "image_path": "str",
            "raw_hw": "tuple[int, int]",
            "network_hw": "tuple[int, int]",
            "det_supervised_classes": "list[str]",
            "det_supervised_class_ids": "list[int] required when source_mask.det is true",
            "det_allow_objectness_negatives": "bool required when source_mask.det is true",
            "det_allow_unmatched_class_negatives": "bool required when source_mask.det is true",
            "transform": {
                "scale": "float",
                "pad_left": "int",
                "pad_top": "int",
                "pad_right": "int",
                "pad_bottom": "int",
                "resized_hw": "tuple[int, int]",
            },
        },
    }


def _build_encoded_batch_contract() -> dict[str, Any]:
    return {
        "image": "float32[B, 3, 608, 800]",
        "det_gt": "detector-native GT batch derived from N_gt_det",
        "tl_attr_gt_bits": "float32[B, N_gt_det_max, 4]",
        "tl_attr_gt_mask": "bool[B, N_gt_det_max]",
        "lane": "float32[B, 24, 38]",
        "stop_line": "float32[B, 8, 9]",
        "crosswalk": "float32[B, 8, 33]",
        "det_supervision": {
            "det_supervised_class_mask": "bool[B, C_det] with at least one true class for det_source rows",
            "det_allow_objectness_negatives": "bool[B]",
            "det_allow_unmatched_class_negatives": "bool[B]",
        },
        "det_assignment_binding": "computed inside loss and maps Q_det positives to N_gt_det indices",
    }


def _build_transform_contract() -> dict[str, Any]:
    return {
        "dataset_raw_hw": "variable",
        "vehicle_camera_raw_hw": [600, 800],
        "network_hw": [608, 800],
        "scale_formula": "r = min(W_net / W_src, H_net / H_src)",
        "resize_formula": [
            "W_resize = round(W_src * r)",
            "H_resize = round(H_src * r)",
        ],
        "padding_formula": [
            "pad_w = W_net - W_resize",
            "pad_h = H_net - H_resize",
            "pad_left = floor(pad_w / 2)",
            "pad_right = pad_w - pad_left",
            "pad_top = floor(pad_h / 2)",
            "pad_bottom = pad_h - pad_top",
        ],
        "forward_mapping": {
            "x_prime": "x * r + pad_left",
            "y_prime": "y * r + pad_top",
        },
        "inverse_mapping": {
            "x_src": "(x_prime - pad_left) / r",
            "y_src": "(y_prime - pad_top) / r",
        },
        "interpolation": "bilinear",
        "padding_fill_uint8": 114,
        "padding_fill_normalized": 114.0 / 255.0,
        "coordinate_dtype": "float32 until visualization/export raster step",
        "clipping": {
            "x_range": "[0, W_net - 1]",
            "y_range": "[0, H_net - 1]",
            "det_row_policy": "drop row entirely when clipped width <= 1 or height <= 1",
            "tl_attr_invalid_policy": "keep det row, zero bits, valid_mask false",
            "lane_invalid_policy": "keep row, valid_mask false when fewer than 2 unique points after clipping",
            "crosswalk_invalid_policy": "keep row, valid_mask false when fewer than 3 unique points after clipping",
        },
    }


def _build_canonical_target_rules() -> dict[str, Any]:
    return {
        "lane": [
            "Sort bottom-to-top, then project to 16 fixed anchor rows.",
            "Keep original color and solid/dotted attributes in metadata.",
            "Preserve source visibility when available; otherwise derive pseudo-visibility from point span.",
        ],
        "stop_line": [
            "Project to a canonical centerline, then sample 4 ordered polyline points.",
        ],
        "crosswalk": [
            "Order contour clockwise, preserve original vertices when possible, then sample a 16-point contour sequence.",
        ],
        "external_contract": [
            "Training/inference IO stays AIHUB-compatible.",
            "Fixed-length vectors are internal training targets only.",
        ],
    }


def _build_dataset_masking() -> dict[str, Any]:
    return {
        "bdd100k": {
            "det": 1,
            "tl_attr": 0,
            "lane": 0,
            "stop_line": 0,
            "crosswalk": 0,
        },
        "aihub_traffic": {
            "det": 1,
            "tl_attr": "valid_mask_only",
            "lane": 0,
            "stop_line": 0,
            "crosswalk": 0,
        },
        "aihub_obstacle": {
            "det": 1,
            "tl_attr": 0,
            "lane": 0,
            "stop_line": 0,
            "crosswalk": 0,
        },
        "aihub_lane": {
            "det": 0,
            "tl_attr": 0,
            "lane": 1,
            "stop_line": 1,
            "crosswalk": 1,
        },
    }


def _build_tl_attr_policy() -> dict[str, Any]:
    return {
        "base_detector_class": "traffic_light",
        "valid_source_type": "car",
        "arrow_sources": ["left_arrow", "others_arrow"],
        "training_binding": [
            "Reuse the detector assignment result.",
            "Each detector positive reads TL bits from its matched GT index.",
            "No TL attr loss is applied when the matched GT class is not traffic_light.",
            "No TL attr loss is applied when the matched GT row is invalid-masked.",
        ],
        "valid_examples": [
            "off -> [0,0,0,0]",
            "red -> [1,0,0,0]",
            "yellow -> [0,1,0,0]",
            "green -> [0,0,1,0]",
            "arrow_only -> [0,0,0,1]",
            "red+arrow -> [1,0,0,1]",
            "green+arrow -> [0,0,1,1]",
        ],
        "masked_cases": [
            "non_car_traffic_light",
            "missing_attribute_map",
            "x_light_active",
            "multi_color_active",
        ],
        "loss": {
            "type": "sigmoid_focal_bce",
            "bit_weights": {
                "red": 1.0,
                "yellow": 2.5,
                "green": 1.0,
                "arrow": 1.8,
            },
        },
    }


def _build_inference_contract() -> dict[str, Any]:
    return {
        "raw_model_output": {
            "det": "float32[B, Q_det, 12]",
            "tl_attr": "float32[B, Q_det, 4]",
        },
        "prediction_bundle": [
            "box_xyxy",
            "score",
            "class_id",
            "class_name",
            "tl_attr_scores",
        ],
        "ignore_non_tl_attr_scores": True,
    }


def _build_losses() -> dict[str, Any]:
    return {
        "total": "L_total = det + tl_attr + lane + stop_line + crosswalk",
        "det": {
            "type": "yolo_detect_loss",
            "terms": ["box", "obj", "cls"],
        },
        "tl_attr": {
            "type": "masked_sigmoid_focal_bce",
            "normalize_by": "matched_valid_tl_positive_count",
        },
        "lane": {
            "type": "matched_query_loss",
            "subterms": {
                "objectness": 1.0,
                "color_ce": 1.0,
                "type_ce": 0.5,
                "anchor_x_smooth_l1": 5.0,
                "visibility_bce": 1.0,
                "smoothness": 0.25,
                "visibility_tv": 0.1,
            },
        },
        "stop_line": {
            "type": "matched_query_loss",
            "subterms": {
                "objectness": 1.0,
                "polyline_smooth_l1": 6.0,
                "angle_length": 0.5,
            },
        },
        "crosswalk": {
            "type": "matched_query_loss",
            "subterms": {
                "objectness": 1.0,
                "contour_smooth_l1": 4.0,
                "shape_regularizer": 0.5,
            },
        },
    }


def _build_matching() -> dict[str, Any]:
    return {
        "det": "upstream_yolo_assignment",
        "lane": {
            "matcher": "hungarian",
            "costs": {
                "anchor_x_l1": 3.0,
                "color_ce": 1.0,
                "type_ce": 0.5,
                "visibility_bce": 0.5,
            },
        },
        "stop_line": {
            "matcher": "hungarian",
            "costs": {
                "polyline_l1": 4.0,
                "angle_length_cost": 0.5,
            },
        },
        "crosswalk": {
            "matcher": "hungarian",
            "costs": {
                "contour_l1": 3.0,
                "polygon_overlap_cost": 1.0,
            },
        },
    }


def _build_sampler() -> dict[str, Any]:
    return {
        "type": "dataset_balanced",
        "ratios": {
            "bdd100k": 0.30,
            "aihub_traffic": 0.30,
            "aihub_lane": 0.25,
            "aihub_obstacle": 0.15,
        },
    }


def _build_validation() -> dict[str, Any]:
    return {
        "traffic_light": [
            "bit-level AP/F1",
            "decoded combination accuracy",
        ],
        "lane": [
            "polyline point distance",
            "lane color accuracy",
            "lane type accuracy",
        ],
        "stop_line": [
            "centerline point distance",
            "segment angle error",
        ],
        "crosswalk": [
            "polygon IoU",
            "contour vertex distance",
        ],
    }


def build_loss_spec() -> LossSpec:
    stages = _build_training_stages()
    return {
        "version": SPEC_VERSION,
        "model_contract": _build_model_contract(),
        "heads": _build_heads(),
        "sample_contract": _build_sample_contract(),
        "encoded_batch_contract": _build_encoded_batch_contract(),
        "transform_contract": _build_transform_contract(),
        "canonical_target_rules": _build_canonical_target_rules(),
        "dataset_masking": _build_dataset_masking(),
        "tl_attr_policy": _build_tl_attr_policy(),
        "inference_contract": _build_inference_contract(),
        "losses": _build_losses(),
        "matching": _build_matching(),
        "sampler": _build_sampler(),
        "validation": _build_validation(),
        "training_schedule": [asdict(stage) for stage in stages],
    }


def render_loss_spec_markdown(spec: dict[str, Any] | None = None) -> str:
    spec = spec or build_loss_spec()
    lines = [
        "# PV26 Loss Design Spec",
        "",
        f"- Version: `{spec['version']}`",
        "",
        "## Model Contract",
        "",
        f"- OD classes: `{', '.join(spec['model_contract']['od_classes'])}`",
        f"- TL bits: `{', '.join(spec['model_contract']['tl_bits'])}`",
        f"- Lane classes: `{', '.join(spec['model_contract']['lane_classes'])}`",
        f"- Lane types: `{', '.join(spec['model_contract']['lane_types'])}`",
        "",
        "## Heads",
        "",
    ]
    for head_name, head_spec in spec["heads"].items():
        lines.append(f"- `{head_name}`: `{head_spec['shape']}`")
    lines.extend(
        [
            "",
            "## TL Attribute Policy",
            "",
            f"- Base detector class: `{spec['tl_attr_policy']['base_detector_class']}`",
            f"- Valid source type: `{spec['tl_attr_policy']['valid_source_type']}`",
            f"- Arrow sources: `{', '.join(spec['tl_attr_policy']['arrow_sources'])}`",
            f"- Training binding: `{'; '.join(spec['tl_attr_policy']['training_binding'])}`",
            f"- Masked cases: `{', '.join(spec['tl_attr_policy']['masked_cases'])}`",
            "",
            "## Sample Contract",
            "",
            f"- Naming: `N_gt_det={spec['sample_contract']['naming']['N_gt_det']}`",
            f"- Naming: `Q_det={spec['sample_contract']['naming']['Q_det']}`",
            f"- Image shape: `{spec['sample_contract']['image']['shape']}`",
            f"- Network size: `{spec['transform_contract']['network_hw']}`",
            f"- Vehicle camera reference: `{spec['transform_contract']['vehicle_camera_raw_hw']}`",
            f"- Lane query count: `{spec['heads']['lane']['query_count']}`",
            f"- Stop-line query count: `{spec['heads']['stop_line']['query_count']}`",
            f"- Crosswalk query count: `{spec['heads']['crosswalk']['query_count']}`",
            "",
            "## Dataset Masking",
            "",
        ]
    )
    for dataset_name, mask in spec["dataset_masking"].items():
        lines.append(f"- `{dataset_name}`: `{mask}`")
    lines.extend(
        [
            "",
            "## Schedule",
            "",
        ]
    )
    for stage in spec["training_schedule"]:
        lines.append(
            f"- `{stage['name']}`: freeze=`{stage['freeze']}`, objective=`{stage['objective']}`, "
            f"weights=`{stage['loss_weights']}`"
        )
    return "\n".join(lines) + "\n"


__all__ = ["SPEC_VERSION", "build_loss_spec", "render_loss_spec_markdown"]
