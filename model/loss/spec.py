from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..preprocess.aihub_standardize import LANE_CLASSES, LANE_TYPES, OD_CLASSES, TL_BITS

SPEC_VERSION = "pv26-loss-v2"


@dataclass(frozen=True)
class StageSpec:
    name: str
    objective: str
    freeze: str
    loss_weights: dict[str, float]


def build_loss_spec() -> dict[str, Any]:
    stages = [
        StageSpec(
            name="stage_0_smoke",
            objective="shape, target encoder, NaN guard 확인",
            freeze="none",
            loss_weights={
                "det": 1.0,
                "tl_attr": 0.5,
                "lane": 1.0,
                "stop_line": 1.0,
                "crosswalk": 0.0,
            },
        ),
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
    ]
    return {
        "version": SPEC_VERSION,
        "model_contract": {
            "od_classes": list(OD_CLASSES),
            "tl_bits": list(TL_BITS),
            "lane_classes": list(LANE_CLASSES),
            "lane_types": list(LANE_TYPES),
        },
        "heads": {
            "det": {
                "shape": "B x N_det x (4 bbox + 1 obj + 7 cls)",
                "notes": [
                    "OD taxonomy is fixed to 7 classes.",
                    "traffic_light remains a generic detector class.",
                ],
            },
            "tl_attr": {
                "shape": "B x N_det x 4",
                "bits": list(TL_BITS),
                "notes": [
                    "Attached only to traffic_light predictions/matches.",
                    "Training reads the same matched GT index produced by detector assignment.",
                    "Uses independent sigmoid logits, not softmax.",
                ],
            },
            "lane": {
                "shape": "B x 12 x 54",
                "query_count": 12,
                "target_encoding": {
                    "objectness": 1,
                    "color_logits": 3,
                    "type_logits": 2,
                    "polyline_points": 16,
                    "point_coordinates": 32,
                    "visibility_logits": 16,
                },
            },
            "stop_line": {
                "shape": "B x 6 x 9",
                "query_count": 6,
                "target_encoding": {
                    "objectness": 1,
                    "polyline_points": 4,
                    "point_coordinates": 8,
                },
            },
            "crosswalk": {
                "shape": "B x 4 x 17",
                "query_count": 4,
                "target_encoding": {
                    "objectness": 1,
                    "polygon_points": 8,
                    "point_coordinates": 16,
                },
            },
        },
        "sample_contract": {
            "image": {
                "dtype": "float32",
                "shape": [3, 608, 800],
                "range": [0.0, 1.0],
                "color_order": "RGB",
            },
            "det_targets": {
                "boxes_xyxy": "float32[N_det, 4] in network pixel space",
                "classes": "int64[N_det]",
            },
            "tl_attr_targets": {
                "bits": "float32[N_det, 4] aligned 1:1 with det_targets",
                "is_traffic_light": "bool[N_det]",
                "collapse_reason": "list[str] aligned 1:1 with det_targets",
            },
            "lane_targets": {
                "lanes": "list[{points_xy: float32[P,2], color: int, lane_type: int}]",
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
                "det": "bool[N_det]",
                "tl_attr": "bool[N_det]",
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
                "transform": {
                    "scale": "float",
                    "pad_left": "int",
                    "pad_top": "int",
                    "pad_right": "int",
                    "pad_bottom": "int",
                    "resized_hw": "tuple[int, int]",
                },
            },
        },
        "transform_contract": {
            "runtime_raw_hw": [600, 800],
            "network_hw": [608, 800],
            "scale_formula": "r = min(W_net / W_raw, H_net / H_raw)",
            "resize_formula": [
                "W_resize = round(W_raw * r)",
                "H_resize = round(H_raw * r)",
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
                "x_raw": "(x_prime - pad_left) / r",
                "y_raw": "(y_prime - pad_top) / r",
            },
            "clipping": {
                "x_range": "[0, W_net - 1]",
                "y_range": "[0, H_net - 1]",
                "drop_small_bbox": "width <= 1 or height <= 1",
                "invalid_polyline": "fewer than 2 unique points after clipping",
            },
        },
        "canonical_target_rules": {
            "lane": [
                "Sort bottom-to-top, then resample by arc length to 16 points.",
                "Keep original color and solid/dotted attributes in metadata.",
            ],
            "stop_line": [
                "Sort left-to-right, then resample to 4 points.",
            ],
            "crosswalk": [
                "Order contour clockwise, then resample to 8 polygon points.",
            ],
            "external_contract": [
                "Training/inference IO stays AIHUB-compatible.",
                "Fixed-length vectors are internal training targets only.",
            ],
        },
        "dataset_masking": {
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
            "aihub_lane": {
                "det": 0,
                "tl_attr": 0,
                "lane": 1,
                "stop_line": 1,
                "crosswalk": 1,
            },
        },
        "tl_attr_policy": {
            "base_detector_class": "traffic_light",
            "valid_source_type": "car",
            "arrow_sources": ["left_arrow", "others_arrow"],
            "training_binding": [
                "Reuse the detector assignment result.",
                "Each detector positive reads TL bits from its matched GT index.",
                "No TL attr loss is applied when the matched GT class is not traffic_light.",
                "No TL attr loss is applied when the matched GT row is invalid-masked.",
            ],
            "inference_output": {
                "prediction_bundle": [
                    "box_xyxy",
                    "score",
                    "class_id",
                    "class_name",
                    "tl_attr_scores",
                ],
                "ignore_non_tl_attr_scores": True,
            },
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
        },
        "losses": {
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
                    "points_l1": 5.0,
                    "visibility_bce": 1.0,
                    "smoothness": 0.25,
                },
            },
            "stop_line": {
                "type": "matched_query_loss",
                "subterms": {
                    "objectness": 1.0,
                    "points_l1": 6.0,
                    "straightness": 0.5,
                },
            },
            "crosswalk": {
                "type": "matched_query_loss",
                "subterms": {
                    "objectness": 1.0,
                    "polygon_l1": 4.0,
                    "shape_regularizer": 0.5,
                },
            },
        },
        "matching": {
            "det": "upstream_yolo_assignment",
            "lane": {
                "matcher": "hungarian",
                "costs": {
                    "points_l1": 3.0,
                    "color_ce": 1.0,
                    "type_ce": 0.5,
                    "visibility_bce": 0.5,
                },
            },
            "stop_line": {
                "matcher": "hungarian",
                "costs": {
                    "points_l1": 4.0,
                    "angle_length_cost": 0.5,
                },
            },
            "crosswalk": {
                "matcher": "hungarian",
                "costs": {
                    "polygon_l1": 3.0,
                    "polygon_overlap_cost": 1.0,
                },
            },
        },
        "sampler": {
            "type": "dataset_balanced",
            "ratios": {
                "bdd100k": 0.35,
                "aihub_traffic": 0.35,
                "aihub_lane": 0.30,
            },
        },
        "validation": {
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
                "point distance",
                "segment angle error",
            ],
            "crosswalk": [
                "polygon IoU",
                "vertex distance",
            ],
        },
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
            f"- Image shape: `{spec['sample_contract']['image']['shape']}`",
            f"- Network size: `{spec['transform_contract']['network_hw']}`",
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
