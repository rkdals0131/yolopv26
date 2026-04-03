from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, TypedDict

from common.io import now_iso
from common.pv26_schema import LANE_CLASSES, LANE_TYPES, OD_CLASSES, TL_BITS

from ..shared.summary import counter_to_dict


class AIHubDebugVisSummary(TypedDict):
    selection_count: int
    seed: int


class AIHubQADatasetSummary(TypedDict):
    dataset_key: str
    processed_samples: int
    fresh_processed_count: int
    resume_skipped_count: int
    failure_count: int
    empty_scene_count: int
    traffic_light_count: int
    traffic_sign_count: int
    detection_count: int
    lane_count: int
    top_held_reasons: list[tuple[str, int]]
    top_tl_invalid_reasons: list[tuple[str, int]]


class AIHubQASummary(TypedDict):
    version: str
    generated_at: str
    output_root: str
    debug_vis: AIHubDebugVisSummary
    failure_count: int
    datasets: list[AIHubQADatasetSummary]


def aggregate_results(
    lane_root: Path,
    traffic_root: Path,
    obstacle_root: Path,
    output_root: Path,
    workers: int,
    max_samples_per_dataset: int | None,
    debug_vis_count: int,
    source_inventory: dict[str, Any],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    pipeline_version: str,
    scene_version: str,
    source_root_for_dataset: Callable[..., Path],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        grouped[item["dataset_key"]].append(item)

    datasets: list[dict[str, Any]] = []
    for dataset_key, items in sorted(grouped.items()):
        split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        det_class_counts = Counter()
        lane_class_counts = Counter()
        lane_type_counts = Counter()
        tl_combo_counts = Counter()
        tl_invalid_reason_counts = Counter()
        held_reason_counts = Counter()
        materialization_counts = Counter()
        total_stop_lines = 0
        total_crosswalks = 0
        total_tl_valid = 0
        total_tl_invalid = 0
        total_lights = 0
        total_signs = 0
        total_dets = 0
        total_resume_skipped = 0
        empty_scene_count = 0
        for item in items:
            split = item["split"]
            split_counts[split]["samples"] += 1
            split_counts[split]["detections"] += item["det_count"]
            split_counts[split]["lanes"] += item["lane_count"]
            split_counts[split]["stop_lines"] += item["stop_line_count"]
            split_counts[split]["crosswalks"] += item["crosswalk_count"]
            split_counts[split]["traffic_lights"] += item["traffic_light_count"]
            split_counts[split]["traffic_signs"] += item["traffic_sign_count"]
            total_stop_lines += item["stop_line_count"]
            total_crosswalks += item["crosswalk_count"]
            total_tl_valid += item["tl_attr_valid_count"]
            total_tl_invalid += item["tl_attr_invalid_count"]
            total_lights += item["traffic_light_count"]
            total_signs += item["traffic_sign_count"]
            total_dets += item["det_count"]
            total_resume_skipped += int(item.get("resume_skipped", 0))
            materialization_counts[item["image_materialization"]] += 1
            det_class_counts.update(item["det_class_counts"])
            lane_class_counts.update(item["lane_class_counts"])
            lane_type_counts.update(item["lane_type_counts"])
            tl_combo_counts.update(item["tl_combo_counts"])
            tl_invalid_reason_counts.update(item["tl_invalid_reason_counts"])
            held_reason_counts.update(item["held_reason_counts"])
            if (
                item["det_count"] == 0
                and item["lane_count"] == 0
                and item["stop_line_count"] == 0
                and item["crosswalk_count"] == 0
            ):
                empty_scene_count += 1

        dataset_failures = [item for item in failures if item["dataset_key"] == dataset_key]

        datasets.append(
            {
                "dataset_key": dataset_key,
                "source_root": str(
                    source_root_for_dataset(
                        dataset_key,
                        lane_root=lane_root,
                        traffic_root=traffic_root,
                        obstacle_root=obstacle_root,
                    )
                ),
                "processed_samples": len(items),
                "fresh_processed_count": len(items) - total_resume_skipped,
                "resume_skipped_count": total_resume_skipped,
                "failure_count": len(dataset_failures),
                "per_split_counts": {
                    split: {key: value for key, value in sorted(counts.items())}
                    for split, counts in sorted(split_counts.items())
                },
                "det_class_counts": counter_to_dict(det_class_counts),
                "lane_class_counts": counter_to_dict(lane_class_counts),
                "lane_type_counts": counter_to_dict(lane_type_counts),
                "stop_line_count": total_stop_lines,
                "crosswalk_count": total_crosswalks,
                "traffic_light_count": total_lights,
                "traffic_sign_count": total_signs,
                "detection_count": total_dets,
                "tl_attr_valid_count": total_tl_valid,
                "tl_attr_invalid_count": total_tl_invalid,
                "tl_combo_counts": counter_to_dict(tl_combo_counts),
                "tl_invalid_reason_counts": counter_to_dict(tl_invalid_reason_counts),
                "held_reason_counts": counter_to_dict(held_reason_counts),
                "image_materialization": counter_to_dict(materialization_counts),
                "empty_scene_count": empty_scene_count,
            }
        )

    return {
        "version": pipeline_version,
        "scene_version": scene_version,
        "generated_at": now_iso(),
        "settings": {
            "workers": workers,
            "max_samples_per_dataset": max_samples_per_dataset,
            "debug_vis_count": debug_vis_count,
            "output_root": str(output_root),
            "failure_count": len(failures),
        },
        "det_class_map": {str(index): class_name for index, class_name in enumerate(OD_CLASSES)},
        "lane_class_map": {str(index): class_name for index, class_name in enumerate(LANE_CLASSES)},
        "tl_bits": TL_BITS,
        "datasets": datasets,
        "failures": failures,
        "source_inventory_snapshot": source_inventory,
    }


def conversion_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PV26 AIHUB Conversion Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Version: `{report['version']}`",
        f"- Output root: `{report['settings']['output_root']}`",
        f"- Workers: `{report['settings']['workers']}`",
        f"- Max samples per dataset: `{report['settings']['max_samples_per_dataset']}`",
        "",
    ]
    for dataset in report["datasets"]:
        lines.extend(
            [
                f"## {dataset['dataset_key']}",
                "",
                f"- Processed samples: `{dataset['processed_samples']}`",
                f"- Fresh processed samples: `{dataset['fresh_processed_count']}`",
                f"- Resume skipped samples: `{dataset['resume_skipped_count']}`",
                f"- Failure count: `{dataset['failure_count']}`",
                f"- Detection count: `{dataset['detection_count']}`",
                f"- Lane count: `{sum(item['lanes'] for item in dataset['per_split_counts'].values()) if dataset['per_split_counts'] else 0}`",
                f"- Stop line count: `{dataset['stop_line_count']}`",
                f"- Crosswalk count: `{dataset['crosswalk_count']}`",
                f"- Traffic light count: `{dataset['traffic_light_count']}`",
                f"- Traffic sign count: `{dataset['traffic_sign_count']}`",
                "",
                "| Split | Samples | Detections | Lanes | Stop lines | Crosswalks | TL | Sign |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for split, counts in dataset["per_split_counts"].items():
            lines.append(
                f"| {split} | {counts.get('samples', 0)} | {counts.get('detections', 0)} | {counts.get('lanes', 0)} "
                f"| {counts.get('stop_lines', 0)} | {counts.get('crosswalks', 0)} | {counts.get('traffic_lights', 0)} "
                f"| {counts.get('traffic_signs', 0)} |"
            )
        if dataset["det_class_counts"]:
            lines.extend(["", "### Detection Class Counts", ""])
            for key, value in dataset["det_class_counts"].items():
                lines.append(f"- `{key}`: {value}")
        if dataset["lane_class_counts"]:
            lines.extend(["", "### Lane Class Counts", ""])
            for key, value in dataset["lane_class_counts"].items():
                lines.append(f"- `{key}`: {value}")
        if dataset["tl_combo_counts"]:
            lines.extend(["", "### TL Combo Counts", ""])
            for key, value in dataset["tl_combo_counts"].items():
                lines.append(f"- `{key}`: {value}")
        if dataset["tl_invalid_reason_counts"]:
            lines.extend(["", "### TL Invalid Reasons", ""])
            for key, value in dataset["tl_invalid_reason_counts"].items():
                lines.append(f"- `{key}`: {value}")
        if dataset["held_reason_counts"]:
            lines.extend(["", "### Held Reasons", ""])
            for key, value in dataset["held_reason_counts"].items():
                lines.append(f"- `{key}`: {value}")
        lines.append("")
    if report["failures"]:
        lines.extend(["## Failure Manifest", ""])
        for item in report["failures"][:32]:
            lines.append(
                f"- dataset=`{item['dataset_key']}` split=`{item['split']}` raw_id=`{item['raw_id']}` error=`{item['error_type']}`"
            )
    return "\n".join(lines) + "\n"


def failure_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# PV26 AIHUB Failure Manifest",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Version: `{manifest['version']}`",
        f"- Failure count: `{manifest['failure_count']}`",
        "",
    ]
    for item in manifest["items"]:
        lines.extend(
            [
                f"## {item['dataset_key']} / {item['split']} / {item['raw_id']}",
                "",
                f"- Error type: `{item['error_type']}`",
                f"- Error message: `{item['error_message']}`",
                f"- Image path: `{item['image_path']}`",
                f"- Label path: `{item['label_path']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def qa_summary(report: dict[str, Any], debug_vis_index: dict[str, Any], failure_manifest: dict[str, Any]) -> AIHubQASummary:
    return {
        "version": report["version"],
        "generated_at": now_iso(),
        "output_root": report["settings"]["output_root"],
        "debug_vis": {
            "selection_count": int(debug_vis_index.get("selection_count", 0)),
            "seed": int(debug_vis_index.get("seed", 0)),
        },
        "failure_count": int(failure_manifest["failure_count"]),
        "datasets": [
            {
                "dataset_key": item["dataset_key"],
                "processed_samples": item["processed_samples"],
                "fresh_processed_count": item["fresh_processed_count"],
                "resume_skipped_count": item["resume_skipped_count"],
                "failure_count": item["failure_count"],
                "empty_scene_count": item["empty_scene_count"],
                "traffic_light_count": item["traffic_light_count"],
                "traffic_sign_count": item["traffic_sign_count"],
                "detection_count": item["detection_count"],
                "lane_count": sum(split.get("lanes", 0) for split in item["per_split_counts"].values()),
                "top_held_reasons": list(item["held_reason_counts"].items())[:5],
                "top_tl_invalid_reasons": list(item["tl_invalid_reason_counts"].items())[:5],
            }
            for item in report["datasets"]
        ],
    }


def qa_summary_markdown(summary: AIHubQASummary) -> str:
    lines = [
        "# PV26 AIHUB QA Summary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Output root: `{summary['output_root']}`",
        f"- Debug vis selections: `{summary['debug_vis']['selection_count']}`",
        f"- Failure count: `{summary['failure_count']}`",
        "",
    ]
    for item in summary["datasets"]:
        lines.extend(
            [
                f"## {item['dataset_key']}",
                "",
                f"- Processed samples: `{item['processed_samples']}`",
                f"- Fresh processed samples: `{item['fresh_processed_count']}`",
                f"- Resume skipped samples: `{item['resume_skipped_count']}`",
                f"- Failure count: `{item['failure_count']}`",
                f"- Empty scenes: `{item['empty_scene_count']}`",
                f"- Detection count: `{item['detection_count']}`",
                f"- Lane count: `{item['lane_count']}`",
                f"- Traffic lights: `{item['traffic_light_count']}`",
                f"- Traffic signs: `{item['traffic_sign_count']}`",
                "",
            ]
        )
        if item["top_held_reasons"]:
            lines.append("### Top Held Reasons")
            lines.append("")
            for key, value in item["top_held_reasons"]:
                lines.append(f"- `{key}`: {value}")
            lines.append("")
        if item["top_tl_invalid_reasons"]:
            lines.append("### Top TL Invalid Reasons")
            lines.append("")
            for key, value in item["top_tl_invalid_reasons"]:
                lines.append(f"- `{key}`: {value}")
            lines.append("")
    return "\n".join(lines) + "\n"


def det_class_map_yaml() -> str:
    return shared_det_class_map_yaml()


def scene_class_map_yaml() -> str:
    lines = [
        "version: pv26-scene-aihub-v1",
        "lane_classes:",
    ]
    for class_name in LANE_CLASSES:
        lines.append(f"  - {class_name}")
    lines.extend(
        [
            "lane_types:",
        ]
    )
    for lane_type in LANE_TYPES:
        lines.append(f"  - {lane_type}")
    lines.extend(
        [
            "other_geometry_classes:",
            "  - stop_line",
            "  - crosswalk",
        ]
    )
    return "\n".join(lines) + "\n"
