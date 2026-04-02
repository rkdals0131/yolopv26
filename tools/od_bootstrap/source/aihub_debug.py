from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
from typing import Any, Callable

from common.overlay import render_overlay

from .types import DebugVisManifest, DebugVisManifestItem, DebugVisOutputs, DebugVisSummaryRow


def _debug_vis_priority(item: DebugVisSummaryRow, *, obstacle_dataset_key: str) -> tuple[int, int, int]:
    annotation_score = (
        int(item.get("det_count", 0))
        + int(item.get("traffic_light_count", 0))
        + int(item.get("traffic_sign_count", 0))
        + int(item.get("lane_count", 0))
        + int(item.get("stop_line_count", 0))
        + int(item.get("crosswalk_count", 0))
    )
    obstacle_positive = int(item.get("dataset_key") == obstacle_dataset_key and int(item.get("det_count", 0)) > 0)
    non_empty = int(annotation_score > 0)
    return obstacle_positive, non_empty, annotation_score


def _select_debug_vis_summaries(
    summaries: list[DebugVisSummaryRow],
    *,
    count: int,
    seed: int,
    obstacle_dataset_key: str,
) -> list[DebugVisSummaryRow]:
    if count <= 0 or not summaries:
        return []

    grouped: dict[str, list[DebugVisSummaryRow]] = {}
    for item in summaries:
        grouped.setdefault(str(item["dataset_key"]), []).append(item)

    rng = random.Random(seed)
    dataset_keys = sorted(grouped)
    for dataset_key in dataset_keys:
        rng.shuffle(grouped[dataset_key])
        grouped[dataset_key].sort(
            key=lambda item: _debug_vis_priority(item, obstacle_dataset_key=obstacle_dataset_key)
        )

    selected: list[DebugVisSummaryRow] = []
    remaining = min(count, len(summaries))
    active_keys = [key for key in dataset_keys if grouped[key]]
    while remaining > 0 and active_keys:
        next_active: list[str] = []
        for dataset_key in active_keys:
            bucket = grouped[dataset_key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            remaining -= 1
            if bucket:
                next_active.append(dataset_key)
            if remaining == 0:
                break
        active_keys = next_active
    return selected


def build_debug_vis_manifest(
    *,
    generated_at: str,
    debug_vis_seed: int,
    items: list[DebugVisManifestItem],
) -> DebugVisManifest:
    return {
        "generated_at": generated_at,
        "selection_count": len(items),
        "seed": debug_vis_seed,
        "items": sorted(items, key=lambda item: (item["dataset_key"], item["split"], item["sample_id"])),
    }


def _render_debug_vis_entry(
    scene_path: str,
    output_path: str,
    *,
    load_json_fn: Callable[[Path], dict[str, Any]],
    prepare_scene_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, str]:
    scene = load_json_fn(Path(scene_path))
    render_overlay(prepare_scene_fn(scene), Path(output_path))
    return {
        "scene_path": scene_path,
        "output_path": output_path,
    }


def _generate_debug_vis(
    output_root: Path,
    summaries: list[DebugVisSummaryRow],
    *,
    debug_vis_count: int,
    debug_vis_seed: int,
    obstacle_dataset_key: str,
    logger: Any,
    debug_vis_dirname: str,
    now_iso_fn: Callable[[], str],
    write_json_fn: Callable[[Path, dict[str, Any]], None],
    load_json_fn: Callable[[Path], dict[str, Any]],
    prepare_scene_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> DebugVisOutputs:
    debug_vis_dir = output_root / "meta" / debug_vis_dirname
    debug_vis_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_debug_vis_summaries(
        summaries,
        count=debug_vis_count,
        seed=debug_vis_seed,
        obstacle_dataset_key=obstacle_dataset_key,
    )
    index_path = debug_vis_dir / "index.json"
    if not selected:
        write_json_fn(
            index_path,
            build_debug_vis_manifest(
                generated_at=now_iso_fn(),
                debug_vis_seed=debug_vis_seed,
                items=[],
            ),
        )
        return {
            "debug_vis_dir": debug_vis_dir,
            "debug_vis_index": index_path,
        }

    logger.stage(
        "debug_vis",
        "표준화 결과를 사람이 빠르게 검수할 수 있도록 랜덤 샘플에 라벨 오버레이 PNG를 생성합니다.",
        total=len(selected) + 1,
    )

    completed = 0
    items: list[DebugVisManifestItem] = []
    max_workers = max(1, min(8, len(selected)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for item in selected:
            output_path = debug_vis_dir / item["split"] / f"{item['sample_id']}.png"
            future = executor.submit(
                _render_debug_vis_entry,
                item["scene_path"],
                str(output_path),
                load_json_fn=load_json_fn,
                prepare_scene_fn=prepare_scene_fn,
            )
            future_map[future] = (item, output_path)

        for future in as_completed(future_map):
            item, output_path = future_map[future]
            result = future.result()
            completed += 1
            items.append(
                {
                    "dataset_key": item["dataset_key"],
                    "split": item["split"],
                    "sample_id": item["sample_id"],
                    "scene_path": result["scene_path"],
                    "image_path": item["image_path"],
                    "output_path": result["output_path"],
                }
            )
            logger.progress(completed, {"rendered": completed}, force=True)

    write_json_fn(
        index_path,
        build_debug_vis_manifest(
            generated_at=now_iso_fn(),
            debug_vis_seed=debug_vis_seed,
            items=items,
        ),
    )
    logger.progress(len(selected) + 1, {"rendered": len(items), "index_written": 1}, force=True)
    return {
        "debug_vis_dir": debug_vis_dir,
        "debug_vis_index": index_path,
    }


def select_debug_vis_summaries(
    summaries: list[DebugVisSummaryRow],
    *,
    count: int,
    seed: int,
    obstacle_dataset_key: str,
) -> list[DebugVisSummaryRow]:
    return _select_debug_vis_summaries(
        summaries,
        count=count,
        seed=seed,
        obstacle_dataset_key=obstacle_dataset_key,
    )


def generate_debug_vis_outputs(
    output_root: Path,
    summaries: list[DebugVisSummaryRow],
    *,
    debug_vis_count: int,
    debug_vis_seed: int,
    obstacle_dataset_key: str,
    logger: Any,
    debug_vis_dirname: str,
    now_iso_fn: Callable[[], str],
    write_json_fn: Callable[[Path, dict[str, Any]], None],
    load_json_fn: Callable[[Path], dict[str, Any]],
    prepare_scene_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> DebugVisOutputs:
    return _generate_debug_vis(
        output_root,
        summaries,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        obstacle_dataset_key=obstacle_dataset_key,
        logger=logger,
        debug_vis_dirname=debug_vis_dirname,
        now_iso_fn=now_iso_fn,
        write_json_fn=write_json_fn,
        load_json_fn=load_json_fn,
        prepare_scene_fn=prepare_scene_fn,
    )


__all__ = [
    "build_debug_vis_manifest",
    "generate_debug_vis_outputs",
    "select_debug_vis_summaries",
]
