from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TextIO

from common.io import now_iso, write_text
from common.pv26_schema import (
    AIHUB_LANE_DATASET_KEY,
    AIHUB_OBSTACLE_DATASET_KEY,
    AIHUB_TRAFFIC_DATASET_KEY,
    TL_BITS,
)
from .reports import (
    aggregate_results as _aggregate_results,
    conversion_report_markdown as _conversion_report_markdown,
    failure_manifest_markdown as _failure_manifest_markdown,
    qa_summary as _qa_summary,
    qa_summary_markdown as _qa_summary_markdown,
    scene_class_map_yaml as _scene_class_map_yaml,
)
from .source_meta import (
    build_source_inventory as _build_source_inventory,
    source_inventory_markdown as _source_inventory_markdown,
    source_root_for_dataset as _source_root_for_dataset,
    write_source_readmes as _write_source_readmes,
)
from .worker_common import SCENE_VERSION, StandardizeTask
from .workers import (
    existing_output_summary as _existing_output_summary,
    prepare_debug_scene_for_overlay as _prepare_debug_scene_for_overlay,
    worker_chunk_entry as _worker_chunk_entry,
)
from ..raw_common import (
    LANE_DATASET_KEY as DISCOVERY_LANE_KEY,
    OBSTACLE_DATASET_KEY as DISCOVERY_OBSTACLE_KEY,
    TRAFFIC_DATASET_KEY as DISCOVERY_TRAFFIC_KEY,
    PairRecord,
)
from ..shared.debug import (
    generate_debug_vis as _generate_debug_vis_impl,
    select_debug_vis_summaries as _select_debug_vis_summaries_impl,
)
from ..shared.io import load_json, write_json
from ..shared.parallel import (
    LiveLogger,
    PARALLEL_INFLIGHT_CHUNKS_PER_WORKER,
    PARALLEL_SUBMIT_LOG_INTERVAL,
    PARALLEL_WAIT_HEARTBEAT_SECONDS,
    default_workers,
    iter_task_chunks,
    parallel_chunk_size,
)
from ..shared.raw import (
    discover_pairs as _discover_pairs,
    env_path as _env_path,
    extract_annotations as _extract_annotations,
    normalize_text as _normalize_text,
    repo_root as _repo_root,
    safe_slug as _safe_slug,
    seg_dataset_root as _seg_dataset_root,
)
from ..shared.reports import det_class_map_yaml as _det_class_map_yaml
from ..types import DebugVisOutputs, DebugVisSummaryRow

PIPELINE_VERSION = "pv26-aihub-standardize-v1"
DEFAULT_REPO_ROOT = _repo_root()
DEFAULT_SEG_DATASET_ROOT = _seg_dataset_root(DEFAULT_REPO_ROOT)
DEFAULT_AIHUB_ROOT = _env_path("PV26_AIHUB_ROOT", DEFAULT_SEG_DATASET_ROOT / "AIHUB")
DEFAULT_DOCS_ROOT = DEFAULT_AIHUB_ROOT / "docs"
DEFAULT_LANE_ROOT = DEFAULT_AIHUB_ROOT / "차선-횡단보도 인지 영상(수도권)"
DEFAULT_OBSTACLE_ROOT = DEFAULT_AIHUB_ROOT / "도로장애물·표면 인지 영상(수도권)"
DEFAULT_TRAFFIC_ROOT = DEFAULT_AIHUB_ROOT / "신호등-도로표지판 인지 영상(수도권)"
DEFAULT_OUTPUT_ROOT = _env_path("PV26_AIHUB_OUTPUT_ROOT", DEFAULT_AIHUB_ROOT.parent / "pv26_aihub_standardized")
CACHE_DIR_NAME = "_cache"
DEBUG_VIS_DIRNAME = "debug_vis"
DEFAULT_DEBUG_VIS_COUNT = 20
DEFAULT_DEBUG_VIS_SEED = 26
OUTPUT_LANE_KEY = AIHUB_LANE_DATASET_KEY
OUTPUT_OBSTACLE_KEY = AIHUB_OBSTACLE_DATASET_KEY
OUTPUT_TRAFFIC_KEY = AIHUB_TRAFFIC_DATASET_KEY
DOCUMENTED_STATS = {
    OUTPUT_LANE_KEY: {
        "json_count_seoul": 1_147_048,
        "lane_objects_seoul": 6_115_856,
        "crosswalk_objects_seoul": 407_494,
        "stop_line_objects_seoul": 271_461,
        "white_lane_objects_seoul": 4_380_045,
        "yellow_lane_objects_seoul": 1_612_795,
        "blue_lane_objects_seoul": 123_016,
        "solid_lane_objects_seoul": 3_242_630,
        "dotted_lane_objects_seoul": 2_873_226,
        "doc_reference": "차선_횡단보도_인지_영상(수도권)_데이터_구축_가이드라인.pdf",
    },
    OUTPUT_TRAFFIC_KEY: {
        "json_count_seoul": 1_106_612,
        "traffic_light_objects_seoul": 1_970_735,
        "traffic_sign_objects_seoul": 1_608_805,
        "traffic_light_red_seoul": 474_481,
        "traffic_light_yellow_seoul": 51_639,
        "traffic_light_green_seoul": 587_682,
        "traffic_light_left_arrow_seoul": 100_816,
        "traffic_sign_instruction_seoul": 598_752,
        "traffic_sign_caution_seoul": 138_707,
        "traffic_sign_restriction_seoul": 800_992,
        "doc_reference": "수도권신호등표지판_인공지능 데이터 구축활용 가이드라인_통합수정_210607.pdf",
    },
}


def _archive_paths(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in {".zip", ".tar"})


def _has_extracted_assets(dataset_root: Path) -> bool:
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"}:
            return True
    return False


def _extract_archives_if_needed(dataset_root: Path, cache_root: Path, logger: LiveLogger) -> Path:
    if _has_extracted_assets(dataset_root):
        logger.info(f"archive_extract skip | root={dataset_root} | reason=extracted assets already available")
        return dataset_root

    archives = _archive_paths(dataset_root)
    if not archives:
        raise FileNotFoundError(f"no extracted assets or archives found under {dataset_root}")

    extract_root = cache_root / _safe_slug(dataset_root.name)
    extract_root.mkdir(parents=True, exist_ok=True)
    logger.stage(
        f"extract:{dataset_root.name}",
        "원본 디렉터리에 추출본이 없어서 output cache에 archive를 풀어 작업 가능한 파일 트리를 만듭니다.",
        total=len(archives),
    )

    completed = 0
    for archive_path in archives:
        target_dir = extract_root / _safe_slug(str(archive_path.relative_to(dataset_root).with_suffix("")))
        done_marker = target_dir / ".done"
        if done_marker.is_file():
            completed += 1
            logger.progress(completed, {"archives": completed}, force=True)
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as archive_file:
                archive_file.extractall(target_dir)
        else:
            with tarfile.open(archive_path) as archive_file:
                archive_file.extractall(target_dir)
        write_text(done_marker, "ok\n")
        completed += 1
        logger.progress(completed, {"archives": completed}, force=True)

    return extract_root


def _select_debug_vis_summaries(
    summaries: list[DebugVisSummaryRow],
    *,
    count: int,
    seed: int,
) -> list[DebugVisSummaryRow]:
    return _select_debug_vis_summaries_impl(
        summaries,
        count=count,
        seed=seed,
        obstacle_dataset_key=OUTPUT_OBSTACLE_KEY,
    )


def _generate_debug_vis(
    output_root: Path,
    summaries: list[DebugVisSummaryRow],
    *,
    debug_vis_count: int,
    debug_vis_seed: int,
    logger: LiveLogger,
) -> DebugVisOutputs:
    return _generate_debug_vis_impl(
        output_root,
        summaries,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        obstacle_dataset_key=OUTPUT_OBSTACLE_KEY,
        logger=logger,
        debug_vis_dirname=DEBUG_VIS_DIRNAME,
        now_iso_fn=now_iso,
        write_json_fn=write_json,
        load_json_fn=load_json,
        prepare_scene_fn=_prepare_debug_scene_for_overlay,
    )


def _scan_existing_outputs(
    tasks: list[StandardizeTask],
    *,
    force_reprocess: bool,
    logger: LiveLogger,
) -> tuple[list[dict[str, Any]], list[StandardizeTask]]:
    summaries: list[dict[str, Any]] = []
    pending_tasks: list[StandardizeTask] = []
    logger.stage(
        "resume_scan",
        "기존 표준화 결과가 있으면 재처리하지 않고 기존 산출물을 summary로 재사용해 장시간 작업을 이어갑니다.",
        total=len(tasks),
    )
    resume_progress = Counter()
    if tasks:
        for index, task in enumerate(tasks, start=1):
            existing = None if force_reprocess else _existing_output_summary(task)
            if existing is not None:
                summaries.append(existing)
                resume_progress["reused"] += 1
            else:
                pending_tasks.append(task)
                resume_progress["pending"] += 1
            logger.progress(index, dict(resume_progress))
        logger.progress(len(tasks), dict(resume_progress), force=True)
    else:
        logger.progress(0, {"pending": 0, "reused": 0}, force=True)
    return summaries, pending_tasks


def _run_pending_standardization(
    pending_tasks: list[StandardizeTask],
    *,
    workers: int,
    logger: LiveLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    logger.stage(
        "parallel_standardize",
        "JSON 파싱, class remap, scene/det 직렬화, 이미지 materialization을 프로세스 풀로 병렬 실행합니다.",
        total=len(pending_tasks),
    )
    completed = 0
    progress_counters = Counter()
    if not pending_tasks:
        logger.progress(0, {"samples": 0, "failed": 0}, force=True)
        return summaries, failures

    chunk_size = parallel_chunk_size(len(pending_tasks), workers)
    max_inflight_chunks = max(1, workers * PARALLEL_INFLIGHT_CHUNKS_PER_WORKER)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        future_map: dict[Any, list[StandardizeTask]] = {}
        submitted = 0
        next_submit_log = PARALLEL_SUBMIT_LOG_INTERVAL
        chunk_iter = iter(iter_task_chunks(pending_tasks, chunk_size))

        def submit_chunks() -> None:
            nonlocal submitted, next_submit_log
            while len(future_map) < max_inflight_chunks:
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    return
                future_map[executor.submit(_worker_chunk_entry, chunk)] = chunk
                submitted += len(chunk)
                if submitted == len(chunk) or submitted == len(pending_tasks) or submitted >= next_submit_log:
                    logger.info(
                        f"stage=parallel_standardize submit_progress={submitted}/{len(pending_tasks)} "
                        f"workers={workers} chunk_size={chunk_size} inflight_chunks={len(future_map)}"
                    )
                    next_submit_log = ((submitted // PARALLEL_SUBMIT_LOG_INTERVAL) + 1) * PARALLEL_SUBMIT_LOG_INTERVAL

        submit_chunks()

        logger.info(
            f"stage=parallel_standardize waiting_for_results submitted={submitted}/{len(pending_tasks)} "
            f"completed={completed} chunk_size={chunk_size} inflight_chunks={len(future_map)} "
            f"heartbeat_interval_s={PARALLEL_WAIT_HEARTBEAT_SECONDS:.0f}"
        )

        while future_map:
            done, _ = wait(
                set(future_map),
                timeout=PARALLEL_WAIT_HEARTBEAT_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                logger.info(
                    f"stage=parallel_standardize heartbeat completed={completed}/{len(pending_tasks)} "
                    f"submitted={submitted}/{len(pending_tasks)} inflight_chunks={len(future_map)} workers={workers}"
                )
                continue

            for future in done:
                chunk = future_map.pop(future)
                try:
                    results = future.result()
                except Exception as exc:
                    logger.info(
                        f"stage=parallel_standardize chunk_error size={len(chunk)} error={type(exc).__name__}: {exc}"
                    )
                    for task in chunk:
                        failures.append(
                            {
                                "dataset_kind": task.dataset_kind,
                                "dataset_key": task.output_dataset_key,
                                "split": task.pair.split,
                                "raw_id": task.pair.relative_id,
                                "image_path": str(task.pair.image_path),
                                "label_path": str(task.pair.label_path),
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        completed += 1
                        progress_counters["failed"] += 1
                    logger.progress(completed, dict(progress_counters), force=True)
                    submit_chunks()
                    continue

                for result in results:
                    failure = result.get("failure")
                    if failure is not None:
                        failures.append(failure)
                        logger.info(
                            f"stage=parallel_standardize error dataset={failure['dataset_kind']} "
                            f"sample={failure['raw_id']} error={failure['error_type']}: {failure['error_message']}"
                        )
                        completed += 1
                        progress_counters["failed"] += 1
                        logger.progress(completed, dict(progress_counters), force=True)
                        continue

                    summary = result["summary"]
                    summaries.append(summary)
                    completed += 1
                    progress_counters["samples"] = completed
                    progress_counters["detections"] += summary["det_count"]
                    progress_counters["lanes"] += summary["lane_count"]
                    progress_counters["stop_lines"] += summary["stop_line_count"]
                    progress_counters["crosswalks"] += summary["crosswalk_count"]
                    progress_counters["tl_valid"] += summary["tl_attr_valid_count"]
                    progress_counters["held"] += sum(summary["held_reason_counts"].values())
                    logger.progress(completed, dict(progress_counters))
                submit_chunks()
    logger.progress(completed, dict(progress_counters), force=True)
    return summaries, failures


def _write_standardization_outputs(
    *,
    lane_root: Path,
    obstacle_root: Path,
    traffic_root: Path,
    output_root: Path,
    workers: int,
    max_samples_per_dataset: int | None,
    debug_vis_count: int,
    debug_vis_seed: int,
    source_inventory: dict[str, Any],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    logger: LiveLogger,
) -> dict[str, Path]:
    logger.stage(
        "report_write",
        "클래스 맵, 변환 리포트, source inventory를 남겨 다음 모델 설계와 데이터 검증의 입력으로 사용합니다.",
        total=8,
    )
    conversion_report = _aggregate_results(
        lane_root=lane_root,
        traffic_root=traffic_root,
        obstacle_root=obstacle_root,
        output_root=output_root,
        workers=workers,
        max_samples_per_dataset=max_samples_per_dataset,
        debug_vis_count=debug_vis_count,
        source_inventory=source_inventory,
        summaries=summaries,
        failures=failures,
        pipeline_version=PIPELINE_VERSION,
        scene_version=SCENE_VERSION,
        source_root_for_dataset=_source_root_for_dataset,
    )

    meta_root = output_root / "meta"
    conversion_json = meta_root / "conversion_report.json"
    conversion_md = meta_root / "conversion_report.md"
    inventory_json = meta_root / "source_inventory.json"
    inventory_md = meta_root / "source_inventory.md"
    det_map_yaml_path = meta_root / "class_map_det.yaml"
    scene_map_yaml = meta_root / "class_map_scene.yaml"
    failure_json = meta_root / "failure_manifest.json"
    failure_md = meta_root / "failure_manifest.md"

    write_json(conversion_json, conversion_report)
    logger.progress(1, {"files_written": 1}, force=True)
    write_text(conversion_md, _conversion_report_markdown(conversion_report))
    logger.progress(2, {"files_written": 2}, force=True)
    write_json(inventory_json, source_inventory)
    logger.progress(3, {"files_written": 3}, force=True)
    write_text(inventory_md, _source_inventory_markdown(source_inventory))
    logger.progress(4, {"files_written": 4}, force=True)
    write_text(det_map_yaml_path, _det_class_map_yaml())
    logger.progress(5, {"files_written": 5}, force=True)
    write_text(scene_map_yaml, _scene_class_map_yaml())
    logger.progress(6, {"files_written": 6}, force=True)
    failure_manifest = {
        "version": PIPELINE_VERSION,
        "generated_at": now_iso(),
        "failure_count": len(failures),
        "items": failures,
    }
    write_json(failure_json, failure_manifest)
    logger.progress(7, {"files_written": 7}, force=True)
    write_text(failure_md, _failure_manifest_markdown(failure_manifest))
    logger.progress(8, {"files_written": 8}, force=True)

    debug_vis_outputs = _generate_debug_vis(
        output_root,
        summaries,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        logger=logger,
    )

    logger.stage(
        "qa_write",
        "resume/실패/debug-vis 선택 결과를 묶어 full-dataset 전처리 전 QA summary를 남깁니다.",
        total=2,
    )
    debug_vis_index = load_json(debug_vis_outputs["debug_vis_index"])
    qa_json = meta_root / "qa_summary.json"
    qa_md = meta_root / "qa_summary.md"
    qa_summary = _qa_summary(conversion_report, debug_vis_index, failure_manifest)
    write_json(qa_json, qa_summary)
    logger.progress(1, {"files_written": 1}, force=True)
    write_text(qa_md, _qa_summary_markdown(qa_summary))
    logger.progress(2, {"files_written": 2}, force=True)

    return {
        "output_root": output_root,
        "conversion_json": conversion_json,
        "conversion_md": conversion_md,
        "inventory_json": inventory_json,
        "inventory_md": inventory_md,
        "det_map_yaml": det_map_yaml_path,
        "scene_map_yaml": scene_map_yaml,
        "failure_json": failure_json,
        "failure_md": failure_md,
        "qa_json": qa_json,
        "qa_md": qa_md,
        "debug_vis_dir": debug_vis_outputs["debug_vis_dir"],
        "debug_vis_index": debug_vis_outputs["debug_vis_index"],
    }


def run_standardization(
    *,
    lane_root: Path = DEFAULT_LANE_ROOT,
    obstacle_root: Path = DEFAULT_OBSTACLE_ROOT,
    traffic_root: Path = DEFAULT_TRAFFIC_ROOT,
    docs_root: Path = DEFAULT_DOCS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int | None = None,
    max_samples_per_dataset: int | None = None,
    debug_vis_count: int = DEFAULT_DEBUG_VIS_COUNT,
    debug_vis_seed: int = DEFAULT_DEBUG_VIS_SEED,
    write_dataset_readmes: bool = True,
    force_reprocess: bool = False,
    fail_on_error: bool = False,
    log_stream: TextIO | None = None,
) -> dict[str, Path]:
    lane_root = lane_root.resolve()
    obstacle_root = obstacle_root.resolve()
    traffic_root = traffic_root.resolve()
    docs_root = docs_root.resolve()
    output_root = output_root.resolve()
    workers = workers or default_workers()

    logger = LiveLogger(log_stream)
    logger.info(f"pv26_aihub_standardize version={PIPELINE_VERSION}")
    logger.info(f"lane_root={lane_root}")
    logger.info(f"obstacle_root={obstacle_root}")
    logger.info(f"traffic_root={traffic_root}")
    logger.info(f"output_root={output_root}")
    logger.info(
        f"workers={workers} max_samples_per_dataset={max_samples_per_dataset} debug_vis_count={debug_vis_count} "
        f"force_reprocess={force_reprocess} fail_on_error={fail_on_error}"
    )

    if not lane_root.is_dir():
        raise FileNotFoundError(f"lane root does not exist: {lane_root}")
    if not obstacle_root.is_dir():
        raise FileNotFoundError(f"obstacle root does not exist: {obstacle_root}")
    if not traffic_root.is_dir():
        raise FileNotFoundError(f"traffic root does not exist: {traffic_root}")
    if not docs_root.is_dir():
        raise FileNotFoundError(f"docs root does not exist: {docs_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / CACHE_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)

    readme_paths = {"lane_readme": "", "traffic_readme": "", "obstacle_readme": ""}
    if write_dataset_readmes:
        readme_paths = _write_source_readmes(
            lane_root,
            traffic_root,
            obstacle_root,
            docs_root,
            logger,
            documented_stats=DOCUMENTED_STATS,
        )

    logger.stage(
        "source_inventory",
        "원본 데이터셋의 현재 추출 상태와 문서 참조 상태를 메타데이터로 남겨 나중에 재현성을 보장합니다.",
        total=1,
    )
    source_inventory = _build_source_inventory(
        lane_root,
        traffic_root,
        obstacle_root,
        docs_root,
        readme_paths,
        pipeline_version=PIPELINE_VERSION,
        documented_stats=DOCUMENTED_STATS,
    )
    logger.progress(1, {"docs": len(source_inventory["docs"]), "datasets": len(source_inventory["datasets"])}, force=True)

    working_lane_root = _extract_archives_if_needed(lane_root, cache_root, logger)
    working_obstacle_root = _extract_archives_if_needed(obstacle_root, cache_root, logger)
    working_traffic_root = _extract_archives_if_needed(traffic_root, cache_root, logger)

    logger.stage(
        "pair_discovery",
        "AIHUB 디렉터리 구조가 split/subfolder/archive 혼합이라 실제 image-label 짝을 먼저 확정해야 합니다.",
        total=3,
    )
    lane_discovery = _discover_pairs(DISCOVERY_LANE_KEY, working_lane_root)
    logger.progress(1, {"lane_pairs": len(lane_discovery.pairs)}, force=True)
    obstacle_discovery = _discover_pairs(DISCOVERY_OBSTACLE_KEY, working_obstacle_root)
    logger.progress(
        2,
        {"lane_pairs": len(lane_discovery.pairs), "obstacle_pairs": len(obstacle_discovery.pairs)},
        force=True,
    )
    traffic_discovery = _discover_pairs(DISCOVERY_TRAFFIC_KEY, working_traffic_root)
    logger.progress(
        3,
        {
            "lane_pairs": len(lane_discovery.pairs),
            "obstacle_pairs": len(obstacle_discovery.pairs),
            "traffic_pairs": len(traffic_discovery.pairs),
        },
        force=True,
    )

    lane_pairs = sorted(lane_discovery.pairs, key=lambda item: (item.split, item.relative_id))
    obstacle_pairs = sorted(obstacle_discovery.pairs, key=lambda item: (item.split, item.relative_id))
    traffic_pairs = sorted(traffic_discovery.pairs, key=lambda item: (item.split, item.relative_id))
    if max_samples_per_dataset is not None:
        lane_pairs = lane_pairs[:max_samples_per_dataset]
        obstacle_pairs = obstacle_pairs[:max_samples_per_dataset]
        traffic_pairs = traffic_pairs[:max_samples_per_dataset]

    tasks = [
        StandardizeTask("lane", OUTPUT_LANE_KEY, pair, str(output_root))
        for pair in lane_pairs
    ] + [
        StandardizeTask("obstacle", OUTPUT_OBSTACLE_KEY, pair, str(output_root))
        for pair in obstacle_pairs
    ] + [
        StandardizeTask("traffic", OUTPUT_TRAFFIC_KEY, pair, str(output_root))
        for pair in traffic_pairs
    ]

    summaries, pending_tasks = _scan_existing_outputs(
        tasks,
        force_reprocess=force_reprocess,
        logger=logger,
    )
    fresh_summaries, failures = _run_pending_standardization(
        pending_tasks,
        workers=workers,
        logger=logger,
    )
    summaries.extend(fresh_summaries)
    outputs = _write_standardization_outputs(
        lane_root=lane_root,
        obstacle_root=obstacle_root,
        traffic_root=traffic_root,
        output_root=output_root,
        workers=workers,
        max_samples_per_dataset=max_samples_per_dataset,
        debug_vis_count=debug_vis_count,
        debug_vis_seed=debug_vis_seed,
        source_inventory=source_inventory,
        summaries=summaries,
        failures=failures,
        logger=logger,
    )

    logger.info("standardization complete")
    if failures and fail_on_error:
        raise RuntimeError(f"AIHUB standardization completed with failures: {len(failures)}")
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hardcoded AIHUB standardization pipeline.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional process pool size. Defaults to CPU count minus one.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional per-dataset sample limit for faster verification runs. Source inventory and README generation still scan the full source tree.",
    )
    parser.add_argument(
        "--debug-vis-count",
        type=int,
        default=DEFAULT_DEBUG_VIS_COUNT,
        help="Random QA overlay count written under meta/debug_vis after standardization. Set 0 to disable.",
    )
    parser.add_argument(
        "--skip-readmes",
        action="store_true",
        help="Skip dataset-local README generation under the AIHUB source roots.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore existing standardized outputs and rebuild every discovered sample from source.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero after writing failure manifests if any sample conversion fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    outputs = run_standardization(
        workers=args.workers,
        max_samples_per_dataset=args.max_samples_per_dataset,
        debug_vis_count=args.debug_vis_count,
        write_dataset_readmes=not args.skip_readmes,
        force_reprocess=args.force_reprocess,
        fail_on_error=args.fail_on_error,
    )
    print(f"output_root={outputs['output_root']}")
    print(f"conversion_json={outputs['conversion_json']}")
    print(f"inventory_json={outputs['inventory_json']}")
    print(f"failure_json={outputs['failure_json']}")
    print(f"qa_json={outputs['qa_json']}")
    print(f"debug_vis_dir={outputs['debug_vis_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
