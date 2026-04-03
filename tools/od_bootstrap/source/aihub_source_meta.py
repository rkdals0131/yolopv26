from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.io import now_iso, write_text
from common.pv26_schema import (
    AIHUB_LANE_DATASET_KEY,
    AIHUB_OBSTACLE_DATASET_KEY,
    AIHUB_TRAFFIC_DATASET_KEY,
)

from .shared_summary import counter_to_dict

README_TREE_DEPTH = 3
MAX_TREE_LINES = 96


def _inventory_image_json_archives(dataset_root: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "root": str(dataset_root),
        "splits": {},
    }
    for split in ("Training", "Validation", "Test"):
        split_root = dataset_root / split
        if not split_root.exists():
            inventory["splits"][split] = {"present": False}
            continue
        images = sum(
            1
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        json_files = sum(1 for path in split_root.rglob("*.json") if path.is_file())
        archives = Counter(
            path.suffix.lower()
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".zip", ".tar"}
        )
        inventory["splits"][split] = {
            "present": True,
            "images": images,
            "json_files": json_files,
            "archives": counter_to_dict(archives),
        }
    return inventory


def _inventory_lane_root(dataset_root: Path) -> dict[str, Any]:
    return _inventory_image_json_archives(dataset_root)


def _inventory_obstacle_root(dataset_root: Path) -> dict[str, Any]:
    return _inventory_image_json_archives(dataset_root)


def _inventory_traffic_root(dataset_root: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "root": str(dataset_root),
        "splits": {},
    }
    for split in ("Training", "Validation", "Test"):
        split_root = dataset_root / split
        if not split_root.exists():
            inventory["splits"][split] = {"present": False}
            continue
        raw_images = 0
        crop_images = 0
        for path in split_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if "표지판코드분류crop데이터" in str(path):
                crop_images += 1
            else:
                raw_images += 1
        json_files = sum(1 for path in split_root.rglob("*.json") if path.is_file())
        archives = Counter(
            path.suffix.lower()
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".zip", ".tar"}
        )
        inventory["splits"][split] = {
            "present": True,
            "raw_images": raw_images,
            "crop_images": crop_images,
            "json_files": json_files,
            "archives": counter_to_dict(archives),
        }
    return inventory


def _docs_inventory(docs_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "file_name": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(docs_root.glob("*.pdf"))
        if path.is_file()
    ]


def _source_pdf_inventory(dataset_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "file_name": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(dataset_root.glob("*.pdf"))
        if path.is_file()
    ]


def _tree_markdown(root: Path, *, max_depth: int = README_TREE_DEPTH, max_lines: int = MAX_TREE_LINES) -> str:
    lines = [f"{root.name}/"]

    def walk(current: Path, depth: int) -> None:
        if len(lines) >= max_lines or depth >= max_depth:
            return
        children = sorted(current.iterdir(), key=lambda item: (item.is_file(), item.name))
        for child in children:
            if len(lines) >= max_lines:
                break
            indent = "  " * depth
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{indent}- {child.name}{suffix}")
            if child.is_dir():
                walk(child, depth + 1)

    walk(root, 1)
    if len(lines) >= max_lines:
        lines.append("  - ...")
    return "\n".join(lines)


def _lane_readme(dataset_root: Path, docs_root: Path, inventory: dict[str, Any], documented_stats: dict[str, Any]) -> str:
    lines = [
        "# 차선-횡단보도 인지 영상(수도권)",
        "",
        "AIHUB 원본 데이터셋의 로컬 구조, 문서 기준 통계, 실제 JSON 스키마를 정리한 원본용 README다.",
        "",
        "## 문서 기준 통계",
        "",
        f"- 문서 기준 수도권 json 수량: `{documented_stats['json_count_seoul']:,}`",
        f"- 문서 기준 수도권 차선 객체 수: `{documented_stats['lane_objects_seoul']:,}`",
        f"- 문서 기준 수도권 횡단보도 객체 수: `{documented_stats['crosswalk_objects_seoul']:,}`",
        f"- 문서 기준 수도권 정지선 객체 수: `{documented_stats['stop_line_objects_seoul']:,}`",
        f"- 문서 기준 차선 색상 분포: `white {documented_stats['white_lane_objects_seoul']:,} / yellow {documented_stats['yellow_lane_objects_seoul']:,} / blue {documented_stats['blue_lane_objects_seoul']:,}`",
        f"- 문서 기준 차선 타입 분포: `solid {documented_stats['solid_lane_objects_seoul']:,} / dotted {documented_stats['dotted_lane_objects_seoul']:,}`",
        "",
        "## 현재 로컬 보유 상태",
        "",
        "| Split | Present | Images | JSON | Archives |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for split, split_info in inventory["splits"].items():
        if not split_info["present"]:
            lines.append(f"| {split} | no | 0 | 0 | - |")
            continue
        archives = ", ".join(f"{key}:{value}" for key, value in split_info["archives"].items()) or "-"
        lines.append(
            f"| {split} | yes | {split_info['images']:,} | {split_info['json_files']:,} | {archives} |"
        )
    lines.extend(
        [
            "",
            "## 로컬 디렉터리 구조",
            "",
            "```text",
            _tree_markdown(dataset_root),
            "```",
            "",
            "## 원본 어노테이션 스키마 요약",
            "",
            "- 이미지 메타: `image.file_name`, `image.image_size`",
            "- 라벨 객체 리스트: `annotations[]`",
            "- 차선: `class=traffic_lane`, `category=polyline`, `attributes=[lane_color, lane_type]`, `data=[{x,y}, ...]`",
            "- 정지선: `class=stop_line`, `category=polyline`, `data=[{x,y}, ...]`",
            "- 횡단보도: `class=crosswalk`, `category=polygon`, `data=[{x,y}, ...]`",
            "",
            "## 표준화 관점 메모",
            "",
            "- 차선은 `white/yellow/blue`와 `solid/dotted` 속성을 모두 가진다.",
            "- 정지선과 횡단보도는 차선과 별도 객체로 존재한다.",
            "- 현재 표준화 스크립트는 원본 geometry를 유지한 scene JSON을 만들고, 학습용 fixed-length target은 후단 encoder가 만든다.",
            "",
            "## 참조 문서",
            "",
        ]
    )
    for item in _docs_inventory(docs_root):
        lines.append(f"- `{item['file_name']}`")
    return "\n".join(lines) + "\n"


def _traffic_readme(dataset_root: Path, docs_root: Path, inventory: dict[str, Any], documented_stats: dict[str, Any]) -> str:
    lines = [
        "# 신호등-도로표지판 인지 영상(수도권)",
        "",
        "AIHUB 원본 데이터셋의 로컬 구조, 문서 기준 통계, 실제 JSON 스키마를 정리한 원본용 README다.",
        "",
        "## 문서 기준 통계",
        "",
        f"- 문서 기준 수도권 json 수량: `{documented_stats['json_count_seoul']:,}`",
        f"- 문서 기준 수도권 신호등 객체 수: `{documented_stats['traffic_light_objects_seoul']:,}`",
        f"- 문서 기준 수도권 표지판 객체 수: `{documented_stats['traffic_sign_objects_seoul']:,}`",
        f"- 문서 기준 수도권 TL 상태 분포: `red {documented_stats['traffic_light_red_seoul']:,} / yellow {documented_stats['traffic_light_yellow_seoul']:,} / green {documented_stats['traffic_light_green_seoul']:,} / left_arrow {documented_stats['traffic_light_left_arrow_seoul']:,}`",
        f"- 문서 기준 수도권 표지판 세부 분포: `instruction {documented_stats['traffic_sign_instruction_seoul']:,} / caution {documented_stats['traffic_sign_caution_seoul']:,} / restriction {documented_stats['traffic_sign_restriction_seoul']:,}`",
        "",
        "## 현재 로컬 보유 상태",
        "",
        "| Split | Present | Raw Images | Crop Images | JSON | Archives |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for split, split_info in inventory["splits"].items():
        if not split_info["present"]:
            lines.append(f"| {split} | no | 0 | 0 | 0 | - |")
            continue
        archives = ", ".join(f"{key}:{value}" for key, value in split_info["archives"].items()) or "-"
        lines.append(
            f"| {split} | yes | {split_info['raw_images']:,} | {split_info['crop_images']:,} | {split_info['json_files']:,} | {archives} |"
        )
    lines.extend(
        [
            "",
            "## 로컬 디렉터리 구조",
            "",
            "```text",
            _tree_markdown(dataset_root),
            "```",
            "",
            "## 원본 어노테이션 스키마 요약",
            "",
            "- 이미지 메타: `image.filename`, `image.imsize`",
            "- 라벨 객체 리스트: `annotation[]`",
            "- 신호등: `class=traffic_light`, `box=[x1,y1,x2,y2]`, `attribute=[{red, yellow, green, left_arrow, others_arrow, x_light}]`, `type`, `direction`, `light_count`",
            "- 표지판: `class=traffic_sign`, `box=[x1,y1,x2,y2]`, `shape`, `color`, `kind`, `type`, `text`",
            "- 보조 정보: `class=traffic_information`, `type`",
            "",
            "## 표준화 관점 메모",
            "",
            "- 원천 주행 이미지와 `표지판코드분류crop데이터*`가 같은 split 아래 공존한다. detector 표준화는 원천 주행 이미지와 JSON만 사용하고 crop 분류셋은 별도 auxiliary 자산으로 본다.",
            "- detector class는 `traffic_light` generic bbox와 `sign`으로 정규화한다.",
            "- TL 상태는 후단 crop 분류기가 아니라 같은 박스에 붙는 `red/yellow/green/arrow` 4-bit supervision으로 유지한다.",
            "- `left_arrow`와 `others_arrow`는 `arrow=1`로 접는다. `off`는 네 bit가 모두 0인 상태로 유지한다.",
            "- `x_light`, non-car signal, base color 다중 on 조합은 TL attribute 학습 마스크로 빠진다.",
            "",
            "## 참조 문서",
            "",
        ]
    )
    for item in _docs_inventory(docs_root):
        lines.append(f"- `{item['file_name']}`")
    return "\n".join(lines) + "\n"


def _obstacle_readme(dataset_root: Path, inventory: dict[str, Any]) -> str:
    local_docs = _source_pdf_inventory(dataset_root)
    lines = [
        "# 도로장애물·표면 인지 영상(수도권)",
        "",
        "AIHUB 원본 데이터셋의 로컬 구조와 detector-only 표준화 규칙을 정리한 원본용 README다.",
        "",
        "## 현재 로컬 보유 상태",
        "",
        "| Split | Present | Images | JSON | Archives |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for split, split_info in inventory["splits"].items():
        if not split_info["present"]:
            lines.append(f"| {split} | no | 0 | 0 | - |")
            continue
        archives = ", ".join(f"{key}:{value}" for key, value in split_info["archives"].items()) or "-"
        lines.append(
            f"| {split} | yes | {split_info['images']:,} | {split_info['json_files']:,} | {archives} |"
        )
    lines.extend(
        [
            "",
            "## 로컬 디렉터리 구조",
            "",
            "```text",
            _tree_markdown(dataset_root),
            "```",
            "",
            "## 원본 어노테이션 스키마 요약",
            "",
            "- 이미지 메타: `images.file_name`, `images.width`, `images.height`",
            "- 라벨 객체 리스트: `annotations[]`",
            "- 카테고리 정의: `categories[{id, name}]`",
            "- 박스 포맷: `bbox=[x, y, w, h]`",
            "",
            "## 표준화 관점 메모",
            "",
            "- detector class는 `traffic_cone`와 `obstacle`만 유지한다.",
            "- remap: `Traffic cone -> traffic_cone`",
            "- remap: `Animals(Dolls) / Garbage bag & sacks / Construction signs & Parking prohibited board / Box / Stones on road -> obstacle`",
            "- exclusion: `Person / Manhole / Pothole on road / Filled pothole`는 detector canonical output에서 제외한다.",
            "- lane, stop-line, crosswalk, traffic-light attribute supervision은 이 source에서 제공하지 않는다.",
            "",
            "## 참조 문서",
            "",
        ]
    )
    for item in local_docs:
        lines.append(f"- `{item['file_name']}`")
    return "\n".join(lines) + "\n"


def write_source_readmes(
    lane_root: Path,
    traffic_root: Path,
    obstacle_root: Path,
    docs_root: Path,
    logger: Any,
    *,
    documented_stats: dict[str, dict[str, Any]],
) -> dict[str, str]:
    logger.stage(
        "source_readme",
        "원본 데이터셋을 다시 열지 않고도 구조와 스키마를 확인할 수 있게 dataset-local README를 생성합니다.",
        total=3,
    )
    lane_inventory = _inventory_lane_root(lane_root)
    traffic_inventory = _inventory_traffic_root(traffic_root)
    obstacle_inventory = _inventory_obstacle_root(obstacle_root)
    lane_readme_path = lane_root / "README.md"
    traffic_readme_path = traffic_root / "README.md"
    obstacle_readme_path = obstacle_root / "README.md"
    write_text(lane_readme_path, _lane_readme(lane_root, docs_root, lane_inventory, documented_stats[AIHUB_LANE_DATASET_KEY]))
    logger.progress(1, {"written": 1}, force=True)
    write_text(
        traffic_readme_path,
        _traffic_readme(traffic_root, docs_root, traffic_inventory, documented_stats[AIHUB_TRAFFIC_DATASET_KEY]),
    )
    logger.progress(2, {"written": 2}, force=True)
    write_text(obstacle_readme_path, _obstacle_readme(obstacle_root, obstacle_inventory))
    logger.progress(3, {"written": 3}, force=True)
    return {
        "lane_readme": str(lane_readme_path),
        "traffic_readme": str(traffic_readme_path),
        "obstacle_readme": str(obstacle_readme_path),
    }


def build_source_inventory(
    lane_root: Path,
    traffic_root: Path,
    obstacle_root: Path,
    docs_root: Path,
    readme_paths: dict[str, str],
    *,
    pipeline_version: str,
    documented_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": pipeline_version,
        "generated_at": now_iso(),
        "docs": _docs_inventory(docs_root),
        "datasets": [
            {
                "dataset_key": AIHUB_LANE_DATASET_KEY,
                "documented_stats": documented_stats[AIHUB_LANE_DATASET_KEY],
                "local_inventory": _inventory_lane_root(lane_root),
                "readme_path": readme_paths["lane_readme"],
            },
            {
                "dataset_key": AIHUB_TRAFFIC_DATASET_KEY,
                "documented_stats": documented_stats[AIHUB_TRAFFIC_DATASET_KEY],
                "local_inventory": _inventory_traffic_root(traffic_root),
                "readme_path": readme_paths["traffic_readme"],
            },
            {
                "dataset_key": AIHUB_OBSTACLE_DATASET_KEY,
                "documented_stats": {
                    "doc_references": [item["file_name"] for item in _source_pdf_inventory(obstacle_root)],
                },
                "local_inventory": _inventory_obstacle_root(obstacle_root),
                "readme_path": readme_paths["obstacle_readme"],
            },
        ],
    }


def source_root_for_dataset(
    dataset_key: str,
    *,
    lane_root: Path,
    traffic_root: Path,
    obstacle_root: Path,
) -> Path:
    if dataset_key == AIHUB_LANE_DATASET_KEY:
        return lane_root
    if dataset_key == AIHUB_TRAFFIC_DATASET_KEY:
        return traffic_root
    if dataset_key == AIHUB_OBSTACLE_DATASET_KEY:
        return obstacle_root
    raise KeyError(f"unsupported AIHUB dataset key: {dataset_key}")


def source_inventory_markdown(source_inventory: dict[str, Any]) -> str:
    lines = [
        "# PV26 AIHUB Source Inventory",
        "",
        f"- Generated: `{source_inventory['generated_at']}`",
        f"- Version: `{source_inventory['version']}`",
        "",
        "## Docs",
        "",
    ]
    for item in source_inventory["docs"]:
        lines.append(f"- `{item['file_name']}` ({item['size_bytes']:,} bytes)")
    for dataset in source_inventory["datasets"]:
        lines.extend(
            [
                "",
                f"## {dataset['dataset_key']}",
                "",
                f"- Root: `{dataset['local_inventory']['root']}`",
                f"- README: `{dataset['readme_path']}`",
                "",
                "| Split | Present | Images/Raw | Crop | JSON | Archives |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for split, split_info in dataset["local_inventory"]["splits"].items():
            if not split_info["present"]:
                lines.append(f"| {split} | no | 0 | 0 | 0 | - |")
                continue
            if dataset["dataset_key"] == AIHUB_TRAFFIC_DATASET_KEY:
                images_or_raw = split_info["raw_images"]
                crop_images = split_info["crop_images"]
            else:
                images_or_raw = split_info["images"]
                crop_images = 0
            archives = ", ".join(f"{key}:{value}" for key, value in split_info["archives"].items()) or "-"
            lines.append(
                f"| {split} | yes | {images_or_raw:,} | {crop_images:,} | {split_info['json_files']:,} | {archives} |"
            )
    return "\n".join(lines) + "\n"
