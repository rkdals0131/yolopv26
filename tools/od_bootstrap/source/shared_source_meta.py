from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from common.pv26_schema import BDD100K_DATASET_KEY

from .raw_common import IMAGE_EXTENSIONS
from .shared_io import now_iso

README_TREE_DEPTH = 3
MAX_TREE_LINES = 96


def _count_files(root: Path, suffixes: set[str]) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def tree_markdown(
    root: Path,
    *,
    max_depth: int = README_TREE_DEPTH,
    max_lines: int = MAX_TREE_LINES,
) -> str:
    lines = [f"{root.name}/"]

    def walk(current: Path, depth: int) -> None:
        if len(lines) >= max_lines or depth >= max_depth:
            return
        for child in sorted(current.iterdir(), key=lambda item: (item.is_file(), item.name)):
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


def build_bdd_inventory(
    bdd_root: Path,
    images_root: Path,
    labels_root: Path,
    *,
    splits: Iterable[str],
    official_split_sizes: dict[str, int],
) -> dict[str, Any]:
    inventory_splits: dict[str, Any] = {}
    for split in splits:
        image_dir = images_root / split
        label_dir = labels_root / split
        inventory_splits[split] = {
            "images_present": image_dir.is_dir(),
            "labels_present": label_dir.is_dir(),
            "images": _count_files(image_dir, IMAGE_EXTENSIONS) if image_dir.is_dir() else 0,
            "json_files": _count_files(label_dir, {".json"}) if label_dir.is_dir() else 0,
            "official_images": official_split_sizes[split],
        }

    return {
        "root": str(bdd_root),
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "extra_assets": {
            "bdd100k_det_20_labels": (bdd_root / "bdd100k_det_20_labels").is_dir(),
            "bdd100k_drivable_maps": (bdd_root / "bdd100k_drivable_maps").is_dir(),
            "bdd100k_seg_maps": (bdd_root / "bdd100k_seg_maps").is_dir(),
            "bdd100k_gh_toolkit": (bdd_root / "bdd100k-gh").is_dir(),
        },
        "splits": inventory_splits,
    }


def bdd_readme(bdd_root: Path, inventory: dict[str, Any]) -> str:
    lines = [
        "# BDD100K",
        "",
        "PV26에서 사용하는 BDD100K 원본 구조와 detection-only 표준화 관점을 정리한 원본용 README다.",
        "",
        "## PV26 사용 범위",
        "",
        "- 사용 목적: `7-class object detection` 중 non-signal class 보강",
        "- 사용 원천: `bdd100k_images_100k/100k/<split>` + `bdd100k_labels/100k/<split>/*.json`",
        "- 비사용 원천: drivable map, segmentation map, det_20 preview asset",
        "- BDD canonical output은 `vehicle / bike / pedestrian`만 detector supervision으로 남긴다.",
        "- `traffic light`, `traffic sign`는 AIHUB signal source가 담당하므로 BDD canonical output에서 제외한다.",
        "- TL 4-bit supervision은 BDD source에서 사용하지 않는다.",
        "",
        "## 공식 split 크기",
        "",
    ]
    for split, item in inventory["splits"].items():
        lines.append(f"- {split}: `{item['official_images']:,}`")
    lines.extend(
        [
            "",
            "## 현재 로컬 보유 상태",
            "",
            "| Split | Images Present | Labels Present | Local Images | Local JSON | Official Images |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split, item in inventory["splits"].items():
        lines.append(
            f"| {split} | {'yes' if item['images_present'] else 'no'} | {'yes' if item['labels_present'] else 'no'} "
            f"| {item['images']:,} | {item['json_files']:,} | {item['official_images']:,} |"
        )
    lines.extend(
        [
            "",
            "## 원본 어노테이션 스키마 요약",
            "",
            "- 파일 단위: 이미지 1장당 JSON 1개",
            "- top-level: `name`, `attributes`, `frames[]`",
            "- detection 객체: `frames[0].objects[]`",
            "- detection box: `box2d = {x1, y1, x2, y2}`",
            "- scene width/height와 YOLO normalization은 실제 image file size probe 기준으로 계산한다.",
            "- 문맥 메타: `attributes.weather`, `attributes.scene`, `attributes.timeofday`",
            "",
            "## PV26 클래스 collapse 규칙",
            "",
            "- `car/truck/bus/train/(other vehicle, van, caravan, trailer alias)` -> `vehicle`",
            "- `bike/motor/(bicycle, motorcycle alias)` -> `bike`",
            "- `person/rider/(other person alias)` -> `pedestrian`",
            "- `traffic light` -> excluded (`AIHUB-owned signal class`)",
            "- `traffic sign` -> excluded (`AIHUB-owned signal class`)",
            "- `lane/*`, `area/*` 등 non-box driving map 계열은 detector 표준화에서 제외",
            "",
            "## 로컬 디렉터리 구조",
            "",
            "```text",
            tree_markdown(bdd_root),
            "```",
            "",
            "## 추가 자산 존재 여부",
            "",
        ]
    )
    for key, value in sorted(inventory["extra_assets"].items()):
        lines.append(f"- `{key}`: {'yes' if value else 'no'}")
    return "\n".join(lines) + "\n"


def build_bdd_source_inventory(
    *,
    pipeline_version: str,
    readme_path: str,
    inventory: dict[str, Any],
    dataset_key: str = BDD100K_DATASET_KEY,
) -> dict[str, Any]:
    return {
        "version": pipeline_version,
        "generated_at": now_iso(),
        "dataset": {
            "dataset_key": dataset_key,
            "readme_path": readme_path,
            "local_inventory": inventory,
        },
    }


def bdd_source_inventory_markdown(source_inventory: dict[str, Any]) -> str:
    dataset = source_inventory["dataset"]
    inventory = dataset["local_inventory"]
    lines = [
        "# PV26 BDD100K Source Inventory",
        "",
        f"- Generated: `{source_inventory['generated_at']}`",
        f"- Version: `{source_inventory['version']}`",
        f"- Dataset key: `{dataset['dataset_key']}`",
        f"- Root: `{inventory['root']}`",
        f"- README: `{dataset['readme_path']}`",
        "",
        "| Split | Images Present | Labels Present | Local Images | Local JSON | Official Images |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for split, item in inventory["splits"].items():
        lines.append(
            f"| {split} | {'yes' if item['images_present'] else 'no'} | {'yes' if item['labels_present'] else 'no'} "
            f"| {item['images']:,} | {item['json_files']:,} | {item['official_images']:,} |"
        )
    lines.extend(["", "## Extra Assets", ""])
    for key, value in sorted(inventory["extra_assets"].items()):
        lines.append(f"- `{key}`: {'yes' if value else 'no'}")
    return "\n".join(lines) + "\n"


__all__ = [
    "MAX_TREE_LINES",
    "README_TREE_DEPTH",
    "bdd_readme",
    "bdd_source_inventory_markdown",
    "build_bdd_inventory",
    "build_bdd_source_inventory",
    "tree_markdown",
]
