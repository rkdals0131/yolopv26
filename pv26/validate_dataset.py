from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from .manifest import read_manifest_csv, validate_manifest_row_basic
from .masks import IGNORE_VALUE, validate_binary_mask_u8, validate_semantic_id_u8


@dataclass
class ValidationSummary:
    num_rows: int
    errors: List[str]
    warnings: List[str]


def _load_u8_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"mask not single-channel: {path}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _load_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def load_class_map_allowed_seg_ids(class_map_yaml_path: Path) -> Set[int]:
    """
    Minimal YAML parsing for the generated class_map.yaml format.
    Extracts seg IDs under:
      segmentation:
        id_to_name:
          <id>: <name>
    """
    allowed: Set[int] = set()
    in_seg = False
    in_id_to_name = False
    for line in class_map_yaml_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s == "segmentation:":
            in_seg = True
            in_id_to_name = False
            continue
        if in_seg and s == "id_to_name:":
            in_id_to_name = True
            continue
        if in_seg and s == "detection:":
            break
        if in_seg and in_id_to_name:
            # Expect: "<int>: <name>"
            if ":" not in s:
                continue
            left = s.split(":", 1)[0].strip()
            try:
                allowed.add(int(left))
            except ValueError:
                continue
    return allowed


def validate_pv26_dataset(out_root: Path) -> ValidationSummary:
    errors: List[str] = []
    warnings: List[str] = []

    meta = out_root / "meta"
    manifest_path = meta / "split_manifest.csv"
    class_map_path = meta / "class_map.yaml"
    if not manifest_path.exists():
        return ValidationSummary(num_rows=0, errors=[f"missing_manifest:{manifest_path}"], warnings=[])
    if not class_map_path.exists():
        errors.append(f"missing_class_map:{class_map_path}")
        allowed_seg_ids: Set[int] = set()
    else:
        allowed_seg_ids = load_class_map_allowed_seg_ids(class_map_path)

    rows = read_manifest_csv(manifest_path)
    for i, row in enumerate(rows):
        prefix = f"row[{i}] sample_id={row.get('sample_id','')}: "
        basic_errs = validate_manifest_row_basic(row)
        for e in basic_errs:
            errors.append(prefix + e)
        if basic_errs:
            continue

        split = row["split"]
        # Resolve and check files.
        def _p(rel: str) -> Path:
            return out_root / rel

        img_p = _p(row["image_relpath"])
        det_p = _p(row["det_relpath"])
        da_p = _p(row["da_relpath"])
        rm_lane_p = _p(row["rm_lane_marker_relpath"])
        rm_road_p = _p(row["rm_road_marker_non_lane_relpath"])
        rm_stop_p = _p(row["rm_stop_line_relpath"])
        sem_rel = row.get("semantic_relpath", "")
        sem_p = _p(sem_rel) if sem_rel else None

        for p in [img_p, det_p, da_p, rm_lane_p, rm_road_p, rm_stop_p]:
            if not p.exists():
                errors.append(prefix + f"missing_file:{p}")

        if sem_p is not None and row["has_semantic_id"] == "1" and not sem_p.exists():
            errors.append(prefix + f"missing_semantic_file:{sem_p}")

        # Size consistency.
        try:
            w, h = _load_image_size(img_p)
        except Exception as ex:
            errors.append(prefix + f"image_open_failed:{img_p} err={ex}")
            continue

        if str(w) != row["width"] or str(h) != row["height"]:
            errors.append(prefix + f"manifest_size_mismatch manifest=({row['width']},{row['height']}) image=({w},{h})")

        # Mask domain + size checks.
        def _check_mask(mask_path: Path, *, allow_ignore: bool, name: str) -> Optional[np.ndarray]:
            try:
                arr = _load_u8_mask(mask_path)
            except Exception as ex:
                errors.append(prefix + f"mask_open_failed:{name}:{mask_path} err={ex}")
                return None
            if arr.shape != (h, w):
                errors.append(prefix + f"mask_size_mismatch:{name} mask={arr.shape} image={(h,w)}")
                return None
            try:
                validate_binary_mask_u8(arr, allow_ignore=allow_ignore, name=name)
            except Exception as ex:
                errors.append(prefix + f"mask_invalid:{name} err={ex}")
            return arr

        da = _check_mask(da_p, allow_ignore=True, name="da")
        rm_lane = _check_mask(rm_lane_p, allow_ignore=True, name="rm_lane_marker")
        rm_road = _check_mask(rm_road_p, allow_ignore=True, name="rm_road_marker_non_lane")
        rm_stop = _check_mask(rm_stop_p, allow_ignore=True, name="rm_stop_line")

        # Partial-label contract: has_*==0 => all-255
        def _expect_all_ignore(arr: Optional[np.ndarray], flag: str, name: str) -> None:
            if arr is None:
                return
            if row[flag] == "0":
                if np.any(arr != IGNORE_VALUE):
                    errors.append(prefix + f"partial_label_violation:{name} flag={flag}=0 but mask has non-255 values")

        _expect_all_ignore(da, "has_da", "da")
        _expect_all_ignore(rm_lane, "has_rm_lane_marker", "rm_lane_marker")
        _expect_all_ignore(rm_road, "has_rm_road_marker_non_lane", "rm_road_marker_non_lane")
        _expect_all_ignore(rm_stop, "has_rm_stop_line", "rm_stop_line")

        # Semantic validation.
        if row["has_semantic_id"] == "1":
            if sem_p is None:
                errors.append(prefix + "semantic_relpath_missing_but_has_semantic_id=1")
            else:
                try:
                    sem = _load_u8_mask(sem_p)
                except Exception as ex:
                    errors.append(prefix + f"semantic_open_failed:{sem_p} err={ex}")
                else:
                    if sem.shape != (h, w):
                        errors.append(prefix + f"semantic_size_mismatch:{sem.shape} image={(h,w)}")
                    if allowed_seg_ids:
                        try:
                            validate_semantic_id_u8(sem, allowed_ids=allowed_seg_ids, name="semantic_id")
                        except Exception as ex:
                            errors.append(prefix + f"semantic_invalid err={ex}")

        # Det scope consistency (minimal): det_label_scope=none => det file must be empty
        if row["det_label_scope"] == "none":
            try:
                txt = det_p.read_text(encoding="utf-8").strip()
            except Exception as ex:
                errors.append(prefix + f"det_read_failed:{det_p} err={ex}")
            else:
                if txt:
                    errors.append(prefix + "det_scope_none_but_det_file_not_empty")

    return ValidationSummary(num_rows=len(rows), errors=errors, warnings=warnings)

