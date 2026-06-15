from __future__ import annotations

from importlib import import_module
from typing import Any

_DRY_RUN_EXPORTS = (
    "BLOCKED_STATUS",
    "CANDIDATE_REASON_INVALID_SPLIT",
    "CANDIDATE_REASON_LABEL_PARSE_FAILURE",
    "CANDIDATE_REASON_MISSING_IMAGE_SIZE",
    "CANDIDATE_REASON_MISSING_SEMANTIC_LABEL",
    "CANDIDATE_REASON_SAMPLE_ID_MISMATCH",
    "DATASET_KEY",
    "EtriCandidateError",
    "EtriDryRunNotReadyError",
    "EtriDryRunResult",
    "EtriDryRunSample",
    "RAW_SCAN_REASON_LIDAR",
    "RAW_SCAN_REASON_MONO_CAMERA",
    "RAW_SCAN_REASON_RIGHT_IMG",
    "READY_STATUS",
    "RELEASE_BLOCKER_ZERO_SAMPLES",
    "VALID_SPLITS",
    "build_dry_run_manifest",
    "build_dry_run_sample",
    "build_ready_dry_run_manifest",
    "dry_run_release_blockers",
    "is_dry_run_ready",
    "is_kcity_leftimg_candidate",
    "is_semantic_label_candidate",
    "raw_scan_ignored_reason",
    "require_dry_run_ready",
    "sample_id_from_path",
    "scan_dry_run",
    "write_dry_run_manifest",
    "write_ready_dry_run_manifest",
)

__all__ = list(_DRY_RUN_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _DRY_RUN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.dry_run")
    for export in _DRY_RUN_EXPORTS:
        globals()[export] = getattr(module, export)
    return globals()[name]
