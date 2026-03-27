from __future__ import annotations

from dataclasses import asdict, dataclass


RUN_MANIFEST_VERSION = "od-bootstrap-run-v1"
JOB_MANIFEST_VERSION = "od-bootstrap-job-v1"
LABEL_ORIGINS = {"raw_source", "bootstrap"}


@dataclass(frozen=True)
class BoxProvenance:
    label_origin: str
    teacher_name: str
    confidence: float
    model_version: str
    run_id: str
    created_at: str

    def __post_init__(self) -> None:
        if self.label_origin not in LABEL_ORIGINS:
            raise ValueError(f"unsupported label_origin: {self.label_origin}")
        if not self.teacher_name:
            raise ValueError("teacher_name must not be empty")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.created_at:
            raise ValueError("created_at must not be empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    scenario_path: str
    execution_mode: str
    dry_run: bool
    run_dir: str
    image_pool_manifest: str
    image_count: int
    teacher_names: tuple[str, ...]
    manifest_version: str = RUN_MANIFEST_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherJobManifest:
    run_id: str
    created_at: str
    teacher_name: str
    base_model: str
    model_version: str
    checkpoint_path: str
    classes: tuple[str, ...]
    image_count: int
    predictions_path: str
    dry_run: bool
    manifest_version: str = JOB_MANIFEST_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
