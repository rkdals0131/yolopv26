"""PV26 training and evaluation engine.

The public package surface intentionally re-exports only the stable engine
entry points; underscore-prefixed helper modules remain internal.
"""

from .batch import augment_lane_family_metrics, move_batch_to_device, raw_batch_for_metrics
from .evaluator import PV26Evaluator
from .loss import PV26DetAssignmentUnavailable, PV26MultiTaskLoss
from .metrics import PV26MetricConfig, summarize_pv26_metrics
from .postprocess import PV26PostprocessConfig, postprocess_pv26_batch
from .spec import SPEC_VERSION, build_loss_spec, render_loss_spec_markdown
from .trainer import (
    PV26Trainer,
    STAGE_NAMES,
    TIMING_KEYS,
    TENSORBOARD_LOSS_KEYS,
    build_pv26_optimizer,
    build_pv26_scheduler,
    configure_pv26_train_stage,
    run_pv26_tiny_overfit,
)

__all__ = [
    "PV26DetAssignmentUnavailable",
    "PV26Evaluator",
    "PV26MetricConfig",
    "PV26MultiTaskLoss",
    "PV26PostprocessConfig",
    "PV26Trainer",
    "SPEC_VERSION",
    "STAGE_NAMES",
    "TIMING_KEYS",
    "TENSORBOARD_LOSS_KEYS",
    "augment_lane_family_metrics",
    "build_loss_spec",
    "build_pv26_optimizer",
    "build_pv26_scheduler",
    "configure_pv26_train_stage",
    "move_batch_to_device",
    "postprocess_pv26_batch",
    "raw_batch_for_metrics",
    "render_loss_spec_markdown",
    "run_pv26_tiny_overfit",
    "summarize_pv26_metrics",
]
