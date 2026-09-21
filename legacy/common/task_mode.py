from __future__ import annotations

from typing import Any


LANE_FAMILY_TASK_MODE = "lane_family"
ROADMARK_JOINT_TASK_MODE = "roadmark_joint"
LANE_ONLY_TASK_MODE = "lane_only"
STOPLINE_ONLY_TASK_MODE = "stopline_only"
CROSSWALK_ONLY_TASK_MODE = "crosswalk_only"

TASK_MODE_ALIASES = {
    "family": LANE_FAMILY_TASK_MODE,
    "lane_family": LANE_FAMILY_TASK_MODE,
    "roadmark_joint": ROADMARK_JOINT_TASK_MODE,
    "joint": ROADMARK_JOINT_TASK_MODE,
    "lane": LANE_ONLY_TASK_MODE,
    "lane_only": LANE_ONLY_TASK_MODE,
    "stop_line": STOPLINE_ONLY_TASK_MODE,
    "stop_line_only": STOPLINE_ONLY_TASK_MODE,
    "stopline": STOPLINE_ONLY_TASK_MODE,
    "stopline_only": STOPLINE_ONLY_TASK_MODE,
    "crosswalk": CROSSWALK_ONLY_TASK_MODE,
    "crosswalk_only": CROSSWALK_ONLY_TASK_MODE,
}

TASK_MODE_TO_ACTIVE_TASKS = {
    LANE_FAMILY_TASK_MODE: ("lane", "stop_line", "crosswalk"),
    ROADMARK_JOINT_TASK_MODE: ("lane", "stop_line", "crosswalk"),
    LANE_ONLY_TASK_MODE: ("lane",),
    STOPLINE_ONLY_TASK_MODE: ("stop_line",),
    CROSSWALK_ONLY_TASK_MODE: ("crosswalk",),
}

TASK_MODE_TO_SELECTION_METRIC = {
    LANE_FAMILY_TASK_MODE: "val.metrics.lane_family.selection_score",
    ROADMARK_JOINT_TASK_MODE: "val.metrics.lane_family.strict_selection_score",
    LANE_ONLY_TASK_MODE: "val.metrics.lane.f1",
    STOPLINE_ONLY_TASK_MODE: "val.metrics.stop_line.f1",
    CROSSWALK_ONLY_TASK_MODE: "val.metrics.crosswalk.f1",
}

TASK_NAME_TO_COLLECTION = {
    "lane": "lanes",
    "stop_line": "stop_lines",
    "crosswalk": "crosswalks",
}


def canonicalize_task_mode(task_mode: Any) -> str:
    key = str(task_mode or LANE_FAMILY_TASK_MODE).strip().lower()
    try:
        return TASK_MODE_ALIASES[key]
    except KeyError as exc:
        raise KeyError(f"unsupported task mode: {task_mode!r}") from exc


def active_tasks_for_mode(task_mode: Any) -> tuple[str, ...]:
    return TASK_MODE_TO_ACTIVE_TASKS[canonicalize_task_mode(task_mode)]


def task_mode_selection_metric_path(task_mode: Any) -> str:
    return TASK_MODE_TO_SELECTION_METRIC[canonicalize_task_mode(task_mode)]


def collection_name_for_task(task_name: str) -> str:
    return TASK_NAME_TO_COLLECTION[str(task_name)]


def filter_loss_weights_for_task_mode(loss_weights: dict[str, float], task_mode: Any) -> dict[str, float]:
    resolved_mode = canonicalize_task_mode(task_mode)
    filtered = {
        "det": float(loss_weights.get("det", 0.0)),
        "tl_attr": float(loss_weights.get("tl_attr", 0.0)),
        "lane": float(loss_weights.get("lane", 0.0)),
        "stop_line": float(loss_weights.get("stop_line", 0.0)),
        "crosswalk": float(loss_weights.get("crosswalk", 0.0)),
    }
    if resolved_mode in {LANE_FAMILY_TASK_MODE, ROADMARK_JOINT_TASK_MODE}:
        return filtered
    active_tasks = set(active_tasks_for_mode(resolved_mode))
    filtered["det"] = 0.0
    filtered["tl_attr"] = 0.0
    for task_name in ("lane", "stop_line", "crosswalk"):
        if task_name not in active_tasks:
            filtered[task_name] = 0.0
    return filtered


def filter_source_mask_for_task_mode(source_mask: Any, task_mode: Any) -> dict[str, bool]:
    raw = dict(source_mask) if isinstance(source_mask, dict) else {}
    resolved_mode = canonicalize_task_mode(task_mode)
    active_tasks = set(active_tasks_for_mode(resolved_mode))
    filtered = {
        "det": bool(raw.get("det", False)),
        "tl_attr": bool(raw.get("tl_attr", False)),
        "lane": bool(raw.get("lane", False)),
        "stop_line": bool(raw.get("stop_line", False)),
        "crosswalk": bool(raw.get("crosswalk", False)),
    }
    if resolved_mode not in {LANE_FAMILY_TASK_MODE, ROADMARK_JOINT_TASK_MODE}:
        filtered["det"] = False
        filtered["tl_attr"] = False
    for task_name in ("lane", "stop_line", "crosswalk"):
        filtered[task_name] = filtered[task_name] and task_name in active_tasks
    return filtered


__all__ = [
    "CROSSWALK_ONLY_TASK_MODE",
    "LANE_FAMILY_TASK_MODE",
    "LANE_ONLY_TASK_MODE",
    "ROADMARK_JOINT_TASK_MODE",
    "STOPLINE_ONLY_TASK_MODE",
    "TASK_MODE_TO_ACTIVE_TASKS",
    "TASK_MODE_TO_SELECTION_METRIC",
    "TASK_NAME_TO_COLLECTION",
    "active_tasks_for_mode",
    "canonicalize_task_mode",
    "collection_name_for_task",
    "filter_loss_weights_for_task_mode",
    "filter_source_mask_for_task_mode",
    "task_mode_selection_metric_path",
]
