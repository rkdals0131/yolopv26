from __future__ import annotations

import argparse
import json
from pathlib import Path
import site
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

import torch


TASK_PREFIXES = {
    "lane": (".lane_head.", "lane_head."),
    "stop_line": (".stop_line_head.", "stop_line_head."),
    "crosswalk": (".crosswalk_head.", "crosswalk_head."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge lane-family task-specific head weights into one PV26 checkpoint. "
            "Use this when lane/stop-line/crosswalk task-best checkpoints peak at different epochs."
        )
    )
    parser.add_argument("--base-checkpoint", required=True, help="Checkpoint providing non-task state and fallback weights.")
    parser.add_argument("--lane-checkpoint", required=True, help="Checkpoint providing lane_head weights.")
    parser.add_argument("--stop-line-checkpoint", required=True, help="Checkpoint providing stop_line_head weights.")
    parser.add_argument("--crosswalk-checkpoint", required=True, help="Checkpoint providing crosswalk_head weights.")
    parser.add_argument("--output", required=True, help="Output merged checkpoint path.")
    parser.add_argument("--metadata", default="", help="Optional JSON metadata string to store in extra_state.")
    parser.add_argument(
        "--allow-source-extra-keys",
        action="store_true",
        help=(
            "Allow source task-head keys that are absent from the base checkpoint. "
            "Use this when transplanting from a newer compatible head architecture."
        ),
    )
    return parser.parse_args()


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict: {checkpoint_path}")
    if not isinstance(checkpoint.get("heads_state_dict"), dict):
        raise KeyError(f"checkpoint is missing heads_state_dict: {checkpoint_path}")
    return checkpoint


def _matches_task(key: str, task_name: str) -> bool:
    return any(prefix in key for prefix in TASK_PREFIXES[task_name])


def _replace_task_weights(
    merged_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    *,
    task_name: str,
    allow_source_extra_keys: bool = False,
) -> int:
    replaced = 0
    for key, value in source_state.items():
        if not _matches_task(key, task_name):
            continue
        if key not in merged_state:
            if not allow_source_extra_keys:
                raise KeyError(f"source key for {task_name} is missing in base checkpoint: {key}")
            merged_state[key] = value.detach().clone()
            replaced += 1
            continue
        if isinstance(merged_state[key], torch.Tensor) and tuple(merged_state[key].shape) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch for {task_name} key {key}: "
                f"base={tuple(merged_state[key].shape)} source={tuple(value.shape)}"
            )
        merged_state[key] = value.detach().clone()
        replaced += 1
    if replaced == 0:
        raise ValueError(f"no weights replaced for task: {task_name}")
    return replaced


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    lane_path = Path(args.lane_checkpoint).expanduser().resolve()
    stop_line_path = Path(args.stop_line_checkpoint).expanduser().resolve()
    crosswalk_path = Path(args.crosswalk_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    base = _load_checkpoint(base_path)
    sources = {
        "lane": _load_checkpoint(lane_path),
        "stop_line": _load_checkpoint(stop_line_path),
        "crosswalk": _load_checkpoint(crosswalk_path),
    }

    merged_heads = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in base["heads_state_dict"].items()
    }
    replacements = {
        task_name: _replace_task_weights(
            merged_heads,
            source["heads_state_dict"],
            task_name=task_name,
            allow_source_extra_keys=bool(args.allow_source_extra_keys),
        )
        for task_name, source in sources.items()
    }
    base["heads_state_dict"] = merged_heads

    metadata: dict[str, Any] = {}
    if args.metadata:
        metadata = json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise TypeError("--metadata must decode to a JSON object")
    extra_state = base.get("extra_state")
    if not isinstance(extra_state, dict):
        extra_state = {}
    extra_state["lane_family_task_head_merge"] = {
        "base_checkpoint": str(base_path),
        "lane_checkpoint": str(lane_path),
        "stop_line_checkpoint": str(stop_line_path),
        "crosswalk_checkpoint": str(crosswalk_path),
        "allow_source_extra_keys": bool(args.allow_source_extra_keys),
        "replacements": replacements,
        "metadata": metadata,
    }
    base["extra_state"] = extra_state

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "replacements": replacements,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
