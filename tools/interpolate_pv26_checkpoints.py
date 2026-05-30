from __future__ import annotations

import argparse
import json
from pathlib import Path
import site
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PV26 checkpoint by fixed-alpha interpolation between a "
            "retention/base checkpoint and a trained candidate checkpoint."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.50, help="Trained checkpoint weight, in [0, 1].")
    parser.add_argument(
        "--scope",
        choices=("heads", "adapter_heads", "stop_line_head"),
        default="heads",
        help=(
            "heads interpolates all head tensors and keeps the base adapter; "
            "adapter_heads interpolates both adapter and heads; "
            "stop_line_head only interpolates stop-line head tensors."
        ),
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict: {path}")
    for key in ("adapter_state_dict", "heads_state_dict"):
        if not isinstance(checkpoint.get(key), dict):
            raise KeyError(f"checkpoint missing {key}: {path}")
    return checkpoint


def _interpolate_state_dict(
    base_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
    *,
    alpha: float,
    key_prefix: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    output: dict[str, torch.Tensor] = {}
    stats = {
        "copied_from_base": 0,
        "interpolated": 0,
        "missing_in_trained": 0,
        "shape_mismatch": 0,
        "non_float": 0,
        "prefix_skipped": 0,
    }
    for key, base_value in base_state.items():
        trained_value = trained_state.get(key)
        if key_prefix is not None and not key.startswith(key_prefix):
            output[key] = base_value.detach().clone()
            stats["prefix_skipped"] += 1
            continue
        if trained_value is None:
            output[key] = base_value.detach().clone()
            stats["missing_in_trained"] += 1
            continue
        if tuple(base_value.shape) != tuple(trained_value.shape):
            output[key] = base_value.detach().clone()
            stats["shape_mismatch"] += 1
            continue
        if not (torch.is_floating_point(base_value) and torch.is_floating_point(trained_value)):
            output[key] = base_value.detach().clone()
            stats["non_float"] += 1
            continue
        value = (1.0 - alpha) * base_value.detach().to(dtype=torch.float32)
        value = value + alpha * trained_value.detach().to(dtype=torch.float32)
        output[key] = value.to(dtype=base_value.dtype)
        stats["interpolated"] += 1
    stats["copied_from_base"] = (
        stats["missing_in_trained"]
        + stats["shape_mismatch"]
        + stats["non_float"]
        + stats["prefix_skipped"]
    )
    return output, stats


def interpolate_checkpoints(
    *,
    base_checkpoint: Path,
    trained_checkpoint: Path,
    output_path: Path,
    alpha: float,
    scope: str,
) -> dict[str, Any]:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = _load_checkpoint(base_checkpoint)
    trained = _load_checkpoint(trained_checkpoint)
    result = dict(base)
    summary: dict[str, Any] = {
        "base_checkpoint": str(base_checkpoint),
        "trained_checkpoint": str(trained_checkpoint),
        "output": str(output_path),
        "alpha": float(alpha),
        "scope": str(scope),
        "states": {},
    }

    if scope == "adapter_heads":
        adapter_state, adapter_stats = _interpolate_state_dict(
            base["adapter_state_dict"],
            trained["adapter_state_dict"],
            alpha=float(alpha),
        )
        result["adapter_state_dict"] = adapter_state
        summary["states"]["adapter_state_dict"] = adapter_stats
    elif scope in {"heads", "stop_line_head"}:
        summary["states"]["adapter_state_dict"] = {"kept_base": len(base["adapter_state_dict"])}
    else:
        raise ValueError(f"unsupported interpolation scope: {scope}")

    head_prefix = "stop_line_head." if scope == "stop_line_head" else None
    heads_state, heads_stats = _interpolate_state_dict(
        base["heads_state_dict"],
        trained["heads_state_dict"],
        alpha=float(alpha),
        key_prefix=head_prefix,
    )
    result["heads_state_dict"] = heads_state
    summary["states"]["heads_state_dict"] = heads_stats
    result["checkpoint_interpolation"] = dict(summary)
    result["optimizer_state_dict"] = {}
    result.pop("scheduler_state_dict", None)
    result.pop("scaler_state_dict", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return summary


def main() -> int:
    args = parse_args()
    summary = interpolate_checkpoints(
        base_checkpoint=Path(args.base_checkpoint).expanduser().resolve(),
        trained_checkpoint=Path(args.trained_checkpoint).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        alpha=float(args.alpha),
        scope=str(args.scope),
    )
    summary_path = Path(args.output).expanduser().resolve().with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
