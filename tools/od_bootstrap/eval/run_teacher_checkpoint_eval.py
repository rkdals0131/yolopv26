from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.eval.checkpoint_eval import eval_teacher_checkpoint, load_teacher_checkpoint_eval_scenario


DEFAULT_SCENARIO_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "eval" / "teacher_checkpoint_eval.default.yaml"


def load_and_run_default_teacher_checkpoint_eval() -> dict[str, Any]:
    scenario_path = Path(DEFAULT_SCENARIO_PATH).resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"teacher checkpoint eval scenario not found: {scenario_path}")
    scenario = load_teacher_checkpoint_eval_scenario(scenario_path)
    return eval_teacher_checkpoint(scenario=scenario, scenario_path=scenario_path)


def run_teacher_checkpoint_eval_scenario(scenario, *, scenario_path: Path) -> dict[str, Any]:
    return eval_teacher_checkpoint(scenario=scenario, scenario_path=scenario_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a checkpoint evaluation scenario for an Ultralytics teacher model.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SCENARIO_PATH),
        help="Path to a checkpoint eval scenario YAML file.",
    )
    args = parser.parse_args(argv)
    scenario_path = Path(args.config).resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"teacher checkpoint eval scenario not found: {scenario_path}")
    scenario = load_teacher_checkpoint_eval_scenario(scenario_path)
    eval_teacher_checkpoint(scenario=scenario, scenario_path=scenario_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
