from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.od_bootstrap.calibration.policy_calibration import calibrate_class_policy_scenario
from tools.od_bootstrap.calibration.scenario import CalibrationScenario, load_calibration_scenario


DEFAULT_SCENARIO_PATH = REPO_ROOT / "tools" / "od_bootstrap" / "config" / "calibration" / "class_policy.default.yaml"


def load_and_run_default_class_policy_calibration() -> dict[str, Any]:
    scenario_path = Path(DEFAULT_SCENARIO_PATH).resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"class policy calibration scenario not found: {scenario_path}")
    scenario = load_calibration_scenario(scenario_path)
    return calibrate_class_policy_scenario(scenario, scenario_path=scenario_path)


def run_class_policy_calibration_scenario(scenario: CalibrationScenario, *, scenario_path: Path) -> dict[str, Any]:
    return calibrate_class_policy_scenario(scenario, scenario_path=scenario_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate OD bootstrap class policy from teacher validation predictions.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SCENARIO_PATH),
        help="Path to a calibration scenario YAML file.",
    )
    args = parser.parse_args(argv)
    scenario_path = Path(args.config).resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"class policy calibration scenario not found: {scenario_path}")
    scenario = load_calibration_scenario(scenario_path)
    calibrate_class_policy_scenario(scenario, scenario_path=scenario_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
