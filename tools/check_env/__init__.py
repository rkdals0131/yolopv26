from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import Any, Sequence

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(REPO_ROOT))

from .actions import (  # noqa: E402
    ActionSpec,
    _action_catalog,
    _action_config_lines,
)
from .launch import (  # noqa: E402
    _ascii_input,
    _default_stage3_stress_batch_size,
    _interactive_loop,
    _resolve_stage3_stress_action,
)
from .scan import (  # noqa: E402
    PipelinePaths,
    ResumeCandidate,
    StageRow,
    WorkspaceSnapshot,
    _manifest_header,
    _scan_pv26_resume_candidates,
    check_env,
    scan_workspace_status,
)

__all__ = [
    "ActionSpec",
    "PipelinePaths",
    "ResumeCandidate",
    "StageRow",
    "WorkspaceSnapshot",
    "_ascii_input",
    "_action_catalog",
    "_action_config_lines",
    "_default_stage3_stress_batch_size",
    "_interactive_loop",
    "_manifest_header",
    "_resolve_stage3_stress_action",
    "_scan_pv26_resume_candidates",
    "_should_run_interactive",
    "check_env",
    "main",
    "scan_workspace_status",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PV26 runtime environment portability prerequisites.")
    parser.add_argument("--check-yolo-runtime", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", help="Always print JSON instead of launching the interactive TUI.")
    return parser


def _should_run_interactive(
    args: argparse.Namespace,
    *,
    stdin_isatty: bool | None = None,
    stdout_isatty: bool | None = None,
) -> bool:
    in_tty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
    out_tty = sys.stdout.isatty() if stdout_isatty is None else stdout_isatty
    return bool(in_tty and out_tty and not args.strict and not args.json)


def _strict_failures(report: dict[str, Any], *, require_runtime: bool) -> list[str]:
    failures: list[str] = []
    if report["versions"]["torch"] is None:
        failures.append("torch missing")
    if report["checks"]["yolo26"]["importable"] is not True:
        failures.append("ultralytics missing")
    if require_runtime and report["checks"]["yolo26"]["runtime_load_ok"] is not True:
        failures.append("yolo26 runtime load failed")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if _should_run_interactive(args):
        return _interactive_loop(Console())

    report = check_env(check_yolo_runtime=args.check_yolo_runtime)
    print(json_dumps(report))

    if not args.strict:
        return 0

    failures = _strict_failures(report, require_runtime=args.check_yolo_runtime)
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


def json_dumps(report: dict[str, Any]) -> str:
    import json

    return json.dumps(report, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    raise SystemExit(main())
