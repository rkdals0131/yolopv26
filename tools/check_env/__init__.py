"""Status and operator entrypoint for the current PV26 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from common.paths import REPO_ROOT
from .scan import scan_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YOLOPV26 상태 확인 및 학습·추론 작업 허브")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/pv26.yaml")
    parser.add_argument("--signal-config", type=Path, default=REPO_ROOT / "config/signal_attr.yaml")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--json", action="store_true", help="상태만 JSON으로 출력하고 종료")
    display.add_argument("--once", action="store_true", help="상태 화면을 한 번 출력하고 종료")
    args = parser.parse_args(argv)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if args.json or (not interactive and not args.once):
        print(json.dumps(scan_workspace(args.config, args.signal_config), ensure_ascii=False, indent=2))
        return 0

    from rich.console import Console
    from .actions import ACTIONS
    from .launch import interactive_loop
    from .tui import render_dashboard

    console = Console()
    if args.once:
        render_dashboard(console, scan_workspace(args.config, args.signal_config), ACTIONS)
        return 0
    return interactive_loop(console, args.config, args.signal_config)
