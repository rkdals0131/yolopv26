"""Foreground CLI execution and the terminal menu loop."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from common.paths import REPO_ROOT
from .actions import ACTIONS, Cancelled, Command, ask, edit_training_settings, resolve_action, select_run
from .scan import scan_workspace
from .tui import render_dashboard, render_help, render_run_details


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def run_command(console: Console, command: Command) -> int:
    """Keep one owned process group alive until its foreground command exits."""
    process = subprocess.Popen(command.argv, cwd=REPO_ROOT, start_new_session=True)
    previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)}

    def terminate_hub(signum, frame):
        raise SystemExit(128 + signum)

    for signum in previous_handlers:
        signal.signal(signum, terminate_hub)
    try:
        try:
            return process.wait()
        except KeyboardInterrupt:
            console.print("\n중단을 요청했습니다. 학습기는 진행 중인 step을 마치고 저장합니다.", style="yellow")
            _signal_group(process, signal.SIGINT)
            try:
                return process.wait()
            except KeyboardInterrupt:
                console.print("\n종료를 다시 요청했습니다.", style="yellow")
                _signal_group(process, signal.SIGTERM)
                try:
                    return process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _signal_group(process, signal.SIGKILL)
                    return process.wait()
    finally:
        try:
            _signal_group(process, signal.SIGTERM)
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    _signal_group(process, signal.SIGKILL)
                    process.wait()
        finally:
            # The command owns its workers. Release any descendants even if
            # their parent exited without waiting for them.
            _signal_group(process, signal.SIGKILL)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _confirm_and_run(console: Console, command: Command) -> None:
    text = Text("\n".join((*command.notes, "", shlex.join(command.argv))))
    console.print(Panel(text, title=command.title, border_style="cyan"))
    if ask(console, "실행할까요? (y/N)").lower() not in ("y", "yes"):
        return
    result = run_command(console, command)
    console.print(Text(f"종료 코드: {result}", style="green" if result == 0 else "yellow"))
    for note in command.notes:
        console.print(Text(note))


def interactive_loop(console: Console, config_path: Path, signal_config_path: Path) -> int:
    while True:
        snapshot = scan_workspace(config_path, signal_config_path)
        console.clear(home=True)
        render_dashboard(console, snapshot, ACTIONS)
        try:
            key = ask(console, "선택").upper()
            if key == "Q":
                return 0
            if key in ("", "R"):
                continue
            if key == "H":
                render_help(console, snapshot)
            elif key == "L":
                render_run_details(console, select_run(console, snapshot))
            elif key == "S":
                console.print(Text(edit_training_settings(console, snapshot), style="green"))
                continue
            elif any(action.key == key for action in ACTIONS):
                _confirm_and_run(console, resolve_action(key, console, snapshot))
            else:
                console.print("표시된 메뉴 키를 입력하세요.", style="yellow")
            if ask(console, "Enter 화면 갱신 / Q 종료").upper() == "Q":
                return 0
        except Cancelled:
            continue
        except EOFError:
            return 0
        except KeyboardInterrupt:
            console.print("\n선택을 취소했습니다.", style="yellow")
        except (OSError, ValueError, KeyError) as error:
            console.print(Text(str(error), style="red"))
            try:
                if ask(console, "Enter 화면 갱신 / Q 종료").upper() == "Q":
                    return 1
            except (EOFError, KeyboardInterrupt, Cancelled):
                return 1
