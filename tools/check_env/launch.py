from __future__ import annotations

import subprocess
import sys
from typing import Callable

from rich.console import Console
from rich.panel import Panel

from .actions import (
    ActionSpec,
    _action_advisory,
    _action_blockers,
    _action_catalog,
    _action_config_lines,
)
from .scan import (
    ResumeCandidate,
    WorkspaceSnapshot,
    _resolve_pipeline_paths,
    _scan_pv26_resume_candidates,
    check_env,
    scan_workspace_status,
)
from .tui import (
    _render_dashboard,
    _render_help,
    _render_resume_candidates,
    _render_stage3_stress_result,
)


def _default_stage3_stress_batch_size() -> int:
    try:
        from tools.run_pv26_train import load_meta_train_scenario

        scenario = load_meta_train_scenario("default")
        return int(scenario.train_defaults.batch_size)
    except Exception:
        return 40


def _ascii_input(console: Console, prompt: str) -> str | None:
    try:
        raw = input(prompt).strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        console.print("\n[red]입력이 중단되었습니다.[/red]")
        return None
    if raw and not raw.isascii():
        console.print("[red]한글 입력은 받지 않습니다. 숫자/영문만 입력하세요.[/red]")
        return ""
    return raw


def _confirm(console: Console, prompt: str = "yes/no > ") -> bool:
    while True:
        raw = _ascii_input(console, prompt)
        if raw is None:
            return False
        lowered = raw.lower()
        if lowered in {"y", "yes"}:
            return True
        if lowered in {"n", "no", ""}:
            return False
        console.print("[yellow]`yes` 또는 `no`만 입력하세요.[/yellow]")


def _pause(console: Console, *, prompt: str = "Enter=계속 > ") -> str:
    raw = _ascii_input(console, prompt)
    return "" if raw is None else raw.upper()


def _prompt_positive_int(console: Console, prompt: str, *, default: int) -> int | None:
    while True:
        raw = _ascii_input(console, prompt)
        if raw is None:
            return None
        if raw == "":
            return int(default)
        if raw.upper() == "Q":
            return None
        try:
            value = int(raw)
        except ValueError:
            console.print("[yellow]양의 정수만 입력하세요.[/yellow]")
            continue
        if value <= 0:
            console.print("[yellow]0보다 큰 정수만 입력하세요.[/yellow]")
            continue
        return value


def _resolve_stage3_stress_action(console: Console, action: ActionSpec) -> ActionSpec | None:
    default_batch_size = _default_stage3_stress_batch_size()
    batch_size = _prompt_positive_int(
        console,
        f"stage_3 stress batch size (Enter={default_batch_size}, Q=취소) > ",
        default=default_batch_size,
    )
    if batch_size is None:
        return None
    stress_iters = _prompt_positive_int(
        console,
        "stage_3 stress iterations (Enter=12, 권장 10-20, Q=취소) > ",
        default=12,
    )
    if stress_iters is None:
        return None
    argv = tuple(action.argv) + (
        "--stress-batch-size",
        str(batch_size),
        "--stress-iters",
        str(stress_iters),
    )
    command_display = (
        "interactive: "
        f"stage_3 peak VRAM probe (batch_size={batch_size}, stress_iters={stress_iters})"
    )
    return ActionSpec(
        key=action.key,
        label=action.label,
        command_display=command_display,
        argv=argv,
        output_hint=action.output_hint,
        rerun_contract=action.rerun_contract,
    )


def _render_panel(text: str, *, title: str, border_style: str) -> Panel:
    return Panel(text, title=title, border_style=border_style)


def _pause_to_continue(console: Console, *, prompt: str) -> bool:
    return _pause(console, prompt=prompt) != "Q"


def _run_stage3_stress_probe(
    console: Console,
    snapshot: WorkspaceSnapshot,
    action: ActionSpec,
) -> bool:
    resolved_action = _resolve_stage3_stress_action(console, action)
    if resolved_action is None:
        return True

    console.clear(home=True)
    lines = [
        f"🎯 작업: {resolved_action.label}",
        f"- mode: {resolved_action.command_display}",
        f"- 출력 위치: {resolved_action.output_hint}",
    ]
    if resolved_action.rerun_contract is not None:
        lines.append(f"- 재실행 계약: {resolved_action.rerun_contract}")
    advisory = _action_advisory(resolved_action, snapshot)
    if advisory is not None:
        lines.append(f"- 주의: {advisory}")
    console.print(_render_panel("\n".join(lines), title=f"{resolved_action.key} 실행 확인", border_style="cyan"))
    if not _confirm(console):
        return True

    console.print("[bold green]stage_3 VRAM stress probe를 시작합니다. 진행 로그가 아래에 이어집니다.[/bold green]")
    from tools.run_pv26_train import PRESET_PATH_ROOT, load_meta_train_scenario, run_stage3_vram_stress

    try:
        scenario = load_meta_train_scenario("default")
        result = run_stage3_vram_stress(
            scenario,
            scenario_path=PRESET_PATH_ROOT / "default",
            batch_size=int(resolved_action.argv[-3]),
            stress_iters=int(resolved_action.argv[-1]),
        )
    except KeyboardInterrupt:
        console.print("\n[red]실행 중단됨[/red]")
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    except SystemExit as exc:
        console.print(_render_panel(str(exc), title="Stage 3 VRAM Probe 실패", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    except Exception as exc:
        console.print(_render_panel(str(exc), title="Stage 3 VRAM Probe 예외", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    _render_stage3_stress_result(console, result)
    return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")


def _select_resume_candidate(console: Console, candidates: list[ResumeCandidate]) -> ResumeCandidate | None:
    while True:
        raw = _ascii_input(console, f"resume 번호 선택 (1-{len(candidates)}, Enter=취소) > ")
        if raw is None or raw == "":
            return None
        if not raw.isdigit():
            console.print("[yellow]숫자만 입력하세요.[/yellow]")
            continue
        index = int(raw)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
        console.print("[yellow]목록에 있는 번호만 입력하세요.[/yellow]")


def _run_subprocess_action(
    console: Console,
    snapshot: WorkspaceSnapshot,
    resolved_action: ActionSpec,
    *,
    extra_lines: list[str] | None = None,
    config_line_factory: Callable[[ActionSpec], list[str]] | None = _action_config_lines,
) -> bool:
    console.clear(home=True)
    lines = [
        f"🎯 작업: {resolved_action.label}",
        f"- 명령: {resolved_action.command_display}",
        f"- 출력 위치: {resolved_action.output_hint}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    if config_line_factory is not None:
        try:
            config_lines = config_line_factory(resolved_action)
        except Exception as exc:
            config_lines = [f"- config load error: {exc}"]
        lines.extend(config_lines)
    if resolved_action.rerun_contract is not None:
        lines.append(f"- 재실행 계약: {resolved_action.rerun_contract}")
    advisory = _action_advisory(resolved_action, snapshot)
    if advisory is not None:
        lines.append(f"- 주의: {advisory}")
    console.print(_render_panel("\n".join(lines), title=f"{resolved_action.key} 실행 확인", border_style="cyan"))
    if not _confirm(console):
        return True

    console.print("[bold green]실행을 시작합니다. 로그는 아래에 그대로 이어집니다.[/bold green]")
    try:
        completed = subprocess.run(resolved_action.argv, cwd=str(snapshot.paths.repo_root), check=False)
    except KeyboardInterrupt:
        console.print("\n[red]실행 중단됨[/red]")
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    console.print(
        _render_panel(
            f"return code: {completed.returncode}\n다시 스캔해서 상태를 갱신합니다.",
            title="실행 종료",
            border_style="green" if completed.returncode == 0 else "yellow",
        )
    )
    return _pause_to_continue(console, prompt="Enter=상태 새로고침, Q=종료 > ")


def _run_pv26_resume(console: Console, snapshot: WorkspaceSnapshot, action: ActionSpec) -> bool:
    candidates = _scan_pv26_resume_candidates(snapshot.paths.pv26_run_root)
    console.clear(home=True)
    if not candidates:
        console.print(
            _render_panel(
                f"resumable run이 없습니다.\n검색 위치: {snapshot.paths.pv26_run_root}",
                title=f"{action.key} {action.label}",
                border_style="yellow",
            )
        )
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    _render_resume_candidates(console, candidates)
    selected = _select_resume_candidate(console, candidates)
    if selected is None:
        return True

    resolved_action = ActionSpec(
        key=action.key,
        label=f"{action.label}: {selected.run_name}",
        command_display=f"python3 tools/run_pv26_train.py --resume-run {selected.run_dir}",
        argv=(sys.executable, "tools/run_pv26_train.py", "--resume-run", str(selected.run_dir)),
        output_hint=str(selected.run_dir),
        rerun_contract=action.rerun_contract,
    )
    extra_lines = [
        f"- 다음 phase: {selected.next_phase_name} ({selected.next_phase_stage})",
        f"- resume source: {selected.resume_source}",
        "- config source: selected run의 meta_manifest.json / scenario snapshot",
    ]
    return _run_subprocess_action(
        console,
        snapshot,
        resolved_action,
        extra_lines=extra_lines,
        config_line_factory=None,
    )


def _run_action(console: Console, action: ActionSpec, snapshot: WorkspaceSnapshot) -> bool:
    blockers = _action_blockers(action, snapshot)
    if blockers:
        console.print(
            _render_panel(
                "\n".join(f"- {item}" for item in blockers),
                title=f"{action.key} {action.label} 실행 차단",
                border_style="red",
            )
        )
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    if action.key == "D":
        return _run_stage3_stress_probe(console, snapshot, action)
    if action.key == "E":
        return _run_pv26_resume(console, snapshot, action)
    return _run_subprocess_action(console, snapshot, action)


def _interactive_loop(console: Console) -> int:
    actions = _action_catalog(_resolve_pipeline_paths())
    while True:
        report = check_env(check_yolo_runtime=True)
        snapshot = scan_workspace_status(report)
        actions = _action_catalog(snapshot.paths)
        _render_dashboard(console, snapshot, actions)

        raw = _ascii_input(console, "선택 (1-9/A-E/H/R/Q) > ")
        if raw is None:
            return 0
        if raw == "":
            continue
        choice = raw.upper()
        if choice == "Q":
            return 0
        if choice == "R":
            continue
        if choice == "H":
            _render_help(console, snapshot)
            _pause(console, prompt="Enter=메인으로 돌아가기 > ")
            continue
        action = next((item for item in actions if item.key == choice), None)
        if action is None:
            console.print("[yellow]알 수 없는 입력입니다. 숫자/영문 키만 사용하세요.[/yellow]")
            _pause(console)
            continue
        if not _run_action(console, action, snapshot):
            return 0
