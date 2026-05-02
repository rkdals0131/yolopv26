from __future__ import annotations

import subprocess
import sys
from typing import Callable

from rich.console import Console
from rich.panel import Panel

from tools.model_export import artifact_paths_for_checkpoint
from tools.model_export import export_pv26_torchscript
from tools.model_export import export_teacher_torchscript
from tools.od_bootstrap.presets import build_teacher_eval_preset

from .actions import (
    ActionSpec,
    _action_advisory,
    _action_blockers,
    _action_catalog,
    _action_config_lines,
)
from .scan import (
    Pv26ExportCandidate,
    RetrainCandidate,
    ResumeCandidate,
    WorkspaceSnapshot,
    _resolve_pipeline_paths,
    _scan_pv26_export_candidates,
    _scan_pv26_retrain_candidates,
    _scan_pv26_resume_candidates,
    check_env,
    scan_workspace_status,
)
from .tui import (
    _render_dashboard,
    _render_export_candidates,
    _render_final_dataset_stats,
    _render_help,
    _render_retrain_candidates,
    _render_resume_candidates,
    _render_phase_stress_result,
)


def _default_phase_stress_batch_size(stage: str = "stage_3_end_to_end_finetune") -> int:
    try:
        from tools.run_pv26_train import _scenario_phase_defaults, load_meta_train_scenario

        scenario = load_meta_train_scenario("default")
        phase = next((item for item in scenario.phases if str(item.stage) == str(stage)), None)
        if phase is None:
            return int(scenario.train_defaults.batch_size)
        phase_train = _scenario_phase_defaults(scenario.train_defaults, phase.overrides)
        return int(phase_train.batch_size)
    except Exception:
        return 40


def _default_stage3_stress_batch_size() -> int:
    return _default_phase_stress_batch_size("stage_3_end_to_end_finetune")


def _ascii_input(console: Console, prompt: str) -> str | None:
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        stdin_buffer = getattr(sys.stdin, "buffer", None)
        if stdin_buffer is not None:
            raw_bytes = stdin_buffer.readline()
            if raw_bytes == b"":
                return None
            raw = raw_bytes.decode("utf-8", errors="ignore").strip()
        else:
            raw = sys.stdin.readline()
            if raw == "":
                return None
            raw = raw.strip()
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


def _prompt_batch_size_csv(console: Console, prompt: str, *, default: str) -> str | None:
    while True:
        raw = _ascii_input(console, prompt)
        if raw is None:
            return None
        if raw == "":
            raw = default
        if raw.upper() == "Q":
            return None
        try:
            from tools.pv26_train.runtime import parse_phase_vram_sweep_batch_sizes

            parse_phase_vram_sweep_batch_sizes(raw)
        except ValueError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            continue
        return raw


def _prompt_stress_stage(console: Console) -> str | None:
    while True:
        raw = _ascii_input(
            console,
            "stress stage (1=stage_1, 2=stage_2, 3=stage_3, 4=stage_4, Enter=3, Q=취소) > ",
        )
        if raw is None:
            return None
        if raw == "":
            return "stage_3_end_to_end_finetune"
        if raw.upper() == "Q":
            return None
        mapping = {
            "1": "stage_1_frozen_trunk_warmup",
            "2": "stage_2_partial_unfreeze",
            "3": "stage_3_end_to_end_finetune",
            "4": "stage_4_lane_family_finetune",
        }
        stage = mapping.get(raw)
        if stage is not None:
            return stage
        console.print("[yellow]1-4 중 하나만 입력하세요.[/yellow]")


def _resolve_phase_stress_action(console: Console, action: ActionSpec) -> ActionSpec | None:
    stage = _prompt_stress_stage(console)
    if stage is None:
        return None
    default_batch_size = _default_phase_stress_batch_size(stage)
    batch_size = _prompt_positive_int(
        console,
        f"{stage} stress batch size (Enter={default_batch_size}, Q=취소) > ",
        default=default_batch_size,
    )
    if batch_size is None:
        return None
    stress_iters = _prompt_positive_int(
        console,
        "phase stress iterations (Enter=12, 권장 10-20, Q=취소) > ",
        default=12,
    )
    if stress_iters is None:
        return None
    argv = tuple(action.argv) + (
        "--stress-stage",
        stage,
        "--stress-batch-size",
        str(batch_size),
        "--stress-iters",
        str(stress_iters),
    )
    command_display = (
        "interactive: "
        f"{stage} VRAM probe (batch_size={batch_size}, stress_iters={stress_iters})"
    )
    return ActionSpec(
        key=action.key,
        label=action.label,
        command_display=command_display,
        argv=argv,
        output_hint=action.output_hint,
        rerun_contract=action.rerun_contract,
    )


def _resolve_stage3_stress_action(console: Console, action: ActionSpec) -> ActionSpec | None:
    return _resolve_phase_stress_action(console, action)


def _resolve_phase_sweep_action(console: Console, action: ActionSpec) -> ActionSpec | None:
    batch_sizes = _prompt_batch_size_csv(
        console,
        "phase sweep batch sizes CSV (Enter=1,2,4,6,8,12, Q=취소) > ",
        default="1,2,4,6,8,12",
    )
    if batch_sizes is None:
        return None
    stress_iters = _prompt_positive_int(
        console,
        "phase sweep iterations per attempt (Enter=8, 권장 6-12, Q=취소) > ",
        default=8,
    )
    if stress_iters is None:
        return None
    argv = (
        sys.executable,
        "tools/run_pv26_train.py",
        "--preset",
        "default",
        "--phase-vram-sweep",
        "--stress-batch-sizes",
        batch_sizes,
        "--stress-iters",
        str(stress_iters),
    )
    command_display = (
        "python3 tools/run_pv26_train.py --preset default --phase-vram-sweep "
        f"--stress-batch-sizes {batch_sizes} --stress-iters {stress_iters}"
    )
    return ActionSpec(
        key=action.key,
        label=action.label,
        command_display=command_display,
        argv=argv,
        output_hint=action.output_hint,
        rerun_contract=action.rerun_contract,
    )


def _argv_flag_value(argv: tuple[str, ...], flag: str) -> str | None:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _render_panel(text: str, *, title: str, border_style: str) -> Panel:
    return Panel(text, title=title, border_style=border_style)


def _pause_to_continue(console: Console, *, prompt: str) -> bool:
    return _pause(console, prompt=prompt) != "Q"


def _run_stage3_stress_probe(
    console: Console,
    snapshot: WorkspaceSnapshot,
    action: ActionSpec,
) -> bool:
    resolved_action = _resolve_phase_stress_action(console, action)
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

    console.print("[bold green]PV26 phase VRAM stress probe를 시작합니다. 진행 로그가 아래에 이어집니다.[/bold green]")
    from tools.run_pv26_train import PRESET_PATH_ROOT, load_meta_train_scenario, run_phase_vram_stress

    try:
        scenario = load_meta_train_scenario("default")
        result = run_phase_vram_stress(
            scenario,
            scenario_path=PRESET_PATH_ROOT / "default",
            stage=_argv_flag_value(resolved_action.argv, "--stress-stage"),
            batch_size=int(_argv_flag_value(resolved_action.argv, "--stress-batch-size") or "0"),
            stress_iters=int(_argv_flag_value(resolved_action.argv, "--stress-iters") or "0"),
        )
    except KeyboardInterrupt:
        console.print("\n[red]실행 중단됨[/red]")
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    except SystemExit as exc:
        console.print(_render_panel(str(exc), title="PV26 Phase VRAM Probe 실패", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    except Exception as exc:
        console.print(_render_panel(str(exc), title="PV26 Phase VRAM Probe 예외", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    _render_phase_stress_result(console, result)
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


def _select_export_candidate(console: Console, candidates: list[Pv26ExportCandidate]) -> Pv26ExportCandidate | None:
    while True:
        raw = _ascii_input(console, f"export 번호 선택 (1-{len(candidates)}, Enter=취소) > ")
        if raw is None or raw == "":
            return None
        if not raw.isdigit():
            console.print("[yellow]숫자만 입력하세요.[/yellow]")
            continue
        index = int(raw)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
        console.print("[yellow]목록에 있는 번호만 입력하세요.[/yellow]")


def _select_retrain_candidate(console: Console, candidates: list[RetrainCandidate]) -> RetrainCandidate | None:
    while True:
        raw = _ascii_input(console, f"retrain 번호 선택 (1-{len(candidates)}, Enter=취소) > ")
        if raw is None or raw == "":
            return None
        if not raw.isdigit():
            console.print("[yellow]숫자만 입력하세요.[/yellow]")
            continue
        index = int(raw)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
        console.print("[yellow]목록에 있는 번호만 입력하세요.[/yellow]")


def _phase_stage_entries() -> list[tuple[int, str, str]]:
    from tools.run_pv26_train import load_meta_train_scenario

    scenario = load_meta_train_scenario("default")
    return [
        (phase_index, str(phase.name), str(phase.stage))
        for phase_index, phase in enumerate(scenario.phases, start=1)
    ]


def _prompt_phase_window(console: Console) -> tuple[str, str] | None:
    phase_entries = _phase_stage_entries()
    console.print(
        _render_panel(
            "\n".join(
                f"{phase_index}. {phase_name} ({phase_stage})"
                for phase_index, phase_name, phase_stage in phase_entries
            ),
            title="Stage Window",
            border_style="cyan",
        )
    )
    while True:
        raw_start = _ascii_input(console, f"start stage 번호 선택 (1-{len(phase_entries)}, Enter=취소) > ")
        if raw_start is None or raw_start == "":
            return None
        if not raw_start.isdigit():
            console.print("[yellow]숫자만 입력하세요.[/yellow]")
            continue
        start_index = int(raw_start)
        if not 1 <= start_index <= len(phase_entries):
            console.print("[yellow]목록에 있는 번호만 입력하세요.[/yellow]")
            continue
        break
    while True:
        raw_end = _ascii_input(console, f"end stage 번호 선택 ({start_index}-{len(phase_entries)}, Enter=취소) > ")
        if raw_end is None or raw_end == "":
            return None
        if not raw_end.isdigit():
            console.print("[yellow]숫자만 입력하세요.[/yellow]")
            continue
        end_index = int(raw_end)
        if not start_index <= end_index <= len(phase_entries):
            console.print("[yellow]start 이상, 마지막 stage 이하 번호만 입력하세요.[/yellow]")
            continue
        break
    return phase_entries[start_index - 1][2], phase_entries[end_index - 1][2]


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


def _run_pv26_retrain(console: Console, snapshot: WorkspaceSnapshot, action: ActionSpec) -> bool:
    candidates = _scan_pv26_retrain_candidates(snapshot.paths.pv26_run_root)
    console.clear(home=True)
    if not candidates:
        console.print(
            _render_panel(
                f"retrain 가능한 source run이 없습니다.\n검색 위치: {snapshot.paths.pv26_run_root}",
                title=f"{action.key} {action.label}",
                border_style="yellow",
            )
        )
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    _render_retrain_candidates(console, candidates)
    selected = _select_retrain_candidate(console, candidates)
    if selected is None:
        return True
    phase_window = _prompt_phase_window(console)
    if phase_window is None:
        return True
    start_stage, end_stage = phase_window
    resolved_action = ActionSpec(
        key=action.key,
        label=f"{action.label}: {selected.run_name}",
        command_display=(
            "python3 tools/run_pv26_train.py "
            f"--preset default --derive-run {selected.run_dir} "
            f"--start-stage {start_stage} --end-stage {end_stage}"
        ),
        argv=(
            sys.executable,
            "tools/run_pv26_train.py",
            "--preset",
            "default",
            "--derive-run",
            str(selected.run_dir),
            "--start-stage",
            start_stage,
            "--end-stage",
            end_stage,
        ),
        output_hint=str(snapshot.paths.pv26_run_root),
        rerun_contract=action.rerun_contract,
    )
    extra_lines = [
        f"- source run: {selected.run_dir}",
        f"- latest stage: {selected.latest_phase_stage or 'unknown'}",
        f"- stage window: {start_stage} -> {end_stage}",
        "- config source: current preset/default + current user YAML",
    ]
    return _run_subprocess_action(
        console,
        snapshot,
        resolved_action,
        extra_lines=extra_lines,
        config_line_factory=None,
    )


def _run_final_dataset_stats(console: Console, snapshot: WorkspaceSnapshot, action: ActionSpec) -> bool:
    from tools.od_bootstrap.build.final_dataset_stats import analyze_final_dataset, load_final_dataset_stats

    dataset_root = snapshot.paths.final_dataset_root
    stats = load_final_dataset_stats(dataset_root)
    generated = False
    if stats is None:
        console.clear(home=True)
        lines = [
            f"🎯 작업: {action.label}",
            f"- dataset root: {dataset_root}",
            "- final_dataset_stats.json이 없어서 labels_scene 전체를 다시 스캔합니다.",
        ]
        advisory = _action_advisory(action, snapshot)
        if advisory is not None:
            lines.append(f"- 주의: {advisory}")
        console.print(_render_panel("\n".join(lines), title=f"{action.key} 실행 확인", border_style="cyan"))
        if not _confirm(console):
            return True
        console.print("[bold green]final dataset full stats를 생성합니다.[/bold green]")
        try:
            stats = analyze_final_dataset(dataset_root=dataset_root, write_artifacts=True)
            generated = True
        except KeyboardInterrupt:
            console.print("\n[red]실행 중단됨[/red]")
            return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
        except Exception as exc:
            console.print(_render_panel(str(exc), title="Final Dataset Stats 실패", border_style="red"))
            return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    if not isinstance(stats, dict):
        console.print(_render_panel("stats payload를 읽지 못했습니다.", title="Final Dataset Stats 실패", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    console.clear(home=True)
    if generated:
        console.print(
            _render_panel(
                f"stats_path: {stats.get('stats_path')}\nstats_markdown_path: {stats.get('stats_markdown_path')}",
                title="Final Dataset Stats 생성 완료",
                border_style="green",
            )
        )
    _render_final_dataset_stats(console, stats)
    return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")


def _run_pv26_export(console: Console, snapshot: WorkspaceSnapshot, action: ActionSpec) -> bool:
    candidates = _scan_pv26_export_candidates(snapshot.paths.pv26_run_root)
    console.clear(home=True)
    if not candidates:
        console.print(
            _render_panel(
                f"export 가능한 completed PV26 run이 없습니다.\n검색 위치: {snapshot.paths.pv26_run_root}",
                title=f"{action.key} {action.label}",
                border_style="yellow",
            )
        )
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    _render_export_candidates(console, candidates)
    selected = _select_export_candidate(console, candidates)
    if selected is None:
        return True

    console.clear(home=True)
    existing_text = "existing=yes" if selected.artifact_path.is_file() and selected.meta_path.is_file() else "existing=no"
    lines = [
        f"🎯 작업: {action.label}",
        f"- run: {selected.run_name}",
        f"- checkpoint: {selected.checkpoint_path}",
        f"- output: {selected.artifact_path}",
        f"- meta: {selected.meta_path}",
        f"- stage: {selected.latest_phase_stage or 'unknown'}",
        f"- selection: {selected.latest_selection_metric_path or 'unknown'}",
        f"- backbone: {selected.latest_backbone_variant or 'unknown'}",
        f"- 상태: {existing_text}",
    ]
    console.print(_render_panel("\n".join(lines), title=f"{action.key} 실행 확인", border_style="cyan"))
    if not _confirm(console):
        return True

    console.print("[bold green]PV26 TorchScript export를 시작합니다.[/bold green]")
    try:
        result = export_pv26_torchscript(
            checkpoint_path=selected.checkpoint_path,
            overwrite=True,
        )
    except KeyboardInterrupt:
        console.print("\n[red]실행 중단됨[/red]")
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    except Exception as exc:
        console.print(_render_panel(str(exc), title="PV26 TorchScript Export 실패", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    console.print(
        _render_panel(
            "\n".join(
                [
                    f"checkpoint: {result['checkpoint_path']}",
                    f"artifact: {result['artifact_path']}",
                    f"meta: {result['meta_path']}",
                    f"variant: {result.get('checkpoint_variant') or 'unknown'}",
                    "다시 스캔해서 상태를 갱신합니다.",
                ]
            ),
            title="PV26 TorchScript Export 완료",
            border_style="green",
        )
    )
    return _pause_to_continue(console, prompt="Enter=상태 새로고침, Q=종료 > ")


def _run_teacher_export(console: Console, snapshot: WorkspaceSnapshot, action: ActionSpec) -> bool:
    teacher_name = {"G": "mobility", "I": "signal", "J": "obstacle"}[action.key]
    scenario = build_teacher_eval_preset(teacher_name)
    checkpoint_path = scenario.model.checkpoint_path
    artifact_path, meta_path = artifact_paths_for_checkpoint(checkpoint_path)
    console.clear(home=True)
    lines = [
        f"🎯 작업: {action.label}",
        f"- teacher: {teacher_name}",
        f"- checkpoint: {checkpoint_path}",
        f"- output: {artifact_path}",
        f"- meta: {meta_path}",
        f"- classes: {', '.join(scenario.model.class_names)}",
        f"- imgsz: {scenario.eval.imgsz}",
    ]
    console.print(_render_panel("\n".join(lines), title=f"{action.key} 실행 확인", border_style="cyan"))
    if not _confirm(console):
        return True

    console.print("[bold green]teacher TorchScript export를 시작합니다.[/bold green]")
    try:
        result = export_teacher_torchscript(
            teacher_name=teacher_name,
            checkpoint_path=checkpoint_path,
            class_names=scenario.model.class_names,
            imgsz=scenario.eval.imgsz,
            overwrite=True,
        )
    except KeyboardInterrupt:
        console.print("\n[red]실행 중단됨[/red]")
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")
    except Exception as exc:
        console.print(_render_panel(str(exc), title="Teacher TorchScript Export 실패", border_style="red"))
        return _pause_to_continue(console, prompt="Enter=메인으로 복귀, Q=종료 > ")

    console.print(
        _render_panel(
            "\n".join(
                [
                    f"teacher: {result['teacher_name']}",
                    f"artifact: {result['artifact_path']}",
                    f"meta: {result['meta_path']}",
                    f"device: {result['device']}",
                    "다시 스캔해서 상태를 갱신합니다.",
                ]
            ),
            title="Teacher TorchScript Export 완료",
            border_style="green",
        )
    )
    return _pause_to_continue(console, prompt="Enter=상태 새로고침, Q=종료 > ")


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
    if action.key == "M":
        resolved_action = _resolve_phase_sweep_action(console, action)
        if resolved_action is None:
            return True
        return _run_subprocess_action(console, snapshot, resolved_action)
    if action.key == "E":
        return _run_pv26_resume(console, snapshot, action)
    if action.key == "K":
        return _run_pv26_retrain(console, snapshot, action)
    if action.key == "L":
        return _run_final_dataset_stats(console, snapshot, action)
    if action.key == "F":
        return _run_pv26_export(console, snapshot, action)
    if action.key in {"G", "I", "J"}:
        return _run_teacher_export(console, snapshot, action)
    return _run_subprocess_action(console, snapshot, action)


def _interactive_loop(console: Console) -> int:
    actions = _action_catalog(_resolve_pipeline_paths())
    while True:
        report = check_env(check_yolo_runtime=True)
        snapshot = scan_workspace_status(report)
        actions = _action_catalog(snapshot.paths)
        _render_dashboard(console, snapshot, actions)

        raw = _ascii_input(console, "선택 (1-9/A-M/H/R/Q) > ")
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
