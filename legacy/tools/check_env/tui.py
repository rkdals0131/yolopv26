from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .actions import ActionSpec
from .scan import Pv26ExportCandidate, RetrainCandidate, ResumeCandidate, STAGE_ICON, WorkspaceSnapshot


def _format_gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.2f} GiB"


def _render_phase_stress_result(console: Console, result: dict[str, Any]) -> None:
    memory = result.get("memory", {}) if isinstance(result.get("memory"), dict) else {}
    train_summary = result.get("train_summary", {}) if isinstance(result.get("train_summary"), dict) else {}
    lines = [
        f"status: {result.get('status')}",
        f"phase: {result.get('phase_name')} ({result.get('phase_stage')})",
        f"device: {result.get('device')}",
        f"backbone: {result.get('backbone_variant')}",
        f"batch_size: {result.get('batch_size')}",
        f"stress_iters: {result.get('stress_iters')}",
        f"duration: {float(result.get('duration_sec', 0.0)):.2f} sec",
        f"peak_allocated: {_format_gib(memory.get('peak_allocated_gib'))}",
        f"peak_reserved: {_format_gib(memory.get('peak_reserved_gib'))}",
        f"current_allocated: {_format_gib(memory.get('current_allocated_gib'))}",
        f"current_reserved: {_format_gib(memory.get('current_reserved_gib'))}",
    ]
    if train_summary:
        lines.extend(
            [
                f"attempted_batches: {train_summary.get('attempted_batches')}",
                f"successful_batches: {train_summary.get('successful_batches')}",
                f"skipped_batches: {train_summary.get('skipped_batches')}",
            ]
        )
    error = result.get("error")
    if error:
        lines.append(f"error: {error}")
    recommendation = result.get("recommendation")
    if recommendation:
        lines.append(f"recommendation: {recommendation}")
    status = str(result.get("status") or "")
    border_style = "green" if status == "ok" else "yellow"
    console.print(Panel("\n".join(lines), title="PV26 Phase VRAM Probe", border_style=border_style))


def _render_stage3_stress_result(console: Console, result: dict[str, Any]) -> None:
    _render_phase_stress_result(console, result)


def _render_dashboard(console: Console, snapshot: WorkspaceSnapshot, actions: tuple[ActionSpec, ...]) -> None:
    console.clear(home=True)
    console.print(
        Panel.fit(
            f"🧭 PV26 체크 허브\nrepo: {snapshot.paths.repo_root}\n📌 추천: {snapshot.recommendation}",
            border_style="cyan",
        )
    )

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("단계", style="bold")
    table.add_column("성공 조건")
    table.add_column("현재 상태")
    table.add_column("판정", justify="center")
    for row in snapshot.rows:
        table.add_row(row.stage, row.success_condition, row.current_state, f"{STAGE_ICON[row.verdict]} {row.verdict}")
    console.print(table)

    path_table = Table(box=box.SIMPLE, title="📂 현재 기준 경로")
    path_table.add_column("항목", style="bold")
    path_table.add_column("경로")
    path_table.add_row("user_paths", str(snapshot.paths.user_paths_config_path))
    path_table.add_row("bootstrap_root", str(snapshot.paths.bootstrap_root))
    path_table.add_row("final_dataset_root", str(snapshot.paths.final_dataset_root))
    path_table.add_row("pv26_run_root", str(snapshot.paths.pv26_run_root))
    console.print(path_table)

    if snapshot.notes:
        console.print(
            Panel(
                "\n".join(f"- {note}" for note in snapshot.notes),
                title="📝 메모",
                border_style="yellow",
            )
        )

    action_table = Table(box=box.SIMPLE, title="🎛️ 메뉴")
    action_table.add_column("키", style="bold cyan", justify="center")
    action_table.add_column("동작")
    action_table.add_column("명령")
    for action in actions:
        action_table.add_row(action.key, action.label, action.command_display)
    action_table.add_row("H", "Help", "간단 README / config 위치 안내")
    action_table.add_row("R", "Refresh", "상태 다시 스캔")
    action_table.add_row("Q", "Quit", "종료")
    console.print(action_table)


def _render_help(console: Console, snapshot: WorkspaceSnapshot) -> None:
    console.clear(home=True)
    console.print(
        Panel(
            "\n".join(
                [
                    "❓ 이 TUI는 설정 편집기가 아닙니다.",
                    f"- 경로를 바꾸려면: {snapshot.paths.user_paths_config_path}",
                    f"- bootstrap/calibration/exhaustive 숫자 파라미터를 바꾸려면: {snapshot.paths.od_hyperparameters_config_path}",
                    f"- PV26 학습 숫자 파라미터를 바꾸려면: {snapshot.paths.pv26_hyperparameters_config_path}",
                    f"- 전체 흐름 설명: {snapshot.paths.repo_root / 'README.md'}",
                    f"- bootstrap 전용 설명: {snapshot.paths.repo_root / 'tools' / 'od_bootstrap' / 'README.md'}",
                    "- 코드에서 빠른 조절 지점을 찾고 싶으면 `USER CONFIG`, `HYPERPARAMETERS`, `PHASE HYPERPARAMETERS`를 검색하세요.",
                    "- `E` resume는 exact resume only입니다. 같은 run을 그대로 이어서만 재개합니다.",
                    "- `K` retrain은 source run을 seed로 새 derived run을 만듭니다. stage window만 고르고 숫자 파라미터는 config에서 관리합니다.",
                    "- `L`은 최종 병합 데이터셋의 full stats를 보여줍니다. stats 파일이 없으면 labels_scene를 다시 스캔해 생성합니다.",
                    "- `F/G/I/J` export는 checkpoint 옆에 TorchScript artifact와 .meta.json을 씁니다.",
                    "- `N`은 고정 eval dataset 하나와 ready TorchScript model 여러 개를 골라 metrics/plots/overlays를 생성합니다.",
                    "- 입력은 숫자/영문만 받습니다. yes/no 또는 y/n만 사용하세요.",
                ]
            ),
            title="H 도움말",
            border_style="green",
        )
    )


def _render_resume_candidates(console: Console, candidates: list[ResumeCandidate]) -> None:
    table = Table(box=box.SIMPLE_HEAVY, title="PV26 Exact Resume Candidates")
    table.add_column("번호", justify="right", style="bold cyan")
    table.add_column("Run")
    table.add_column("상태")
    table.add_column("진행도")
    table.add_column("다음 phase")
    table.add_column("Resume source")
    table.add_column("Updated")
    for index, item in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            item.run_name,
            item.status,
            f"{item.completed_phases}/{item.total_phases}",
            f"{item.next_phase_name} ({item.next_phase_stage})",
            item.resume_source,
            item.updated_at or "-",
        )
    console.print(table)


def _render_export_candidates(console: Console, candidates: list[Pv26ExportCandidate]) -> None:
    table = Table(box=box.SIMPLE_HEAVY, title="PV26 TorchScript Export Candidates")
    table.add_column("번호", justify="right", style="bold cyan")
    table.add_column("Run")
    table.add_column("Checkpoint")
    table.add_column("Stage")
    table.add_column("Backbone")
    table.add_column("Selection")
    table.add_column("Updated")
    for index, item in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            item.run_name,
            item.checkpoint_path.name,
            item.latest_phase_stage or "-",
            item.latest_backbone_variant or "-",
            item.latest_selection_metric_path or "-",
            item.updated_at or "-",
        )
    console.print(table)


def _render_retrain_candidates(console: Console, candidates: list[RetrainCandidate]) -> None:
    table = Table(box=box.SIMPLE_HEAVY, title="PV26 Retrain / Fine-tune Candidates")
    table.add_column("번호", justify="right", style="bold cyan")
    table.add_column("Run")
    table.add_column("상태")
    table.add_column("완료")
    table.add_column("Latest stage")
    table.add_column("Seed stages")
    table.add_column("Updated")
    for index, item in enumerate(candidates, start=1):
        latest_stage = item.latest_phase_stage or "-"
        table.add_row(
            str(index),
            item.run_name,
            item.status,
            f"{item.completed_phases}/{item.total_executable_phases}",
            latest_stage,
            ", ".join(item.available_seed_stages) if item.available_seed_stages else "-",
            item.updated_at or "-",
        )
    console.print(table)


def _render_final_dataset_stats(console: Console, stats: dict[str, Any]) -> None:
    detector = stats.get("detector", {}) if isinstance(stats.get("detector"), dict) else {}
    detector_classes = detector.get("classes", {}) if isinstance(detector.get("classes"), dict) else {}
    traffic_light_attr = (
        stats.get("traffic_light_attr", {})
        if isinstance(stats.get("traffic_light_attr"), dict)
        else {}
    )
    lane = stats.get("lane", {}) if isinstance(stats.get("lane"), dict) else {}
    lane_classes = lane.get("classes", {}) if isinstance(lane.get("classes"), dict) else {}
    lane_types = lane.get("types", {}) if isinstance(lane.get("types"), dict) else {}
    audit = stats.get("audit", {}) if isinstance(stats.get("audit"), dict) else {}
    warnings = stats.get("warnings", []) if isinstance(stats.get("warnings"), list) else []

    summary_lines = [
        f"dataset_root: {stats.get('dataset_root')}",
        f"samples: {stats.get('sample_count')}",
        f"dataset_counts: {stats.get('dataset_counts')}",
        f"split_counts: {stats.get('split_counts')}",
        f"source_kinds: {stats.get('source_kind_counts')}",
        f"presence: {stats.get('presence_counts')}",
        f"warnings: {warnings if warnings else '[]'}",
    ]
    console.print(Panel("\n".join(summary_lines), title="Final Dataset Stats", border_style="cyan"))

    det_table = Table(box=box.SIMPLE_HEAVY, title="Detector Classes")
    det_table.add_column("Class", style="bold")
    det_table.add_column("Images", justify="right")
    det_table.add_column("Instances", justify="right")
    det_table.add_column("Split Images")
    for class_name, payload in detector_classes.items():
        if not isinstance(payload, dict):
            continue
        det_table.add_row(
            str(class_name),
            str(payload.get("image_count", 0)),
            str(payload.get("instance_count", 0)),
            str(payload.get("split_image_counts", {})),
        )
    console.print(det_table)

    tl_lines = [
        f"valid_count: {traffic_light_attr.get('valid_count', 0)}",
        f"invalid_count: {traffic_light_attr.get('invalid_count', 0)}",
        f"valid_image_count: {traffic_light_attr.get('valid_image_count', 0)}",
        f"combo_counts: {traffic_light_attr.get('combo_counts', {})}",
        f"bit_positive_counts: {traffic_light_attr.get('bit_positive_counts', {})}",
        f"invalid_reason_counts: {traffic_light_attr.get('invalid_reason_counts', {})}",
        f"bbox_buckets: {detector.get('traffic_light_bbox_buckets', {})}",
    ]
    console.print(Panel("\n".join(tl_lines), title="Traffic Light Attr", border_style="green"))

    lane_table = Table(box=box.SIMPLE_HEAVY, title="Lane Classes")
    lane_table.add_column("Lane Class", style="bold")
    lane_table.add_column("Images", justify="right")
    lane_table.add_column("Instances", justify="right")
    for class_name, payload in lane_classes.items():
        if not isinstance(payload, dict):
            continue
        lane_table.add_row(
            str(class_name),
            str(payload.get("image_count", 0)),
            str(payload.get("instance_count", 0)),
        )
    console.print(lane_table)

    lane_type_table = Table(box=box.SIMPLE_HEAVY, title="Lane Types")
    lane_type_table.add_column("Lane Type", style="bold")
    lane_type_table.add_column("Images", justify="right")
    lane_type_table.add_column("Instances", justify="right")
    for lane_type, payload in lane_types.items():
        if not isinstance(payload, dict):
            continue
        lane_type_table.add_row(
            str(lane_type),
            str(payload.get("image_count", 0)),
            str(payload.get("instance_count", 0)),
        )
    console.print(lane_type_table)

    audit_lines = [
        f"manifest_found: {audit.get('manifest_found', False)}",
        f"manifest_sample_count: {audit.get('manifest_sample_count', 0)}",
        f"scanned_scene_count: {audit.get('scanned_scene_count', 0)}",
        f"scene_path_valid: {audit.get('manifest_scene_path_valid_count', 0)} / invalid: {audit.get('manifest_scene_path_invalid_count', 0)}",
        f"image_path_valid: {audit.get('manifest_image_path_valid_count', 0)} / invalid: {audit.get('manifest_image_path_invalid_count', 0)}",
        f"det_path_present: {audit.get('manifest_det_path_present_count', 0)} / valid: {audit.get('manifest_det_path_valid_count', 0)} / invalid: {audit.get('manifest_det_path_invalid_count', 0)}",
        f"rebuild_needed: {audit.get('rebuild_needed', False)}",
    ]
    console.print(Panel("\n".join(audit_lines), title="Audit", border_style="yellow"))
