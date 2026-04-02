from __future__ import annotations

from dataclasses import dataclass
import sys

from tools.check_env_scan import PipelinePaths, TEACHER_NAMES, WorkspaceSnapshot
from tools.od_bootstrap.presets import build_teacher_train_preset


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    command_display: str
    argv: tuple[str, ...]
    output_hint: str
    rerun_contract: str | None = None


def _action_catalog(paths: PipelinePaths) -> tuple[ActionSpec, ...]:
    python_exe = sys.executable
    return (
        ActionSpec("1", "OD bootstrap 소스 준비", "python -m tools.od_bootstrap prepare-sources", (python_exe, "-m", "tools.od_bootstrap", "prepare-sources"), str(paths.bootstrap_root)),
        ActionSpec("2", "Teacher dataset 생성", "python -m tools.od_bootstrap build-teacher-datasets", (python_exe, "-m", "tools.od_bootstrap", "build-teacher-datasets"), str(paths.teacher_dataset_root)),
        ActionSpec("3", "Mobility teacher 학습", "python -m tools.od_bootstrap train --teacher mobility", (python_exe, "-m", "tools.od_bootstrap", "train", "--teacher", "mobility"), str(paths.teacher_train_root / "mobility")),
        ActionSpec("4", "Signal teacher 학습", "python -m tools.od_bootstrap train --teacher signal", (python_exe, "-m", "tools.od_bootstrap", "train", "--teacher", "signal"), str(paths.teacher_train_root / "signal")),
        ActionSpec("5", "Obstacle teacher 학습", "python -m tools.od_bootstrap train --teacher obstacle", (python_exe, "-m", "tools.od_bootstrap", "train", "--teacher", "obstacle"), str(paths.teacher_train_root / "obstacle")),
        ActionSpec("6", "Mobility teacher 평가", "python -m tools.od_bootstrap eval --teacher mobility", (python_exe, "-m", "tools.od_bootstrap", "eval", "--teacher", "mobility"), str(paths.teacher_eval_root / "mobility")),
        ActionSpec("7", "Signal teacher 평가", "python -m tools.od_bootstrap eval --teacher signal", (python_exe, "-m", "tools.od_bootstrap", "eval", "--teacher", "signal"), str(paths.teacher_eval_root / "signal")),
        ActionSpec("8", "Obstacle teacher 평가", "python -m tools.od_bootstrap eval --teacher obstacle", (python_exe, "-m", "tools.od_bootstrap", "eval", "--teacher", "obstacle"), str(paths.teacher_eval_root / "obstacle")),
        ActionSpec("9", "Calibration", "python -m tools.od_bootstrap calibrate", (python_exe, "-m", "tools.od_bootstrap", "calibrate"), str(paths.calibration_root)),
        ActionSpec("A", "Exhaustive OD 생성", "python -m tools.od_bootstrap build-exhaustive-od", (python_exe, "-m", "tools.od_bootstrap", "build-exhaustive-od"), str(paths.exhaustive_dataset_root)),
        ActionSpec("B", "최종 병합 데이터셋 생성", "python -m tools.od_bootstrap build-final-dataset", (python_exe, "-m", "tools.od_bootstrap", "build-final-dataset"), str(paths.final_dataset_root), rerun_contract="overwrite: staging build 후 final root swap"),
        ActionSpec("C", "PV26 기본 학습", "python3 tools/run_pv26_train.py --preset default", (python_exe, "tools/run_pv26_train.py", "--preset", "default"), str(paths.pv26_run_root)),
        ActionSpec("D", "PV26 stage_3 VRAM stress", "interactive: batch/iter 입력 후 stage_3 peak VRAM probe", (), "TUI result panel only (no checkpoints / no run dir)", rerun_contract="short probe only: stage_3 train loop 일부만 실행"),
        ActionSpec("E", "PV26 exact resume", "interactive: resumable run 목록에서 골라 same run dir exact resume", (), str(paths.pv26_run_root), rerun_contract="exact resume only: same run dir / same scenario"),
    )


def _action_blockers(action: ActionSpec, snapshot: WorkspaceSnapshot) -> list[str]:
    flags = snapshot.flags
    blockers: list[str] = []
    if action.key == "1":
        if not flags.get("raw_roots", False):
            blockers.append("raw dataset root가 config 기준으로 안 맞습니다.")
    elif action.key == "2":
        if not flags.get("source_prep", False):
            blockers.append("source prep이 아직 준비되지 않았습니다.")
    elif action.key in {"3", "4", "5"}:
        teacher_name = {"3": "mobility", "4": "signal", "5": "obstacle"}[action.key]
        if not flags.get("runtime_core", False):
            blockers.append("학습에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get(f"teacher_dataset.{teacher_name}", False):
            blockers.append(f"{teacher_name} teacher dataset이 아직 없습니다.")
    elif action.key in {"6", "7", "8"}:
        teacher_name = {"6": "mobility", "7": "signal", "8": "obstacle"}[action.key]
        if not flags.get("runtime_core", False):
            blockers.append("평가에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get(f"teacher_train.{teacher_name}", False):
            blockers.append(f"{teacher_name} teacher checkpoint가 아직 없습니다.")
    elif action.key == "9":
        if not flags.get("runtime_core", False):
            blockers.append("calibration에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        missing = [name for name in TEACHER_NAMES if not flags.get(f"teacher_train.{name}", False)]
        if missing:
            blockers.append(f"teacher checkpoint가 부족합니다: {', '.join(missing)}")
    elif action.key == "A":
        if not flags.get("runtime_core", False):
            blockers.append("exhaustive OD에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get("source_prep", False):
            blockers.append("source prep이 먼저 준비되어야 합니다.")
        missing = [name for name in TEACHER_NAMES if not flags.get(f"teacher_train.{name}", False)]
        if missing:
            blockers.append(f"teacher checkpoint가 부족합니다: {', '.join(missing)}")
    elif action.key == "B":
        if not flags.get("source_prep", False):
            blockers.append("lane canonical/source prep이 먼저 필요합니다.")
        if not flags.get("exhaustive", False):
            blockers.append("최신 exhaustive OD run이 아직 없습니다.")
    elif action.key in {"C", "D"}:
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 학습에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
        if not flags.get("final_dataset", False):
            blockers.append("최종 병합 데이터셋이 아직 없습니다.")
    elif action.key == "E":
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 학습에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
    return blockers


def _action_advisory(action: ActionSpec, snapshot: WorkspaceSnapshot) -> str | None:
    if action.key == "A" and not snapshot.flags.get("calibration", False):
        return "calibration이 없어서 fallback class policy로 진행될 수 있습니다."
    if action.key == "B":
        return "fixed output root를 직접 덮어쓰지 않고 staging build 후 atomic swap합니다."
    if action.key == "D":
        return "stage_3는 현재 전체 학습 단계 중 VRAM 상한을 보는 가장 좋은 proxy입니다. stage_4는 trunk/lane-family만 학습하므로 보통 더 낮습니다."
    if action.key == "E":
        return "resume는 exact resume only입니다. batch_size 변경이나 best/epoch 재시작은 별도 흐름으로 다루는 편이 안전합니다."
    return None


def _bool_flag(value: bool) -> str:
    return "true" if bool(value) else "false"


def _teacher_action_config_lines(action: ActionSpec) -> list[str]:
    if action.key not in {"3", "4", "5"}:
        return []
    teacher_name = {"3": "mobility", "4": "signal", "5": "obstacle"}[action.key]
    scenario = build_teacher_train_preset(teacher_name)
    return [
        f"- config: teacher={scenario.teacher_name}, model=yolo26{scenario.model.model_size}, weights={scenario.model.weights}",
        f"- classes: {', '.join(scenario.model.class_names)}",
        f"- train: epochs={scenario.train.epochs}, batch={scenario.train.batch}, imgsz={scenario.train.imgsz}, device={scenario.train.device}",
        f"- loader: workers={scenario.train.workers}, pin_memory={_bool_flag(scenario.train.pin_memory)}, persistent_workers={_bool_flag(scenario.train.persistent_workers)}, prefetch_factor={scenario.train.prefetch_factor}",
        f"- runtime: amp={_bool_flag(scenario.train.amp)}, cache={_bool_flag(scenario.train.cache)}, optimizer={scenario.train.optimizer}, patience={scenario.train.patience}, save_period={scenario.train.save_period}",
        f"- dataset root: {scenario.dataset.root}",
    ]


def _pv26_action_config_lines() -> list[str]:
    from tools.run_pv26_train import (
        _resolve_phase_selection,
        _scenario_phase_defaults,
        load_meta_train_scenario,
    )

    scenario = load_meta_train_scenario("default")
    dataset_roots = [str(path) for path in scenario.dataset.roots]
    train_defaults = scenario.train_defaults
    lines = [
        f"- dataset roots: {dataset_roots}",
        f"- defaults: device={train_defaults.device}, backbone={train_defaults.backbone_variant}, batch={train_defaults.batch_size}, workers={train_defaults.num_workers}, amp={_bool_flag(train_defaults.amp)}",
        f"- optimizer: trunk_lr={train_defaults.trunk_lr}, head_lr={train_defaults.head_lr}, weight_decay={train_defaults.weight_decay}, schedule={train_defaults.schedule}",
        f"- runtime: val_every={train_defaults.val_every}, checkpoint_every={train_defaults.checkpoint_every}, accumulate_steps={train_defaults.accumulate_steps}, grad_clip={train_defaults.grad_clip_norm}",
        f"- preview: enabled={_bool_flag(scenario.preview.enabled)}, split={scenario.preview.split}, per_dataset={scenario.preview.max_samples_per_dataset}, keys={list(scenario.preview.dataset_keys)}",
    ]
    for phase_index, phase in enumerate(scenario.phases, start=1):
        phase_train = _scenario_phase_defaults(scenario.train_defaults, phase.overrides)
        phase_selection = _resolve_phase_selection(scenario.selection, phase)
        lines.append(
            f"- phase_{phase_index} {phase.name}: epochs={phase.min_epochs}-{phase.max_epochs}, patience={phase.patience}, metric={phase_selection.metric_path}({phase_selection.mode}), trunk_lr={phase_train.trunk_lr}, head_lr={phase_train.head_lr}"
        )
    return lines


def _action_config_lines(action: ActionSpec) -> list[str]:
    if action.key in {"3", "4", "5"}:
        return _teacher_action_config_lines(action)
    if action.key == "C":
        return _pv26_action_config_lines()
    return []
