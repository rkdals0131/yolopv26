from __future__ import annotations

from dataclasses import dataclass
import sys

from tools.model_export import artifact_paths_for_checkpoint
from .scan import PipelinePaths, TEACHER_NAMES, WorkspaceSnapshot
from tools.od_bootstrap.presets import build_teacher_train_preset
from tools.od_bootstrap.presets import build_teacher_eval_preset


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
        ActionSpec("D", "PV26 phase VRAM stress", "interactive: phase/batch/iter 입력 후 short VRAM probe", (), "TUI result panel only (no checkpoints / no run dir)", rerun_contract="short probe only: 선택한 phase train loop 일부만 실행"),
        ActionSpec("E", "PV26 exact resume", "interactive: resumable run 목록에서 골라 same run dir exact resume", (), str(paths.pv26_run_root), rerun_contract="exact resume only: same run dir / same scenario"),
        ActionSpec("F", "PV26 TorchScript export", "interactive: completed PV26 run 선택 후 adjacent TorchScript export", (), str(paths.pv26_run_root)),
        ActionSpec("G", "Mobility teacher TorchScript export", "interactive: stable weights/best.pt -> adjacent TorchScript export", (), str(paths.teacher_train_root / "mobility" / "weights")),
        ActionSpec("I", "Signal teacher TorchScript export", "interactive: stable weights/best.pt -> adjacent TorchScript export", (), str(paths.teacher_train_root / "signal" / "weights")),
        ActionSpec("J", "Obstacle teacher TorchScript export", "interactive: stable weights/best.pt -> adjacent TorchScript export", (), str(paths.teacher_train_root / "obstacle" / "weights")),
        ActionSpec("K", "PV26 retrain / fine-tune", "interactive: source run 선택 후 stage window derived run", (), str(paths.pv26_run_root), rerun_contract="derived run only: source run + selected stage window / current config"),
        ActionSpec("L", "최종 데이터셋 full stats", "interactive: final dataset class/task/audit 통계 표시", (), str(paths.final_dataset_root / "meta")),
        ActionSpec("M", "PV26 phase VRAM ceiling sweep", "interactive: phase 1-4 batch list probe", (), "stdout JSON summary (no checkpoints / no run dir)", rerun_contract="short probe only: phase별 batch 후보를 순차 확인"),
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
    elif action.key in {"C", "D", "M"}:
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 학습에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
        if not flags.get("final_dataset", False):
            blockers.append("최종 병합 데이터셋이 아직 없습니다.")
    elif action.key == "E":
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 학습에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
    elif action.key == "K":
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 retrain/fine-tune에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
        if not flags.get("final_dataset", False):
            blockers.append("최종 병합 데이터셋이 아직 없습니다.")
    elif action.key == "L":
        if not flags.get("final_dataset", False):
            blockers.append("최종 병합 데이터셋이 아직 없습니다.")
    elif action.key == "F":
        if not flags.get("pv26_runtime", False):
            blockers.append("PV26 TorchScript export에 필요한 YOLO26 runtime이 아직 정상 로드되지 않습니다.")
        if not flags.get("pv26_export_available", False):
            blockers.append("export 가능한 completed PV26 run이 없습니다.")
    elif action.key in {"G", "I", "J"}:
        teacher_name = {"G": "mobility", "I": "signal", "J": "obstacle"}[action.key]
        if not flags.get("runtime_core", False):
            blockers.append("teacher TorchScript export에 필요한 torch / ultralytics 환경이 아직 깨져 있습니다.")
        if not flags.get(f"teacher_train.{teacher_name}", False):
            blockers.append(f"{teacher_name} teacher stable checkpoint가 아직 없습니다.")
    return blockers


def _action_advisory(action: ActionSpec, snapshot: WorkspaceSnapshot) -> str | None:
    if action.key == "A" and not snapshot.flags.get("calibration", False):
        return "calibration이 없어서 fallback class policy로 진행될 수 있습니다."
    if action.key == "B":
        return "fixed output root를 직접 덮어쓰지 않고 staging build 후 atomic swap합니다."
    if action.key == "D":
        return "기본은 stage_3지만, 이제 phase별 probe를 실행할 수 있습니다. stage_3가 대체로 VRAM 상한 proxy이고 stage_4는 보통 더 가볍습니다."
    if action.key == "M":
        return "CUDA가 보이는 환경에서만 실행됩니다. 결과의 max_ok_batch_size는 ceiling_observed=false이면 하한값입니다."
    if action.key == "E":
        return "resume는 exact resume only입니다. batch_size 변경이나 best/epoch 재시작은 별도 흐름으로 다루는 편이 안전합니다."
    if action.key == "K":
        return "retrain은 새 derived run을 만듭니다. 숫자 파라미터는 config를 그대로 읽고, launcher에서는 source run과 stage window만 고릅니다."
    if action.key == "L":
        return "stats 파일이 있으면 그대로 읽고, 없으면 final dataset labels_scene를 다시 스캔해 생성합니다."
    if action.key == "F":
        return "선택한 run의 final checkpoint 옆에 best.torchscript.pt / .meta.json을 씁니다."
    if action.key in {"G", "I", "J"}:
        return "teacher별 stable weights/best.pt 옆에 best.torchscript.pt / .meta.json을 씁니다."
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
        f"- runtime: checkpoint_every={train_defaults.checkpoint_every}, accumulate_steps={train_defaults.accumulate_steps}, grad_clip={train_defaults.grad_clip_norm}, amp_init_scale={train_defaults.amp_init_scale}, skip_non_finite={_bool_flag(train_defaults.skip_non_finite_loss)}, oom_guard={_bool_flag(train_defaults.oom_guard)}",
        f"- preview: enabled={_bool_flag(scenario.preview.enabled)}, split={scenario.preview.split}, per_dataset={scenario.preview.max_samples_per_dataset}, keys={list(scenario.preview.dataset_keys)}",
    ]
    for phase_index, phase in enumerate(scenario.phases, start=1):
        phase_train = _scenario_phase_defaults(scenario.train_defaults, phase.overrides)
        phase_selection = _resolve_phase_selection(scenario.selection, phase)
        stop_policy = (
            f"min_delta_abs={float(phase.min_delta_abs):.4f}"
            if phase.min_delta_abs is not None
            else f"min_improvement_pct={float(phase.min_improvement_pct):.3f}"
        )
        lines.append(
            f"- phase_{phase_index} {phase.name}: epochs={phase.min_epochs}-{phase.max_epochs}, patience={phase.patience}, {stop_policy}, metric={phase_selection.metric_path}({phase_selection.mode}), batch={phase_train.batch_size}, trunk_lr={phase_train.trunk_lr}, head_lr={phase_train.head_lr}"
        )
    return lines


def _teacher_export_config_lines(action: ActionSpec) -> list[str]:
    teacher_name = {"G": "mobility", "I": "signal", "J": "obstacle"}[action.key]
    train_scenario = build_teacher_train_preset(teacher_name)
    eval_scenario = build_teacher_eval_preset(teacher_name)
    artifact_path, meta_path = artifact_paths_for_checkpoint(eval_scenario.model.checkpoint_path)
    return [
        f"- teacher={teacher_name}, classes={', '.join(train_scenario.model.class_names)}",
        f"- checkpoint: {eval_scenario.model.checkpoint_path}",
        f"- export: {artifact_path}",
        f"- meta: {meta_path}",
        f"- imgsz={eval_scenario.eval.imgsz}, device=auto, format=torchscript",
    ]


def _action_config_lines(action: ActionSpec) -> list[str]:
    if action.key in {"3", "4", "5"}:
        return _teacher_action_config_lines(action)
    if action.key in {"C", "K"}:
        return _pv26_action_config_lines()
    if action.key in {"G", "I", "J"}:
        return _teacher_export_config_lines(action)
    return []
