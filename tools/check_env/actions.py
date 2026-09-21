"""Select current CLI inputs and preview the exact command to execute."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Sequence

import yaml
from rich.console import Console
from rich.text import Text

from common.paths import REPO_ROOT
from .scan import scan_run


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class Command:
    title: str
    argv: tuple[str, ...]
    notes: tuple[str, ...] = ()
    training_view: "TrainingViewSpec | None" = None


@dataclass(frozen=True)
class TrainingViewSpec:
    kind: str
    stage: str
    output: Path
    start_step: int
    stop_step: int
    planned_steps: int
    logical_batch_size: int


ACTIONS = (
    ActionSpec("1", "SignalAttr crop 생성", "원본 신호등 라벨 → crop 데이터셋"),
    ActionSpec("C", "PV26 학습", "현재 설정으로 본체의 새 실행"),
    ActionSpec("A", "SignalAttr 학습", "crop 데이터셋으로 상태 분류 학습"),
    ActionSpec("D", "짧은 학습 실행", "지정한 업데이트 수까지 실행 후 저장"),
    ActionSpec("E", "기존 실행 재개", "선택한 실행의 설정과 학습 상태 복원"),
    ActionSpec("K", "기존 가중치로 새 학습", "현재 설정과 새 optimizer로 시작"),
    ActionSpec("F", "PV26 TorchScript 내보내기", "본체 체크포인트 선택"),
    ActionSpec("G", "SignalAttr TorchScript 내보내기", "상태 분류 체크포인트 선택"),
    ActionSpec("P", "이미지 추론", "박스·상태·점열 JSON과 overlay 저장"),
    ActionSpec("L", "실행·평가 결과 보기", "저장된 진행도, 지표와 산출물"),
)


_EDITABLE_SETTINGS = {
    "pv26": (
        ("학습 단계", "stage", str, ("joint", "detector", "roadmark")),
        ("장치", "device", str, None),
        ("정밀도", "amp_dtype", str, ("bfloat16", "float16", "float32")),
        ("업데이트 배치", "logical_batch_size", int, None),
        ("GPU 1회 배치", "microbatch_size", int, None),
        ("전체 step", "max_steps", int, None),
        ("Backbone 학습률", "backbone_lr", float, None),
        ("Head 학습률", "head_lr", float, None),
        ("Weight decay", "weight_decay", float, None),
        ("Gradient clip", "grad_clip_norm", float, None),
        ("검증 주기", "validation_every", int, None),
    ),
    "signal_attr": (
        ("장치", "device", str, None),
        ("정밀도", "precision", str, ("bf16", "fp16", "fp32")),
        ("업데이트 배치", "logical_batch_size", int, None),
        ("GPU 1회 배치", "microbatch_size", int, None),
        ("전체 step", "max_steps", int, None),
        ("학습률", "learning_rate", float, None),
        ("Weight decay", "weight_decay", float, None),
        ("좌회전 loss 가중치", "arrow_loss_weight", float, None),
        ("검증 주기", "validation_every", int, None),
        ("검증 표본", "validation_samples", int, None),
        ("표본 선택", "sampling", str, ("balanced", "natural")),
    ),
}


class Cancelled(Exception):
    pass


def ask(console: Console, prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = console.input(Text(f"{prompt}{suffix} > ")).strip()
    if value.upper() == "B":
        raise Cancelled
    return value or (default if default is not None else "")


def ask_path(console: Console, prompt: str, *, default: str | Path | None = None,
             kind: str = "output", optional: bool = False) -> Path | None:
    while True:
        value = ask(console, prompt, default=str(default) if default is not None else None)
        if not value:
            if optional:
                return None
            raise Cancelled
        if value[:1] in ("'", '"') and value[-1:] == value[:1]:
            value = value[1:-1]
        path = Path(value).expanduser()
        path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
        if kind == "file" and not path.is_file():
            console.print(Text(f"파일을 찾을 수 없습니다: {path}", style="yellow"))
        elif kind == "directory" and not path.is_dir():
            console.print(Text(f"디렉터리를 찾을 수 없습니다: {path}", style="yellow"))
        else:
            return path


def ask_count(console: Console, prompt: str, *, default: int | None = None) -> int | None:
    while True:
        value = ask(console, prompt, default=str(default) if default is not None else None)
        if not value:
            return None
        try:
            count = int(value)
            if count > 0:
                return count
        except ValueError:
            pass
        console.print("양의 정수를 입력하세요. B는 이전 화면입니다.", style="yellow")


def choose(console: Console, title: str, items: Sequence[Any], label: Callable[[Any], str],
           *, manual: bool = False, optional: bool = False) -> Any:
    console.print(Text(title, style="bold cyan"))
    for index, item in enumerate(items, 1):
        console.print(Text(f"  {index}. {label(item)}"))
    if manual:
        console.print("  M. 경로 직접 입력")
    if optional:
        console.print("  0. 생략")
    while True:
        value = ask(console, "번호 선택 (Enter/B 취소)").upper()
        if not value:
            raise Cancelled
        if value == "M" and manual:
            return "manual"
        if value == "0" and optional:
            return None
        if value.isdigit() and 1 <= int(value) <= len(items):
            return items[int(value) - 1]
        console.print("표시된 번호를 입력하세요.", style="yellow")


def _write_yaml(path: Path, document: dict) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = stream.name
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def edit_training_settings(console: Console, snapshot: dict) -> str:
    kind = choose(console, "수정할 모델", ("pv26", "signal_attr"),
                  lambda value: "PV26" if value == "pv26" else "SignalAttr")
    config_key = "config" if kind == "pv26" else "signal_config"
    path_key = "config_path" if kind == "pv26" else "signal_config_path"
    document = snapshot[config_key]
    train = document.get("train") if isinstance(document, dict) else None
    if not isinstance(train, dict):
        raise ValueError("학습 설정을 읽지 못했습니다.")
    fields = _EDITABLE_SETTINGS[kind]
    field = choose(console, "수정할 설정", fields,
                   lambda item: f"{item[0]}  {train.get(item[1], '미설정')}")
    label, key, value_type, choices = field
    if choices is not None:
        value = choose(console, label, choices, str)
    else:
        raw = ask(console, label, default=str(train.get(key, "")))
        try:
            value = value_type(raw)
        except ValueError as exc:
            raise ValueError(f"{label} 값을 해석할 수 없습니다: {raw}") from exc
        positive = {"logical_batch_size", "microbatch_size", "max_steps",
                    "backbone_lr", "head_lr", "learning_rate", "validation_samples"}
        nonnegative = {"weight_decay", "grad_clip_norm", "arrow_loss_weight",
                       "validation_every"}
        if key in positive and value <= 0:
            raise ValueError(f"{label} 값은 0보다 커야 합니다.")
        if key in nonnegative and value < 0:
            raise ValueError(f"{label} 값은 0 이상이어야 합니다.")
    old = train.get(key)
    console.print(Text(f"{label}: {old} → {value}"))
    if ask(console, "저장할까요? (y/N)").lower() not in ("y", "yes"):
        raise Cancelled
    train[key] = value
    _write_yaml(Path(snapshot[path_key]), document)
    return f"{label}을(를) {value}(으)로 저장했습니다."


def select_run(console: Console, snapshot: dict, *, kind: str | None = None,
               resumable: bool = False) -> dict:
    runs = [run for run in snapshot["runs"] if kind is None or run["kind"] == kind]
    if resumable:
        runs = [run for run in runs if not run["running"] and not (
            run["step"] is not None and run["max_steps"] is not None
            and run["step"] >= run["max_steps"])]
    selected = choose(console, "실행 선택", runs,
        lambda run: f"{run['kind']} | {run['state']} | {run['path']}", manual=True)
    if selected == "manual":
        selected = scan_run(ask_path(console, "실행 폴더", kind="directory"))
    if kind is not None and selected["kind"] != kind:
        raise ValueError(f"{kind} 실행 폴더를 선택하세요.")
    if resumable and selected["running"]:
        raise ValueError("현재 다른 프로세스가 이 실행을 사용하고 있습니다.")
    return selected


def _checkpoint(console: Console, snapshot: dict, kind: str, *, optional: bool = False) -> Path | None:
    candidates = []
    roles = ("published",) if kind == "signal_attr" else ("best", "latest", "previous")
    for run in snapshot["runs"]:
        if run["kind"] != kind:
            continue
        for role in roles:
            path = run["checkpoints"].get(role)
            if path:
                candidates.append((Path(path), role, run))
    selected = choose(console, f"{kind} 체크포인트", candidates,
        lambda item: f"{item[1]} | {item[0]}", manual=True, optional=optional)
    if selected is None:
        return None
    if selected == "manual":
        return ask_path(console, "체크포인트 파일", kind="file")
    return selected[0]


def _crop_dataset(console: Console, snapshot: dict) -> Path:
    choices = [item for item in snapshot["crop_datasets"] if item["usable"]]
    selected = choose(console, "SignalAttr crop 데이터셋", choices,
        lambda item: f"train {item['train_count']} / val {item['val_count']} | {item['path']}", manual=True)
    return ask_path(console, "crop 데이터셋 폴더", kind="directory") if selected == "manual" else Path(selected["path"])


def _settings(snapshot: dict, kind: str) -> dict:
    config = snapshot["config" if kind == "pv26" else "signal_config"]
    if not config:
        raise ValueError("설정 파일을 읽지 못했습니다. S 메뉴에서 파일 경로를 선택하세요.")
    return config


def _root(snapshot: dict, kind: str) -> Path:
    path = Path(_settings(snapshot, kind)["train"]["output_root"]).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _new_output(console: Console, snapshot: dict, kind: str, suffix: str) -> Path:
    proposed = _root(snapshot, kind) / f"{datetime.now():%Y%m%d_%H%M%S}_{suffix}"
    path = ask_path(console, "새 출력 폴더", default=proposed)
    if path.exists() and any(path.iterdir()):
        raise ValueError("출력 폴더에 파일이 있습니다. 새 폴더를 선택하거나 E 메뉴로 재개하세요.")
    return path


def _argv(script: str, *args: Any) -> tuple[str, ...]:
    return (sys.executable, str(REPO_ROOT / "tools" / script), *(str(arg) for arg in args))


def _train(console: Console, snapshot: dict, kind: str, *, short: bool = False,
           initial: Path | None = None) -> Command:
    settings = _settings(snapshot, kind)
    if kind == "pv26":
        stage = ask(console, "학습 단계 (joint/detector/roadmark)", default=settings["train"]["stage"])
        if stage not in ("joint", "detector", "roadmark"):
            raise ValueError("학습 단계는 joint, detector, roadmark 중 하나입니다.")
        output = _new_output(console, snapshot, kind, stage)
        argv = _argv("run_pv26_train.py", "--config", snapshot["config_path"],
                     "--stage", stage, "--output-dir", output)
    else:
        dataset = _crop_dataset(console, snapshot)
        output = _new_output(console, snapshot, kind, "signal_attr")
        argv = _argv("train_signal_attr.py", "train", "--config", snapshot["signal_config_path"],
                     "--dataset", dataset, "--output-dir", output)
    if initial is not None:
        argv += ("--initial-checkpoint", str(initial))
    added_steps = None
    if short:
        steps = ask_count(console, "업데이트 수", default=2)
        added_steps = steps
        argv += ("--steps", str(steps))
        if kind == "pv26":
            limit = ask_count(console, "출처별 원본 수 제한", default=64)
            argv += ("--sample-limit", str(limit))
    train = settings["train"]
    precision = train.get("amp_dtype", train.get("precision"))
    notes = (f"출력: {output}", f"정밀도: {precision} | 논리/물리 배치: "
             f"{train['logical_batch_size']}/{train['microbatch_size']} | 전체 계획: {train['max_steps']} step")
    if short:
        notes += ("짧은 실행은 전체 학습률 계획을 유지하고 지정한 업데이트 수에서 저장합니다.",)
    planned_steps = int(train["max_steps"])
    stop_step = min(planned_steps, int(added_steps)) if added_steps is not None else planned_steps
    view = TrainingViewSpec(
        kind=kind,
        stage=stage if kind == "pv26" else "signal_attr",
        output=output,
        start_step=0,
        stop_step=stop_step,
        planned_steps=planned_steps,
        logical_batch_size=int(train["logical_batch_size"]),
    )
    return Command(f"{kind} {'짧은 실행' if short else '새 학습'}", argv, notes, view)


def resolve_action(key: str, console: Console, snapshot: dict) -> Command:
    if key == "1":
        _settings(snapshot, "signal_attr")
        policy = choose(console, "모든 램프가 off인 라벨의 처리", ("exclude", "off"),
            lambda value: "상태 학습에서 제외" if value == "exclude" else "소등 상태로 학습")
        output = _new_output(console, snapshot, "signal_attr", "signal_crops")
        limit = ask_count(console, "split별 원본 수 제한 (Enter 전체)")
        argv = _argv("train_signal_attr.py", "prepare", "--config", snapshot["signal_config_path"],
                     "--output-dir", output, "--all-off-policy", policy)
        if limit is not None:
            argv += ("--sample-limit", str(limit))
        return Command("SignalAttr crop 생성", argv, (f"출력: {output}", f"소등 라벨 정책: {policy}"))
    if key in ("C", "A"):
        return _train(console, snapshot, "pv26" if key == "C" else "signal_attr")
    if key in ("D", "K"):
        kind = choose(console, "학습 대상", ("pv26", "signal_attr"), str)
        initial = _checkpoint(console, snapshot, kind) if key == "K" else None
        return _train(console, snapshot, kind, short=key == "D", initial=initial)
    if key == "E":
        run = select_run(console, snapshot, resumable=True)
        script = "run_pv26_train.py" if run["kind"] == "pv26" else "train_signal_attr.py"
        prefix = () if run["kind"] == "pv26" else ("train",)
        argv = _argv(script, *prefix, "--resume-run", run["path"])
        steps = ask_count(console, "추가 업데이트 수 (Enter 저장된 계획 끝까지)")
        micro = ask_count(console, "물리 배치 변경 (Enter 저장된 값)")
        if steps is not None:
            argv += ("--steps", str(steps))
        if micro is not None:
            argv += ("--microbatch-size", str(micro))
        saved = run.get("config") or {}
        train = saved.get("train") if run["kind"] == "pv26" else saved
        train = train if isinstance(train, dict) else {}
        start_step = int(run.get("step") or 0)
        planned_steps = int(run.get("max_steps") or train.get("max_steps") or start_step)
        stop_step = min(planned_steps, start_step + steps) if steps is not None else planned_steps
        view = TrainingViewSpec(
            kind=str(run["kind"]),
            stage=str(run.get("stage") or ("signal_attr" if run["kind"] == "signal_attr" else "?")),
            output=Path(run["path"]),
            start_step=start_step,
            stop_step=stop_step,
            planned_steps=planned_steps,
            logical_batch_size=int(train.get("logical_batch_size") or 1),
        )
        return Command(
            "기존 실행 재개", argv,
            (f"실행: {run['path']}", "선택한 실행의 설정, optimizer와 데이터 위치를 복구합니다."),
            view,
        )
    if key in ("F", "G"):
        kind = "pv26" if key == "F" else "signal_attr"
        checkpoint = _checkpoint(console, snapshot, kind)
        output_root = _root(snapshot, kind)
        default = checkpoint.with_suffix(".torchscript.pt")
        if not default.is_relative_to(output_root):
            default = output_root / "exports" / f"{kind}_{checkpoint.stem}.torchscript.pt"
        output = ask_path(console, "TorchScript 출력 파일", default=default)
        script = "export_pv26_torchscript.py" if kind == "pv26" else "export_signal_attr_torchscript.py"
        argv = _argv(script, "--checkpoint", checkpoint, "--output", output)
        if kind == "pv26":
            argv += ("--device", str(_settings(snapshot, kind)["train"]["device"]))
        notes = (f"출력: {output}",)
        if output.exists() or output.with_suffix(".meta.json").exists():
            argv += ("--overwrite",)
            notes += ("기존 TorchScript와 metadata를 이 출력 경로에서 교체합니다.",)
        return Command(f"{kind} TorchScript 내보내기", argv, notes)
    if key == "P":
        checkpoint = _checkpoint(console, snapshot, "pv26")
        signal = _checkpoint(console, snapshot, "signal_attr", optional=True)
        images = []
        while True:
            path = ask_path(console, f"영상 {len(images) + 1} 경로 (Enter 입력 완료)",
                            kind="file", optional=True)
            if path is None:
                break
            images.append(path)
        if not images:
            raise Cancelled
        output = _new_output(console, snapshot, "pv26", "inference")
        argv = _argv("predict_pv26.py", "--checkpoint", checkpoint, "--images", *images,
                     "--device", _settings(snapshot, "pv26")["train"]["device"],
                     "--output", output / "observations.json", "--overlay", output / "overlays")
        if signal is not None:
            argv += ("--signal-checkpoint", str(signal))
        return Command("이미지 추론", argv, (f"출력: {output}", f"영상 {len(images)}장 | 상태 판독 {'포함' if signal else '생략'}"))
    raise ValueError(f"지원하지 않는 메뉴: {key}")
