"""Rich rendering for the YOLOPV26 training status hub."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Mapping, Sequence

from rich import box
from rich.console import Console, Group, RenderHook
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


class ScreenPadding(RenderHook):
    """Keep screens and input prompts inset from the terminal edge."""

    def process_renderables(self, renderables):
        if len(renderables) == 1 and isinstance(renderables[0], Text) and renderables[0].end == "":
            prompt = Text("  ", end="")
            prompt.append_text(renderables[0])
            return [prompt]
        return [Padding(Group(*renderables), (0, 2))]


def _section(console: Console, title: str) -> None:
    console.print()
    console.print(Text(title, style="bold"))
    console.print(Rule(style="dim"))


def _table() -> Table:
    return Table(box=box.ROUNDED, expand=True, pad_edge=True,
                 collapse_padding=True, padding=(0, 1), border_style="dim",
                 header_style="dim", show_edge=True)


def _keys(items: Sequence[tuple[str, str]]) -> Text:
    text = Text()
    for key, label in items:
        if text:
            text.append("   ")
        text.append(key, style="bold cyan")
        text.append(f" {label}")
    return text


def _text(value: Any, *, missing: str = "미확인") -> Text:
    return Text(str(value) if value is not None and value != "" else missing)


def _number(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "미측정"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _bytes(value: Any) -> str:
    if value is None:
        return "미확인"
    try:
        size = float(value)
    except (TypeError, ValueError):
        return str(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "미확인"


def _scope(value: Any) -> str:
    return {"pv26": "PV26 본체", "signal_attr": "SignalAttr"}.get(str(value), str(value or "미확인"))


def _progress(step: Any, maximum: Any) -> str:
    return f"{step if step is not None else '?'} / {maximum if maximum is not None else '?'}"


def _integer(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "?"


def _batch(train: Mapping[str, Any]) -> str:
    logical = train.get("logical_batch_size")
    micro = train.get("microbatch_size")
    try:
        accumulation = math.ceil(int(logical) / int(micro))
    except (TypeError, ValueError, ZeroDivisionError):
        accumulation = "?"
    return f"{logical or '?'} / {micro or '?'}  ×{accumulation}"


def _compact_name(path: Path) -> str:
    parts = list(path.parts[-2:])
    if parts and len(parts[0]) > 9 and parts[0][:8].isdigit() and parts[0][8] == "_":
        parts[0] = parts[0][9:]
    if len(parts) > 1 and parts[0].endswith("_run"):
        parts[0] = parts[0][:-4]
    return "/".join(parts)


def _source_ratio(sources: Any) -> str:
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return "미확인"
    parts = []
    for source in sources:
        if isinstance(source, Mapping):
            parts.append(f"{source.get('name', source.get('kind', '?'))}:{source.get('weight', '?')}")
    return "  ".join(parts) or "미확인"


def _setting_rows(config: Any, kind: str) -> list[tuple[str, Any]]:
    if not isinstance(config, Mapping):
        return []
    if kind == "pv26":
        train = config.get("train") or {}
        model = config.get("model") or {}
        data = config.get("data") or {}
        return [
            ("단계", train.get("stage")),
            ("장치 / 정밀도", f"{train.get('device', '?')} / {train.get('amp_dtype', '?')}"),
            ("논리 / 물리 배치", f"{train.get('logical_batch_size', '?')} / {train.get('microbatch_size', '?')}"),
            ("목표 업데이트", train.get("max_steps")),
            ("학습률 (본체 / Head)", f"{train.get('backbone_lr', '?')} / {train.get('head_lr', '?')}"),
            ("Weight decay / Grad clip", f"{train.get('weight_decay', '?')} / {train.get('grad_clip_norm', '?')}"),
            ("검증 주기 / 출처당 표본", f"{train.get('validation_every', '?')} / {train.get('validation_samples_per_source', '?')}"),
            ("기본 가중치", model.get("weights")),
            ("본체 / 입력", f"{model.get('variant', '?')} / {model.get('image_hw', '?')}"),
            ("원본 비율", _source_ratio(data.get("sources"))),
        ]
    train = config.get("train", config)
    if not isinstance(train, Mapping):
        return []
    prepare = config.get("prepare")
    prepare = prepare if isinstance(prepare, Mapping) else {}
    crop = prepare.get("crop")
    crop = crop if isinstance(crop, Mapping) else {}
    return [
        ("장치 / 정밀도", f"{train.get('device', '?')} / {train.get('precision', '?')}"),
        ("논리 / 물리 배치", f"{train.get('logical_batch_size', '?')} / {train.get('microbatch_size', '?')}"),
        ("목표 업데이트", train.get("max_steps")),
        ("학습률 / Weight decay", f"{train.get('learning_rate', '?')} / {train.get('weight_decay', '?')}"),
        ("검증 주기 / 표본", f"{train.get('validation_every', '?')} / {train.get('validation_samples', '?')}"),
        ("좌회전 loss 가중치", train.get("arrow_loss_weight")),
        ("표본 선택 / 모델 선택", f"{train.get('sampling', 'natural')} / {train.get('selection_metric', 'combo_accuracy')}"),
        ("초기 가중치", train.get("initial_checkpoint")),
        ("원본 경로 / Crop", f"{prepare.get('raw_root', '?')} / {crop.get('input_size', '?')}px"),
        ("Crop 데이터", train.get("dataset")),
    ]


def _settings_panel(config: Any, path: Any, kind: str, title: str) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="dim", width=20, overflow="fold")
    table.add_column(overflow="fold")
    table.add_row("설정 파일", _text(path))
    for label, value in _setting_rows(config, kind):
        table.add_row(label, _text(value))
    return Panel(table, title=title, title_align="left", box=box.ROUNDED, border_style="dim")


def _paths_panel(rows: Sequence[tuple[str, Any]], title: str) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="dim", width=20, overflow="fold")
    table.add_column(overflow="fold")
    for label, path in rows:
        table.add_row(label, _text(path))
    return Panel(table, title=title, title_align="left", box=box.ROUNDED, border_style="dim")


def _settings_card(title: str, lines: Sequence[tuple[str, Any]]) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in lines:
        table.add_row(label, _text(value))
    return Panel(table, title=title, title_align="left", border_style="dim", expand=True)


def _brief_settings(config: Any, signal_config: Any, *, stacked: bool = False) -> Table:
    pv = config if isinstance(config, Mapping) else {}
    pv_train = pv.get("train") if isinstance(pv.get("train"), Mapping) else {}
    signal = signal_config if isinstance(signal_config, Mapping) else {}
    signal_train = signal.get("train") if isinstance(signal.get("train"), Mapping) else {}
    pv = _settings_card("PV26", (
        ("운용", f"{pv_train.get('stage', '?')} · {pv_train.get('amp_dtype', '?')} · {pv_train.get('device', '?')}"),
        ("배치", f"update / GPU · 누적 = {_batch(pv_train)}"),
        ("학습률", f"backbone {pv_train.get('backbone_lr', '?')} · head {pv_train.get('head_lr', '?')}"),
        ("규제", f"WD {pv_train.get('weight_decay', '?')} · clip {pv_train.get('grad_clip_norm', '?')}"),
        ("계획", f"{_integer(pv_train.get('max_steps'))} step · {_integer(pv_train.get('validation_every'))}마다 검증"
                 f" ({_integer(pv_train.get('validation_samples_per_source'))}/source)"),
    ))
    signal_attr = _settings_card("SignalAttr", (
        ("운용", f"{signal_train.get('sampling', '?')} · {signal_train.get('precision', '?')} · {signal_train.get('device', '?')}"),
        ("배치", f"update / GPU · 누적 = {_batch(signal_train)}"),
        ("학습률", signal_train.get("learning_rate")),
        ("규제", f"WD {signal_train.get('weight_decay', '?')} · arrow {signal_train.get('arrow_loss_weight', '?')}"),
        ("계획", f"{_integer(signal_train.get('max_steps'))} step · {_integer(signal_train.get('validation_every'))}마다 검증"
                 f" ({_integer(signal_train.get('validation_samples'))} samples)"),
    ))
    cards = Table.grid(expand=True, padding=(0, 1))
    if stacked:
        cards.add_column()
        cards.add_row(pv)
        cards.add_row(signal_attr)
    else:
        cards.add_column(ratio=1)
        cards.add_column(ratio=1)
        cards.add_row(pv, signal_attr)
    return cards


def _brief_environment(environment: Any, storage: Any) -> Table:
    env = environment if isinstance(environment, Mapping) else {}
    gpus = env.get("gpus") or []
    gpu = gpus[0] if gpus and isinstance(gpus[0], Mapping) else None
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=10, no_wrap=True)
    table.add_column(overflow="fold")
    if gpu is not None:
        extra = f" 외 {len(gpus) - 1}개" if len(gpus) > 1 else ""
        table.add_row("GPU", _text(f"{gpu.get('name', '?')}{extra} · VRAM {gpu.get('free_mb', '?')}/{gpu.get('total_mb', '?')} MiB"))
    else:
        table.add_row("GPU", _text(env.get("error"), missing="정보 없음"))
    stores = [item for item in (storage or []) if isinstance(item, Mapping)]
    if stores:
        table.add_row("저장소", _text(" · ".join(
            f"{Path(str(item.get('root') or '?')).name}: {_bytes(item.get('free_bytes'))} 여유"
            for item in stores
        )))
    return table


def _brief_data(sources: Any, crops: Any) -> Table:
    table = _table()
    table.add_column("데이터", ratio=2, overflow="fold")
    table.add_column("용도")
    table.add_column("train", justify="right")
    table.add_column("val", justify="right")
    for source in sources or []:
        if isinstance(source, Mapping):
            table.add_row(
                _text(source.get("name")),
                Text({"traffic": "신호등", "roadmark": "도로표식"}.get(
                    str(source.get("kind")), str(source.get("kind") or "?"))),
                Text(_integer(source.get("train_count"))), Text(_integer(source.get("val_count"))),
            )
    for crop in crops or []:
        if not isinstance(crop, Mapping):
            continue
        path = Path(str(crop.get("path") or "?"))
        table.add_row(Text(_compact_name(path)), Text("신호 속성 Crop"),
                      Text(_integer(crop.get("train_count"))),
                      Text(_integer(crop.get("val_count"))))
    if table.row_count == 0:
        table.add_row(Text("등록된 데이터 없음"), Text(""), Text(""), Text(""))
    return table


def _brief_runs(runs: Any, storage: Any) -> Table:
    roots = [Path(str(item["root"])) for item in (storage or [])
             if isinstance(item, Mapping) and item.get("root")]
    table = _table()
    table.add_column("실행", ratio=1, overflow="fold")
    table.add_column("모델 / 단계", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("상태", overflow="fold")
    for run in (runs or [])[:5]:
        if not isinstance(run, Mapping):
            continue
        path = Path(str(run.get("path") or "?"))
        display = str(path)
        for root in roots:
            try:
                display = str(path.relative_to(root))
                break
            except ValueError:
                pass
        display_path = Path(display)
        display = _compact_name(display_path)
        state = str(run.get("state") or ("실행 중" if run.get("running") else "미확인")).split(" · ", 1)[0]
        kind = "SignalAttr" if run.get("kind") == "signal_attr" else f"PV26/{run.get('stage') or '?'}"
        table.add_row(
            Text(display), Text(kind),
            Text(_progress(run.get("step"), run.get("max_steps"))), Text(state),
        )
    if table.row_count == 0:
        table.add_row(Text("최근 실행 없음"), Text(""), Text(""), Text(""))
    return table


def render_dashboard(console: Console, snapshot: dict, actions: tuple[Any, ...]) -> None:
    """Render a compact scanner snapshot without filesystem or model reads."""
    console.print()
    console.print(Text("YOLOPV26  /  학습 상태", style="bold"))
    _section(console, "환경")
    console.print(_brief_environment(snapshot.get("environment"), snapshot.get("storage")))
    weights = snapshot.get("weights") or []
    missing_weights = [str(item.get("role") or item.get("path")) for item in weights
                       if isinstance(item, Mapping) and item.get("exists") is False]
    if missing_weights:
        console.print(Text("기본 가중치 없음: " + ", ".join(missing_weights), style="yellow"))
    _section(console, "다음 실행 설정")
    console.print(_brief_settings(snapshot.get("config"), snapshot.get("signal_config"),
                                  stacked=console.width < 84))
    _section(console, "데이터")
    crops = snapshot.get("crop_datasets") or []
    console.print(_brief_data(snapshot.get("sources"), crops))
    _section(console, "최근 실행")
    console.print(_brief_runs(snapshot.get("runs"), snapshot.get("storage")))
    errors = snapshot.get("errors") or []
    if errors:
        console.print(Text(f"확인 필요 {len(errors)}건 · H 도움말에서 자세히 보기", style="yellow"))
    _section(console, "작업")
    menu = Table.grid(padding=(0, 2))
    menu.add_column(style="dim", no_wrap=True)
    menu.add_column()
    groups = (("준비", "1"), ("학습", "CAD"), ("이어하기", "EK"),
              ("내보내기", "FG"), ("결과", "PL"))
    for label, keys in groups:
        menu.add_row(label, _keys([(action.key, action.label)
                                  for action in actions if action.key in keys]))
    console.print(menu)
    console.print()
    console.print(_keys((("S", "설정 수정"), ("H", "도움말"),
                         ("R", "새로고침"), ("Q", "종료"))))


def _metrics_table(validation: Mapping[str, Any], kind: str, stage: str | None) -> Table:
    table = _table()
    table.add_column("대상", ratio=1, overflow="fold")
    table.add_column("TP / FP / FN", justify="right", no_wrap=True)
    table.add_column("Precision", justify="right", no_wrap=True)
    table.add_column("Recall", justify="right", no_wrap=True)
    table.add_column("F1", justify="right", no_wrap=True)

    def add_scores(label: str, scores: Mapping[str, Any]) -> None:
        counts = [scores.get(name) for name in ("tp", "fp", "fn")]
        observed = all(value is not None for value in counts) and sum(int(value) for value in counts) > 0
        table.add_row(
            Text(label), Text(" / ".join(str(value) if value is not None else "?" for value in counts)),
            Text(_number(scores.get("precision")) if observed else "해당 없음"),
            Text(_number(scores.get("recall")) if observed else "해당 없음"),
            Text(_number(scores.get("f1")) if observed else "해당 없음"),
        )

    if kind == "signal_attr":
        by_type = validation.get("by_light_type") or {}
        for light_type, group in by_type.items():
            if not isinstance(group, Mapping):
                continue
            for state, scores in (group.get("states") or {}).items():
                if isinstance(scores, Mapping):
                    add_scores(f"{light_type} · {state}", scores)
    else:
        if stage != "roadmark":
            scores = validation.get("signal_detection_total")
            if isinstance(scores, Mapping):
                add_scores("신호등 검출 합계", scores)
            for class_name, scores in (validation.get("signal_detection") or {}).items():
                if isinstance(scores, Mapping):
                    add_scores(str(class_name), scores)
        if stage != "detector":
            scores = validation.get("roadmark_lines_total")
            if isinstance(scores, Mapping):
                add_scores("도로표식 점열 합계", scores)
            for class_name, scores in (validation.get("roadmark_lines") or {}).items():
                if isinstance(scores, Mapping):
                    add_scores(str(class_name), scores)
    if table.row_count == 0:
        table.add_row(Text("평가 결과 없음"), Text(""), Text(""), Text(""), Text(""))
    return table


def render_run_details(console: Console, run: dict) -> None:
    """Render saved run information from a scanner row."""
    kind = str(run.get("kind") or "")
    info = Table.grid(expand=True, padding=(0, 1))
    info.add_column(style="dim", width=20, no_wrap=True)
    info.add_column(overflow="fold")
    info.add_row("실행 경로", _text(run.get("path")))
    info.add_row("종류 / 단계", _text(f"{_scope(kind)} / {run.get('stage') or '미확인'}"))
    info.add_row("진행", Text(_progress(run.get("step"), run.get("max_steps"))))
    info.add_row("상태", _text(run.get("state") or ("실행 중" if run.get("running") else None)))
    summary = run.get("summary") or {}
    if isinstance(summary, Mapping):
        for label, key in (("경과 시간", "elapsed_sec"), ("OOM 재시도", "oom_retries"),
                           ("건너뛴 업데이트", "skipped_updates")):
            if summary.get(key) is not None:
                value = f"{_number(summary[key], digits=1)}초" if key == "elapsed_sec" else summary[key]
                info.add_row(label, _text(value))
        if summary.get("peak_allocated_bytes") is not None:
            info.add_row("Peak VRAM", Text(_bytes(summary["peak_allocated_bytes"])))
    console.print(Panel(info, title="실행 정보", title_align="left", border_style="dim"))

    config = run.get("config")
    console.print(_settings_panel(config, "실행 당시 저장된 설정", kind, "저장된 학습 설정"))
    if kind == "pv26" and isinstance(config, Mapping):
        data = config.get("data") if isinstance(config.get("data"), Mapping) else {}
        sources = data.get("sources") or []
        source_paths = [(str(source.get("name") or source.get("kind") or "원본"), source.get("root"))
                        for source in sources if isinstance(source, Mapping)]
        if source_paths:
            console.print(_paths_panel(source_paths, "저장된 원본 경로"))
    validation = run.get("validation")
    if not isinstance(validation, Mapping):
        console.print(Panel(Text("평가 결과 없음 · 아직 평가하지 않았거나 읽을 수 없습니다."),
                            title="검증 결과", title_align="left", border_style="dim"))
    else:
        highlights = Table.grid(padding=(0, 1))
        highlights.add_column(style="dim", width=20, no_wrap=True)
        highlights.add_column(overflow="fold")
        if kind == "signal_attr":
            highlights.add_row("Macro 상태 F1", Text(_number(validation.get("macro_state_f1"))))
            highlights.add_row("유효 판독 비율", Text(_number(validation.get("valid_coverage"))))
            highlights.add_row("조합 정확도", Text(_number(validation.get("combo_accuracy"))))
        else:
            highlights.add_row("평가 표본", _text(validation.get("samples")))
            highlights.add_row("선택 지표", Text(_number(validation.get("selection_metric"))))
            highlights.add_row("도로표식 중심선", Text("흰색·노란색 차선 / 정지선"))
        console.print(Panel(Group(highlights, _metrics_table(validation, kind, run.get("stage"))),
                            title="검증 결과", title_align="left", border_style="dim"))

    artifacts = Table.grid(expand=True, padding=(0, 1))
    artifacts.add_column(style="dim", width=20, no_wrap=True)
    artifacts.add_column(overflow="fold")
    for label, key in (("최근 재개", "latest"), ("이전 재개", "previous"),
                       ("최고 가중치", "best"), ("배포 가중치", "published")):
        path = (run.get("checkpoints") or {}).get(key)
        if path:
            artifacts.add_row(label, _text(path))
    for index, path in enumerate(run.get("exports") or [], start=1):
        artifacts.add_row(f"내보내기 {index}", _text(path))
    if artifacts.row_count == 0:
        artifacts.add_row("산출물", Text("확인된 파일 없음"))
    console.print(Panel(artifacts, title="체크포인트와 배포 파일", title_align="left", border_style="dim"))


def render_help(console: Console, snapshot: dict) -> None:
    """Explain current data ownership and how to read status without inventing readiness gates."""
    lines = [
        "PV26 본체에는 차량용·보행자용 신호등 검출과 흰색·노란색 차선·정지선 출력이 있습니다.",
        "SignalAttr는 검출된 신호등 Crop에서 색상과 좌회전을 읽는 별도 모델입니다.",
        "SignalAttr Crop 생성에서는 소등 표본을 포함할지 명시합니다. 판독 불가는 소등 정답이 아닙니다.",
        "신호등 원본은 검출 감독, 차선 원본은 도로표식 감독에 사용합니다. 표시되지 않은 라벨을 음성으로 단정하지 않습니다.",
        "원본 수량은 학습 로더가 읽는 [라벨] JSON 수입니다. Crop 수량은 각 manifest의 train/val 값입니다.",
        "현재 설정은 다음 실행의 기본값입니다. 실행 상세의 저장된 설정은 해당 실행이 실제 사용한 값입니다.",
        "Precision/Recall/F1은 검증 표본 기준이며, 정보가 없다는 표시는 정확도 실패를 뜻하지 않습니다.",
        "L에서 실행·평가 결과를 열어 저장된 지표와 파일 경로를 확인하세요. 메인 실행표의 행은 메뉴 번호가 아닙니다.",
        "S에서 다음 실행에 사용할 핵심 학습 설정을 수정합니다. 저장된 기존 실행 설정은 바뀌지 않습니다.",
        "1은 SignalAttr Crop 생성, C는 PV26 학습, A는 SignalAttr 학습입니다.",
        "D는 짧은 실행, E는 저장된 설정 그대로 재개, K는 기존 가중치로 현재 설정의 새 실행을 시작합니다.",
        "F/G는 각각 PV26/SignalAttr TorchScript 내보내기, P는 이미지 추론, L은 실행 상세입니다.",
        "실행 디렉터리 직접 입력은 M, 도움말은 H, 새로고침은 R, 종료는 Q입니다.",
        "질문에서는 B로 뒤로 가고, 최종 명령을 확인한 뒤 y로 실행합니다.",
        "자식 학습 실행에서 Ctrl+C는 정상 종료를 요청합니다. 종료·저장이 끝나면 이 화면으로 돌아옵니다.",
    ]
    body = Text("\n\n".join(lines))
    console.print(Panel(body, title="화면 도움말", title_align="left", border_style="dim"))
    console.print(_settings_panel(snapshot.get("config"), snapshot.get("config_path"), "pv26", "현재 PV26 설정"))
    console.print(_settings_panel(snapshot.get("signal_config"), snapshot.get("signal_config_path"),
                                  "signal_attr", "현재 SignalAttr 설정"))
    path_rows: list[tuple[str, Any]] = []
    for source in snapshot.get("sources") or []:
        if isinstance(source, Mapping):
            path_rows.append((str(source.get("name") or source.get("kind") or "원본"), source.get("root")))
    for dataset in snapshot.get("crop_datasets") or []:
        if isinstance(dataset, Mapping):
            path_rows.append(("SignalAttr Crop", dataset.get("path")))
    for store in snapshot.get("storage") or []:
        if isinstance(store, Mapping):
            path_rows.append(("저장소", store.get("root")))
    weights = snapshot.get("weights") or []
    for weight in weights:
        if isinstance(weight, Mapping):
            path_rows.append((str(weight.get("role") or "기본 가중치"), weight.get("path")))
    if path_rows:
        console.print(_paths_panel(path_rows, "현재 데이터·저장 경로"))
    errors = snapshot.get("errors") or []
    if errors:
        console.print(Panel(Group(*[Text(str(error)) for error in errors]),
                            title="확인 필요", border_style="yellow"))


__all__ = ["render_dashboard", "render_run_details", "render_help"]
