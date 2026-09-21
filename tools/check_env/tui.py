"""Rich rendering for the YOLOPV26 training status hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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


def _exists(value: Any) -> str:
    if value is None:
        return "미확인"
    return "있음" if value else "없음"


def _scope(value: Any) -> str:
    return {"pv26": "PV26 본체", "signal_attr": "SignalAttr"}.get(str(value), str(value or "미확인"))


def _progress(step: Any, maximum: Any) -> str:
    return f"{step if step is not None else '?'} / {maximum if maximum is not None else '?'}"


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
        ("표본 선택 / 모델 선택", f"{train.get('sampling', 'natural')} / {train.get('selection_metric', 'combo_accuracy')}"),
        ("초기 가중치", train.get("initial_checkpoint")),
        ("원본 경로 / Crop", f"{prepare.get('raw_root', '?')} / {crop.get('input_size', '?')}px"),
        ("Crop 데이터", train.get("dataset")),
    ]


def _settings_panel(config: Any, path: Any, kind: str, title: str) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="cyan", width=20, no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("설정 파일", _text(path))
    for label, value in _setting_rows(config, kind):
        table.add_row(label, _text(value))
    return Panel(table, title=title, border_style="blue")


def _paths_panel(rows: Sequence[tuple[str, Any]], title: str) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="cyan", width=20, no_wrap=True)
    table.add_column(overflow="fold")
    for label, path in rows:
        table.add_row(label, _text(path))
    return Panel(table, title=title, border_style="green")


def _brief_settings(config: Any, signal_config: Any) -> Table:
    pv = config if isinstance(config, Mapping) else {}
    pv_train = pv.get("train") if isinstance(pv.get("train"), Mapping) else {}
    signal = signal_config if isinstance(signal_config, Mapping) else {}
    signal_train = signal.get("train") if isinstance(signal.get("train"), Mapping) else {}
    table = Table(box=box.SIMPLE, expand=True, pad_edge=False)
    for heading, ratio in (("모델", 2), ("단계", 2), ("정밀도", 2), ("논리/물리", 2), ("목표 step", 2), ("표본 선택", 2)):
        table.add_column(heading, ratio=ratio, overflow="ellipsis")
    table.add_row(
        Text("PV26"), _text(pv_train.get("stage")), _text(pv_train.get("amp_dtype")),
        Text(f"{pv_train.get('logical_batch_size', '?')}/{pv_train.get('microbatch_size', '?')}"),
        _text(pv_train.get("max_steps")), Text("원본 비율"),
    )
    table.add_row(
        Text("SignalAttr"), Text("Crop 판독"), _text(signal_train.get("precision")),
        Text(f"{signal_train.get('logical_batch_size', '?')}/{signal_train.get('microbatch_size', '?')}"),
        _text(signal_train.get("max_steps")), _text(signal_train.get("sampling")),
    )
    return table


def _brief_environment(environment: Any, storage: Any) -> Table:
    env = environment if isinstance(environment, Mapping) else {}
    versions = env.get("versions") if isinstance(env.get("versions"), Mapping) else {}
    gpus = env.get("gpus") or []
    gpu = gpus[0] if gpus and isinstance(gpus[0], Mapping) else None
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="cyan", width=12, no_wrap=True)
    table.add_column(overflow="fold")
    if gpu is not None:
        extra = f" 외 {len(gpus) - 1}개" if len(gpus) > 1 else ""
        table.add_row("GPU", _text(f"{gpu.get('name', '?')}{extra} · VRAM {gpu.get('free_mb', '?')}/{gpu.get('total_mb', '?')} MiB"))
    else:
        table.add_row("GPU", _text(env.get("error"), missing="정보 없음"))
    table.add_row("라이브러리", _text(
        f"torch {versions.get('torch') or '?'} · ultralytics {versions.get('ultralytics') or '?'}"
    ))
    stores = [item for item in (storage or []) if isinstance(item, Mapping)]
    if stores:
        table.add_row("저장소", _text(" · ".join(
            f"{Path(str(item.get('root') or '?')).name}: {_bytes(item.get('free_bytes'))} 여유"
            for item in stores
        )))
    return table


def _brief_sources(sources: Any) -> Table:
    table = Table(box=box.SIMPLE, expand=True, pad_edge=False)
    table.add_column("원본", width=18, no_wrap=True)
    table.add_column("종류", width=10)
    table.add_column("비율", width=5)
    table.add_column("학습/검증", width=12, no_wrap=True)
    table.add_column("경로 이름", ratio=1, overflow="ellipsis")
    for source in sources or []:
        if isinstance(source, Mapping):
            root_name = Path(str(source.get("root") or "")).name.split(" 인지", 1)[0]
            table.add_row(
                _text(source.get("name")), _text(source.get("kind")), _text(source.get("weight")),
                Text(f"{_exists(source.get('train_exists'))}/{_exists(source.get('val_exists'))}"),
                _text(root_name),
            )
    if table.row_count == 0:
        table.add_row(Text("등록된 원본 없음"), Text(""), Text(""), Text(""), Text(""))
    return table


def _brief_runs(runs: Any, storage: Any) -> Table:
    roots = [Path(str(item["root"])) for item in (storage or [])
             if isinstance(item, Mapping) and item.get("root")]
    table = Table(box=box.SIMPLE, expand=True, pad_edge=False)
    table.add_column("최근 실행 · L 상세보기", ratio=4, overflow="ellipsis")
    table.add_column("종류 / 단계", ratio=2)
    table.add_column("진행", ratio=2)
    table.add_column("상태", ratio=2)
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
        if len(display) > 29:
            display = "…" + display[-28:]
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
    console.print(Text("YOLOPV26 · 학습 상태", style="bold bright_blue"))
    console.print(Text("현재 설정", style="bold cyan"))
    console.print(_brief_settings(snapshot.get("config"), snapshot.get("signal_config")))
    console.print(_brief_environment(snapshot.get("environment"), snapshot.get("storage")))
    weights = snapshot.get("weights") or []
    missing_weights = [str(item.get("role") or item.get("path")) for item in weights
                       if isinstance(item, Mapping) and item.get("exists") is False]
    if missing_weights:
        console.print(Text("기본 가중치 없음: " + ", ".join(missing_weights), style="yellow"))
    console.print(Text("데이터 원본", style="bold green"))
    console.print(_brief_sources(snapshot.get("sources")))
    crops = snapshot.get("crop_datasets") or []
    usable_count = sum(item.get("usable") is True for item in crops if isinstance(item, Mapping))
    console.print(Text(f"SignalAttr Crop {len(crops)}개 · 라벨·설정 파일 확인 {usable_count}개 · 학습 선택은 A 메뉴"))
    console.print(Text("최근 실행", style="bold magenta"))
    console.print(_brief_runs(snapshot.get("runs"), snapshot.get("storage")))
    errors = snapshot.get("errors") or []
    if errors:
        console.print(Text(f"확인 필요 {len(errors)}건 · H 도움말에서 자세히 보기", style="yellow"))
    console.print(Text("메뉴", style="bold cyan"))
    menu = Table.grid(expand=True, padding=(0, 2))
    menu.add_column(ratio=1)
    menu.add_column(ratio=1)
    for start in range(0, len(actions), 2):
        pair = actions[start:start + 2]
        values = [Text(f"{getattr(action, 'key', '?')}  {getattr(action, 'label', '')}") for action in pair]
        menu.add_row(*values, *([Text("")] if len(values) == 1 else []))
    console.print(menu)
    console.print(Text("H 도움말   R 새로고침   S 설정 파일 경로   Q 종료", style="dim"))


def _metrics_table(validation: Mapping[str, Any], kind: str, stage: str | None) -> Table:
    table = Table(box=box.SIMPLE, expand=True, show_header=True, pad_edge=False)
    table.add_column("대상", ratio=2)
    table.add_column("TP / FP / FN", ratio=1)
    table.add_column("Precision", ratio=1)
    table.add_column("Recall", ratio=1)
    table.add_column("F1", ratio=1)

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
    info.add_column(style="cyan", width=20, no_wrap=True)
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
    console.print(Panel(info, title="실행 정보", border_style="bright_blue"))

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
                            title="검증 결과", border_style="yellow"))
    else:
        highlights = Table.grid(expand=True, padding=(0, 1))
        highlights.add_column(style="cyan", width=20, no_wrap=True)
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
                            title="검증 결과", border_style="green"))

    artifacts = Table.grid(expand=True, padding=(0, 1))
    artifacts.add_column(style="cyan", width=20, no_wrap=True)
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
    console.print(Panel(artifacts, title="체크포인트와 배포 파일", border_style="magenta"))


def render_help(console: Console, snapshot: dict) -> None:
    """Explain current data ownership and how to read status without inventing readiness gates."""
    lines = [
        "PV26 본체에는 차량용·보행자용 신호등 검출과 흰색·노란색 차선·정지선 출력이 있습니다.",
        "SignalAttr는 검출된 신호등 Crop에서 색상과 좌회전을 읽는 별도 모델입니다.",
        "SignalAttr Crop 생성에서는 소등 표본을 포함할지 명시합니다. 판독 불가는 소등 정답이 아닙니다.",
        "신호등 원본은 검출 감독, 차선 원본은 도로표식 감독에 사용합니다. 표시되지 않은 라벨을 음성으로 단정하지 않습니다.",
        "원본의 학습/검증 표시는 디렉터리 존재 여부만 확인합니다. Crop 개수는 manifest에 기록된 값입니다.",
        "현재 설정은 다음 실행의 기본값입니다. 실행 상세의 저장된 설정은 해당 실행이 실제 사용한 값입니다.",
        "Precision/Recall/F1은 검증 표본 기준이며, 정보가 없다는 표시는 정확도 실패를 뜻하지 않습니다.",
        "L에서 실행·평가 결과를 열어 저장된 지표와 파일 경로를 확인하세요. 메인 실행표의 행은 메뉴 번호가 아닙니다.",
        "YAML에서 수치를 편집합니다. S는 설정 파일 경로만 바꾸며, 수치를 직접 수정하지 않습니다.",
        "1은 SignalAttr Crop 생성, C는 PV26 학습, A는 SignalAttr 학습입니다.",
        "D는 짧은 실행, E는 저장된 설정 그대로 재개, K는 기존 가중치로 현재 설정의 새 실행을 시작합니다.",
        "F/G는 각각 PV26/SignalAttr TorchScript 내보내기, P는 이미지 추론, L은 실행 상세입니다.",
        "실행 디렉터리 직접 입력은 M, 도움말은 H, 새로고침은 R, 종료는 Q입니다.",
        "질문에서는 B로 뒤로 가고, 최종 명령을 확인한 뒤 y로 실행합니다.",
        "자식 학습 실행에서 Ctrl+C는 정상 종료를 요청합니다. 종료·저장이 끝나면 이 화면으로 돌아옵니다.",
    ]
    body = Text("\n\n".join(lines))
    console.print(Panel(body, title="화면 도움말", border_style="cyan"))
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
