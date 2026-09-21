"""Live terminal view for a foreground training subprocess."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .actions import TrainingViewSpec


_DATA_LINE = re.compile(r"data: train=(\d+), val=(\d+), run=(.+)")
_SPARK = "▁▂▃▄▅▆▇█"


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _metric(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _sparkline(values: deque[float]) -> str:
    if not values:
        return ""
    points = list(values)
    low, high = min(points), max(points)
    if high <= low:
        return _SPARK[len(_SPARK) // 2] * len(points)
    return "".join(_SPARK[min(len(_SPARK) - 1, int((value - low) / (high - low) * len(_SPARK)))]
                   for value in points)


@dataclass
class TrainingViewState:
    spec: TrainingViewSpec
    started_at: float = field(default_factory=time.monotonic)
    step: int = field(init=False)
    sampler_position: int | None = None
    microbatch_size: int | None = None
    skipped_updates: int = 0
    oom_retries: int = 0
    training_elapsed_sec: float | None = None
    samples_per_sec: float | None = None
    batch_wait_sec: float | None = None
    update_wall_sec: float | None = None
    batch_wait_history: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    update_wall_history: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    train_count: int | None = None
    val_count: int | None = None
    losses: dict[str, float] = field(default_factory=dict)
    loss_history: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=36)))
    validation: dict[str, Any] | None = None
    validation_step: int | None = None
    status: str = "데이터 인덱싱과 모델 초기화 중"
    messages: deque[str] = field(default_factory=lambda: deque(maxlen=4))
    raw_tail: deque[str] = field(default_factory=lambda: deque(maxlen=30))
    gpu: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step = self.spec.start_step

    def consume(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        self.raw_tail.append(line)
        match = _DATA_LINE.fullmatch(line)
        if match:
            self.train_count, self.val_count = int(match.group(1)), int(match.group(2))
            self.status = "학습 배치 준비 중"
            self.messages.append(f"데이터 train {self.train_count:,} / val {self.val_count:,}")
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if line.startswith("Overriding model.yaml"):
                self.status = "모델 초기화 중"
                self.messages.append("검출 클래스를 PV26 2종으로 구성")
            elif not line.startswith(("{", "}", '"')):
                self.messages.append(line)
            return
        if not isinstance(payload, dict):
            return
        validation = payload.get("validation")
        if isinstance(validation, dict):
            self.validation = validation
            self.validation_step = int(validation.get("global_step") or payload.get("step") or self.step)
            self.status = f"step {self.validation_step:,} 검증 완료"
        if payload.get("global_step") is None or not isinstance(payload.get("losses"), dict):
            return
        self.step = int(payload["global_step"])
        self.sampler_position = int(payload.get("sampler_position") or 0)
        self.microbatch_size = int(payload.get("microbatch_size") or 0)
        self.skipped_updates = int(payload.get("skipped_updates") or 0)
        self.oom_retries = int(payload.get("oom_retries") or 0)
        self.training_elapsed_sec = float(payload["elapsed_sec"]) if payload.get("elapsed_sec") is not None else None
        self.samples_per_sec = float(payload["samples_per_sec"]) if payload.get("samples_per_sec") is not None else None
        self.batch_wait_sec = float(payload["batch_wait_sec"]) if payload.get("batch_wait_sec") is not None else None
        self.update_wall_sec = float(payload["update_wall_sec"]) if payload.get("update_wall_sec") is not None else None
        if self.batch_wait_sec is not None:
            self.batch_wait_history.append(self.batch_wait_sec)
        if self.update_wall_sec is not None:
            self.update_wall_history.append(self.update_wall_sec)
        self.losses = {str(name): float(value) for name, value in payload["losses"].items()}
        for name, value in self.losses.items():
            self.loss_history[name].append(value)
        self.status = "학습 중"


def _gpu_snapshot() -> dict[str, float]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        fields = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(("utilization", "memory_used", "memory_total", "power"), map(float, fields)))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def _loss_panel(state: TrainingViewState) -> Panel:
    table = Table(box=None, expand=True, padding=(0, 1), show_header=True, header_style="dim")
    table.add_column("loss", no_wrap=True)
    table.add_column("현재", justify="right", no_wrap=True)
    table.add_column("최근 기록", ratio=1, overflow="crop")
    preferred = ("det", "roadmark_bce", "roadmark_dice", "base_color", "left_arrow", "total")
    names = [name for name in preferred if name in state.losses]
    names.extend(name for name in state.losses if name not in names)
    for name in names:
        table.add_row(name, _metric(state.losses.get(name)), _sparkline(state.loss_history[name]))
    if not names:
        table.add_row("준비 중", "-", "")
    return Panel(table, title="학습 loss", title_align="left", border_style="dim")


def _validation_panel(state: TrainingViewState) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    validation = state.validation or {}
    if validation.get("elapsed_sec") is not None:
        table.add_row("검증 시간", _duration(float(validation["elapsed_sec"])))
    if state.spec.kind == "pv26":
        loss = validation.get("loss") or {}
        if isinstance(loss, dict) and loss:
            table.add_row("val loss", _metric(sum(float(value) for value in loss.values())))
        signal = validation.get("signal_detection_total") or {}
        road = validation.get("roadmark_lines_total") or {}
        pixels = validation.get("roadmark_pixels_total") or {}
        table.add_row("신호등 F1", _metric(signal.get("f1")))
        table.add_row("도로표식 선 F1", _metric(road.get("f1")))
        table.add_row("도로표식 픽셀 F1", _metric(pixels.get("f1")))
    else:
        table.add_row("Macro 상태 F1", _metric(validation.get("macro_state_f1")))
        table.add_row("유효 판독", _metric(validation.get("valid_coverage")))
        table.add_row("조합 정확도", _metric(validation.get("combo_accuracy")))
    if not validation:
        table.add_row("상태", "첫 검증 대기")
    title = "검증" + (f" · step {state.validation_step:,}" if state.validation_step is not None else "")
    return Panel(table, title=title, title_align="left", border_style="dim")


def render_training_view(state: TrainingViewState) -> Group:
    spec = state.spec
    elapsed = time.monotonic() - state.started_at
    remaining = max(0, spec.stop_step - state.step)
    eta = None
    if state.samples_per_sec and state.samples_per_sec > 0:
        eta = remaining * spec.logical_batch_size / state.samples_per_sec
    progress = ProgressBar(total=max(1, spec.stop_step), completed=min(state.step, spec.stop_step), width=None)
    header = Table.grid(expand=True, padding=(0, 1))
    header.add_column(style="dim", no_wrap=True)
    header.add_column(overflow="fold")
    header.add_column(style="dim", no_wrap=True)
    header.add_column(justify="right", no_wrap=True)
    header.add_row("실행", f"{spec.kind} / {spec.stage}", "상태", state.status)
    header.add_row("경로", str(spec.output), "목표", f"{state.step:,} / {spec.stop_step:,} step")
    header.add_row("진행", progress, "경과 / ETA", f"{_duration(state.training_elapsed_sec or elapsed)} / {_duration(eta)}")
    speed = f"{state.samples_per_sec:.1f} img/s" if state.samples_per_sec is not None else "측정 중"
    micro = state.microbatch_size if state.microbatch_size is not None else "?"
    header.add_row("처리량", speed, "microbatch", str(micro))
    current_total = ((state.batch_wait_sec or 0.0) + (state.update_wall_sec or 0.0)
                     if state.batch_wait_sec is not None or state.update_wall_sec is not None else None)
    recent_wait = (sum(state.batch_wait_history) / len(state.batch_wait_history)
                   if state.batch_wait_history else None)
    recent_update = (sum(state.update_wall_history) / len(state.update_wall_history)
                     if state.update_wall_history else None)
    recent_total = ((recent_wait or 0.0) + (recent_update or 0.0)
                    if recent_wait is not None or recent_update is not None else None)
    current_timing = (f"load {_metric(state.batch_wait_sec, 3)} + update {_metric(state.update_wall_sec, 3)}"
                      f" = {_metric(current_total, 3)}s" if current_total is not None else "측정 중")
    recent_timing = (f"load {_metric(recent_wait, 3)} + update {_metric(recent_update, 3)}"
                     f" = {_metric(recent_total, 3)}s" if recent_total is not None else "측정 중")
    end_to_end = (spec.logical_batch_size / state.samples_per_sec
                  if state.samples_per_sec is not None and state.samples_per_sec > 0 else None)
    header.add_row("이번 step", current_timing, "최근 30 step", recent_timing)
    header.add_row("전체 평균", f"{_metric(end_to_end, 3)}s/step", "논리 배치", str(spec.logical_batch_size))
    header.add_row("표본 위치", f"{state.sampler_position:,}" if state.sampler_position is not None else "-",
                   "OOM / skip", f"{state.oom_retries} / {state.skipped_updates}")
    gpu = state.gpu
    if gpu:
        header.add_row(
            "GPU", f"{gpu['utilization']:.0f}% · {gpu['power']:.0f} W",
            "VRAM", f"{gpu['memory_used']:.0f} / {gpu['memory_total']:.0f} MiB",
        )

    metrics = Table.grid(expand=True, padding=(0, 1))
    metrics.add_column(ratio=3)
    metrics.add_column(ratio=2)
    metrics.add_row(_loss_panel(state), _validation_panel(state))
    messages = Text("\n".join(state.messages) if state.messages else "학습기 출력을 기다리는 중", style="dim")
    footer = Text("Ctrl+C  안전 종료 요청 · 전체 원문 로그: ", style="dim")
    footer.append(str(spec.output / "training.log"))
    return Group(
        Text("YOLOPV26  /  학습 실행", style="bold"),
        Panel(header, border_style="cyan"),
        metrics,
        Panel(messages, title="최근 상태", title_align="left", border_style="dim"),
        footer,
    )


def run_training_view(console: Console, process: subprocess.Popen[str], spec: TrainingViewSpec) -> int:
    """Drain child output while a compact alternate-screen dashboard is visible."""
    if process.stdout is None:
        raise RuntimeError("training view requires captured subprocess output")
    spec.output.mkdir(parents=True, exist_ok=True)
    state = TrainingViewState(spec)
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                lines.put(line.rstrip("\r\n"))
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, name="training-output", daemon=True)
    reader.start()
    ended = False
    last_gpu_at = 0.0
    log_path = spec.output / "training.log"
    original_width = console.width
    available_width = max(original_width, shutil.get_terminal_size().columns)
    console.width = min(available_width, max(original_width, round(original_width * 1.2)))
    try:
        with log_path.open("a", encoding="utf-8") as log, Live(
            render_training_view(state), console=console, screen=console.is_terminal,
            auto_refresh=False, transient=False,
        ) as live:
            while not ended or process.poll() is None:
                while True:
                    try:
                        line = lines.get_nowait()
                    except queue.Empty:
                        break
                    if line is None:
                        ended = True
                        break
                    log.write(line + "\n")
                    log.flush()
                    state.consume(line)
                now = time.monotonic()
                if now - last_gpu_at >= 1.0:
                    state.gpu = _gpu_snapshot()
                    last_gpu_at = now
                if process.poll() is not None and ended:
                    state.status = "완료" if process.returncode == 0 else f"실패 · 종료 코드 {process.returncode}"
                live.update(render_training_view(state), refresh=True)
                if process.poll() is not None and ended:
                    break
                time.sleep(0.2)
    finally:
        console.width = original_width
    reader.join(timeout=2)
    result = int(process.wait())
    if result != 0:
        console.print(Panel(Text("\n".join(state.raw_tail) or "오류 출력 없음"),
                            title="학습 오류 · 마지막 원문", border_style="red"))
    return result


__all__ = ["TrainingViewState", "render_training_view", "run_training_view"]
