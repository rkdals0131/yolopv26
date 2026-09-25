"""Summarize seeded PV26 DoE cells and their measured interactions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
import os
from pathlib import Path
import site
import statistics
import tempfile

site.addsitedir(str(Path(__file__).resolve().parents[1]))


RESPONSES = ("lane_macro_f1", "signal_f1", "stop_f1", "line_total_f1")
REFERENCE_CELLS = {
    "lr_3x3": ("m", "m"),
    "optimizer_2x_gradient_3": ("adamw_cosine", "sum"),
    "source_ratio_3x_loss_weight_3": ("equal", "one"),
    "optimizer_recipe_nested": ("adamw_cosine",),
}


def _responses(validation: dict) -> dict[str, float]:
    roadmark = validation["roadmark_lines"]
    return {
        "lane_macro_f1": 0.5 * (roadmark["white_lane"]["f1"]
                                + roadmark["yellow_lane"]["f1"]),
        "signal_f1": validation["signal_detection_total"]["f1"],
        "stop_f1": roadmark["stop_line"]["f1"],
        "line_total_f1": validation["roadmark_lines_total"]["f1"],
    }


def analyze(run_dir: Path, resource_images: int) -> str:
    history = json.loads((run_dir / "search_results.json").read_text(encoding="utf-8"))
    design = history["experiment"]["design"]
    factor_names = tuple(design["factors"])
    levels = tuple(tuple(values) for values in design["factors"].values())
    seeds = tuple(int(value) for value in design["seeds"])
    rung = next((item for item in history["rungs"]
                 if int(item["resource_images"]) == resource_images), None)
    if rung is None:
        raise ValueError(f"resource rung has no saved results: {resource_images}")
    cells: dict[tuple, dict[int, dict[str, float]]] = defaultdict(dict)
    failures = []
    for result in rung["results"]:
        if result["status"] != "complete":
            failures.append(result["name"])
            continue
        cell = tuple(result["factors"][name] for name in factor_names)
        seed = int(result["seed"])
        if seed in cells[cell]:
            raise ValueError(f"duplicate DoE cell/seed: {cell}, {seed}")
        validation = json.loads((Path(result["run_dir"]) / "evaluation"
                                 / f"common_{resource_images}.json").read_text(encoding="utf-8"))
        cells[cell][seed] = _responses(validation)

    expected = set(product(*levels, seeds))
    observed = {(cell + (seed,)) for cell, by_seed in cells.items() for seed in by_seed}
    missing = sorted(expected - observed)
    lines = [f"# PV26 DoE: {design['block']} · {resource_images:,} image draws",
             "", f"반복 seed: {', '.join(map(str, seeds))}.",
             f"완료한 cell×seed: {len(observed)}/{len(expected)}.",
             f"실패 후보: {', '.join(failures) if failures else '없음'}.", ""]
    if missing:
        lines.extend(["교차 효과는 균형 잡힌 모든 cell×seed가 끝난 후 계산한다.",
                      f"누락 cell×seed: {missing}", ""])
    lines.extend(["lane_macro_f1은 흰 차선과 노란 차선의 F1 평균이다. "
                  "signal_f1과 stop_f1은 별도로 제시한다.",
                  "", "| " + " / ".join(factor_names)
                  + " | 완료 seed | lane macro | signal | stop | line total | paired Δ lane | paired Δ signal |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    reference = cells.get(REFERENCE_CELLS[design["block"]], {})
    for cell in product(*levels):
        by_seed = cells.get(cell, {})
        if not by_seed:
            continue
        means = {name: statistics.fmean(row[name] for row in by_seed.values())
                 for name in RESPONSES}
        paired = {name: [by_seed[seed][name] - reference[seed][name]
                         for seed in by_seed if seed in reference]
                  for name in ("lane_macro_f1", "signal_f1")}
        def delta(values: list[float]) -> str:
            return f"{statistics.fmean(values):+.4f}" if len(values) == len(seeds) else "n/a"
        lines.append("| " + " / ".join(map(str, cell)) + " | "
                     + f"{len(by_seed)} | {means['lane_macro_f1']:.4f} | "
                     + f"{means['signal_f1']:.4f} | {means['stop_f1']:.4f} | "
                     + f"{means['line_total_f1']:.4f} | "
                     + f"{delta(paired['lane_macro_f1'])} | {delta(paired['signal_f1'])} |")

    if not missing and len(factor_names) == 2:
        lines.extend(["", "## 교차 효과", "",
                      "각 수치는 같은 seed와 전체 cell의 균형 잡힌 평균으로 계산한 기술 통계다. "
                      "세 seed만으로 배포 일반화나 유의확률을 주장하지 않는다.", ""])
        for response in RESPONSES:
            grand = statistics.fmean(cells[cell][seed][response]
                                     for cell in product(*levels) for seed in seeds)
            main_first = {value: statistics.fmean(
                cells[(value, other)][seed][response]
                for other in levels[1] for seed in seeds) - grand
                for value in levels[0]}
            main_second = {value: statistics.fmean(
                cells[(other, value)][seed][response]
                for other in levels[0] for seed in seeds) - grand
                for value in levels[1]}
            interaction = {
                cell: statistics.fmean(cells[cell][seed][response] for seed in seeds)
                      - grand - main_first[cell[0]] - main_second[cell[1]]
                for cell in product(*levels)
            }
            lines.extend([f"### {response}", "",
                          f"전체 평균 {grand:.5f}; "
                          f"{factor_names[0]} 주효과 "
                          + ", ".join(f"{value}={effect:+.5f}" for value, effect in main_first.items())
                          + "; " + f"{factor_names[1]} 주효과 "
                          + ", ".join(f"{value}={effect:+.5f}" for value, effect in main_second.items())
                          + ".", "",
                          "| cell | 평균에서 두 주효과를 뺀 상호작용 잔차 |",
                          "| --- | ---: |"])
            for cell, residual in interaction.items():
                lines.append(f"| {' / '.join(map(str, cell))} | {residual:+.5f} |")
            lines.append("")
    elif not missing:
        lines.extend(["", "이 블록은 optimizer 방식의 개별 레시피 비교다. "
                      "동일 LR 의미의 완전 교차실험으로 해석하지 않는다."])
    return "\n".join(lines) + "\n"


def save_report(report: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(report)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resource-images", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    report = analyze(run, args.resource_images)
    destination = args.output or run / f"doe_analysis_{args.resource_images}.md"
    save_report(report, destination)
    print(destination)


if __name__ == "__main__":
    main()
