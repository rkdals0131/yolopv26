"""Run the fixed PV26 DoE blocks serially on one GPU and summarize each."""

from __future__ import annotations

import argparse
from pathlib import Path
import site

import yaml

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from common.paths import REPO_ROOT
from tools.analyze_pv26_stage2_doe import analyze, save_report
from tools.run_pv26_method_search import run_search


BLOCKS = {
    "lr": "pv26_stage2_doe_lr.yaml",
    "optimizer_gradient": "pv26_stage2_doe_optimizer_gradient.yaml",
    "balance": "pv26_stage2_doe_balance.yaml",
    "optimizer_nested": "pv26_stage2_doe_optimizer_nested.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", choices=tuple(BLOCKS),
                        default=list(BLOCKS))
    parser.add_argument("--through-images", type=int, default=9600)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs")
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    base = REPO_ROOT / "config/pv26_stage2.yaml"
    for block in args.blocks:
        configuration = REPO_ROOT / "config" / BLOCKS[block]
        search = yaml.safe_load(configuration.read_text(encoding="utf-8"))
        if args.through_images not in search["experiment"]["resource_images"]:
            parser.error(f"{block}: --through-images must name a configured rung")
        output = root / search["experiment"]["name"]
        print(f"DoE {block}: {len(search['candidates'])} candidates; "
              f"through {args.through_images} images", flush=True)
        history = run_search(base, configuration, output,
                             through_images=args.through_images, artifact_root=root)
        rung = next(row for row in history["rungs"]
                    if row["resource_images"] == args.through_images)
        complete = sum(row["status"] == "complete" for row in rung["results"])
        report_path = output / f"doe_analysis_{args.through_images}.md"
        save_report(analyze(output, args.through_images), report_path)
        print(f"DoE {block}: {complete}/{len(rung['results'])} complete; "
              f"analysis={report_path}", flush=True)


if __name__ == "__main__":
    main()
