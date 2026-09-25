"""Write the fixed, seeded PV26 stage-2 experiment blocks for method_search."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import random
import site

import yaml

site.addsitedir(str(Path(__file__).resolve().parents[1]))

from common.paths import REPO_ROOT


SEEDS = (26, 27, 28)
RESOURCE_IMAGES = (9600, 32000, 128000, 384000)
REFERENCE_RUN = "runs/20260922_1607_joint_lr3x_80k"


def _search(name: str, candidates: list[dict], *, block: str,
            factors: dict[str, list]) -> dict:
    randomization_seed = 260923 + sum(ord(character) for character in block)
    random.Random(randomization_seed).shuffle(candidates)
    return {
        "experiment": {
            "name": name,
            "seed": SEEDS[0],
            "validation_samples_per_source": 1024,
            "common_eval_run": REFERENCE_RUN,
            "base_overrides": {"train": {"detector_loss_schedule": "restart"}},
            "resource_images": list(RESOURCE_IMAGES),
            "promotion_policy": "all",
            "design": {"block": block, "factors": factors,
                       "seeds": list(SEEDS), "randomization_seed": randomization_seed},
        },
        "candidates": candidates,
    }


def designs() -> dict[str, dict]:
    lr_levels = {"l": 0.3, "m": 1.0, "h": 3.0}
    lr_candidates = []
    for body, roadmark, seed in product(lr_levels, lr_levels, SEEDS):
        lr_candidates.append({
            "name": f"body_{body}_road_{roadmark}_s{seed}", "seed": seed,
            "factors": {"body_lr": body, "roadmark_lr": roadmark},
            "overrides": {"train": {
                "stage": "joint",
                "backbone_lr": 3e-5 * lr_levels[body],
                "head_lr": 3e-4,
                "roadmark_lr": 9e-4 * lr_levels[roadmark],
            }},
        })

    optimizer_candidates = []
    recipes = {
        "adamw_cosine": {"optimizer": "adamw", "lr_schedule": "cosine"},
        "schedulefree": {"optimizer": "schedulefree_adamw", "lr_schedule": "constant",
                         "schedulefree_warmup_steps": 50,
                         "schedulefree_bn_samples_per_source": 32},
    }
    for recipe, gradient, seed in product(recipes, ("sum", "pcgrad", "gradnorm"), SEEDS):
        optimizer_candidates.append({
            "name": f"{recipe}_{gradient}_s{seed}", "seed": seed,
            "factors": {"optimizer_recipe": recipe, "gradient_strategy": gradient},
            "overrides": {"train": {
                **recipes[recipe], "stage": "joint", "microbatch_size": 8,
                "backbone_lr": 3e-5, "head_lr": 3e-4, "roadmark_lr": 9e-4,
                "gradient_strategy": gradient, "gradnorm_alpha": 1.5,
            }},
        })

    balance_candidates = []
    ratios = {"traffic2": (2.0, 1.0), "equal": (1.0, 1.0),
              "roadmark2": (1.0, 2.0)}
    loss_weights = {"half": 0.5, "one": 1.0, "double": 2.0}
    for ratio, weight, seed in product(ratios, loss_weights, SEEDS):
        balance_candidates.append({
            "name": f"{ratio}_loss_{weight}_s{seed}", "seed": seed,
            "factors": {"source_ratio": ratio, "roadmark_loss_weight": weight},
            "source_weights": list(ratios[ratio]),
            "overrides": {"train": {"stage": "joint",
                                    "roadmark_loss_weight": loss_weights[weight]}},
        })

    recipe_candidates = []
    for recipe, seed in product(
        ("adamw_cosine", "adamw_constant", "prodigy_d0p3", "prodigy_d1", "prodigy_d3"),
        SEEDS,
    ):
        d_coef = ({"prodigy_d0p3": 0.3, "prodigy_d1": 1.0,
                   "prodigy_d3": 3.0}[recipe] if recipe.startswith("prodigy") else None)
        settings = ({"optimizer": "prodigy", "lr_schedule": "constant",
                     "prodigy_d_coef": d_coef} if d_coef is not None else
                    {"optimizer": "adamw", "lr_schedule":
                     "constant" if recipe == "adamw_constant" else "cosine"})
        recipe_candidates.append({
            "name": f"{recipe}_s{seed}", "seed": seed,
            "factors": {"optimizer_recipe": recipe},
            "overrides": {"train": {**settings, "stage": "joint",
                                    "gradient_strategy": "sum"}},
        })

    return {
        "pv26_stage2_doe_lr.yaml": _search(
            "20260923_stage2_doe_lr", lr_candidates, block="lr_3x3",
            factors={"body_lr": list(lr_levels), "roadmark_lr": list(lr_levels)}),
        "pv26_stage2_doe_optimizer_gradient.yaml": _search(
            "20260923_stage2_doe_optimizer_gradient", optimizer_candidates,
            block="optimizer_2x_gradient_3",
            factors={"optimizer_recipe": list(recipes),
                     "gradient_strategy": ["sum", "pcgrad", "gradnorm"]}),
        "pv26_stage2_doe_balance.yaml": _search(
            "20260923_stage2_doe_balance", balance_candidates,
            block="source_ratio_3x_loss_weight_3",
            factors={"source_ratio": list(ratios),
                     "roadmark_loss_weight": list(loss_weights)}),
        "pv26_stage2_doe_optimizer_nested.yaml": _search(
            "20260923_stage2_doe_optimizer_nested", recipe_candidates,
            block="optimizer_recipe_nested",
            factors={"optimizer_recipe": ["adamw_cosine", "adamw_constant",
                                          "prodigy_d0p3", "prodigy_d1", "prodigy_d3"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "config")
    args = parser.parse_args()
    target = args.output_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in designs().items():
        path = target / filename
        if path.exists():
            if yaml.safe_load(path.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"existing DoE design differs: {path}")
        else:
            path.write_text(yaml.safe_dump(payload, allow_unicode=True,
                                           sort_keys=False), encoding="utf-8")
        print(f"{path}: {len(payload['candidates'])} candidates")


if __name__ == "__main__":
    main()
