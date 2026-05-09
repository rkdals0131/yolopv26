from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import site
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root not in sys.path:
    site.addsitedir(repo_root)

from tools.pv26_train import cli as train_cli


STAGE4 = "stage_4_lane_family_finetune"
EXPERIMENTS = {
    "upper_trunk_rebalance": {
        "freeze_policy": "lane_family_plus_upper_trunk",
        "trunk_lr": 2.0e-6,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 1.25,
            "stop_line": 2.0,
            "crosswalk": 1.5,
        },
    },
    "heads_rebalance": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 1.25,
            "stop_line": 2.0,
            "crosswalk": 1.5,
        },
    },
    "dense_sharpen_rebalance": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 1.75,
            "stop_line": 2.25,
            "crosswalk": 1.0,
        },
        "overrides": {
            "lane_segfirst_loss_weights": {
                "centerline_bce": 1.5,
                "centerline_dice": 1.5,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "core_centerline_rebalance": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.25,
            "stop_line": 1.75,
            "crosswalk": 1.25,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "core",
            "lane_segfirst_loss_weights": {
                "centerline_bce": 2.0,
                "centerline_dice": 2.0,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "hybrid_centerline_rebalance": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.0,
            "stop_line": 1.75,
            "crosswalk": 1.5,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "hybrid",
            "lane_segfirst_loss_weights": {
                "centerline_bce": 1.75,
                "centerline_dice": 1.75,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "core_cross_retain": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.25,
            "stop_line": 1.75,
            "crosswalk": 1.75,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "core",
            "lane_segfirst_loss_weights": {
                "centerline_bce": 2.0,
                "centerline_dice": 2.0,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "core_centerline_low_lr": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 5.0e-5,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.25,
            "stop_line": 1.75,
            "crosswalk": 1.25,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "core",
            "lane_segfirst_loss_weights": {
                "centerline_bce": 2.0,
                "centerline_dice": 2.0,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "core_centerline_posw8": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.25,
            "stop_line": 1.75,
            "crosswalk": 1.25,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "core",
            "lane_segfirst_centerline_max_positive_weight": 8.0,
            "lane_segfirst_loss_weights": {
                "centerline_bce": 2.0,
                "centerline_dice": 2.0,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "core_centerline_refine": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.25,
            "stop_line": 1.75,
            "crosswalk": 1.25,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "core",
            "lane_segfirst_loss_weights": {
                "centerline_bce": 2.0,
                "centerline_dice": 2.0,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
    "core_centerline_refine_cross_retain": {
        "freeze_policy": "lane_family_heads_only",
        "trunk_lr": 0.0,
        "head_lr": 1.0e-4,
        "loss_weights": {
            "det": 0.0,
            "tl_attr": 0.0,
            "lane": 2.25,
            "stop_line": 1.75,
            "crosswalk": 1.75,
        },
        "overrides": {
            "lane_segfirst_centerline_target_mode": "core",
            "lane_segfirst_loss_weights": {
                "centerline_bce": 2.0,
                "centerline_dice": 2.0,
                "support_bce": 0.15,
                "tangent": 0.35,
                "color": 0.5,
                "type": 0.25,
            },
            "stopline_center_target_mode": "heatmap",
            "stopline_selector_aux_weight": 0.5,
            "stopline_geometry_aux_weight": 1.5,
            "stopline_local_x_aux_weight": 0.5,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short derived phase-4 PV26 lane-family probe from an existing checkpoint. "
            "This is intended to test 60% headroom signals before committing to a multi-day run."
        )
    )
    parser.add_argument("--source-run", required=True, help="Source PV26 meta-train run directory.")
    parser.add_argument("--seed-checkpoint", default="", help="Seed checkpoint. Defaults to <source-run>/phase_4/checkpoints/best.pt.")
    parser.add_argument("--preset", default="default", help="Base PV26 preset.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="upper_trunk_rebalance")
    parser.add_argument("--epochs", type=int, default=3, help="Probe phase-4 epochs.")
    parser.add_argument("--train-batches", type=int, default=512, help="Train batches per probe epoch.")
    parser.add_argument("--val-batches", type=int, default=128, help="Validation batches per probe epoch.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-root", default="", help="Run root. Defaults to the base scenario run root.")
    parser.add_argument("--preview", action="store_true", help="Enable preview generation for the probe.")
    return parser.parse_args()


def _resolve_seed_checkpoint(source_run: Path, seed_checkpoint: str) -> Path:
    checkpoint = Path(seed_checkpoint).expanduser() if seed_checkpoint else source_run / "phase_4" / "checkpoints" / "best.pt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"seed checkpoint not found: {checkpoint}")
    return checkpoint


def _lane60_scenario(args: argparse.Namespace, *, source_run: Path, seed_checkpoint: Path) -> tuple[Any, Path, dict[str, Any]]:
    scenario = train_cli.load_meta_train_scenario(args.preset)
    scenario_path = train_cli._default_scenario_path(args.preset)
    phase_index, phase = train_cli._find_phase_by_stage(scenario, stage=STAGE4)
    experiment = dict(EXPERIMENTS[str(args.experiment)])

    phase_overrides = dict(phase.overrides)
    phase_overrides.update(
        {
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "train_batches": int(args.train_batches),
            "val_batches": int(args.val_batches),
            "trunk_lr": float(experiment["trunk_lr"]),
            "head_lr": float(experiment["head_lr"]),
            "checkpoint_every": 0,
            "task_positive_task": "multi:lane,stopline,crosswalk",
            "task_positive_fraction": 1.0,
            "sampler_ratios": {
                "bdd100k": 0.0,
                "aihub_traffic": 0.0,
                "aihub_lane": 1.0,
                "aihub_obstacle": 0.0,
            },
        }
    )
    phase_overrides.update(dict(experiment.get("overrides", {})))
    probe_phase = replace(
        phase,
        name=f"{phase.name}_{args.experiment}",
        min_epochs=int(args.epochs),
        max_epochs=int(args.epochs),
        patience=int(args.epochs),
        min_delta_abs=0.0,
        loss_weights=dict(experiment["loss_weights"]),
        freeze_policy=str(experiment["freeze_policy"]),
        overrides=phase_overrides,
    )
    phases = list(scenario.phases)
    phases[phase_index - 1] = probe_phase

    train_defaults = replace(
        scenario.train_defaults,
        device=str(args.device),
        batch_size=int(args.batch_size),
        train_batches=int(args.train_batches),
        val_batches=int(args.val_batches),
        checkpoint_every=0,
        profile_device_sync=False,
        step_history_every_n_steps=100,
        step_history_include_grad_details=False,
        pcgrad_aggregate_every_n_steps=100,
        pcgrad_keep_raw_every_n_steps=1000,
    )
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else scenario.run.run_root
    run_config = replace(
        scenario.run,
        run_root=run_root,
        run_dir=None,
        run_name_prefix=f"lane60_{args.experiment}_from_{source_run.name}",
    )
    preview = replace(scenario.preview, enabled=bool(args.preview), epoch_comparison_grid=bool(args.preview))
    scenario = replace(
        scenario,
        train_defaults=train_defaults,
        run=run_config,
        preview=preview,
        phases=tuple(phases),
    )
    lineage = {
        "mode": "lane60_probe",
        "experiment": str(args.experiment),
        "source_run_dir": str(source_run),
        "seed_checkpoint_path": str(seed_checkpoint),
        "selected_phase_index": int(phase_index),
        "probe_phase": {
            "epochs": int(args.epochs),
            "train_batches": int(args.train_batches),
            "val_batches": int(args.val_batches),
            "freeze_policy": str(probe_phase.freeze_policy),
            "loss_weights": dict(probe_phase.loss_weights),
            "overrides": dict(phase_overrides),
        },
    }
    return scenario, scenario_path, {
        "selected_phase_indices": (phase_index,),
        "initial_best_checkpoint": seed_checkpoint,
        "lineage": lineage,
    }


def main() -> int:
    args = parse_args()
    source_run = Path(args.source_run).expanduser().resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"source run not found: {source_run}")
    seed_checkpoint = _resolve_seed_checkpoint(source_run, str(args.seed_checkpoint or ""))
    scenario, scenario_path, options = _lane60_scenario(args, source_run=source_run, seed_checkpoint=seed_checkpoint)
    summary = train_cli.run_meta_train_scenario(
        scenario,
        scenario_path=scenario_path,
        selected_phase_indices=options["selected_phase_indices"],
        initial_best_checkpoint=options["initial_best_checkpoint"],
        lineage=options["lineage"],
    )
    print(json.dumps(train_cli.train_artifacts.json_ready({"summary": summary, "lineage": options["lineage"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
