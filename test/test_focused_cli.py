from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from common.io import atomic_write_json, read_json
from common.train_runtime import training_run_lock
from tools.pv26_train import cli as focused_cli
from tools import train_signal_attr as signal_cli


def test_training_run_rejects_another_writer_and_keeps_lock_inode(tmp_path: Path) -> None:
    lock_path = tmp_path / ".train.lock"
    with training_run_lock(tmp_path):
        inode = lock_path.stat().st_ino
        with pytest.raises(RuntimeError, match="already has a writer"):
            with training_run_lock(tmp_path):
                pytest.fail("a second writer entered the same run")
    with training_run_lock(tmp_path):
        assert lock_path.stat().st_ino == inode


def test_focused_run_config_preserves_pre_checkpoint_initializer(tmp_path: Path) -> None:
    initial = tmp_path / "initial.pt"
    args = argparse.Namespace(
        resume_run=None, initial_checkpoint=initial, stage="roadmark",
        sample_limit=1, config=focused_cli.DEFAULT_CONFIG, device="cpu",
        num_workers=0, microbatch_size=1,
    )
    with training_run_lock(tmp_path):
        focused_cli._configuration(args, tmp_path)
        configured = read_json(tmp_path / "run_config.json")
        assert configured["initial_checkpoint"] == str(initial.resolve())
        assert configured["model"]["weights"] == str((focused_cli.REPO_ROOT / "yolo26s.pt").resolve())
        assert all(Path(source["root"]).is_absolute() for source in configured["data"]["sources"])
        with pytest.raises(FileExistsError, match="run already exists"):
            focused_cli._configuration(args, tmp_path)
        args.resume_run = tmp_path
        args.initial_checkpoint = None
        args.stage = None
        args.sample_limit = None
        resumed = focused_cli._configuration(args, tmp_path)
        assert resumed["initial_checkpoint"] == str(initial.resolve())


def test_signal_run_config_reaches_builder_after_early_restart(tmp_path: Path) -> None:
    dataset = tmp_path / "crops"
    initial = tmp_path / "initial_signal.pt"
    args = argparse.Namespace(
        resume_run=None, dataset=dataset, initial_checkpoint=initial,
        precision="fp32", max_steps=2, logical_batch_size=2,
        microbatch_size=1, validation_every=0, validation_samples=1,
        num_workers=0, device="cpu", steps=1, sampling="balanced",
    )
    interrupted = RuntimeError("stop before first checkpoint")
    with patch.object(signal_cli, "build_signal_attr_focused_run", side_effect=interrupted) as build:
        with training_run_lock(tmp_path):
            with pytest.raises(RuntimeError, match="stop before first checkpoint"):
                signal_cli._train_locked(args, argparse.ArgumentParser(), tmp_path)
            assert read_json(tmp_path / "signal_config.json")["initial_checkpoint"] == str(initial.resolve())
            with pytest.raises(FileExistsError, match="run already exists"):
                signal_cli._train_locked(args, argparse.ArgumentParser(), tmp_path)
            args.resume_run = tmp_path
            args.initial_checkpoint = None
            with pytest.raises(RuntimeError, match="stop before first checkpoint"):
                signal_cli._train_locked(args, argparse.ArgumentParser(), tmp_path)
    assert build.call_count == 2
    assert build.call_args.kwargs["resume"] is True
    assert build.call_args.kwargs["initial_checkpoint"] == initial


def test_signal_yaml_controls_new_run_and_prepare_crop(tmp_path: Path) -> None:
    settings = yaml.safe_load(signal_cli.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    settings["train"]["learning_rate"] = 0.002
    settings["train"]["checkpoint_interval_sec"] = 42
    settings["prepare"]["crop"]["padding_ratio"] = 0.2
    path = tmp_path / "signal.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    args = argparse.Namespace(
        config=path, resume_run=None, dataset=tmp_path / "crops",
        initial_checkpoint=None, device=None, num_workers=None,
        microbatch_size=None, steps=1,
    )
    with patch.object(signal_cli, "build_signal_attr_focused_run", side_effect=RuntimeError("builder reached")) as build:
        with pytest.raises(RuntimeError, match="builder reached"):
            signal_cli._train_locked(args, argparse.ArgumentParser(), tmp_path / "run")
    snapshot = read_json(tmp_path / "run" / "signal_config.json")
    assert snapshot["learning_rate"] == 0.002
    assert snapshot["checkpoint_interval_sec"] == 42
    assert build.call_args.kwargs["learning_rate"] == 0.002
    assert build.call_args.kwargs["checkpoint_interval_sec"] == 42
    assert build.call_args.kwargs["initial_checkpoint"] == (
        signal_cli.REPO_ROOT / "models/signal_attr/best_signal_attr.pt"
    ).resolve()

    with patch.object(signal_cli, "_output_directory", return_value=tmp_path / "prepared"), \
         patch.object(signal_cli, "materialize_product_signal_attr_crop_dataset_from_root", return_value={}) as prepare:
        signal_cli.main(["prepare", "--config", str(path), "--output-dir", str(tmp_path / "prepared"),
                         "--all-off-policy", "exclude"])
    assert prepare.call_args.kwargs["all_off_is_valid"] is False
    assert prepare.call_args.kwargs["crop_config"].padding_ratio == 0.2
    assert prepare.call_args.kwargs["workers"] == 4


def test_signal_old_snapshot_keeps_historical_optimizer_defaults(tmp_path: Path) -> None:
    old = {
        "dataset": str((tmp_path / "crops").resolve()), "precision": "fp32",
        "max_steps": 2, "logical_batch_size": 2, "microbatch_size": 1,
        "validation_every": 0, "validation_samples": 1, "seed": 26,
        "sampling": "balanced", "selection_metric": "macro_state_f1",
        "initial_checkpoint": str((tmp_path / "initial.pt").resolve()),
    }
    atomic_write_json(tmp_path / "signal_config.json", old)
    args = argparse.Namespace(
        resume_run=tmp_path, dataset=None, initial_checkpoint=None,
        device="cpu", num_workers=0, microbatch_size=1, steps=1,
    )
    with patch.object(signal_cli, "build_signal_attr_focused_run", side_effect=RuntimeError("builder reached")) as build:
        with pytest.raises(RuntimeError, match="builder reached"):
            signal_cli._train_locked(args, argparse.ArgumentParser(), tmp_path)
    assert build.call_args.kwargs["learning_rate"] == 1.0e-3
    assert build.call_args.kwargs["weight_decay"] == 1.0e-4
    assert build.call_args.kwargs["checkpoint_interval_sec"] == 600.0
    assert build.call_args.kwargs["resume"] is True
