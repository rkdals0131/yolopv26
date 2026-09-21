from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from model.engine.trainer import FocusedTrainer, FocusedTrainerConfig


class _Sampler:
    def __init__(self) -> None:
        self.position = 0

    def commit(self, n_samples: int) -> None:
        assert n_samples == 4
        self.position += n_samples

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]


class _Model(nn.Module):
    def __init__(self, *, oom_above: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.register_buffer("forwards", torch.tensor(0))
        self.oom_above = oom_above

    def forward_for_loss(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forwards += 1
        if self.oom_above is not None and len(images) > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        return {"prediction": images.mean(dim=(1, 2, 3)) * self.weight}


class _Loss(nn.Module):
    det_weight = 2.0
    roadmark_weight = 0.5

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict) -> dict[str, torch.Tensor]:
        prediction = outputs["prediction"]
        det_mask = batch["det_labeled"]
        det = (prediction[det_mask] - 0.5).square().mean() if det_mask.any() else prediction.sum() * 0
        roadmark_mask = batch["roadmark_valid"].flatten(1).any(dim=1)
        roadmark = (
            (prediction[roadmark_mask] - 1.0).square().mean()
            if roadmark_mask.any() else prediction.sum() * 0
        )
        positive = ((batch["roadmark_target"] > 0) & batch["roadmark_valid"]).flatten(1).any(dim=1)
        dice = (prediction[positive] - 0.3).square().mean() if positive.any() else prediction.sum() * 0
        return {"det": det, "roadmark_bce": roadmark, "roadmark_dice": dice}


class _NonfiniteLoss(_Loss):
    def forward(self, outputs: dict[str, torch.Tensor], batch: dict) -> dict[str, torch.Tensor]:
        losses = super().forward(outputs, batch)
        losses["det"] = losses["det"] * float("inf")
        return losses


def _batch() -> dict:
    roadmark_target = torch.zeros(4, 3, 1, 1)
    roadmark_target[[0, 2, 3], 0, 0, 0] = 1
    return {
        "image": torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).view(4, 3, 4, 4) / 100,
        "det_labeled": torch.tensor([True, False, True, False]),
        "roadmark_target": roadmark_target,
        "roadmark_valid": torch.ones(4, 3, 1, 1, dtype=torch.bool),
        "batch_idx": torch.tensor([0, 2]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.ones(2, 4) / 4,
        "meta": [{"id": index} for index in range(4)],
    }


def _trainer(path: Path, *, oom_above: int | None = None, microbatch_size: int = 4,
             criterion: nn.Module | None = None, max_failures: int = 3) -> FocusedTrainer:
    model = _Model(oom_above=oom_above)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    config = FocusedTrainerConfig(
        output_dir=path, device="cpu", precision="fp32", microbatch_size=microbatch_size,
        max_consecutive_failures=max_failures,
    )
    return FocusedTrainer(model, criterion or _Loss(), optimizer, scheduler, _Sampler(), config)


def test_recoverable_oom_retries_same_logical_batch_and_restores_buffers() -> None:
    with tempfile.TemporaryDirectory() as directory:
        recovered = _trainer(Path(directory) / "recovered", oom_above=2)
        direct = _trainer(Path(directory) / "direct", microbatch_size=2)
        assert recovered.train_batch(_batch())
        assert direct.train_batch(_batch())
        assert recovered.microbatch_size == 2
        assert recovered.sampler.position == direct.sampler.position == 4
        assert recovered.global_step == direct.global_step == 1
        assert recovered.model.forwards.item() == direct.model.forwards.item() == 2
        torch.testing.assert_close(recovered.model.weight, direct.model.weight)


def test_fit_checkpoint_restores_optimizer_scheduler_rng_and_committed_position() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        trainer = _trainer(path)
        callbacks = []
        summary = trainer.fit([_batch()], max_steps=1, on_step=lambda _, state: callbacks.append(state))
        assert summary["global_step"] == 1
        assert callbacks[0]["sampler_position"] == 4
        assert trainer.latest_path.is_file() and trainer.previous_path.is_file()
        restored = _trainer(path)
        restored.load_checkpoint()
        assert restored.global_step == 1 and restored.sampler.position == 4
        assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
        assert restored.optimizer.state_dict()["state"]
        torch.testing.assert_close(restored.model.weight, trainer.model.weight)


def test_nonfinite_updates_do_not_step_or_replace_normal_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        trainer = _trainer(Path(directory), criterion=_NonfiniteLoss(), max_failures=2)
        trainer.save_checkpoint()
        assert not trainer.train_batch(_batch())
        with pytest.raises(RuntimeError, match="consecutive"):
            trainer.train_batch(_batch())
        assert trainer.global_step == 0
        assert trainer.scheduler.last_epoch == 0
        assert trainer.model.forwards.item() == 0
        assert trainer.sampler.position == 8
        trainer.load_checkpoint()
        assert trainer.sampler.position == 0


def test_optimizer_oom_halts_with_last_normal_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        trainer = _trainer(Path(directory))
        trainer.save_checkpoint()

        def partial_step(*, closure=None):
            del closure
            with torch.no_grad():
                trainer.model.weight.add_(1)
            raise RuntimeError("CUDA out of memory")

        trainer.optimizer.step = partial_step
        with pytest.raises(RuntimeError, match="optimizer-step OOM"):
            trainer.train_batch(_batch())
        with pytest.raises(RuntimeError, match="uncertain"):
            trainer.save_checkpoint()
        trainer.load_checkpoint()
        assert trainer.global_step == 0 and trainer.sampler.position == 0
        torch.testing.assert_close(trainer.model.weight, torch.tensor(0.25))


class _AttrAdapter:
    def slice_batch(self, batch: dict, start: int, stop: int) -> dict:
        return {key: value[start:stop] for key, value in batch.items()}

    def term_counts(self, batch: dict) -> dict[str, int]:
        return {"base": len(batch["image"]), "arrow": int(batch["arrow_valid"].sum().item())}

    def weighted_terms(self, losses: dict, criterion: nn.Module) -> dict:
        return {"base": losses["base"], "arrow": losses["arrow"] * criterion.arrow_weight}

    def sample_ids(self, batch: dict) -> list[str]:
        return [row["sample_id"] for row in batch["rows"]]


class _AttrModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"score": image.flatten(1).mean(dim=1) * self.weight}


class _AttrLoss(nn.Module):
    arrow_weight = 0.7

    def __init__(self) -> None:
        super().__init__()
        self.progress_calls: list[tuple[int, int]] = []

    def set_progress(self, step: int, total: int) -> None:
        self.progress_calls.append((step, total))

    def forward(self, output: dict, batch: dict) -> dict:
        score = output["score"]
        base = (score - batch["base_target"]).square().mean()
        valid = batch["arrow_valid"]
        arrow = (score[valid] - batch["arrow_target"][valid]).square().mean() if valid.any() else score.sum() * 0
        return {"base": base, "arrow": arrow}


def _attr_batch() -> dict:
    return {
        "image": torch.tensor([1., 2., 3., 4.]).view(4, 1, 1, 1),
        "base_target": torch.tensor([0., 1., 0., 1.]),
        "arrow_target": torch.tensor([1., 0., 0., 1.]),
        "arrow_valid": torch.tensor([True, False, True, False]),
        "rows": [{"sample_id": str(index)} for index in range(4)],
    }


def _attr_trainer(path: Path, microbatch_size: int) -> FocusedTrainer:
    model = _AttrModel()
    criterion = _AttrLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return FocusedTrainer(
        model, criterion, optimizer, None, _Sampler(),
        FocusedTrainerConfig(output_dir=path, device="cpu", precision="fp32",
                             microbatch_size=microbatch_size, stage="signal_attr"),
        batch_adapter=_AttrAdapter(),
    )


def test_second_consumer_normalization_and_planned_progress_survive_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        direct = _attr_trainer(path / "direct", 4)
        split = _attr_trainer(path / "split", 1)
        assert direct.train_batch(_attr_batch())
        assert split.train_batch(_attr_batch())
        torch.testing.assert_close(direct.model.weight, split.model.weight)
        callbacks = []
        split.fit([_attr_batch()], max_steps=2, planned_steps=10,
                  on_step=lambda _, summary: callbacks.append(summary))
        assert callbacks[0]["losses"]["total"] > 0
        assert split.criterion.progress_calls == [(1, 10), (2, 10)]
        resumed = _attr_trainer(path / "split", 1)
        resumed.load_checkpoint()
        resumed.fit([_attr_batch()], max_steps=3, planned_steps=10)
        assert resumed.criterion.progress_calls == [(2, 10), (3, 10)]
        assert resumed.global_step == 3 and resumed.sampler.position == 12


def test_best_weights_survive_interruption_before_full_resume_publication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        trainer = _trainer(path)
        trainer.save_checkpoint()
        trainer.global_step = 1

        def interrupted_save() -> None:
            raise OSError("interrupted save")

        trainer.save_checkpoint = interrupted_save
        with pytest.raises(OSError, match="interrupted save"):
            trainer.update_best(0.9)
        restored = _trainer(path)
        restored.load_checkpoint()
        assert restored.global_step == 0
        assert restored.best_metric == 0.9 and restored.best_step == 1
        assert not restored.update_best(0.8)
        best = torch.load(restored.best_path, map_location="cpu", weights_only=False)
        assert best["metric"] == 0.9 and best["model_config"] is None
