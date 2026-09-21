from __future__ import annotations

import torch

from model.data.geometry import compute_letterbox_transform
from model.data.dataset import slice_focused_batch
from model.engine.loss import PV26FocusedLoss
from model.engine.postprocess import decode_focused_detections, decode_roadmark_points
from model.net.pv26 import PV26FocusedModel


def _meta(raw_hw: tuple[int, int] = (64, 64), network_hw: tuple[int, int] = (64, 64)) -> dict:
    return {
        "raw_hw": raw_hw,
        "network_hw": network_hw,
        "transform": compute_letterbox_transform(raw_hw, network_hw).as_meta(),
    }


def test_partial_detection_labels_exclude_unlabeled_image() -> None:
    torch.set_num_threads(4)
    model = PV26FocusedModel(weights=None).train()
    criterion = PV26FocusedLoss(model)
    image = torch.rand(2, 3, 64, 64)
    outputs = model.forward_for_loss(image)
    batch = {
        "image": image,
        "det_labeled": torch.tensor([False, True]),
        "batch_idx": torch.tensor([1]),
        "cls": torch.tensor([[1]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "roadmark_target": torch.zeros(2, 3, 16, 16),
        "roadmark_valid": torch.zeros(2, 3, 16, 16, dtype=torch.bool),
    }
    criterion.set_progress(5, 10)
    first = criterion(outputs, batch)
    assert abs(criterion._official.o2m - 0.45) < 1.0e-6
    assert abs(criterion._official.o2o - 0.55) < 1.0e-6
    changed = {"det": {}, "roadmark_logits": outputs["roadmark_logits"]}
    for branch_name, branch in outputs["det"].items():
        changed_branch = dict(branch)
        scores = branch["scores"].clone()
        scores[0] = scores[0] + 100.0
        changed_branch["scores"] = scores
        changed["det"][branch_name] = changed_branch
    second = criterion(changed, batch)
    torch.testing.assert_close(first["det"], second["det"])
    assert int(first["det_count"]) == 1
    assert int(first["roadmark_valid_count"]) == 0
    first["total"].backward()
    assert model.detector.model[0].conv.weight.grad is not None


def test_stage_freeze_and_inactive_loss() -> None:
    torch.set_num_threads(4)
    model = PV26FocusedModel(weights=None)
    model.set_train_stage("roadmark")
    model.train()
    assert not model.detector.training
    assert model.roadmark_decoder.training
    assert not any(parameter.requires_grad for parameter in model.detector.parameters())
    image = torch.rand(1, 3, 64, 64)
    outputs = model.forward_for_loss(image)
    assert outputs["det"] is None
    batch = {
        "image": image,
        "det_labeled": torch.tensor([False]),
        "batch_idx": torch.empty(0, dtype=torch.long),
        "cls": torch.empty(0, 1, dtype=torch.long),
        "bboxes": torch.empty(0, 4),
        "roadmark_target": torch.zeros(1, 3, 16, 16),
        "roadmark_valid": torch.ones(1, 3, 16, 16, dtype=torch.bool),
    }
    losses = PV26FocusedLoss(model)(outputs, batch)
    assert int(losses["det_count"]) == 0
    assert int(losses["roadmark_valid_count"]) == 3 * 16 * 16
    losses["total"].backward()
    assert model.roadmark_decoder.logits.weight.grad is not None


def test_official_per_image_detection_loss_is_microbatch_invariant() -> None:
    torch.set_num_threads(4)
    model = PV26FocusedModel(weights=None).train()
    criterion = PV26FocusedLoss(model)
    image = torch.rand(2, 3, 64, 64)
    outputs = model.forward_for_loss(image)
    batch = {
        "image": image,
        "det_labeled": torch.tensor([True, True]),
        "batch_idx": torch.tensor([0]),  # image 1 is a labeled negative
        "cls": torch.tensor([[0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "roadmark_target": torch.zeros(2, 3, 16, 16),
        "roadmark_valid": torch.zeros(2, 3, 16, 16, dtype=torch.bool),
        "meta": [{}, {}],
    }
    logical = criterion(outputs, batch)["det"]
    micro_losses = []
    for index in range(2):
        raw = {}
        for branch_name, branch in outputs["det"].items():
            raw[branch_name] = {
                "boxes": branch["boxes"][index:index + 1],
                "scores": branch["scores"][index:index + 1],
                "feats": [feature[index:index + 1] for feature in branch["feats"]],
            }
        micro_outputs = {
            "det": raw,
            "roadmark_logits": outputs["roadmark_logits"][index:index + 1],
        }
        micro_batch = slice_focused_batch(batch, index, index + 1)
        micro_losses.append(criterion(micro_outputs, micro_batch)["det"])
    torch.testing.assert_close(logical, (micro_losses[0] + micro_losses[1]) / 2)


def test_roadmark_ridges_keep_distinct_sloped_lanes_and_horizontal_stops() -> None:
    logits = torch.full((1, 3, 16, 16), -10.0)
    for row in range(2, 15):
        logits[0, 0, row, 2 + row // 4] = 10.0
        logits[0, 0, row, 13 - row // 4] = 10.0
    for column in range(2, 15):
        logits[0, 2, 4, column] = 10.0
        logits[0, 2, 11, column] = 10.0
    lines = decode_roadmark_points(logits, [_meta()])[0]
    assert [line["class_name"] for line in lines].count("white_lane") == 2
    assert [line["class_name"] for line in lines].count("stop_line") == 2
    assert all(len(line["points_xy"]) >= 4 for line in lines)


def test_two_signal_classes_survive_raw_image_decode() -> None:
    detections = torch.tensor([[[0.0, 0.0, 20.0, 20.0, 0.9, 0.0],
                                [15.0, 15.0, 30.0, 30.0, 0.8, 1.0]]])
    observations = decode_focused_detections(detections, [_meta()])[0]
    assert [item["class_id"] for item in observations] == [0, 1]
    assert [item["class_name"] for item in observations] == ["vehicle_signal", "pedestrian_signal"]


def test_raw_detection_decodes_like_eval_forward() -> None:
    torch.set_num_threads(4)
    model = PV26FocusedModel(weights=None).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        raw = model.forward_for_loss(image)["det"]
        decoded = model.decode_raw_detection(raw)
        direct = model(image)["det"]
    torch.testing.assert_close(decoded, direct)
