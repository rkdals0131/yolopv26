from __future__ import annotations

from pathlib import Path
import random

from PIL import Image
import pytest
import torch

from common.io import read_jsonl, write_json
from model.signal_attr import (
    BASE_COLOR_TO_INDEX,
    SignalAttrCropConfig,
    SignalAttrRuntime,
    SignalAttrThresholdPolicy,
    build_signal_attr_focused_run,
    evaluate_signal_attr_classifier,
    extract_product_signal_attr_target,
    load_signal_attr_classifier_checkpoint,
    materialize_product_signal_attr_crop_dataset_from_root,
    product_signal_attr_prediction_from_logits,
    signal_attr_collate,
)
from model.signal_attr.augment import (
    SignalAttrAugmentConfig,
    augment_signal_attr_crop,
    signal_attr_augment_config_from_mapping,
    signal_attr_augment_config_to_dict,
)


def _annotation(light_type: str, color: str = "off", *, left: bool = False, other: bool = False) -> dict:
    return {
        "class": "traffic_light", "type": light_type, "box": [8, 8, 28, 36],
        "attribute": [{
            "red": "on" if color == "red" else "off",
            "yellow": "on" if color == "yellow" else "off",
            "green": "on" if color == "green" else "off",
            "left_arrow": "on" if left else "off",
            "others_arrow": "on" if other else "off",
            "x_light": "off",
        }],
    }


def _write_aihub_pair(root: Path, split: str, name: str, annotations: list[dict]) -> None:
    split_dir = "Training" if split == "train" else "Validation"
    label = root / split_dir / "[라벨]group" / "group" / f"{name}.json"
    image = root / split_dir / "[원천]group" / "group" / f"{name}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), "black").save(image)
    write_json(label, {
        "image": {"filename": image.name, "imsize": [64, 48]},
        "annotation": annotations,
    })


def _materialized_crops(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write_aihub_pair(raw, "train", "train_a", [
        _annotation("car", "red", left=True),
        _annotation("car", other=True),
        _annotation("pedestrian", "green"),
        _annotation("car"),
    ])
    _write_aihub_pair(raw, "val", "val_a", [_annotation("pedestrian", "red")])
    crops = tmp_path / "crops"
    manifest = materialize_product_signal_attr_crop_dataset_from_root(
        raw, crops, all_off_is_valid=False,
        crop_config=SignalAttrCropConfig(input_size=32), workers=2,
        max_samples_per_split=1,
    )
    assert manifest["state_semantics"] == "left_arrow"
    assert manifest["accepted_count_by_split"] == {"train": 3, "val": 1}
    return crops


def test_raw_policy_and_crop_rows_keep_other_arrow_negative(tmp_path: Path) -> None:
    other_only = extract_product_signal_attr_target(_annotation("car", other=True), all_off_is_valid=False)
    all_off = extract_product_signal_attr_target(_annotation("car"), all_off_is_valid=False)
    pedestrian_off = extract_product_signal_attr_target(_annotation("pedestrian"), all_off_is_valid=True)
    assert (other_only.base_color, other_only.left_arrow, other_only.state_valid) == ("off", 0, True)
    assert (all_off.state_valid, all_off.reason) == (False, "all_off_unverified")
    assert (pedestrian_off.base_color, pedestrian_off.state_valid) == ("off", True)
    assert product_signal_attr_prediction_from_logits(
        [8, -4, -4, -4], 8, light_type="pedestrian", all_off_is_valid=True,
    ).tl_attr_valid == 1

    crops = _materialized_crops(tmp_path)
    train = read_jsonl(crops / "labels" / "train.jsonl")
    rejected = read_jsonl(crops / "meta" / "rejected_rows.jsonl")
    assert [(row["light_type"], row["base_color"], row["arrow"]) for row in train] == [
        ("car", "red", 1), ("car", "off", 0), ("pedestrian", "green", 0),
    ]
    assert [row["reject_reason"] for row in rejected] == ["all_off_unverified"]
    assert all((crops / row["crop_path"]).is_file() for row in train)


def test_included_checkpoint_and_batched_runtime_keep_semantics() -> None:
    baseline = Path(__file__).resolve().parents[2] / "models" / "signal_attr" / "best_signal_attr.pt"
    loaded = load_signal_attr_classifier_checkpoint(baseline, device="cpu")
    assert loaded["model_config"].input_size == 128
    legacy = SignalAttrRuntime.from_checkpoint(baseline, device="cpu")
    row = legacy.predict(Image.new("RGB", (64, 48)), [
        {"id": 9, "class_name": "vehicle_signal", "bbox_xyxy": [8, 8, 28, 36]},
    ])[0]
    assert (row["detection_id"], row["state_valid"], row["reason"]) == (9, False, "left_arrow_untrained")

    class Fixed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            self.calls += 1
            return {
                "base_color_logits": images.new_tensor([[-4, 8, -4, -4]]).expand(images.shape[0], -1),
                "arrow_logit": images.new_full((images.shape[0],), 8.0),
            }

    fixed = Fixed()
    runtime = SignalAttrRuntime(
        fixed, crop_config=SignalAttrCropConfig(input_size=32),
        threshold_policy=SignalAttrThresholdPolicy(), state_semantics="left_arrow",
        device=torch.device("cpu"),
    )
    rows = runtime.predict(Image.new("RGB", (64, 48)), [
        {"id": 13, "class_name": "vehicle_signal", "bbox_xyxy": [8, 8, 28, 36]},
        {"id": 4, "class_name": "pedestrian_signal", "bbox_xyxy": [8, 8, 28, 36]},
    ])
    assert fixed.calls == 1
    assert [row["detection_id"] for row in rows] == [13, 4]
    assert rows[0]["left_arrow"] == 1 and rows[0]["state_valid"]
    assert rows[1]["left_arrow"] is None and rows[1]["state_valid"]


def test_training_snapshot_resume_and_checkpoint_format(tmp_path: Path) -> None:
    crops = _materialized_crops(tmp_path)
    options = dict(
        logical_batch_size=2, microbatch_size=1, sampling="balanced",
        device="cpu", precision="fp32", num_workers=0, initial_checkpoint=None,
    )
    run = build_signal_attr_focused_run(crops, tmp_path / "run", **options)
    assert run.group_counts == {"car:off:left=0": 1, "car:red:left=1": 1, "pedestrian:green": 1}
    result = run.trainer.fit(run.train_loader, max_steps=1, planned_steps=2)
    assert result["global_step"] == 1
    metrics = run.evaluate()
    assert "macro_state_f1" in metrics and "valid_coverage" in metrics
    run.trainer.update_best(metrics["macro_state_f1"])
    published = run.publish_checkpoint(tmp_path / "best_signal_attr.pt")
    assert load_signal_attr_classifier_checkpoint(published)["payload"]["state_semantics"] == "left_arrow"

    expected_keys = next(iter(run.trainer.sampler))
    resumed = build_signal_attr_focused_run(crops, tmp_path / "run", resume=True, **options)
    assert resumed.trainer.global_step == 1
    assert next(iter(resumed.trainer.sampler)) == expected_keys
    assert resumed.trainer.run_metadata["sampling"] == "balanced"


def test_macro_state_f1_counts_unsupported_false_left() -> None:
    class Green(torch.nn.Module):
        def __init__(self, arrow: float) -> None:
            super().__init__()
            self.arrow = arrow

        def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "base_color_logits": images.new_tensor([[-4, -4, -4, 8]]),
                "arrow_logit": images.new_tensor([self.arrow]),
            }

    batch = signal_attr_collate([{
        "image": torch.zeros((3, 2, 2)),
        "base_color_target": BASE_COLOR_TO_INDEX["green"],
        "arrow_target": 0.0, "arrow_target_valid": 1.0,
        "row": {"light_type": "car", "base_color": "green", "arrow": 0},
    }])
    correct = evaluate_signal_attr_classifier(Green(-8), [batch], device=torch.device("cpu"))
    false_left = evaluate_signal_attr_classifier(Green(8), [batch], device=torch.device("cpu"))
    assert correct["macro_state_f1"] == 1.0
    assert false_left["macro_state_f1"] == 0.5
    assert false_left["by_light_type"]["car"]["states"]["left_arrow"]["fp"] == 1


def test_augment_keeps_size_repeats_by_seed_and_shrinks_to_source_pixels() -> None:
    image = Image.new("RGB", (32, 32), "black")
    image.paste((255, 0, 0), (12, 4, 20, 12))
    image.paste((0, 255, 0), (4, 20, 12, 28))
    everything = SignalAttrAugmentConfig(jitter_prob=1.0, photometric_prob=1.0, lowres_prob=1.0,
                                         blur_prob=1.0, jpeg_prob=1.0)
    first = augment_signal_attr_crop(image, [0, 0, 64, 16], random.Random(7), everything)
    again = augment_signal_attr_crop(image, [0, 0, 64, 16], random.Random(7), everything)
    assert first.size == (32, 32) and first.mode == "RGB"
    assert first.tobytes() == again.tobytes()

    nothing = SignalAttrAugmentConfig(jitter_prob=0.0, photometric_prob=0.0, lowres_prob=0.0,
                                      blur_prob=0.0, jpeg_prob=0.0)
    assert augment_signal_attr_crop(image, None, random.Random(7), nothing).tobytes() == image.tobytes()
    # A crop already at the minimum source size has nothing farther to simulate.
    lowres_only = SignalAttrAugmentConfig(jitter_prob=0.0, photometric_prob=0.0, lowres_prob=1.0,
                                          blur_prob=0.0, jpeg_prob=0.0, lowres_min_short_side_px=5.0)
    assert augment_signal_attr_crop(image, [0, 0, 5, 5], random.Random(7), lowres_only).tobytes() == image.tobytes()
    assert augment_signal_attr_crop(image, [0, 0, 400, 400], random.Random(7), lowres_only).tobytes() != image.tobytes()


def test_augment_config_mapping_round_trip() -> None:
    assert signal_attr_augment_config_from_mapping(None) is None
    assert signal_attr_augment_config_from_mapping({"enabled": False, "jitter_prob": 1.0}) is None
    config = signal_attr_augment_config_from_mapping({"enabled": True, "jpeg_quality": [40, 80]})
    assert config == SignalAttrAugmentConfig(jpeg_quality=(40, 80))
    assert signal_attr_augment_config_from_mapping(signal_attr_augment_config_to_dict(config)) == config
    with pytest.raises(ValueError, match="unknown SignalAttr augment keys"):
        signal_attr_augment_config_from_mapping({"hflip_prob": 0.5})


def test_augmented_training_resumes_and_reports_state_and_size_slices(tmp_path: Path) -> None:
    crops = _materialized_crops(tmp_path)
    augment = SignalAttrAugmentConfig(jitter_prob=1.0, lowres_prob=1.0, blur_prob=1.0, jpeg_prob=1.0)
    options = dict(
        logical_batch_size=2, microbatch_size=1, sampling="balanced",
        device="cpu", precision="fp32", num_workers=0, initial_checkpoint=None,
    )
    run = build_signal_attr_focused_run(crops, tmp_path / "run", augment=augment, **options)
    keys = next(iter(run.trainer.sampler))
    assert torch.equal(run.train_loader.dataset[keys[0]]["image"], run.train_loader.dataset[keys[0]]["image"])
    assert run.trainer.fit(run.train_loader, max_steps=1, planned_steps=2)["global_step"] == 1
    metrics = run.evaluate()
    assert set(metrics["by_state_group"]) == {"pedestrian:red"}
    assert set(metrics["by_short_side"]) == {"16to32px"}
    run.trainer.update_best(metrics["macro_state_f1"])
    published = run.publish_checkpoint(tmp_path / "best_signal_attr.pt")
    assert load_signal_attr_classifier_checkpoint(published)["payload"]["augment"]["enabled"] is True

    with pytest.raises(ValueError, match="augmentation differs"):
        build_signal_attr_focused_run(crops, tmp_path / "run", resume=True, **options)
    resumed = build_signal_attr_focused_run(crops, tmp_path / "run", resume=True, augment=augment, **options)
    assert resumed.trainer.global_step == 1


def test_sqrt_sampling_weights_groups_by_square_root(tmp_path: Path) -> None:
    crops = _materialized_crops(tmp_path)
    run = build_signal_attr_focused_run(
        crops, tmp_path / "run", logical_batch_size=2, microbatch_size=1, sampling="sqrt",
        device="cpu", precision="fp32", num_workers=0, initial_checkpoint=None,
    )
    weights = {source.name: source.weight for source in run.train_loader.dataset.sources}
    assert weights == {key: count ** 0.5 for key, count in run.group_counts.items()}


def test_wider_deeper_model_trains_from_scratch_and_reloads(tmp_path: Path) -> None:
    crops = _materialized_crops(tmp_path)
    options = dict(logical_batch_size=2, microbatch_size=1, sampling="balanced",
                   device="cpu", precision="fp32", num_workers=0)
    with pytest.raises(ValueError, match="omit the initial checkpoint"):
        build_signal_attr_focused_run(crops, tmp_path / "bad", width=48,
                                      initial_checkpoint=Path("models/signal_attr/best_signal_attr.pt"), **options)
    run = build_signal_attr_focused_run(crops, tmp_path / "run", width=16, blocks_per_stage=2,
                                        initial_checkpoint=None, **options)
    run.trainer.fit(run.train_loader, max_steps=1, planned_steps=1)
    run.trainer.update_best(run.evaluate()["macro_state_f1"])
    loaded = load_signal_attr_classifier_checkpoint(run.publish_checkpoint(tmp_path / "best.pt"))
    assert (loaded["model_config"].width, loaded["model_config"].blocks_per_stage) == (16, 2)


def test_cosine_schedule_decays_and_resumes(tmp_path: Path) -> None:
    crops = _materialized_crops(tmp_path)
    options = dict(logical_batch_size=2, microbatch_size=1, sampling="balanced", device="cpu",
                   precision="fp32", num_workers=0, initial_checkpoint=None,
                   lr_schedule="cosine", planned_steps=4)
    run = build_signal_attr_focused_run(crops, tmp_path / "run", **options)
    run.trainer.fit(run.train_loader, max_steps=2, planned_steps=4)
    assert run.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.5e-3)
    resumed = build_signal_attr_focused_run(crops, tmp_path / "run", resume=True, **options)
    assert resumed.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.5e-3)
