import json
from pathlib import Path

from PIL import Image
import torch

from model.data.dataset import FocusedDataset, FocusedSource, LogicalBatchSampler, _roadmark_maps


def test_roadmark_horizontal_flip_preserves_raster_alignment():
    lines = [
        {"class_id": 0, "points_xy": [[8., 8.], [16., 40.]]},
        {"class_id": 2, "points_xy": [[0., 20.], [63., 24.]]},
    ]
    for padding in ((0, 0, 0, 0), (4, 8, 0, 8)):
        for color_known in (True, False):
            target, valid = _roadmark_maps(lines, (64, 64), 4, 1., padding, False, color_known)
            flipped, flipped_valid = _roadmark_maps(lines, (64, 64), 4, 1., padding, True, color_known)
            torch.testing.assert_close(flipped, torch.flip(target, (-1,)))
            torch.testing.assert_close(flipped_valid, torch.flip(valid, (-1,)))


def test_soft_roadmark_target_peaks_at_line_and_preserves_flip_and_validity():
    lines = [{"class_id": 0, "points_xy": [[10.5, 6.], [10.5, 54.]]}]
    target, valid = _roadmark_maps(lines, (64, 64), 4, 1., (0, 0, 0, 0),
                                   False, True, sigma_cells=1.0)
    flipped, flipped_valid = _roadmark_maps(lines, (64, 64), 4, 1., (0, 0, 0, 0),
                                            True, True, sigma_cells=1.0)
    assert 0 < target[0, 7, 1] < target[0, 7, 2] < 1
    assert target[0, 7, 4] < target[0, 7, 1]
    assert target[0, 0, 2] < target[0, 7, 2]  # Finite line endpoint.
    assert target[1:].count_nonzero() == 0
    torch.testing.assert_close(flipped, torch.flip(target, (-1,)))
    torch.testing.assert_close(flipped_valid, torch.flip(valid, (-1,)))


def test_declared_image_name_and_saved_membership(tmp_path):
    root = tmp_path / "traffic"
    label_dir = root / "Training" / "[라벨]sample"
    image_dir = root / "Training" / "[원천]sample" / "nested"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (16, 8), (255, 0, 0)).save(image_dir / "camera.jpg")
    (label_dir / "record.json").write_text(json.dumps({
        "image": {"filename": "camera.jpg", "imsize": [16, 8]},
        "annotation": [],
    }), encoding="utf-8")

    source = FocusedSource("traffic", root, "traffic")
    dataset = FocusedDataset([source], image_hw=(32, 32))
    assert dataset[0]["meta"]["image_path"] == str(image_dir / "camera.jpg")
    assert dataset[0]["det_labeled"]
    assert dataset[0]["cls"].numel() == 0

    index_path = tmp_path / "run" / "train.jsonl"
    dataset.save_index(index_path)
    Image.new("RGB", (16, 8), (0, 255, 0)).save(image_dir / "later.jpg")
    (label_dir / "later.json").write_text(json.dumps({
        "image": {"filename": "later.jpg", "imsize": [16, 8]},
        "annotation": [],
    }), encoding="utf-8")

    resumed = FocusedDataset([source], image_hw=(32, 32), index_path=index_path)
    assert len(resumed) == 1
    assert resumed[0]["meta"]["image_path"] == str(image_dir / "camera.jpg")


def test_joint_sample_index_seeds_a_roadmark_only_run(tmp_path):
    traffic = FocusedSource("traffic", tmp_path / "traffic", "traffic")
    roadmark = FocusedSource("roadmark", tmp_path / "roadmark", "roadmark")
    index = tmp_path / "original" / "train_samples.jsonl"
    index.parent.mkdir()
    rows = [
        {"source": source.name, "kind": source.kind, "split": "train",
         "sample_id": source.name, "image": "Training/image.jpg",
         "label": "Training/label.json"}
        for source in (traffic, roadmark)
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    subset = FocusedDataset([traffic, roadmark], index_path=index,
                            selected_kind="roadmark", image_hw=(32, 32))
    assert [record.sample_id for record in subset.records] == ["roadmark"]
    saved = tmp_path / "new" / "train_samples.jsonl"
    subset.save_index(saved)
    resumed = FocusedDataset([roadmark], index_path=saved, image_hw=(32, 32))
    assert [record.sample_id for record in resumed.records] == ["roadmark"]


def test_unknown_labels_mask_only_untrusted_supervision(tmp_path):
    traffic_root = tmp_path / "traffic"
    traffic_labels = traffic_root / "Training" / "[라벨]sample"
    traffic_images = traffic_root / "Training" / "[원천]sample"
    traffic_labels.mkdir(parents=True)
    traffic_images.mkdir(parents=True)
    Image.new("RGB", (16, 8)).save(traffic_images / "image.jpg")
    (traffic_labels / "image.json").write_text(json.dumps({
        "image": {"filename": "image.jpg", "imsize": [16, 8]},
        "annotation": [
            {"class": "traffic_light", "type": "car", "box": [1, 1, 5, 5]},
            {"class": "traffic_light", "box": [8, 1, 12, 5]},
        ],
    }), encoding="utf-8")
    traffic = FocusedDataset([FocusedSource("traffic", traffic_root, "traffic")], image_hw=(32, 32))[0]
    assert not traffic["det_labeled"]
    assert traffic["cls"].flatten().tolist() == [0]
    (traffic_labels / "image.json").write_text(json.dumps({
        "image": {"filename": "image.jpg", "imsize": [16, 8]},
        "annotation": [{"class": "traffic_light", "type": "bus", "box": [1, 1, 5, 5]}],
    }), encoding="utf-8")
    assert FocusedDataset([FocusedSource("traffic", traffic_root, "traffic")], image_hw=(32, 32))[0]["det_labeled"]

    road_root = tmp_path / "roadmark"
    road_labels = road_root / "Training" / "[라벨]sample"
    road_images = road_root / "Training" / "[원천]sample"
    road_labels.mkdir(parents=True)
    road_images.mkdir(parents=True)
    Image.new("RGB", (16, 8)).save(road_images / "image.jpg")
    (road_labels / "image.json").write_text(json.dumps({
        "image": {"file_name": "image.jpg", "image_size": [8, 16]},
        "annotations": [
            {"class": "traffic_lane", "category": "polyline", "attributes": [],
             "data": [{"x": 1, "y": 3}, {"x": 14, "y": 3}]},
            {"class": "stop_line", "category": "polyline", "attributes": [],
             "data": [{"x": 1, "y": 5}, {"x": 14, "y": 5}]},
        ],
    }), encoding="utf-8")
    roadmark = FocusedDataset([FocusedSource("roadmark", road_root, "roadmark")], image_hw=(32, 32))[0]
    assert roadmark["roadmark_valid"].any(dim=(1, 2)).tolist() == [False, False, True]
    assert roadmark["roadmark_target"].sum(dim=(1, 2)).tolist()[:2] == [0.0, 0.0]
    assert roadmark["roadmark_target"][2].sum() > 0
    assert [line["class_name"] for line in roadmark["meta"]["roadmark_gt"]] == ["stop_line"]
    (road_labels / "image.json").write_text(json.dumps({
        "image": {"file_name": "image.jpg", "image_size": [8, 16]},
        "annotations": [{"class": "traffic_lane", "category": "polyline",
                         "attributes": [{"code": "lane_color", "value": "blue"}],
                         "data": [{"x": 1, "y": 3}, {"x": 14, "y": 3}]}],
    }), encoding="utf-8")
    blue = FocusedDataset([FocusedSource("roadmark", road_root, "roadmark")], image_hw=(32, 32))[0]
    assert blue["roadmark_valid"].any(dim=(1, 2)).tolist() == [True, True, True]
    assert blue["roadmark_target"].sum() == 0


def test_mined_sampling_share_and_resume_after_pool_refresh():
    dataset = type("DatasetStub", (), {
        "sources": [FocusedSource("traffic", Path("."), "traffic", 1),
                    FocusedSource("roadmark", Path("."), "roadmark", 2)],
        "indices_by_source": {"traffic": [0], "roadmark": [1, 2, 3, 4]},
        "__len__": lambda self: 5,
    })()
    sampler = LogicalBatchSampler(dataset, batch_size=32, seed=26)
    sampler.sampling_groups = {"roadmark": {
        "positive": {"weight": .2, "indices": [1]},
        "hard": {"weight": .75, "indices": [2]},
        "regular": {"weight": .05, "indices": [3, 4]},
    }}
    iterator = iter(sampler)
    draws = []
    for _ in range(400):
        draws.extend(index for index, _ in next(iterator))
        sampler.commit(32)
    assert abs(draws.count(2) / len(draws) - .5) < .02
    assert set(draws) == {0, 1, 2, 3, 4}
    # A newly mined example replaces the previous hard example at a boundary.
    sampler.sampling_groups["roadmark"]["hard"]["indices"] = [4]
    sampler.sampling_groups["roadmark"]["regular"]["indices"] = [2, 3]
    sampler.mining_state = {"round": 2, "last_step": 400, "cursor": 128}
    expected = next(iter(sampler))
    resumed = LogicalBatchSampler(dataset, batch_size=32, seed=26)
    resumed.load_state_dict(sampler.state_dict())
    assert next(iter(resumed)) == expected
    assert resumed.mining_state == sampler.mining_state


def test_two_choice_sampler_balances_exposure_and_resumes_from_committed_position():
    dataset = type("Dataset", (), {
        "sources": (FocusedSource("traffic", "/unused", "traffic"),),
        "indices_by_source": {"traffic": list(range(32))},
        "__len__": lambda self: 32,
    })()
    random_sampler = LogicalBatchSampler(dataset, batch_size=8, seed=26)
    balanced_sampler = LogicalBatchSampler(
        dataset, batch_size=8, seed=26, strategy="least_used_of_two"
    )

    random_iterator = iter(random_sampler)
    random_draws = [index for _ in range(40) for index, _ in next(random_iterator)]
    balanced_iterator = iter(balanced_sampler)
    balanced_draws = [index for _ in range(40) for index, _ in next(balanced_iterator)]
    random_counts = [random_draws.count(index) for index in range(32)]
    balanced_counts = [balanced_draws.count(index) for index in range(32)]
    assert max(balanced_counts) - min(balanced_counts) < max(random_counts) - min(random_counts)

    committed = LogicalBatchSampler(
        dataset, batch_size=8, seed=26, strategy="least_used_of_two"
    )
    committed_iterator = iter(committed)
    for _ in range(7):
        next(committed_iterator)
        committed.commit(8)
    expected_next = next(committed_iterator)
    resumed = LogicalBatchSampler(
        dataset, batch_size=8, seed=26, start_position=56,
        strategy="least_used_of_two",
    )
    assert next(iter(resumed)) == expected_next
