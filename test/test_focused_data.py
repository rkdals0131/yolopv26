import json

from PIL import Image
import torch

from model.data.dataset import FocusedDataset, FocusedSource
from model.data.dataset import _roadmark_maps


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
