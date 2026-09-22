"""Direct AIHub feeder for the focused signal/roadmark model.

Only labeled JSON records enter this dataset. An empty annotation list in a
record is a known negative for that source's task; an image without a record is
not silently turned into a negative. The sampler's position is committed by
the trainer, independently of DataLoader's prefetched iterator position.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.utils.data import Dataset, Sampler

from common.schema import (
    DEFAULT_IMAGE_HW,
    ROADMARK_CLASSES,
    ROADMARK_STRIDE,
    SIGNAL_CLASSES,
)


_VEHICLE_ID = SIGNAL_CLASSES.index("vehicle_signal")
_PEDESTRIAN_ID = SIGNAL_CLASSES.index("pedestrian_signal")
_WHITE_ID = ROADMARK_CLASSES.index("white_lane")
_YELLOW_ID = ROADMARK_CLASSES.index("yellow_lane")
_STOP_ID = ROADMARK_CLASSES.index("stop_line")


DEFAULT_TRAFFIC_ROOT = Path(
    "/home/user1/Storage/seg_dataset/AIHUB/신호등-도로표지판 인지 영상(수도권)"
)
DEFAULT_ROADMARK_ROOT = Path(
    "/home/user1/Storage/seg_dataset/AIHUB/차선-횡단보도 인지 영상(수도권)"
)


@dataclass(frozen=True)
class FocusedSource:
    name: str
    root: Path
    kind: str  # "traffic" or "roadmark"
    weight: float = 1.0


@dataclass(frozen=True)
class _Record:
    source: FocusedSource
    sample_id: str
    image_path: Path
    label_path: Path
    split: str


def default_aihub_sources(
    traffic_root: Path = DEFAULT_TRAFFIC_ROOT,
    roadmark_root: Path = DEFAULT_ROADMARK_ROOT,
) -> tuple[FocusedSource, FocusedSource]:
    return (
        FocusedSource("aihub_traffic", Path(traffic_root), "traffic"),
        FocusedSource("aihub_roadmark", Path(roadmark_root), "roadmark"),
    )


def _split_dir(split: str) -> str:
    normalized = split.lower()
    if normalized in {"train", "training"}:
        return "Training"
    if normalized in {"val", "validation"}:
        return "Validation"
    raise ValueError(f"unsupported AIHub split: {split}")


def _read_label(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        label = json.load(file)
    if not isinstance(label, dict):
        raise ValueError(f"AIHub label must be a JSON object: {path}")
    return label


def _image_name(label: Mapping[str, Any], kind: str, path: Path) -> str:
    image = label.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"AIHub image metadata missing: {path}")
    name = image.get("filename" if kind == "traffic" else "file_name")
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ValueError(f"invalid AIHub image filename: {path}")
    return name


def _label_hw(label: Mapping[str, Any], kind: str, path: Path) -> tuple[int, int]:
    image = label.get("image")
    values = image.get("imsize" if kind == "traffic" else "image_size") if isinstance(image, dict) else None
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"invalid AIHub image size: {path}")
    if kind == "traffic":
        width, height = values
    else:
        height, width = values
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (height, width)):
        raise ValueError(f"invalid AIHub image size: {path}")
    return height, width


def _source_records(source: FocusedSource, split: str, limit: int | None) -> list[_Record]:
    split_root = Path(source.root) / _split_dir(split)
    if not split_root.is_dir():
        raise FileNotFoundError(f"AIHub split not found: {split_root}")
    labels = sorted(path for path in split_root.rglob("*.json")
                    if path.relative_to(split_root).parts[0].startswith("[라벨]"))
    if limit is not None:
        labels = labels[:limit]
    records: list[_Record] = []
    image_maps: dict[str, dict[str, Path]] = {}
    for label_path in labels:
        relative = label_path.relative_to(split_root)
        label_group = relative.parts[0]
        if not label_group.startswith("[라벨]"):
            continue
        image_group = "[원천]" + label_group[len("[라벨]"):]
        if image_group not in image_maps:
            image_root = split_root / image_group
            if not image_root.is_dir():
                raise FileNotFoundError(f"AIHub image group not found: {image_root}")
            image_map: dict[str, Path] = {}
            for path in image_root.rglob("*"):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if path.name in image_map:
                    raise ValueError(f"ambiguous AIHub image filename: {path.name} in {image_root}")
                image_map[path.name] = path
            image_maps[image_group] = image_map
        label = _read_label(label_path)
        image_name = _image_name(label, source.kind, label_path)
        image_path = image_maps[image_group].get(image_name)
        if image_path is None:
            raise FileNotFoundError(f"AIHub image {image_name} for label not found: {label_path}")
        records.append(_Record(source, label_path.stem, image_path, label_path, split))
    if not records:
        raise ValueError(f"no labeled AIHub samples in {split_root}")
    return records


def _index_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_index(path: Path, sources: Sequence[FocusedSource], split: str) -> list[_Record]:
    by_name = {source.name: source for source in sources}
    records: list[_Record] = []
    seen_sources: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            source_name = row["source"]
            source = by_name.get(source_name)
            if source is None or row["kind"] != source.kind or row["split"] != split:
                raise ValueError(f"focused index source/split mismatch: {path}:{line_number}")
            relative_paths: list[Path] = []
            for key in ("image", "label"):
                relative = Path(row[key])
                if (relative.is_absolute() or ".." in relative.parts
                        or not relative.parts or relative.parts[0] != _split_dir(split)):
                    raise ValueError(f"focused index path must belong to the selected split: {path}:{line_number}")
                relative_paths.append(relative)
            records.append(_Record(
                source, str(row["sample_id"]), Path(source.root) / relative_paths[0],
                Path(source.root) / relative_paths[1], split,
            ))
            seen_sources.add(source_name)
    if seen_sources != set(by_name):
        raise ValueError(f"focused index does not cover configured sources: {path}")
    return records


def _letterbox(image: Image.Image, image_hw: tuple[int, int]) -> tuple[Image.Image, float, tuple[int, int, int, int]]:
    target_h, target_w = image_hw
    scale = min(target_w / image.width, target_h / image.height)
    new_w = round(image.width * scale)
    new_h = round(image.height * scale)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    right = target_w - new_w - left
    bottom = target_h - new_h - top
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
    return canvas, scale, (left, top, right, bottom)


def _transform_meta(raw_hw: tuple[int, int], image_hw: tuple[int, int],
                    scale: float, padding: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        "original_hw": raw_hw,
        "raw_hw": raw_hw,
        "network_hw": image_hw,
        "scale": scale,
        "padding": padding,
        "transform": {
            "scale": scale,
            "pad_left": padding[0],
            "pad_top": padding[1],
            "pad_right": padding[2],
            "pad_bottom": padding[3],
            "resized_hw": (image_hw[0] - padding[1] - padding[3],
                           image_hw[1] - padding[0] - padding[2]),
        },
        "flipped": False,
    }


def letterbox_focused_image(
    image: Image.Image | np.ndarray,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert RGB image to the exact training geometry and its inverse metadata."""
    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    raw_hw = (pil_image.height, pil_image.width)
    canvas, scale, padding = _letterbox(pil_image, image_hw)
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    return tensor, _transform_meta(raw_hw, image_hw, scale, padding)


def _box_target(box: Any, raw_hw: tuple[int, int], image_hw: tuple[int, int], scale: float,
                padding: tuple[int, int, int, int], flip: bool) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    raw_h, raw_w = raw_hw
    x1, x2 = max(0.0, min(x1, raw_w)), max(0.0, min(x2, raw_w))
    y1, y2 = max(0.0, min(y1, raw_h)), max(0.0, min(y2, raw_h))
    if x2 <= x1 or y2 <= y1:
        return None
    out_h, out_w = image_hw
    left, top, _, _ = padding
    x1, x2 = x1 * scale + left, x2 * scale + left
    y1, y2 = y1 * scale + top, y2 * scale + top
    if flip:
        x1, x2 = out_w - x2, out_w - x1
    return [(x1 + x2) / (2 * out_w), (y1 + y2) / (2 * out_h),
            (x2 - x1) / out_w, (y2 - y1) / out_h]


def _traffic_boxes(label: Mapping[str, Any], raw_hw: tuple[int, int], image_hw: tuple[int, int],
                   scale: float, padding: tuple[int, int, int, int], flip: bool,
                   path: Path) -> tuple[torch.Tensor, torch.Tensor, bool]:
    rows = label.get("annotation")
    if not isinstance(rows, list):
        raise ValueError(f"AIHub traffic annotation missing: {path}")
    classes: list[list[int]] = []
    boxes: list[list[float]] = []
    det_labeled = True
    for row in rows:
        if not isinstance(row, dict) or row.get("class") != "traffic_light":
            continue
        kind = str(row.get("type") or "").strip().lower()
        if kind not in {"car", "pedestrian"}:
            if kind not in {"bus", "bicycle"}:
                det_labeled = False
            continue
        box = _box_target(row.get("box"), raw_hw, image_hw, scale, padding, flip)
        if box is not None:
            classes.append([_VEHICLE_ID if kind == "car" else _PEDESTRIAN_ID])
            boxes.append(box)
    return (torch.tensor(classes, dtype=torch.long).reshape(-1, 1),
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), det_labeled)


def _roadmark_lines(label: Mapping[str, Any], path: Path) -> tuple[list[dict[str, Any]], bool]:
    rows = label.get("annotations")
    if not isinstance(rows, list):
        raise ValueError(f"AIHub roadmark annotations missing: {path}")
    lines: list[dict[str, Any]] = []
    lane_color_known = True
    for row in rows:
        if not isinstance(row, dict):
            continue
        category = row.get("class")
        if category == "stop_line":
            class_id = _STOP_ID
        elif category == "traffic_lane":
            attributes = row.get("attributes")
            colors = [item.get("value") for item in attributes if isinstance(item, dict)
                      and item.get("code") == "lane_color"] if isinstance(attributes, list) else []
            if "white" in colors:
                class_id = _WHITE_ID
            elif "yellow" in colors:
                class_id = _YELLOW_ID
            elif "blue" in colors:
                continue
            else:
                lane_color_known = False
                continue
        else:
            continue
        if row.get("category") != "polyline":
            continue
        points = row.get("data")
        if not isinstance(points, list) or len(points) < 2:
            continue
        try:
            coords = [(float(point["x"]), float(point["y"])) for point in points]
        except (TypeError, ValueError, KeyError):
            continue
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in coords):
            continue
        lines.append({
            "class_id": class_id,
            "class_name": ROADMARK_CLASSES[class_id],
            "points_xy": [[x, y] for x, y in coords],
        })
    return lines, lane_color_known


def _roadmark_maps(lines: Sequence[Mapping[str, Any]], image_hw: tuple[int, int],
                   stride: int, scale: float, padding: tuple[int, int, int, int],
                   flip: bool, lane_color_known: bool) -> tuple[torch.Tensor, torch.Tensor]:
    out_h, out_w = image_hw[0] // stride, image_hw[1] // stride
    masks = [Image.new("L", (out_w, out_h), 0) for _ in ROADMARK_CLASSES]
    draws = [ImageDraw.Draw(mask) for mask in masks]
    left, top, _, _ = padding
    for line in lines:
        transformed = [((x * scale + left) / stride, (y * scale + top) / stride)
                       for x, y in line["points_xy"]]
        draws[line["class_id"]].line(transformed, fill=255, width=1)
    target = torch.from_numpy(np.stack([np.asarray(mask, dtype=np.float32) / 255.0 for mask in masks]))
    if flip:
        # Mirror the rasterized target exactly as the image; out_w - x
        # shifts the supervision one output pixel to the right.
        target = torch.flip(target, dims=(-1,))
    valid_2d = torch.zeros((out_h, out_w), dtype=torch.bool)
    right_exclusive = math.floor((image_hw[1] - padding[2]) / stride)
    bottom_exclusive = math.floor((image_hw[0] - padding[3]) / stride)
    valid_2d[math.ceil(top / stride):bottom_exclusive, math.ceil(left / stride):right_exclusive] = True
    if flip:
        valid_2d = torch.flip(valid_2d, dims=(-1,))
    valid = valid_2d.unsqueeze(0).expand(len(ROADMARK_CLASSES), -1, -1).clone()
    if not lane_color_known:
        valid[[_WHITE_ID, _YELLOW_ID]] = False
    target *= valid
    return target, valid


class FocusedDataset(Dataset[dict[str, Any]]):
    def __init__(self, sources: Sequence[FocusedSource], *, split: str = "train",
                 image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
                 roadmark_stride: int = ROADMARK_STRIDE, augment: bool = False,
                 seed: int = 0, sample_limit_per_source: int | None = None,
                 index_path: Path | None = None) -> None:
        if not sources or len({source.name for source in sources}) != len(sources):
            raise ValueError("focused sources must have unique names")
        if any(source.kind not in {"traffic", "roadmark"} for source in sources):
            raise ValueError("focused source kind must be traffic or roadmark")
        if image_hw[0] <= 0 or image_hw[1] <= 0 or roadmark_stride <= 0 or any(
            value % roadmark_stride for value in image_hw
        ):
            raise ValueError("image dimensions must be positive multiples of roadmark_stride")
        if sample_limit_per_source is not None and sample_limit_per_source <= 0:
            raise ValueError("sample_limit_per_source must be positive")
        if index_path is not None and sample_limit_per_source is not None:
            raise ValueError("sample_limit_per_source cannot trim a saved index")
        self.sources = tuple(sources)
        self.split = "train" if _split_dir(split) == "Training" else "val"
        self.image_hw = tuple(image_hw)
        self.roadmark_stride = roadmark_stride
        self.augment = bool(augment and self.split == "train")
        self.seed = int(seed)
        self.records = (
            _load_index(Path(index_path), sources, self.split)
            if index_path is not None
            else [record for source in sources for record in
                  _source_records(source, self.split, sample_limit_per_source)]
        )
        self.indices_by_source = {
            source.name: [index for index, record in enumerate(self.records)
                          if record.source.name == source.name]
            for source in sources
        }

    def save_index(self, path: Path) -> None:
        """Atomically save this run's ordered membership at the caller's path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp",
                                             delete=False) as file:
                temporary_path = Path(file.name)
                for record in self.records:
                    row = {
                        "source": record.source.name,
                        "kind": record.source.kind,
                        "split": record.split,
                        "sample_id": record.sample_id,
                        "image": _index_relative(record.image_path, Path(record.source.root)),
                        "label": _index_relative(record.label_path, Path(record.source.root)),
                    }
                    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        index, draw_id = key if isinstance(key, tuple) else (key, key)
        record = self.records[index]
        label = _read_label(record.label_path)
        declared_name = _image_name(label, record.source.kind, record.label_path)
        if declared_name != record.image_path.name:
            raise ValueError(f"AIHub image identity mismatch: {record.label_path}")
        declared_hw = _label_hw(label, record.source.kind, record.label_path)
        with Image.open(record.image_path) as opened:
            raw = opened.convert("RGB")
        raw_hw = (raw.height, raw.width)
        if raw_hw != declared_hw:
            raise ValueError(f"AIHub image dimensions mismatch: {record.label_path}")
        image, scale, padding = _letterbox(raw, self.image_hw)
        rng = random.Random((self.seed << 32) + int(draw_id))
        flip = self.augment and rng.random() < 0.5
        if flip:
            image = ImageOps.mirror(image)
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        map_hw = (len(ROADMARK_CLASSES), self.image_hw[0] // self.roadmark_stride,
                  self.image_hw[1] // self.roadmark_stride)
        if record.source.kind == "traffic":
            cls, bboxes, det_labeled = _traffic_boxes(
                label, raw_hw, self.image_hw, scale, padding, flip, record.label_path)
            roadmark_target = torch.zeros(map_hw, dtype=torch.float32)
            roadmark_valid = torch.zeros(map_hw, dtype=torch.bool)
            roadmark_gt: list[dict[str, Any]] = []
        else:
            cls = torch.empty((0, 1), dtype=torch.long)
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            det_labeled = False
            roadmark_gt, lane_color_known = _roadmark_lines(label, record.label_path)
            roadmark_target, roadmark_valid = _roadmark_maps(
                roadmark_gt, self.image_hw, self.roadmark_stride, scale, padding,
                flip, lane_color_known)
        return {
            "image": image_tensor,
            "cls": cls,
            "bboxes": bboxes,
            "det_labeled": det_labeled,
            "roadmark_target": roadmark_target,
            "roadmark_valid": roadmark_valid,
            "meta": {
                "sample_id": record.sample_id,
                "source": record.source.name,
                "split": record.split,
                "image_path": str(record.image_path),
                **_transform_meta(raw_hw, self.image_hw, scale, padding),
                "flipped": flip,
                "draw_id": draw_id,
                "roadmark_gt": roadmark_gt,
            },
        }


def collate_focused(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = [sample["cls"] for sample in samples]
    boxes = [sample["bboxes"] for sample in samples]
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "det_labeled": torch.tensor([sample["det_labeled"] for sample in samples], dtype=torch.bool),
        "roadmark_target": torch.stack([sample["roadmark_target"] for sample in samples]),
        "roadmark_valid": torch.stack([sample["roadmark_valid"] for sample in samples]),
        "batch_idx": torch.cat([torch.full((len(cls),), index, dtype=torch.long)
                                for index, cls in enumerate(classes)]),
        "cls": torch.cat(classes, dim=0),
        "bboxes": torch.cat(boxes, dim=0),
        "meta": [sample["meta"] for sample in samples],
    }


def slice_focused_batch(batch: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Return a physical microbatch with detector indices remapped to zero."""
    count = int(batch["image"].shape[0])
    if not 0 <= start < stop <= count:
        raise ValueError(f"invalid focused batch slice: {start}:{stop} of {count}")
    selected = (batch["batch_idx"] >= start) & (batch["batch_idx"] < stop)
    return {
        "image": batch["image"][start:stop],
        "det_labeled": batch["det_labeled"][start:stop],
        "roadmark_target": batch["roadmark_target"][start:stop],
        "roadmark_valid": batch["roadmark_valid"][start:stop],
        "batch_idx": batch["batch_idx"][selected] - start,
        "cls": batch["cls"][selected],
        "bboxes": batch["bboxes"][selected],
        "meta": batch["meta"][start:stop],
    }


def _radical_inverse_base2(number: int) -> float:
    result = 0.0
    fraction = 0.5
    while number:
        result += (number & 1) * fraction
        number >>= 1
        fraction *= 0.5
    return result


class LogicalBatchSampler(Sampler[list[tuple[int, int]]]):
    """Infinite deterministic source mix, indexed by *consumed* sample position.

    DataLoader may prefetch future batches. Only ``commit(n_samples)`` advances
    the durable position. Resume by creating a new DataLoader iterator after
    ``load_state_dict``; an existing prefetched iterator cannot be rewound.
    """

    def __init__(self, dataset: FocusedDataset, *, batch_size: int,
                 source_ratios: Mapping[str, float] | None = None, seed: int = 0,
                 start_position: int = 0) -> None:
        if batch_size <= 0 or start_position < 0 or start_position % batch_size:
            raise ValueError("batch_size must be positive and position batch-aligned")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.position = int(start_position)
        ratios = {source.name: float(source.weight) for source in dataset.sources}
        if source_ratios is not None:
            if set(source_ratios) != set(ratios):
                raise ValueError("source_ratios must name every configured source")
            ratios = {name: float(source_ratios[name]) for name in ratios}
        if any(not math.isfinite(value) or value < 0 for value in ratios.values()) or sum(ratios.values()) <= 0:
            raise ValueError("source ratios must be finite non-negative with positive total")
        self.source_ratios = ratios
        total = sum(ratios.values())
        cumulative = 0.0
        self._ranges: list[tuple[float, str]] = []
        for name, ratio in ratios.items():
            if ratio == 0:
                continue
            cumulative += ratio / total
            self._ranges.append((cumulative, name))
        self._ranges[-1] = (1.0, self._ranges[-1][1])
        self._offset = random.Random(seed).randrange(1 << 30)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        cursor = self.position
        while True:
            keys: list[tuple[int, int]] = []
            for draw_id in range(cursor, cursor + self.batch_size):
                fraction = _radical_inverse_base2(draw_id + self._offset + 1)
                source_name = next(name for end, name in self._ranges if fraction < end)
                indices = self.dataset.indices_by_source[source_name]
                sample_rng = random.Random((self.seed << 32) + draw_id)
                keys.append((indices[sample_rng.randrange(len(indices))], draw_id))
            yield keys
            cursor += self.batch_size

    def commit(self, n_samples: int) -> None:
        if n_samples != self.batch_size:
            raise ValueError("commit exactly one completed logical batch")
        self.position += n_samples

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position, "batch_size": self.batch_size,
                "seed": self.seed, "source_ratios": dict(self.source_ratios)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (int(state["batch_size"]) != self.batch_size or int(state["seed"]) != self.seed
                or dict(state["source_ratios"]) != self.source_ratios):
            raise ValueError("resume sampler configuration differs from saved run")
        position = int(state["position"])
        if position < 0 or position % self.batch_size:
            raise ValueError("invalid resumed sampler position")
        self.position = position
