"""Refresh stop-line negative sampling from current-model training predictions."""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset

from common.io import atomic_write_json
from model.data.dataset import collate_focused
from model.engine.postprocess import decode_roadmark_points


class StopNegativeMiner:
    def __init__(self, dataset, sampler, config: dict, output: Path) -> None:
        if dataset.split != "train" or sampler.strategy != "random_with_replacement":
            raise ValueError("online mining requires training data and random_with_replacement")
        self.dataset = copy(dataset)
        self.dataset.augment = False
        self.dataset.roadmark_target_sigma_cells = 0.0
        self.sampler, self.config, self.output = sampler, config, output
        source = str(config["source"])
        self.source = source
        with Path(config["positive_index"]).open() as file:
            positives = {(row["source"], row["sample_id"])
                         for row in map(json.loads, file) if row["kind"] == "roadmark"}
        # Existing repeated index entries are sampling tickets, not new images.
        unique = {}
        for index, record in enumerate(dataset.records):
            if record.source.name == source:
                if record.source.kind != "roadmark":
                    raise ValueError("stop mining source must contain roadmark labels")
                unique.setdefault((source, record.sample_id), index)
        if not positives or not positives.issubset(unique):
            raise ValueError("positive membership must belong to this training source")
        self.positive = [index for key, index in unique.items() if key in positives]
        self.negative = [index for key, index in unique.items() if key not in positives]
        random.Random(dataset.seed).shuffle(self.negative)
        self.source_fraction = sampler.source_ratios[source] / sum(sampler.source_ratios.values())
        self.positive_fraction = float(config["positive_fraction"])
        self.fractions = [float(value) for value in config["hard_fractions"]]
        if (not self.negative or not self.fractions or self.positive_fraction <= 0
                or any(not 0 < value < self.source_fraction - self.positive_fraction
                       for value in self.fractions)
                or int(config["interval_steps"]) <= 0 or int(config["refresh_samples"]) <= 0):
            raise ValueError("mining fractions must leave positive support for all sampling groups")

    def due(self, step: int) -> bool:
        state = self.sampler.mining_state
        return not state or step - int(state["last_step"]) >= int(self.config["interval_steps"])

    @torch.inference_mode()
    def refresh(self, trainer) -> bool:
        """Publish pools only after a complete scan; cancellation retains old pools."""
        state = self.sampler.mining_state
        round_number = int(state.get("round", 0))
        cursor = int(state.get("cursor", 0))
        count = int(self.config.get("initial_samples", 0)) if not state else int(self.config["refresh_samples"])
        count = min(count or len(self.negative), len(self.negative))
        selected = [self.negative[(cursor + offset) % len(self.negative)] for offset in range(count)]
        previous = self.sampler.sampling_groups.get(self.source, {}).get("hard", {}).get("indices", [])
        active = set(previous)
        discovered = set()
        loader = DataLoader(Subset(self.dataset, selected), batch_size=int(self.config.get("batch_size", 8)),
                            num_workers=int(self.config.get("num_workers", 4)),
                            pin_memory=True, collate_fn=collate_focused)
        iterator = iter(loader)
        was_training = trainer.model.training
        trainer.begin_evaluation()
        trainer.model.eval()
        started = time.monotonic()
        processed = 0
        print(json.dumps({"mining_start": {"step": trainer.global_step, "round": round_number,
                                          "samples": count}}), flush=True)
        try:
            for batch in iterator:
                if trainer._stop_requested:
                    return False
                # The positive list is a training membership input; check against
                # the actual decoded label before treating an image as negative.
                if any(line["class_id"] == 2 for meta in batch["meta"] for line in meta["roadmark_gt"]):
                    raise ValueError("stop-positive label found in negative mining candidates")
                images = batch["image"].to(trainer.device, non_blocking=True)
                precision = trainer.config.precision
                amp = (torch.autocast(device_type=trainer.device.type,
                       dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
                       if precision != "fp32" else nullcontext())
                with amp:
                    logits = trainer.model.forward_for_loss(images)["roadmark_logits"]
                # Only stop-line predictions are relevant to this mining pass.
                logits[:, :2] = -100
                predictions = decode_roadmark_points(logits, batch["meta"],
                                                     **dict(self.config.get("roadmark_decode") or {}))
                for offset, lines in enumerate(predictions):
                    index = selected[processed + offset]
                    if any(line["class_id"] == 2 for line in lines):
                        active.add(index)
                        discovered.add(index)
                    else:
                        active.discard(index)
                processed += len(predictions)
                if processed % 4096 == 0 or processed == count:
                    print(json.dumps({"mining_progress": {"step": trainer.global_step,
                        "samples": processed, "total": count, "fp_images": len(discovered),
                        "elapsed_sec": time.monotonic() - started}}), flush=True)
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            trainer.model.train(was_training)
            trainer.end_evaluation()
        if trainer._stop_requested:
            return False
        fraction = self.fractions[min(round_number, len(self.fractions) - 1)]
        # If the pool empties, return its allocation to ordinary negatives.
        hard_weight = fraction / self.source_fraction if active else 0.0
        positive_weight = self.positive_fraction / self.source_fraction
        regular = [index for index in self.negative if index not in active]
        regular_weight = 1 - positive_weight - hard_weight
        if not regular:
            positive_weight += regular_weight
            regular_weight = 0.0
        self.sampler.sampling_groups[self.source] = {
            "positive": {"weight": positive_weight, "indices": self.positive},
            "hard": {"weight": hard_weight, "indices": sorted(active)},
            "regular": {"weight": regular_weight, "indices": regular},
        }
        self.sampler.mining_state = {"round": round_number + 1, "last_step": trainer.global_step,
                                    "cursor": (cursor + count) % len(self.negative)}
        trainer.save_checkpoint()
        report = {**self.sampler.mining_state, "scanned": count, "fp_images": len(discovered),
                  "new_active": len(active - set(previous)), "removed_active": len(set(previous) - active),
                  "active_images": len(active), "hard_fraction": hard_weight * self.source_fraction,
                  "elapsed_sec": time.monotonic() - started}
        atomic_write_json(self.output / "hard_negative_pool.json", {
            **report, "source": self.source,
            "sample_ids": [self.dataset.records[index].sample_id for index in sorted(active)]})
        print(json.dumps({"mining": report}), flush=True)
        return True
