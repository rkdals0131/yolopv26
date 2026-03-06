"""Stub multi-task assembly."""

from .multitask_impl import (
    ConvBNAct,
    DetectionHeadDense,
    PV26MultiHead,
    RoadMarkingDecoderDeconv,
    RoadMarkingPredictionHead,
    TinyFPN,
    TinyPV26Backbone,
)

__all__ = [
    "ConvBNAct",
    "DetectionHeadDense",
    "PV26MultiHead",
    "RoadMarkingDecoderDeconv",
    "RoadMarkingPredictionHead",
    "TinyFPN",
    "TinyPV26Backbone",
]

