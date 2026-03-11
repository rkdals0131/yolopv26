from __future__ import annotations

from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DetLabelScope(str, Enum):
    FULL = "full"
    NONE = "none"


class ClassmapVersion(str, Enum):
    V2 = "classmap-v2"
    V3 = "classmap-v3"
