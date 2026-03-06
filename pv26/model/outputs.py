from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class PV26MultiHeadOutput:
    det: Any
    da: Tensor
    rm: Tensor
    rm_lane_subclass: Tensor

