from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

VALID_SEG_OUTPUT_STRIDES = {1, 2}


def validate_seg_output_stride(seg_output_stride: int) -> int:
    stride = int(seg_output_stride)
    if stride not in VALID_SEG_OUTPUT_STRIDES:
        raise ValueError(f"invalid seg_output_stride: {stride}")
    return stride


@dataclass(frozen=True)
class PV26DetBackendOutput:
    det: Any
    p3_backbone: Tensor
    p3_head: Tensor

