from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainRuntime:
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    scaler: Any


@dataclass
class EvalRuntime:
    model: torch.nn.Module
    criterion: torch.nn.Module

