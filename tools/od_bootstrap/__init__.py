"""OD bootstrap tooling for exhaustive 7-class detector supervision."""

from .cli import main
from . import data, teacher

__all__ = ["main", "data", "teacher"]
