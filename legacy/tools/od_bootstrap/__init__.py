"""OD bootstrap tooling for exhaustive 7-class detector supervision."""

from .cli import main
from . import build, source, teacher

__all__ = ["main", "build", "source", "teacher"]
