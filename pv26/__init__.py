"""PV26 package root.

Owns:
- top-level package documentation.

Does not own:
- dataset/model/loss/training implementation details.
"""

from .dataset.labels import DET_CLASSES_CANONICAL  # noqa: F401
from .loss.criterion import PV26Criterion, PV26LossBreakdown  # noqa: F401
from .model.outputs import PV26MultiHeadOutput  # noqa: F401
from .model.multitask_stub import PV26MultiHead  # noqa: F401
