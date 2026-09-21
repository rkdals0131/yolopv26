"""Internal traffic-signal crop classifier and training path."""

from .aihub_policy import ProductSignalAttrTarget, extract_product_signal_attr_target
from .classifier import (
    BASE_COLORS,
    BASE_COLOR_TO_INDEX,
    TL_BITS,
    SignalAttrClassifierConfig,
    SignalAttrCropClassifier,
    SignalAttrCropTorchDataset,
    SignalAttrPrediction,
    SignalAttrThresholdPolicy,
    evaluate_signal_attr_classifier,
    load_signal_attr_classifier_checkpoint,
    product_signal_attr_prediction_from_logits,
    signal_attr_collate,
    signal_attr_crop_image_to_tensor,
    signal_attr_prediction_from_logits,
)
from .crop import (
    DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    SignalAttrCropConfig,
    crop_signal_attr_roi,
    signal_attr_crop_window,
)
from .dataset import materialize_product_signal_attr_crop_dataset_from_root
from .runtime import SignalAttrRuntime
from .training import DEFAULT_INITIAL_CHECKPOINT, SignalAttrFocusedRun, build_signal_attr_focused_run

__all__ = [
    "BASE_COLORS", "BASE_COLOR_TO_INDEX", "TL_BITS", "ProductSignalAttrTarget",
    "SignalAttrClassifierConfig", "SignalAttrCropClassifier", "SignalAttrCropTorchDataset",
    "SignalAttrPrediction", "SignalAttrThresholdPolicy", "SignalAttrCropConfig",
    "DEFAULT_SIGNAL_ATTR_CROP_CONFIG", "DEFAULT_INITIAL_CHECKPOINT", "SignalAttrRuntime",
    "SignalAttrFocusedRun", "extract_product_signal_attr_target",
    "evaluate_signal_attr_classifier", "load_signal_attr_classifier_checkpoint",
    "product_signal_attr_prediction_from_logits", "signal_attr_prediction_from_logits",
    "signal_attr_collate", "signal_attr_crop_image_to_tensor", "crop_signal_attr_roi",
    "signal_attr_crop_window", "materialize_product_signal_attr_crop_dataset_from_root",
    "build_signal_attr_focused_run",
]
