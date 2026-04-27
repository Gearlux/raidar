"""Raidar: object-detection models and adapters for the Marainer training framework."""

from raidar.detection import (
    DetectionBatchInput,
    DetectionMetricAdapter,
    DetectionOutput,
    DetectionPredictions,
    DetectionStepOutput,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)
from raidar.faster_rcnn import FasterRCNNModel
from raidar.yolo26 import YOLO26Model

__all__ = [
    "DetectionBatchInput",
    "DetectionMetricAdapter",
    "DetectionOutput",
    "DetectionPredictions",
    "DetectionStepOutput",
    "FasterRCNNModel",
    "TorchvisionDetectionLoss",
    "YOLO26Model",
    "detection_collate_fn",
]
