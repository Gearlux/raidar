"""Raidar: object-detection models and adapters for the Marainer training framework."""

from raidar.detection import (
    DetectionBatchInput,
    DetectionConfusionMatrix,
    DetectionMetricAdapter,
    DetectionOutput,
    DetectionPredictions,
    DetectionStepOutput,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)
from raidar.faster_rcnn import FasterRCNNModel
from raidar.rfdetr import RFDETRModel
from raidar.rtdetr import RTDETRModel
from raidar.yolo26 import YOLO26Model

__all__ = [
    "DetectionBatchInput",
    "DetectionConfusionMatrix",
    "DetectionMetricAdapter",
    "DetectionOutput",
    "DetectionPredictions",
    "DetectionStepOutput",
    "FasterRCNNModel",
    "RFDETRModel",
    "RTDETRModel",
    "TorchvisionDetectionLoss",
    "YOLO26Model",
    "detection_collate_fn",
]
