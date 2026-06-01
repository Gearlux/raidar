"""Raidar: object-detection models and trainers for the Marainer workflow."""

from raidar.detection import (
    DetectionBatchInput,
    DetectionConfusionMatrix,
    DetectionConfusionMatrixNative,
    DetectionMetricAdapter,
    DetectionOutput,
    DetectionPredictions,
    DetectionStepOutput,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)
from raidar.evaluator import LightningDetectionEvaluator
from raidar.faster_rcnn import FasterRCNNModel
from raidar.lightning_trainer import LightningDetectionTrainer
from raidar.rfdetr import RFDETRModel
from raidar.rfdetr_marainer import RFDETRMarainerModel
from raidar.ultralytics_sink import UltralyticsSink

__all__ = [
    "DetectionBatchInput",
    "DetectionConfusionMatrix",
    "DetectionConfusionMatrixNative",
    "DetectionMetricAdapter",
    "DetectionOutput",
    "DetectionPredictions",
    "DetectionStepOutput",
    "FasterRCNNModel",
    "LightningDetectionEvaluator",
    "LightningDetectionTrainer",
    "RFDETRMarainerModel",
    "RFDETRModel",
    "TorchvisionDetectionLoss",
    "UltralyticsSink",
    "detection_collate_fn",
]
