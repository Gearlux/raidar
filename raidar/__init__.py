"""Raidar: object-detection models and trainers for the Marainer workflow."""

from raidar.detection import (
    DetectionBatchInput,
    DetectionConfusionMatrix,
    DetectionOutput,
    DetectionPredictions,
    detection_collate_fn,
)
from raidar.evaluator import LightningDetectionEvaluator
from raidar.lightning_trainer import LightningDetectionTrainer
from raidar.rfdetr import RFDETRModel
from raidar.yolo_sink import YoloSink

__all__ = [
    "DetectionBatchInput",
    "DetectionConfusionMatrix",
    "DetectionOutput",
    "DetectionPredictions",
    "LightningDetectionEvaluator",
    "LightningDetectionTrainer",
    "RFDETRModel",
    "YoloSink",
    "detection_collate_fn",
]
