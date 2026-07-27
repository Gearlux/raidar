"""Raidar: object-detection models and trainers for the Marainer workflow."""

from raidar.detection import (
    DetectionBatchInput,
    DetectionConfusionMatrix,
    DetectionOutput,
    DetectionPredictions,
    detection_collate_fn,
)
from raidar.detector import LightningDetector

# Hugging Face transformers native detection wrappers (D-FINE / RT-DETRv2 — both
# Apache-2.0, so they live in raidar core). Cheap to import: the `transformers`
# import is deferred into each model's `__init__`, so `import raidar` stays light.
# Discovered via the `confluid.configurables` package scan (their
# `@configurable(task="detection", role="model")` tags), like RFDETRModel.
from raidar.hf_detection import DFINEModel, HFDetectionModel, RTDETRv2Model
from raidar.rfdetr import RFDETRModel

# The native RF-DETR trainer/evaluator are imported here so confluid's
# `confluid.configurables` scan (entry point `raidar = "raidar"`) discovers their
# `@configurable(task="detection", role=…)` tags. The module is CHEAP to import —
# its rfdetr/PTL imports are deferred into method bodies. The Stream dataset bridge
# (`raidar.stream_rfdetr_dataset`) is NOT imported here: it eagerly pulls the heavy
# `rfdetr.training` stack, so it stays import-on-demand (the trainer imports it
# lazily in Stream mode) to keep `import raidar` / navigaitor's registry bootstrap light.
from raidar.rfdetr_trainer import RFDETREvaluator, RFDETRTrainer
from raidar.yolo_sink import YoloSink

__all__ = [
    "DFINEModel",
    "DetectionBatchInput",
    "DetectionConfusionMatrix",
    "DetectionOutput",
    "DetectionPredictions",
    "HFDetectionModel",
    "LightningDetector",
    "RFDETRModel",
    "RFDETREvaluator",
    "RFDETRTrainer",
    "RTDETRv2Model",
    "YoloSink",
    "detection_collate_fn",
]
