"""LightningDetectionTrainer: specialised object-detection trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that bakes in
detection-specific defaults so a minimal YAML config only needs to specify
the dataset, model, and number of classes — no adapter wiring for loss,
collate, or metrics required.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loss_fn`` — :class:`~raidar.detection.TorchvisionDetectionLoss`. Works
  unchanged for every detector raidar ships: Faster R-CNN's multi-component
  loss dict (``loss_classifier``, ``loss_box_reg``, ``loss_objectness``,
  ``loss_rpn_box_reg``) is summed; ultralytics-family detectors emit a
  single-key ``{"loss": total}`` which is also summed (trivially).
* ``train/val/test_loader`` ``collate_fn`` —
  :func:`~raidar.detection.detection_collate_fn`. Variable-length targets
  are preserved as lists.
* ``optimizer`` — ``AdamW(lr=1e-4, weight_decay=0.05)``.
* ``train_metrics`` — ``None``. Detection training metrics break gradient
  flow (predictions are not produced in train mode); leave unset by default.
* ``val_metrics`` / ``test_metrics`` —
  :class:`~raidar.detection.DetectionMetricAdapter` wrapping
  :class:`torchmetrics.detection.MeanAveragePrecision` with ``iou_type=bbox``.
  Surfaces ``map``, ``map_50``, ``map_75``, ``mar_*`` as Lightning-loggable
  scalars under ``val/`` and ``test/`` prefixes.

Minimal YAML example::

    trainer: !class:raidar.trainer.LightningDetectionTrainer
      num_classes: 80
      model: !class:raidar.faster_rcnn.FasterRCNNModel
        num_classes: !ref:num_classes
      train_set: !ref:train_set
      val_set: !ref:val_set

Override with an upstream-shape metric (no raidar adapter needed beyond
the inner-metric unwrapping) — pass any torchmetrics detection metric::

    trainer: !class:raidar.trainer.LightningDetectionTrainer
      num_classes: !ref:num_classes
      model: !ref:model
      train_set: !ref:train_set
      val_set: !ref:val_set
      val_metrics: !class:raidar.detection.DetectionMetricAdapter
        inner: !class:torchmetrics.detection.MeanAveragePrecision
          iou_type: bbox
          class_metrics: true
"""

from typing import Any, Optional

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from confluid import Class, Lazy, configurable
from marainer.trainer import LightningTrainer
from raidar.detection import (
    DetectionMetricAdapter,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)


def _default_detection_metrics(prefix: str) -> DetectionMetricAdapter:
    """Build the default detection metric adapter for a given Lightning split prefix.

    Returns a fresh :class:`DetectionMetricAdapter` over a fresh
    :class:`MeanAveragePrecision` instance — independent state per split so
    val and test accumulations don't bleed into each other.
    """
    return DetectionMetricAdapter(MeanAveragePrecision(iou_type="bbox"), prefix=prefix)


@configurable
class LightningDetectionTrainer(LightningTrainer):
    """Specialised object-detection trainer with baked-in defaults.

    Args:
        model: Detection model that returns a
            :class:`~raidar.detection.DetectionStepOutput`
            (``{losses, predictions}``). All four raidar-shipped detectors
            (Faster R-CNN, YOLO26, RT-DETR, RF-DETR) conform to this contract.
        train_set: Training dataset or Flux pipeline.
        num_classes: Number of output classes including background where
            applicable. Surfaced for parity with sonair / deltaid trainers
            and for downstream tooling (class_names sidecar). The MAP metric
            itself is class-count-agnostic.
        val_set: Validation dataset or Flux pipeline.
        test_set: Held-out test dataset (triggers ``lightning.test()`` after fit).
        optimizer: Deferred optimizer (default: AdamW lr=1e-4, wd=0.05).
        train_loader: Deferred DataLoader (default includes
            ``detection_collate_fn``).
        val_loader: Deferred DataLoader (default includes
            ``detection_collate_fn``).
        test_loader: Deferred DataLoader (default includes
            ``detection_collate_fn``).
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        train_set: Any,
        num_classes: int,
        val_set: Any = None,
        test_set: Any = None,
        optimizer: Lazy[Any] = Class(torch.optim.AdamW, lr=1e-4, weight_decay=0.05),
        train_loader: Lazy[Any] = Class(DataLoader, collate_fn=detection_collate_fn),
        val_loader: Lazy[Any] = Class(DataLoader, collate_fn=detection_collate_fn),
        test_loader: Lazy[Any] = Class(DataLoader, collate_fn=detection_collate_fn),
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "raidar_detection",
        run_name: str = "train",
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lightning=lightning,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        self.num_classes = int(num_classes)
        self.loss_fn = TorchvisionDetectionLoss()
        # Training metrics break gradient flow in detection (no predictions
        # produced during training_step on most detectors). Leave None so the
        # base Trainer skips the train-metric path entirely.
        self.train_metrics: Optional[MetricCollection] = None
        self.val_metrics: Optional[MetricCollection] = _default_detection_metrics(
            prefix="val/"
        )
        self.test_metrics: Optional[MetricCollection] = _default_detection_metrics(
            prefix="test/"
        )
