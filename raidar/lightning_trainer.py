"""LightningDetectionTrainer: specialised object-detection trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that **overrides** the
step functions so it consumes torchvision-native detection models and raw
``torchmetrics`` detection metrics (``MeanAveragePrecision`` and friends)
directly — no :class:`~raidar.detection.TorchvisionDetectionLoss` /
:class:`~raidar.detection.DetectionMetricAdapter` indirection.

Model contract (the same shape torchvision detection models already speak):

* ``model.train(); model(images, targets) → Dict[str, Tensor]`` — loss dict.
  The trainer sums it to get the optimisation scalar.
* ``model.eval();  model(images)         → List[Dict[str, Tensor]]`` — per-image
  ``{boxes, scores, labels}`` in xyxy-pixel space.

Torchvision Faster R-CNN already conforms natively; pass it raw without a
raidar wrapper. The ultralytics-family detectors ship native modules at
:class:`raidar.yolo26.YOLO26Model`, :class:`raidar.rtdetr.RTDETRModel`,
:class:`raidar.rfdetr.RFDETRModel`.

Metric contract: any ``torchmetrics.Metric`` or
``torchmetrics.MetricCollection`` whose ``update(preds, targets)`` consumes
the per-image prediction list directly. Default:
``MeanAveragePrecision(iou_type="bbox")``. Use a ``MetricCollection`` to
combine MAP with e.g. :class:`~raidar.detection.DetectionConfusionMatrixNative`.

The marainer-generic adapter path (model returns ``DetectionStepOutput``,
summed by ``TorchvisionDetectionLoss``, surfaced by
``DetectionMetricAdapter``) is still supported via the base
:class:`~marainer.trainer.LightningTrainer` plus the wrappers in
``raidar.{yolo26,rtdetr,rfdetr}_marainer``.

Minimal YAML (native Faster R-CNN — no raidar model wrapper)::

    trainer: !class:raidar.lightning_trainer.LightningDetectionTrainer
      num_classes: !ref:num_classes
      model: !class:torchvision.models.detection.fasterrcnn_resnet50_fpn
        num_classes: !ref:num_classes
        weights_backbone: DEFAULT
      train_set: !ref:train_set
      val_set: !ref:val_set

Custom metric collection::

    val_metrics: !class:torchmetrics.MetricCollection
      metrics:
        - !class:torchmetrics.detection.MeanAveragePrecision
          iou_type: bbox
        - !class:raidar.detection.DetectionConfusionMatrixNative
          iou_threshold: 0.5
"""

from typing import Any, Dict, List, Optional

import pytorch_lightning as L
import torch
from confluid import Class, Lazy, configurable
from logflow import get_logger
from marainer.trainer import LightningTrainer
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import detection_collate_fn

logger = get_logger(__name__)


def _log_scalar_dict(module: L.LightningModule, metric: Any, prefix: str) -> None:
    """Compute *metric* and log every 0-dim Tensor value under *prefix*.

    Torchmetrics detection metrics typically return a dict (e.g. ``MAP``'s
    ``{map, map_50, map_75, ...}``) — we log every scalar entry. Non-scalar
    entries (``classes``, ``map_per_class`` when ``class_metrics=True``) are
    skipped because Lightning's ``self.log`` only accepts scalars.
    """
    result = metric.compute()
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, Tensor) and v.ndim == 0 and v.is_floating_point():
                module.log(f"{prefix}{k}", v)
    elif isinstance(result, Tensor) and result.ndim == 0 and result.is_floating_point():
        name = type(metric).__name__
        module.log(f"{prefix}{name}", result)


@configurable
class LightningDetectionTrainer(LightningTrainer):
    """Specialised object-detection trainer with overridden step functions.

    Args:
        model: Torchvision-native detection model. Faster R-CNN works raw;
            ultralytics-family detectors ship native modules at
            :class:`raidar.yolo26.YOLO26Model`,
            :class:`raidar.rtdetr.RTDETRModel`, and
            :class:`raidar.rfdetr.RFDETRModel`.
        train_set: Training dataset or Flux pipeline. Each sample carries
            ``input=[3,H,W]`` image, ``target={"boxes":[N,4], "labels":[N]}``.
        num_classes: Number of output classes (including background where
            applicable). Surfaced for parity with sonair / deltaid trainers
            and for downstream tooling (class_names sidecar). The default
            MAP metric is class-count-agnostic.
        val_set: Validation dataset or Flux pipeline.
        test_set: Held-out test dataset (triggers ``lightning.test()`` after fit).
        optimizer: Deferred optimizer (default: AdamW lr=1e-4, wd=0.05).
        train_loader / val_loader / test_loader: Deferred DataLoaders. Default
            collate is :func:`raidar.detection.detection_collate_fn`.
        lightning: PyTorch Lightning Trainer config.
        experiment_name / run_name: MLflow experiment / run names.
        val_metrics: ``torchmetrics.Metric`` or ``MetricCollection`` consumed
            via ``update(predictions, targets)`` in ``validation_step``.
            Default: ``MeanAveragePrecision(iou_type="bbox")``.
        test_metrics: Same shape as ``val_metrics`` — held separately so val
            and test accumulators don't share state. Default: a fresh MAP
            instance.
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
        val_metrics: Any = Class(MeanAveragePrecision, iou_type="bbox"),
        test_metrics: Any = Class(MeanAveragePrecision, iou_type="bbox"),
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
        # Native detection models compute loss inside forward — no separate
        # loss_fn callable is involved. Set to None to make that explicit (and
        # to detach the CrossEntropyLoss default the base trainer sets).
        self.loss_fn = None  # type: ignore[assignment]
        # Predictions are unavailable during training_step for most detection
        # models (the model returns a loss dict, not predictions). Leave train
        # metrics unset.
        self.train_metrics = None
        # val / test metrics may arrive as a Lazy/Class (default) or already
        # materialised. Base trainer's run() calls flow() to materialise them.
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    # --- inference helper ----------------------------------------------------

    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:  # type: ignore[override]
        return self.model(images)  # type: ignore[no-any-return]

    # --- step functions ------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        if targets is None:
            raise ValueError("LightningDetectionTrainer.training_step requires targets on the batch.")
        loss_dict = self.model(images, targets)
        if not isinstance(loss_dict, dict):
            raise TypeError(
                f"Detection model in train mode must return Dict[str, Tensor]; "
                f"got {type(loss_dict).__name__}. Use a raidar.*_marainer model with "
                f"the generic marainer LightningTrainer if you need DetectionStepOutput."
            )
        total: Tensor = sum(loss_dict.values())  # type: ignore[assignment]
        bs = len(images)
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        # Per-component breakdown — skip if the model returned a single "loss"
        # key (otherwise we'd double-log under the same name).
        if not (len(loss_dict) == 1 and "loss" in loss_dict):
            for k, v in loss_dict.items():
                if isinstance(v, Tensor) and v.numel() == 1:
                    self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        predictions = self.model(images)
        if self.val_metrics is not None and targets is not None:
            self.val_metrics.update(predictions, targets)

    def test_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        predictions = self.model(images)
        if self.test_metrics is not None and targets is not None:
            self.test_metrics.update(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is not None:
            _log_scalar_dict(self, self.val_metrics, prefix="val/")
            self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        if self.test_metrics is not None:
            _log_scalar_dict(self, self.test_metrics, prefix="test/")
            self.test_metrics.reset()


__all__ = ["LightningDetectionTrainer"]
