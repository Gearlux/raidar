"""Marainer-generic wrapper around :class:`raidar.yolo26.YOLO26Model`.

Wraps the torchvision-native YOLO26 so it returns
:class:`~raidar.detection.DetectionStepOutput` from a
:class:`~raidar.detection.DetectionBatchInput`, which lets it plug into the
unchanged :class:`marainer.trainer.LightningTrainer` flow alongside
:class:`~raidar.detection.TorchvisionDetectionLoss` and
:class:`~raidar.detection.DetectionMetricAdapter`.

Adds the BN-frozen dual-forward path so ``val/loss`` is directly comparable
to ``train/loss`` — the native model doesn't recompute eval-mode loss because
the specialised :class:`~raidar.lightning_trainer.LightningDetectionTrainer`
doesn't surface ``val/loss``.

Typical wiring (YAML, marainer-generic trainer)::

    trainer: !class:marainer.trainer.LightningTrainer
      ...
    model: !class:raidar.yolo26_marainer.YOLO26MarainerModel
      num_classes: 2
      variant: n
      pretrained_detector: true
      imgsz: 640
    loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
      # No `weights` block — single "loss" key passes through with weight 1.0.
    val_metrics: !class:raidar.detection.DetectionMetricAdapter
      inner: !class:torchmetrics.detection.MeanAveragePrecision
        iou_type: bbox
"""

from typing import Dict, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from torch import Tensor

from raidar.detection import DetectionBatchInput, DetectionStepOutput
from raidar.yolo26 import YOLO26Model, _build_yolo_batch_dict, _decode_yolo_predictions

logger = get_logger(__name__)


def _compute_yolo_loss_without_bn_drift(
    detector: nn.Module,
    batch_dict: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Run YOLO's training loss path under eval-style BN behavior.

    Ultralytics' ``DetectionModel.loss`` only emits its loss tensors after a
    forward pass, and that forward pass updates BN running stats when the
    model is in train mode. To make ``val/loss`` directly comparable to
    ``train/loss`` without polluting the running stats, briefly flip to train
    mode but force every BN layer back to ``eval()`` for the duration. Grad
    is disabled — we only want the scalar.
    """
    was_training = detector.training
    detector.train()
    bn_layers = [m for m in detector.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bn_layers:
        m.eval()
    try:
        with torch.no_grad():
            return cast(Tuple[Tensor, Tensor], detector(batch_dict))
    finally:
        if not was_training:
            detector.eval()


@configurable
class YOLO26MarainerModel(YOLO26Model):
    """Marainer-generic YOLO26 wrapper.

    Inherits the build / state-dict structure from :class:`YOLO26Model` (so
    existing checkpoints round-trip) but overrides :meth:`forward` to consume
    :class:`DetectionBatchInput` and return :class:`DetectionStepOutput` for
    the unchanged base :class:`marainer.trainer.LightningTrainer` flow.

    Args (all inherited from :class:`YOLO26Model`, plus):
        compute_val_loss: When False, skips the BN-frozen loss recompute
            during eval. Default True.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["n", "s", "m", "l", "x"] = "n",
        pretrained_detector: bool = True,
        imgsz: int = 640,
        score_threshold: float = 0.25,
        agnostic_nms: bool = False,
        compute_val_loss: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
        end2end: Optional[bool] = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            variant=variant,
            pretrained_detector=pretrained_detector,
            imgsz=imgsz,
            score_threshold=score_threshold,
            agnostic_nms=agnostic_nms,
            checkpoint_path=checkpoint_path,
            checkpoint_strict=checkpoint_strict,
            end2end=end2end,
        )
        self.compute_val_loss = bool(compute_val_loss)

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:  # type: ignore[override]
        if self.training:
            if x.targets is None:
                raise ValueError("YOLO26MarainerModel in train mode requires targets on DetectionBatchInput.")
            batch_dict = _build_yolo_batch_dict(x.images, x.targets, self.imgsz)
            loss_components, _items = self.detector(batch_dict)
            loss_total = loss_components.sum()
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            batch_dict = _build_yolo_batch_dict(x.images, x.targets, self.imgsz)
            loss_components, _items = _compute_yolo_loss_without_bn_drift(self.detector, batch_dict)
            losses = {"loss": loss_components.sum()}

        img_tensor = torch.stack(x.images)
        raw, _ = self.detector(img_tensor)
        predictions = _decode_yolo_predictions(raw, self.score_threshold)
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["YOLO26MarainerModel", "_compute_yolo_loss_without_bn_drift"]
