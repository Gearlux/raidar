"""Marainer-generic wrapper around :class:`raidar.rtdetr.RTDETRModel`.

Wraps the torchvision-native RT-DETR so it returns
:class:`~raidar.detection.DetectionStepOutput` from a
:class:`~raidar.detection.DetectionBatchInput`, which lets it plug into the
unchanged :class:`marainer.trainer.LightningTrainer` flow alongside
:class:`~raidar.detection.TorchvisionDetectionLoss` and
:class:`~raidar.detection.DetectionMetricAdapter`.

Adds the BN-frozen dual-forward path so ``val/loss`` is directly comparable
to ``train/loss``.
"""

from typing import Dict, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from torch import Tensor

from raidar.detection import DetectionBatchInput, DetectionStepOutput
from raidar.rtdetr import RTDETRModel, _build_rtdetr_batch_dict, _decode_rtdetr_predictions

logger = get_logger(__name__)


def _compute_rtdetr_loss_without_bn_drift(
    detector: nn.Module,
    batch_dict: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Run RT-DETR's training loss path under eval-style BN behavior.

    The first element is **already a 0-dim scalar** (``sum(loss.values())``
    from ``RTDETRDetectionModel.loss``), unlike YOLO26's per-component tensor.
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
class RTDETRMarainerModel(RTDETRModel):
    """Marainer-generic RT-DETR wrapper.

    Inherits from :class:`RTDETRModel` (state-dict layout preserved); overrides
    :meth:`forward` to consume :class:`DetectionBatchInput` and return
    :class:`DetectionStepOutput`.

    Adds ``compute_val_loss`` to control whether the BN-frozen recompute
    fires during eval.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["l", "x", "resnet50", "resnet101"] = "l",
        pretrained_detector: bool = True,
        imgsz: int = 640,
        score_threshold: float = 0.25,
        compute_val_loss: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            variant=variant,
            pretrained_detector=pretrained_detector,
            imgsz=imgsz,
            score_threshold=score_threshold,
            checkpoint_path=checkpoint_path,
            checkpoint_strict=checkpoint_strict,
        )
        self.compute_val_loss = bool(compute_val_loss)

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:  # type: ignore[override]
        if self.training:
            if x.targets is None:
                raise ValueError("RTDETRMarainerModel in train mode requires targets on DetectionBatchInput.")
            batch_dict = _build_rtdetr_batch_dict(x.images, x.targets, self.imgsz)
            loss_total, _items = self.detector(batch_dict)
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            batch_dict = _build_rtdetr_batch_dict(x.images, x.targets, self.imgsz)
            loss_total, _items = _compute_rtdetr_loss_without_bn_drift(self.detector, batch_dict)
            losses = {"loss": loss_total}

        img_tensor = torch.stack(x.images)
        raw, _ = self.detector(img_tensor)
        predictions = _decode_rtdetr_predictions(raw, self.imgsz, self.score_threshold)
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["RTDETRMarainerModel", "_compute_rtdetr_loss_without_bn_drift"]
