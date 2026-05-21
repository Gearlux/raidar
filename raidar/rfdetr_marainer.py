"""Marainer-generic wrapper around :class:`raidar.rfdetr.RFDETRModel`.

Wraps the torchvision-native RF-DETR so it returns
:class:`~raidar.detection.DetectionStepOutput` from a
:class:`~raidar.detection.DetectionBatchInput`, which lets it plug into the
unchanged :class:`marainer.trainer.LightningTrainer` flow alongside
:class:`~raidar.detection.TorchvisionDetectionLoss` and
:class:`~raidar.detection.DetectionMetricAdapter`.

Adds the no-grad val-loss recompute for direct ``train/loss``↔``val/loss``
comparison. (RF-DETR's transformer has no BatchNorm — the BN-freeze trick
used by YOLO26 / RT-DETR / Faster R-CNN doesn't apply; we just call the
criterion under ``torch.no_grad()``.)
"""

from typing import Dict, Literal, Optional

import torch
from confluid import configurable
from logflow import get_logger
from torch import Tensor

from raidar.detection import DetectionBatchInput, DetectionStepOutput
from raidar.rfdetr import RFDETRModel, _build_rfdetr_targets, _decode_rfdetr_predictions

logger = get_logger(__name__)


@configurable
class RFDETRMarainerModel(RFDETRModel):
    """Marainer-generic RF-DETR wrapper.

    Inherits from :class:`RFDETRModel` (state-dict layout preserved); overrides
    :meth:`forward` to consume :class:`DetectionBatchInput` and return
    :class:`DetectionStepOutput`.

    Adds ``compute_val_loss`` to toggle the no-grad eval-mode loss recompute.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["nano", "small", "medium", "large"] = "nano",
        pretrained_detector: bool = True,
        resolution: Optional[int] = None,
        score_threshold: float = 0.5,
        compute_val_loss: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            variant=variant,
            pretrained_detector=pretrained_detector,
            resolution=resolution,
            score_threshold=score_threshold,
            checkpoint_path=checkpoint_path,
            checkpoint_strict=checkpoint_strict,
        )
        self.compute_val_loss = bool(compute_val_loss)

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:  # type: ignore[override]
        if self.training:
            if x.targets is None:
                raise ValueError("RFDETRMarainerModel in train mode requires targets on DetectionBatchInput.")
            normalised = self._preprocess(x.images)
            targets_detr = _build_rfdetr_targets(x.targets, self.resolution, normalised.device)
            outputs = self.detector(normalised, targets_detr)
            loss_total = self._criterion_loss(outputs, targets_detr)
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        normalised = self._preprocess(x.images)
        outputs = self.detector(normalised)
        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            targets_detr = _build_rfdetr_targets(x.targets, self.resolution, normalised.device)
            with torch.no_grad():
                loss_total = self._criterion_loss(outputs, targets_detr)
            losses = {"loss": loss_total}

        target_sizes = torch.tensor(
            [[self.resolution, self.resolution]] * normalised.shape[0],
            device=normalised.device,
        )
        results = self.postprocess(outputs, target_sizes)
        predictions = _decode_rfdetr_predictions(results, self.num_classes, self.score_threshold)
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["RFDETRMarainerModel"]
