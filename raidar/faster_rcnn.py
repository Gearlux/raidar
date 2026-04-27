"""Faster R-CNN model wrapper for the Marainer Trainer flow.

This module provides a torchvision Faster R-CNN wrapped to plug into the
**unchanged** base :class:`~marainer.trainer.Trainer`. The generic detection
plumbing it relies on (:class:`~raidar.detection.DetectionBatchInput`,
:class:`~raidar.detection.DetectionStepOutput`,
:class:`~raidar.detection.TorchvisionDetectionLoss`,
:class:`~raidar.detection.DetectionMetricAdapter`,
:func:`~raidar.detection.detection_collate_fn`) lives in :mod:`raidar.detection`
and is shared with any other torchvision-style detector family added later.

:class:`FasterRCNNModel` always returns a
:class:`~raidar.detection.DetectionStepOutput` — in train mode it's
``{losses: {...}, predictions: None}``; in eval mode (with targets) it's
``{losses: {...}, predictions: [DetectionOutput, ...]}`` via a BN-frozen
dual-forward so val loss is directly comparable to train loss.

Typical wiring (YAML)::

    model: !class:raidar.faster_rcnn.FasterRCNNModel
      num_classes: 2
      variant: v2
      pretrained_detector: true
    loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
      weights: { loss_classifier: 1.0, loss_box_reg: 1.0 }
"""

from typing import Dict, List, Literal, Optional, cast

import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from marainer.outputs import DetectionOutput
from torch import Tensor

from raidar.detection import DetectionBatchInput, DetectionStepOutput

logger = get_logger(__name__)


def _build_faster_rcnn(
    variant: Literal["v1", "v2"],
    num_classes: int,
    pretrained_backbone: bool,
    pretrained_detector: bool,
    trainable_backbone_layers: int,
    min_size: int,
    max_size: int,
) -> nn.Module:
    """Instantiate a torchvision Faster R-CNN with the box predictor head
    replaced to match ``num_classes`` (including background=0).

    With ``pretrained_detector=True`` the model loads the **full** COCO-trained
    weights (91-class predictor included). The 91-class predictor is then
    swapped for a fresh ``FastRCNNPredictor(in_features, num_classes)`` — the
    backbone, FPN, RPN, and ROI pooler keep their pretrained weights, only the
    final classification + box-regression heads start random. This usually
    converges far better on small datasets than training the heads + RPN from
    scratch, since the RPN already produces sensible region proposals.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    factory = fasterrcnn_resnet50_fpn_v2 if variant == "v2" else fasterrcnn_resnet50_fpn
    if pretrained_detector:
        # Full COCO-pretrained model. ``weights_backbone`` is redundant when the
        # full detector weights are loaded — torchvision raises if both are set.
        detector = factory(
            weights="DEFAULT",
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        weights_backbone = "DEFAULT" if pretrained_backbone else None
        detector = factory(
            weights=None,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return cast(nn.Module, detector)


def _compute_losses_without_bn_drift(
    detector: nn.Module,
    images: List[Tensor],
    targets: List[Dict[str, Tensor]],
) -> Dict[str, Tensor]:
    """Run the detector's training forward in eval-like conditions.

    Torchvision's ``GeneralizedRCNN`` only emits its loss dict when
    ``self.training is True``, so we briefly call ``train()``. To prevent
    BatchNorm running statistics from drifting during validation, every BN
    layer is forced back to ``eval()`` for the duration of the pass. Grad is
    disabled — we only want the scalar loss values.
    """
    was_training = detector.training
    detector.train()
    bn_layers = [m for m in detector.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bn_layers:
        m.eval()
    try:
        with torch.no_grad():
            return cast(Dict[str, Tensor], detector(images, targets))
    finally:
        if not was_training:
            detector.eval()


@configurable
class FasterRCNNModel(nn.Module):
    """Torchvision Faster R-CNN wrapped for the Marainer Trainer flow.

    Returns a :class:`~raidar.detection.DetectionStepOutput` in every mode:

    * **Training** (``self.training is True``): ``{losses: <dict>, predictions: None}``.
    * **Validation / test** with targets: ``{losses: <dict>, predictions: [...]}``
      via a BN-frozen dual-forward. This makes ``val/loss`` directly
      comparable to ``train/loss`` — useful for spotting overfitting before
      mAP has had enough samples to stabilize.
    * **Pure prediction** (no targets): ``{losses: None, predictions: [...]}``.

    Args:
        num_classes: Total class count **including** background (class 0).
        variant: ``"v1"`` (:func:`torchvision.models.detection.fasterrcnn_resnet50_fpn`)
            or ``"v2"`` (the improved FPN v2 variant). v2 is the default.
        pretrained_backbone: Load ImageNet-pretrained ResNet-50 weights for
            the backbone. Ignored when ``pretrained_detector=True`` (the full
            detector weights already include a pretrained backbone).
        pretrained_detector: Load **full** COCO-pretrained Faster R-CNN
            weights (backbone + FPN + RPN + ROI pooler + 91-class predictor).
            The 91-class box predictor is then swapped for a fresh head with
            ``num_classes`` outputs, so the body stays pretrained while the
            classifier starts from scratch. Strongly recommended for
            small-dataset training — the pretrained RPN produces sensible
            region proposals from epoch 0, avoiding the score-collapse you
            get when training all heads from scratch with few examples.
        trainable_backbone_layers: Number of trainable trailing ResNet stages
            (0-5). Torchvision's default is 3.
        min_size / max_size: Image resize bounds applied by the detector's
            built-in :class:`GeneralizedRCNNTransform`.
        compute_val_loss: When False, skips the dual-forward during eval so
            val/test throughput matches predictions-only. Defaults to True.

    The output in eval mode wraps torchvision's raw per-image dict as a
    typed :class:`~marainer.outputs.DetectionOutput` so downstream code can
    rely on the three keys ``{boxes, scores, labels}``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        variant: Literal["v1", "v2"] = "v2",
        pretrained_backbone: bool = True,
        pretrained_detector: bool = False,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        compute_val_loss: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.variant = variant
        self.compute_val_loss = bool(compute_val_loss)
        self.detector = _build_faster_rcnn(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_backbone=pretrained_backbone,
            pretrained_detector=pretrained_detector,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        if self.training:
            if x.targets is None:
                raise ValueError("FasterRCNNModel in train mode requires targets on DetectionBatchInput.")
            n_imgs = len(x.images)
            box_counts = [int(t["boxes"].shape[0]) for t in x.targets]
            total_boxes = sum(box_counts)
            train_losses: Dict[str, Tensor] = self.detector(x.images, x.targets)
            losses_summary = ", ".join(f"{k}={float(v):.4f}" for k, v in train_losses.items() if isinstance(v, Tensor))
            if total_boxes == 0:
                logger.warning(
                    f"[diag][faster_rcnn][train] batch sees ZERO target boxes across "
                    f"{n_imgs} image(s) — gradients on classifier/box-reg heads will be "
                    f"trivial. losses=({losses_summary})"
                )
            else:
                logger.info(
                    f"[diag][faster_rcnn][train] images={n_imgs} target_boxes/img={box_counts} "
                    f"total_boxes={total_boxes} losses=({losses_summary})"
                )
            return DetectionStepOutput(losses=train_losses, predictions=None)

        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            losses = _compute_losses_without_bn_drift(self.detector, x.images, x.targets)

        raw = self.detector(x.images)
        predictions: List[DetectionOutput] = [
            DetectionOutput(
                boxes=p["boxes"].float(),
                scores=p["scores"].float(),
                labels=p["labels"].long(),
            )
            for p in raw
        ]

        # Per-batch diagnostic on the raw prediction stream — useful when the
        # downstream sink reports "no boxes" so the user can tell whether the
        # model produces NOTHING (head un-trained) vs. plenty of low-score
        # candidates that downstream score thresholds drop.
        per_image_summaries = []
        for i, p in enumerate(predictions):
            n = int(p["boxes"].shape[0])
            if n == 0:
                per_image_summaries.append(f"img{i}=0")
                continue
            scores = p["scores"]
            per_image_summaries.append(
                f"img{i}={n}@s∈[{float(scores.min()):.3f},{float(scores.max()):.3f}]"
                f"(mean={float(scores.mean()):.3f})"
            )
        total_pred = sum(int(p["boxes"].shape[0]) for p in predictions)
        if total_pred == 0:
            logger.warning(
                f"[diag][faster_rcnn][eval] {len(predictions)} image(s): model produced "
                f"ZERO predictions across the batch — classifier head likely never saw "
                f"positive examples (verify target boxes during training)."
            )
        else:
            losses_summary = ", ".join(f"{k}={float(v):.4f}" for k, v in losses.items()) if losses else "n/a"
            logger.debug(f"[diag][faster_rcnn][eval] {' '.join(per_image_summaries)} | " f"losses=({losses_summary})")
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["FasterRCNNModel"]
