"""Object-detection models and adapters that plug into Marainer's :class:`~marainer.trainer.Trainer`.

This module provides everything needed to train and evaluate torchvision-style
detectors (Faster R-CNN and friends) through the **unchanged** base Trainer
flow — no subclass, no overridden step functions. Marainer's
``training_step`` / ``validation_step`` pattern::

    preds = self(batch.input)
    loss  = self.loss_fn(preds, batch.target)
    self.log_dict(self.val_metrics, ...)

fits detection because we adapt the pieces on either side:

* :class:`DetectionBatchInput` carries both images and targets through
  ``batch.input`` (torchvision detectors need both together during training),
  and exposes a ``shape`` property so ``batch.input.shape[0]`` still works for
  Lightning's batch-size logging.
* :class:`FasterRCNNModel` wraps torchvision and **always** returns a
  :class:`DetectionStepOutput` — in train mode it's ``{losses: {...},
  predictions: None}``; in eval mode (with targets) it's ``{losses: {...},
  predictions: [DetectionOutput, ...]}`` via a BN-frozen dual-forward so val
  loss is directly comparable to train loss.
* :class:`FasterRCNNLoss` sums the loss dict from the step output, returning a
  real scalar in both train and val — no ``tensor(0.0)`` placeholder.
* :class:`DetectionMetricAdapter` unwraps predictions from the step output
  before forwarding to any torchmetrics detection metric (e.g.
  ``MeanAveragePrecision``).
* :func:`detection_collate_fn` replaces the default DataLoader collate so
  variable-length per-image targets are preserved as lists.

The per-image prediction schema is the task-agnostic
:class:`marainer.outputs.DetectionOutput` — the "know exactly what's in the
dict" contract, re-exported from this module for convenience.

Typical wiring (YAML)::

    loss_fn: !class:raidar.detection.FasterRCNNLoss
      weights: { loss_classifier: 1.0, loss_box_reg: 1.0 }
    train_loader: !class:torch.utils.data.DataLoader
      collate_fn: !ref:raidar.detection.detection_collate_fn
    val_metrics: !class:raidar.detection.DetectionMetricAdapter
      inner: !class:torchmetrics.detection.MeanAveragePrecision
        iou_type: bbox
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
from confluid import configurable
from dataflux.sample import Sample
from logflow import get_logger
from marainer.outputs import DetectionOutput, DetectionPredictions
from torch import Tensor
from torchmetrics import Metric, MetricCollection

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Batch type + unified step output
# ---------------------------------------------------------------------------


@dataclass
class DetectionBatchInput:
    """Container that carries images and (optionally) targets through ``batch.input``.

    Torchvision detectors compute loss *inside* their forward by consuming a
    list of targets alongside the images — there is no way to split "run the
    model" from "compute loss". We smuggle both through ``batch.input`` so the
    base :class:`~marainer.trainer.Trainer`'s step functions remain unmodified.

    Attributes:
        images: List of ``[3, H_i, W_i]`` float tensors, one per image. Sizes
            may vary; torchvision handles internal resizing.
        targets: Optional list of per-image ``{"boxes", "labels"}`` dicts in
            torchvision format (xyxy absolute pixels, int64 class ids). Set
            during train and val/test; ``None`` for pure prediction.

    The ``shape`` property returns ``(len(images),)`` so callers that expect
    ``batch.input.shape[0]`` to be the batch size (notably Lightning's
    ``batch_size`` kwarg in ``self.log*``) continue to work.
    """

    images: List[Tensor]
    targets: Optional[List[Dict[str, Tensor]]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self.images),)

    def __len__(self) -> int:
        return len(self.images)


class DetectionStepOutput(TypedDict):
    """Unified model output consumed by :class:`FasterRCNNLoss` and
    :class:`DetectionMetricAdapter`.

    Keys:
        losses: Torchvision loss dict (``loss_classifier``, ``loss_box_reg``,
            ``loss_objectness``, ``loss_rpn_box_reg``) when targets were
            provided; ``None`` otherwise.
        predictions: Per-image :class:`DetectionOutput` list in eval mode;
            ``None`` during training (predictions there would waste compute
            and break gradient flow).
    """

    losses: Optional[Dict[str, Tensor]]
    predictions: Optional[List[DetectionOutput]]


# ---------------------------------------------------------------------------
# Faster R-CNN model wrapper
# ---------------------------------------------------------------------------


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
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                              fasterrcnn_resnet50_fpn_v2)
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
    return detector


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
    bn_layers = [
        m for m in detector.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    for m in bn_layers:
        m.eval()
    try:
        with torch.no_grad():
            return detector(images, targets)
    finally:
        if not was_training:
            detector.eval()


@configurable
class FasterRCNNModel(nn.Module):
    """Torchvision Faster R-CNN wrapped for the Marainer Trainer flow.

    Returns a :class:`DetectionStepOutput` in every mode:

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
    typed :class:`DetectionOutput` so downstream code can rely on the three
    keys ``{boxes, scores, labels}``.
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
                raise ValueError(
                    "FasterRCNNModel in train mode requires targets on DetectionBatchInput."
                )
            n_imgs = len(x.images)
            box_counts = [int(t["boxes"].shape[0]) for t in x.targets]
            total_boxes = sum(box_counts)
            train_losses: Dict[str, Tensor] = self.detector(x.images, x.targets)
            losses_summary = ", ".join(
                f"{k}={float(v):.4f}"
                for k, v in train_losses.items()
                if isinstance(v, Tensor)
            )
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
            losses = _compute_losses_without_bn_drift(
                self.detector, x.images, x.targets
            )

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
            losses_summary = (
                ", ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
                if losses
                else "n/a"
            )
            logger.info(
                f"[diag][faster_rcnn][eval] {' '.join(per_image_summaries)} | "
                f"losses=({losses_summary})"
            )
        return DetectionStepOutput(losses=losses, predictions=predictions)


# ---------------------------------------------------------------------------
# Loss adapter
# ---------------------------------------------------------------------------


@configurable
class FasterRCNNLoss(nn.Module):
    """Sum torchvision's loss dict with optional per-head weights.

    Consumes a :class:`DetectionStepOutput` and returns the weighted sum of
    ``step_output["losses"]`` as a scalar tensor. Because
    :class:`FasterRCNNModel` recomputes losses under BN-frozen eval during
    validation, ``train/loss`` and ``val/loss`` report the **same** torchvision
    objective — they are directly comparable.

    When ``step_output["losses"]`` is ``None`` (pure-prediction path, no
    targets provided), returns a device-matching zero so Lightning's scalar
    contract is satisfied without contaminating the loss curve.

    Args:
        weights: Optional per-loss-key multipliers. Keys missing from the dict
            are weighted 1.0. Extra keys are ignored.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.weights: Dict[str, float] = dict(weights) if weights else {}

    def forward(self, step_output: DetectionStepOutput, target: Any) -> Tensor:
        losses = step_output["losses"]
        if losses is not None:
            total: Optional[Tensor] = None
            for k, v in losses.items():
                term = self.weights.get(k, 1.0) * v
                total = term if total is None else total + term
            assert total is not None  # losses dict is never empty in torchvision
            return total
        preds = step_output["predictions"]
        device = torch.device("cpu")
        if preds:
            b = preds[0]["boxes"]
            if b.numel() > 0:
                device = b.device
        return torch.zeros((), device=device)


# ---------------------------------------------------------------------------
# Metric adapter
# ---------------------------------------------------------------------------


class _ScalarKeyView(Metric):
    """View into a single scalar key of a shared inner metric's dict compute().

    Multiple views can share the same ``inner`` — the dedup bookkeeping lives
    in :class:`DetectionMetricAdapter` which owns exactly one inner metric
    per adapter and routes ``update`` through it once per batch.
    """

    # torchmetrics needs these attributes to treat this as a proper Metric.
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, inner: Metric, key: str) -> None:
        super().__init__()
        self.inner = inner
        self.key = key

    def update(self, *args: Any, **kwargs: Any) -> None:
        # update is driven by DetectionMetricAdapter; views are read-only.
        return None

    def compute(self) -> Tensor:
        result = self.inner.compute()
        if isinstance(result, dict):
            v = result[self.key]
        else:
            v = result
        return v.detach() if isinstance(v, Tensor) else torch.tensor(v)

    def reset(self) -> None:
        # Collection-level reset calls each child; route to inner once.
        return None


# Torchmetrics exposes `map`/`map_50`/`mar_100` etc. as scalars in the
# MeanAveragePrecision.compute() result — these are the defaults we surface
# by default through the adapter. Non-scalar entries (e.g. `map_per_class`,
# `classes`) are intentionally excluded so Lightning's log_dict receives only
# scalar tensors.
DEFAULT_MEAN_AP_SCALAR_KEYS: Tuple[str, ...] = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


class DetectionMetricAdapter(MetricCollection):
    """Wrap a torchmetrics detection metric to consume :class:`DetectionStepOutput`.

    Torchmetrics' :class:`MeanAveragePrecision` expects the call signature
    ``(preds: List[Dict], target: List[Dict])``. Our step output is the richer
    :class:`DetectionStepOutput`, so we override ``update``/``forward`` to
    extract ``step_output["predictions"]`` (and silently no-op when it's
    ``None``, which happens during ``training_step`` when
    ``train_metrics`` is accidentally configured).

    The inner metric's ``compute()`` may return a dict (as MAP does). Lightning's
    ``log_dict`` path iterates a MetricCollection's items and calls ``compute()``
    per entry, and crashes when it finds a non-scalar. To keep the base
    :class:`~marainer.trainer.Trainer`'s ``self.log_dict(self.val_metrics, ...)``
    call unchanged, we expose each scalar key as its own :class:`Metric`-shaped
    view over the shared inner state.

    YAML::

        val_metrics: !class:raidar.detection.DetectionMetricAdapter
          inner: !class:torchmetrics.detection.MeanAveragePrecision
            iou_type: bbox

    Args:
        inner: The detection metric to wrap (typically ``MeanAveragePrecision``).
        scalar_keys: Scalar-valued keys of the inner metric's ``compute()`` dict
            to surface. Defaults to :data:`DEFAULT_MEAN_AP_SCALAR_KEYS`. If the
            inner metric returns a plain tensor (not a dict), pass a single-key
            tuple — e.g. ``("value",)``.
        prefix: Optional prefix applied to every surfaced key (e.g. ``"val/"``).
    """

    def __init__(
        self,
        inner: Metric,
        scalar_keys: Tuple[str, ...] = DEFAULT_MEAN_AP_SCALAR_KEYS,
        prefix: str = "",
    ) -> None:
        # Materialize a still-deferred Confluid marker before wiring it into
        # the views. The adapter needs a live ``Metric`` instance because the
        # views hold a shared reference to ``inner`` and call ``inner.update``
        # at step time — a deferred ``Class`` would fail with a cryptic
        # ``'Class' object has no attribute 'update'`` inside a Lightning
        # worker.
        from confluid import flow as _confluid_flow
        from confluid.fluid import Fluid as _ConfluidFluid

        if isinstance(inner, _ConfluidFluid):
            inner = _confluid_flow(inner)

        views: Dict[str, Metric] = {
            key: _ScalarKeyView(inner, key) for key in scalar_keys
        }
        super().__init__(views, prefix=prefix)
        # Keep a named reference to the shared inner without re-registering it
        # as an extra submodule — each view already holds it under .inner.
        object.__setattr__(self, "_inner_metric", inner)

    def update(self, step_output: DetectionStepOutput, target: Any) -> None:  # type: ignore[override]
        preds = step_output["predictions"]
        if preds is None:
            return
        self._inner_metric.update(preds, target)

    def forward(self, step_output: DetectionStepOutput, target: Any) -> Dict[str, Any]:  # type: ignore[override]
        preds = step_output["predictions"]
        if preds is None:
            return {}
        self._inner_metric.update(preds, target)
        return {k: view.compute() for k, view in self.items()}

    def compute(self) -> Dict[str, Tensor]:  # type: ignore[override]
        return {k: view.compute() for k, view in self.items()}

    def reset(self) -> None:  # type: ignore[override]
        self._inner_metric.reset()
        super().reset()


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def detection_collate_fn(batch: List[Sample]) -> Sample:
    """Collate variable-length detection samples into a batch Trainer can use.

    Each input ``Sample`` is expected to carry:

    * ``sample.input`` — a ``[3, H, W]`` float tensor (e.g. produced by
      :class:`waivefront.detection_ops.SignalRegionsToDetectionTargetOp`).
    * ``sample.target`` — ``{"boxes": [N, 4] float32, "labels": [N] int64}``
      in torchvision xyxy-pixel format.

    The returned :class:`~dataflux.sample.Sample` has:

    * ``.input`` — :class:`DetectionBatchInput` carrying ``images`` (list of
      tensors) AND ``targets`` (list of dicts). This is what the
      :class:`FasterRCNNModel` needs in train and val modes alike.
    * ``.target`` — the same list of target dicts, so
      :class:`DetectionMetricAdapter` can feed torchmetrics.
    * ``.metadata`` — list of the per-sample metadata dicts (preserved for
      sinks that write per-sample predictions).
    """
    images: List[Tensor] = []
    targets: List[Dict[str, Tensor]] = []
    metas: List[Dict[str, Any]] = []
    for s in batch:
        images.append(s.input)
        targets.append(s.target)
        metas.append(s.metadata if s.metadata is not None else {})
    return Sample(
        input=DetectionBatchInput(images=images, targets=targets),
        target=targets,
        metadata={"per_sample": metas},
    )


__all__ = [
    "DetectionBatchInput",
    "DetectionMetricAdapter",
    "DetectionOutput",
    "DetectionPredictions",
    "DetectionStepOutput",
    "FasterRCNNLoss",
    "FasterRCNNModel",
    "detection_collate_fn",
]
