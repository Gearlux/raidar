"""Generic object-detection adapters that plug into Marainer's
:class:`~marainer.trainer.LightningTrainer`.

This module provides the detector-agnostic plumbing needed to train and
evaluate torchvision-style detectors through the **unchanged** base Trainer
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
* :class:`DetectionStepOutput` is the unified ``{losses, predictions}`` dict
  every detector model in ``raidar`` returns from its ``forward``.
* :class:`TorchvisionDetectionLoss` sums the loss dict from the step output,
  returning a real scalar in both train and val — no ``tensor(0.0)``
  placeholder.
* :class:`DetectionMetricAdapter` unwraps predictions from the step output
  before forwarding to any torchmetrics detection metric (e.g.
  ``MeanAveragePrecision``).
* :func:`detection_collate_fn` replaces the default DataLoader collate so
  variable-length per-image targets are preserved as lists.

The per-image prediction schema is the task-agnostic
:class:`marainer.outputs.DetectionOutput` — the "know exactly what's in the
dict" contract, re-exported from this module for convenience.

Detector implementations live in their own sibling modules — see
:mod:`raidar.faster_rcnn` for :class:`~raidar.faster_rcnn.FasterRCNNModel`.

Typical wiring (YAML)::

    model: !class:raidar.faster_rcnn.FasterRCNNModel
    loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
      weights: { loss_classifier: 1.0, loss_box_reg: 1.0 }
    train_loader: !class:torch.utils.data.DataLoader
      collate_fn: !ref:raidar.detection.detection_collate_fn
    val_metrics: !class:raidar.detection.DetectionMetricAdapter
      inner: !class:torchmetrics.detection.MeanAveragePrecision
        iou_type: bbox
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from confluid import configurable
from dataflux.sample import Sample
from logflow import get_logger
from marainer.outputs import DetectionOutput, DetectionPredictions
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchvision.ops import box_iou

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
    base :class:`~marainer.trainer.LightningTrainer`'s step functions remain unmodified.

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
    """Unified model output consumed by :class:`TorchvisionDetectionLoss` and
    :class:`DetectionMetricAdapter`.

    Keys:
        losses: Torchvision loss dict (e.g. ``loss_classifier``,
            ``loss_box_reg``, ``loss_objectness``, ``loss_rpn_box_reg`` for
            Faster R-CNN; the exact keys depend on the detector family) when
            targets were provided; ``None`` otherwise.
        predictions: Per-image :class:`DetectionOutput` list in eval mode;
            ``None`` during training (predictions there would waste compute
            and break gradient flow).
    """

    losses: Optional[Dict[str, Tensor]]
    predictions: Optional[List[DetectionOutput]]


# ---------------------------------------------------------------------------
# Loss adapter
# ---------------------------------------------------------------------------


@configurable
class TorchvisionDetectionLoss(nn.Module):
    """Sum a torchvision detector's loss dict with optional per-head weights.

    Consumes a :class:`DetectionStepOutput` and returns the weighted sum of
    ``step_output["losses"]`` as a scalar tensor. Works for any torchvision
    detector that emits a loss dict during training (Faster R-CNN, RetinaNet,
    FCOS, …) — the keys differ by family but the summing contract is the same.

    When the model recomputes losses under BN-frozen eval during validation
    (as :class:`raidar.faster_rcnn.FasterRCNNModel` does), ``train/loss`` and
    ``val/loss`` report the **same** torchvision objective — they are directly
    comparable.

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
        # update is driven by the surrounding adapter; views are read-only.
        return None

    def compute(self) -> Tensor:
        result = self.inner.compute()
        if isinstance(result, dict):
            v = result[self.key]
        else:
            v = result
        return v.detach() if isinstance(v, Tensor) else torch.tensor(v)

    def reset(self) -> None:
        # Each torchmetrics Metric caches its own ``_computed`` from the last
        # compute() call until update() is called again. The collection-level
        # reset routes through here for every view; clear our own cache so the
        # next compute() picks up the inner metric's fresh state instead of
        # returning the stale cached value.
        self._computed = None


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
    # Class-level annotation so mypy knows the runtime type of ``_inner_metric``.
    # The actual value is stored via ``object.__setattr__`` in __init__ to
    # bypass nn.Module's submodule-registration path (each view already holds
    # a reference under .inner).
    _inner_metric: Metric

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
    :class:`~marainer.trainer.LightningTrainer`'s ``self.log_dict(self.val_metrics, ...)``
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

        # Wider element type satisfies MetricCollection's invariant dict signature.
        views: Dict[str, Union[Metric, MetricCollection]] = {key: _ScalarKeyView(inner, key) for key in scalar_keys}
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
# Confusion-matrix-style detection metric (TP / FP / FN + derived rates)
# ---------------------------------------------------------------------------


class _DetectionConfusionInner(Metric):
    """Single inner ``Metric`` that accumulates TP / FP / FN over a stream of
    ``(predictions, targets)`` pairs and exposes the derived precision /
    recall / F1 in its ``compute()`` dict.

    Matching is greedy and class-aware:

      1. Per image, drop predictions with ``score < score_threshold``.
      2. Sort the survivors by descending score.
      3. For each prediction, pick the highest-IoU **unmatched** ground-truth
         box of the **same class**; if that IoU ``>= iou_threshold``, count
         it as a TP and mark the GT as matched. Otherwise, the prediction is
         an FP.
      4. Any unmatched GT after the loop is an FN.

    True negatives are intentionally absent — TN isn't well-defined for
    object detection (the "negative" universe is uncountable image patches).

    The metric is task-agnostic over class IDs: it works for one foreground
    class (RFUAV) or many, and treats class IDs as opaque integers. A
    per-class breakdown is intentionally **not** surfaced here — the global
    counts are what the FP/FN-sensitive comparison the user asked for needs.

    State (all 0-d ``int64`` tensors so DDP/torchmetrics sync works
    out-of-the-box):

      * ``tp`` — true positives summed over all images / batches
      * ``fp`` — false positives
      * ``fn`` — false negatives
    """

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False  # mixed direction (TP good, FP/FN bad)

    # Class-level annotations for state attributes registered via add_state
    # (torchmetrics manages the values; mypy needs the type to follow them
    # through arithmetic).
    tp: Tensor
    fp: Tensor
    fn: Tensor

    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.25) -> None:
        super().__init__()
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1]; got {iou_threshold}.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be in [0, 1]; got {score_threshold}.")
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: List[DetectionOutput], target: List[Dict[str, Tensor]]) -> None:
        if preds is None or target is None:
            return
        if len(preds) != len(target):
            raise ValueError(
                f"DetectionConfusionMatrix: preds ({len(preds)}) and target ({len(target)}) must match per-image."
            )
        for p, t in zip(preds, target):
            tp, fp, fn = self._match_one_image(p, t)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def _match_one_image(
        self,
        pred: DetectionOutput,
        target: Dict[str, Tensor],
    ) -> Tuple[int, int, int]:
        # Move every input tensor to CPU at function entry. The matcher is
        # integer-counting with no gradient flow — there's no GPU benefit.
        # Keeping everything on CPU also avoids device mismatches between the
        # caller's tensors (often on MPS / CUDA via Lightning) and locally-
        # created tensors like ``gt_matched``, which would otherwise need
        # explicit device threading.
        gt_boxes = target.get("boxes")
        gt_labels = target.get("labels")
        if gt_boxes is None or gt_labels is None:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)
        else:
            gt_boxes = gt_boxes.detach().cpu()
            gt_labels = gt_labels.detach().cpu()
        n_gt = int(gt_boxes.shape[0])

        scores = pred["scores"].detach().cpu()
        keep = scores >= self.score_threshold
        pred_boxes = pred["boxes"].detach().cpu()[keep]
        pred_labels = pred["labels"].detach().cpu()[keep]
        pred_scores = scores[keep]
        n_pred = int(pred_boxes.shape[0])

        if n_pred == 0:
            return 0, 0, n_gt
        if n_gt == 0:
            return 0, n_pred, 0

        # Sort predictions by descending score so the greedy match consumes
        # the highest-confidence boxes first — same convention as COCO eval.
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        ious = box_iou(pred_boxes.float(), gt_boxes.float())  # [n_pred, n_gt]
        gt_matched = torch.zeros(n_gt, dtype=torch.bool)

        tp = 0
        for i in range(n_pred):
            cls = int(pred_labels[i])
            same_class = gt_labels == cls
            available = same_class & (~gt_matched)
            if not available.any():
                continue
            ious_i = ious[i].clone()
            ious_i[~available] = -1.0
            best_iou, best_j = torch.max(ious_i, dim=0)
            if float(best_iou) >= self.iou_threshold:
                gt_matched[int(best_j)] = True
                tp += 1
        fp = n_pred - tp
        fn = n_gt - int(gt_matched.sum())
        return tp, fp, fn

    def compute(self) -> Dict[str, Tensor]:
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()
        precision = tp / (tp + fp) if (tp + fp).item() > 0 else torch.tensor(0.0, device=tp.device)
        recall = tp / (tp + fn) if (tp + fn).item() > 0 else torch.tensor(0.0, device=tp.device)
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom.item() > 0 else torch.tensor(0.0, device=tp.device)
        return {
            "tp": self.tp.detach().clone(),
            "fp": self.fp.detach().clone(),
            "fn": self.fn.detach().clone(),
            "precision": precision.detach(),
            "recall": recall.detach(),
            "f1": f1.detach(),
        }


# Scalar keys surfaced by DetectionConfusionMatrix. Order is stable so logged
# metrics appear in the same TB column order across runs.
DEFAULT_CONFUSION_SCALAR_KEYS: Tuple[str, ...] = (
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
)


@configurable
class DetectionConfusionMatrix(MetricCollection):
    """Confusion-matrix-style detection metric (TP / FP / FN + precision / recall / F1).

    Plugs into the same Marainer / Lightning flow as
    :class:`DetectionMetricAdapter`: consumes a :class:`DetectionStepOutput`
    via ``update`` (silently no-ops when ``predictions`` is ``None``, which
    happens during ``training_step`` before the model has produced predictions
    in train mode), and exposes one scalar per surfaced key for
    ``log_dict``-style logging.

    Pairs naturally with ``DetectionMetricAdapter`` inside a
    :class:`~torchmetrics.MetricCollection` so a single ``val_metrics:`` block
    can carry both mAP **and** TP/FP/FN — useful when the ranking criterion
    is "minimize FN, then FP" (the user's stated case) and you also want mAP
    as a sanity baseline::

        val_metrics: !class:torchmetrics.MetricCollection
          metrics:
            - !class:raidar.detection.DetectionMetricAdapter
              inner: !class:torchmetrics.detection.MeanAveragePrecision
                iou_type: bbox
            - !class:raidar.detection.DetectionConfusionMatrix
              iou_threshold: 0.5
              score_threshold: 0.25

    Args:
        iou_threshold: Minimum IoU (with a same-class GT) required to count a
            prediction as a true positive. Default ``0.5`` (COCO-mAP@50
            convention).
        score_threshold: Predictions with lower confidence are filtered out
            **before** matching. Default ``0.25``. Lower values raise both FP
            (more candidates accepted) and TP (more chances to hit a GT);
            tune for the operating point you actually care about.
        prefix: Optional string prepended to every surfaced key (e.g. ``"det/"``
            yields ``det/tp``, ``det/fp``, ...). Empty by default — the
            outer Lightning logging prefix (``train/`` / ``val/`` / ``test/``)
            already disambiguates.
    """

    _inner_metric: Metric

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.25,
        prefix: str = "",
    ) -> None:
        inner = _DetectionConfusionInner(iou_threshold=iou_threshold, score_threshold=score_threshold)
        views: Dict[str, Union[Metric, MetricCollection]] = {
            key: _ScalarKeyView(inner, key) for key in DEFAULT_CONFUSION_SCALAR_KEYS
        }
        super().__init__(views, prefix=prefix)
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
      tensors) AND ``targets`` (list of dicts). This is what torchvision
      detector wrappers (e.g.
      :class:`raidar.faster_rcnn.FasterRCNNModel`) need in train and val
      modes alike.
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
    "DEFAULT_CONFUSION_SCALAR_KEYS",
    "DEFAULT_MEAN_AP_SCALAR_KEYS",
    "DetectionBatchInput",
    "DetectionConfusionMatrix",
    "DetectionMetricAdapter",
    "DetectionOutput",
    "DetectionPredictions",
    "DetectionStepOutput",
    "TorchvisionDetectionLoss",
    "detection_collate_fn",
]
