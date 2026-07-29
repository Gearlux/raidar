"""Native object-detection plumbing for the specialised
:class:`~raidar.detector.LightningDetector`.

This module holds the small, detector-agnostic pieces the **native** detection
path needs — there is no adapter / indirection layer (the marainer-generic
``DetectionStepOutput`` + ``TorchvisionDetectionLoss`` + ``DetectionMetricAdapter``
path was removed; detection now runs exclusively through the specialised trainer
that drives torchvision-native models and raw ``torchmetrics`` metrics directly):

* :class:`DetectionBatchInput` carries images and (optionally) targets so
  variable-size per-image tensors survive collation. It rides under the batched
  record's ``"detection"`` key; the specialised trainer's step functions recover
  it (``_batch_detection``) and read ``.images`` / ``.targets``, handing them
  straight to the native model.
* :func:`detection_collate_fn` is the DataLoader collate that consumes the
  record dicts the sources now yield, builds a :class:`DetectionBatchInput`,
  and keeps per-image targets as lists (never padded, never dropped).
* :class:`DetectionConfusionMatrix` is a native ``torchmetrics.Metric`` that
  accumulates TP / FP / FN (+ derived precision / recall / F1) and drops
  straight into a ``MetricCollection`` alongside ``MeanAveragePrecision``.

The per-image prediction schema is the task-agnostic
:class:`recordstream.DetectionOutput`, re-exported here for convenience.

Native model contract (the shape torchvision detectors already speak):

* ``model.train(); model(images, targets) → Dict[str, Tensor]`` — loss dict.
* ``model.eval();  model(images)         → List[Dict[str, Tensor]]`` — per-image
  ``{boxes, scores, labels}`` (a :class:`DetectionOutput` list).

Typical wiring (YAML)::

    runnable: !class:raidar.detector.LightningDetector
      task: fit
      num_classes: !ref:num_classes
      model: !class:torchvision.models.detection.fasterrcnn_resnet50_fpn
        num_classes: !ref:num_classes
      train_set: !ref:train_set
      val_set: !ref:val_set
      val_metrics: !class:torchmetrics.MetricCollection
        metrics:
          - !class:torchmetrics.detection.MeanAveragePrecision
            iou_type: bbox
          - !class:raidar.detection.DetectionConfusionMatrix
            iou_threshold: 0.5
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from recordstream import DetectionOutput, DetectionPredictions, Label, Record, Regions, item_data
from torch import Tensor
from torchmetrics import Metric
from torchvision.ops import box_iou

# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------


@dataclass
class DetectionBatchInput:
    """Container that carries images and (optionally) targets as the batch's primary input.

    Torchvision detectors compute loss *inside* their forward by consuming a
    list of targets alongside the images — there is no way to split "run the
    model" from "compute loss". We carry both under the batched record's
    ``"detection"`` key so the specialised
    :class:`~raidar.detector.LightningDetector`'s step functions can unwrap
    ``(images, targets)`` and hand them to the model.

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


# ---------------------------------------------------------------------------
# Confusion-matrix-style detection metric (TP / FP / FN + derived rates)
# ---------------------------------------------------------------------------


class DetectionConfusionMatrix(Metric):
    """``torchmetrics.Metric`` that accumulates TP / FP / FN over a stream of
    ``(predictions, targets)`` pairs and exposes the derived precision /
    recall / F1 in its ``compute()`` dict.

    Native-API metric for the specialised
    :class:`~raidar.detector.LightningDetector`: its
    ``update(predictions: List[DetectionOutput], targets: List[Dict])``
    consumes the same per-image lists ``torchmetrics.MeanAveragePrecision``
    does, so it drops straight into a ``MetricCollection``::

        val_metrics: !class:torchmetrics.MetricCollection
          metrics:
            - !class:torchmetrics.detection.MeanAveragePrecision
              iou_type: bbox
            - !class:raidar.detection.DetectionConfusionMatrix
              iou_threshold: 0.5

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
    counts are what the FP/FN-sensitive comparison needs.

    State (all 0-d ``int64`` tensors so DDP/torchmetrics sync works
    out-of-the-box):

      * ``tp`` — true positives summed over all images / batches
      * ``fp`` — false positives
      * ``fn`` — false negatives

    Args:
        iou_threshold: Minimum IoU (with a same-class GT) required to count a
            prediction as a true positive. Default ``0.5`` (COCO-mAP@50
            convention).
        score_threshold: Predictions with lower confidence are filtered out
            **before** matching. Default ``0.25``. Lower values raise both FP
            (more candidates accepted) and TP (more chances to hit a GT);
            tune for the operating point you actually care about.
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


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def _as_tensor(value: Any) -> Tensor:
    """Coerce an item payload (a torch tensor or numpy array) to a torch tensor (memory-sharing)."""
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _detection_target(record: Record, target_key: str) -> Dict[str, Tensor]:
    """The torchvision ``{boxes, labels}`` target dict under ``target_key`` in a record.

    The typed detection-target op (``recordstream.ops.target.CocoToTorchVisionDetection`` /
    ``MasksToDetectionBoxes``) writes the target as a :class:`~recordstream.Regions` item whose
    ``boxes`` = the ``[N, 4]`` xyxy tensor and ``labels`` = the ``[N]`` class-id tensor; unwrap it
    into the ``{"boxes", "labels"}`` dict the native torchvision model consumes. A value that already
    carries the dict verbatim (a :class:`~recordstream.Label` wrapper or a raw dict) is passed through.
    """
    item = record[target_key]
    if isinstance(item, Regions):
        return {"boxes": _as_tensor(item.boxes), "labels": _as_tensor(item.labels)}
    data = item.value if isinstance(item, Label) else item_data(item)
    return cast(Dict[str, Tensor], data)


def _record_metadata(record: Record, input_key: str, target_key: str) -> Dict[str, Any]:
    """Gather a record's remaining keys (neither input nor target) into a plain metadata dict.

    Every key besides the configured input and target becomes ``{name: value}`` (a
    :class:`~recordstream.Label`'s ``.value``, else the item payload) so string-valued values survive
    collation and predictions sinks can correlate per-image outputs.
    """
    return {
        key: (item.value if isinstance(item, Label) else item_data(item))
        for key, item in record.items()
        if key not in (input_key, target_key)
    }


def detection_collate_fn(batch: List[Record], input_key: str = "image", target_key: str = "target") -> Record:
    """Collate variable-length detection record dicts into ONE batched record.

    Each per-record record carries an :class:`~recordstream.Image` input (``[3, H, W]`` float) under
    ``input_key`` and a :class:`~recordstream.Regions` target (boxes + labels) under ``target_key``.
    This reads the input image (``item_data(record[input_key])``, coerced to a tensor) and the
    target region set (unwrapped to a torchvision ``{boxes, labels}`` dict by
    :func:`_detection_target`), then returns ONE batched record dict:

    * key ``"detection"`` — a :class:`DetectionBatchInput` carried VERBATIM as the value. It is a
      custom container (variable-N images + per-image target dicts), NOT an array item, so it must
      NOT be forced into an :class:`~recordstream.Image` (which would ``np.asarray`` it); the
      specialised :class:`~raidar.detector.LightningDetector` recovers it via ``_batch_detection``
      and unwraps ``(images, targets)`` for the native model.
    * key ``"targets"`` — the SAME list of per-image target dicts, so a metric's
      ``update(preds, targets)`` receives per-image ground truth (``batch["targets"] is
      batch_detection.targets``).
    * key ``"metadata"`` — a ``list`` of the per-record metadata dicts (the remaining keys),
      preserved for sinks that write per-record predictions.

    Per-image targets are kept as lists — never padded, never dropped (the variable-N detection
    batch convention).

    Args:
        batch: List of per-record record dicts.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the target ``Regions`` (default ``"target"``).
    """
    images: List[Tensor] = []
    targets: List[Dict[str, Tensor]] = []
    metas: List[Dict[str, Any]] = []
    for r in batch:
        images.append(_as_tensor(item_data(r[input_key])))
        targets.append(_detection_target(r, target_key))
        metas.append(_record_metadata(r, input_key, target_key))
    batch_input = DetectionBatchInput(images=images, targets=targets)
    return {"detection": batch_input, "targets": targets, "metadata": metas}


__all__ = [
    "DetectionBatchInput",
    "DetectionConfusionMatrix",
    "DetectionOutput",
    "DetectionPredictions",
    "detection_collate_fn",
]
