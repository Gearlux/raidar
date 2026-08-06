"""The TORCH engine's registered detection metrics — one module per engine, no wrappers.

The deltaid/sonair rule applied to detection: an engine's losses/metrics are registered from
that engine's OWN module (`<engine>_<task>.py`), each with its own ``confluid.configurables``
entry point, so a missing engine drops its whole set and a discovery scan offers exactly what
the install can run. Detection has NO ``role="loss"`` entries — the native model contract
computes loss inside ``forward`` — so this module registers METRICS only:

* the ``torchmetrics.detection`` box-overlap family (registered upstream classes, no wrappers);
* :class:`DetectionConfusionMatrix` — the ONE metric this project owns, because torchmetrics has
  no TP/FP/FN accumulator over per-image detection lists.

Every registered class is named in the literal ``__all__``: ``scan_module`` filters by
``__module__`` (which is ``torchmetrics.detection.*`` for the upstream family), so the
``__all__``-gated pass is what carries them into a visual editor's palette.

**There is no detection collate here, and that is the point (2026-08-05):** batching a
detection record is ``recordstream.collate_records`` unchanged — it leaves variable-size images
as a per-image list and a variable-N :class:`~recordstream.Regions` as per-record columns — and
reading it back is ``recordstream.batch_regions`` / ``batch_values`` / ``batch_metadata``. The
~90 lines this project used to carry for it were re-deriving the engine's own collate.

The torchvision detection BUILDERS are registered by the sibling
:mod:`raidar.torchvision_detection`; the CLI's ``Framework.registrations`` names both modules
rather than one importing the other.

Native model contract (the shape torchvision detectors already speak):

* ``model.train(); model(images, targets) → Dict[str, Tensor]`` — loss dict.
* ``model.eval();  model(images)         → List[Dict[str, Tensor]]`` — per-image
  ``{boxes, scores, labels}`` (a :class:`~recordstream.DetectionOutput` list).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from confluid import configurable, register
from recordstream import DetectionOutput, batch_regions, batch_values
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection import (
    CompleteIntersectionOverUnion,
    DistanceIntersectionOverUnion,
    GeneralizedIntersectionOverUnion,
    IntersectionOverUnion,
    MeanAveragePrecision,
)
from torchvision.ops import box_iou

__all__ = [
    "CompleteIntersectionOverUnion",
    "DetectionConfusionMatrix",
    "DistanceIntersectionOverUnion",
    "GeneralizedIntersectionOverUnion",
    "IntersectionOverUnion",
    "MeanAveragePrecision",
    "native_batch",
]


# ---------------------------------------------------------------------------
# Registered metrics
# ---------------------------------------------------------------------------


def native_batch(
    batch: Any, input_key: str = "image", target_key: str = "target"
) -> Tuple[List[torch.Tensor], Optional[List[Dict[str, Any]]]]:
    """The ``(images, targets)`` pair the native contract takes, read from a COLLATED record.

    Every line is a recordstream call. The loaders here DECLARE
    :func:`~recordstream.collate_list` — the unstacked batch — because the native forward takes
    ``List[Tensor]``; :func:`~recordstream.batch_regions` is the transpose back into per-image
    target dicts, and it reads either collate. The comprehension is the one TORCHVISION fact
    left: it normalizes both shapes a batch can hold (a per-record list, or a stacked payload
    if a config chose the ``"record"`` collate) into the list the forward wants.

    This replaced ~90 lines of raidar-owned collate on 2026-08-05 — that code was re-deriving
    the engine's own batching.

    Args:
        batch: A batched record (the output of ``collate_records``).
        input_key: Record key carrying the image.
        target_key: Record key carrying the target ``Regions``; absent, ``targets`` is ``None``
            (the prediction path, where there is no ground truth to read).
    """
    images = [torch.as_tensor(value) for value in batch_values(batch, input_key)]
    targets = batch_regions(batch, target_key) if target_key in batch else None
    return images, targets


def reportable_metrics(values: Any) -> Dict[str, Any]:
    """A computed metrics mapping minus ``MeanAveragePrecision``'s ``classes`` bookkeeping key.

    ``classes`` is the OBSERVED class ids, not a metric. Left in, it is logged as a bogus scalar
    (one class seen → a 0-dim int tensor) or rendered as a bogus per-class table (several seen →
    an int vector that passes the shape-based vector rule). Lives beside the registered metrics
    whose ``compute()`` output it filters, and is shared by every torch-family backend's
    reporting path. Filtered by the trailing path segment because epoch-end values arrive already
    ``<split>/``-prefixed.
    """
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if str(k).rsplit("/", 1)[-1] != "classes"}


#: The upstream torchmetrics detection metrics offered as bare registered names. All consume the
#: same per-image ``(predictions, targets)`` lists the native step functions produce.
_DETECTION_METRICS: Tuple[type, ...] = (
    MeanAveragePrecision,
    IntersectionOverUnion,
    GeneralizedIntersectionOverUnion,
    CompleteIntersectionOverUnion,
    DistanceIntersectionOverUnion,
)

for _metric in _DETECTION_METRICS:
    register(_metric, task="detection", role="metric", framework="torch")


@configurable(task="detection", role="metric", framework="torch")
class DetectionConfusionMatrix(Metric):
    """``torchmetrics.Metric`` that accumulates TP / FP / FN over a stream of
    ``(predictions, targets)`` pairs and exposes the derived precision /
    recall / F1 in its ``compute()`` dict.

    Consumes the same per-image lists ``torchmetrics.MeanAveragePrecision``
    does, so it drops straight into a ``MetricCollection``::

        val_metrics:
          confusion: !lazy:raidar.DetectionConfusionMatrix
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
    class or many, and treats class IDs as opaque integers. A per-class
    breakdown is intentionally **not** surfaced here — the global counts are
    what the FP/FN-sensitive comparison needs.

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
        gt_boxes: Optional[Tensor] = target.get("boxes")
        gt_labels: Optional[Tensor] = target.get("labels")
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
