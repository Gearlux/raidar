"""``DetectionTorchHooks`` — the torch-family implementations of the task hooks.

The detection answers to the trainer API's typed hooks
(:class:`matrainer.torch.runnable.TorchTrainerRunnable`): the COCO-mAP default metric, the
``_reportable`` strip of MAP's pseudo-metric ``classes`` key, and the per-class vector
reporting. Deliberately NO ``_targets`` implementation — a native detection model reads its
batch inside the (overridden) step functions via ``native_batch``, which is why the family
base's default raises with exactly that explanation.

NOT to be confused with :mod:`raidar.torch_detection` (the torch ENGINE's registered metric
set + ``native_batch``): that one registers/adapts, this one implements hooks. This module
imports torch, so the four spec engines must never import it.
"""

from typing import Any, List, Mapping, Optional

from confluid import LazyClass
from loggair import get_logger
from matrainer.torch.metrics import MetricsSpec
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker
from matrainer.tracking.render import report_vectors
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection_base import DetectionBase
from raidar.torch_detection import reportable_metrics

logger = get_logger(__name__)

__all__ = ["DetectionTorchHooks"]


class DetectionTorchHooks(DetectionBase, TorchTrainerRunnable):
    """The torch half of the detection task, mixed into every torch-family backend."""

    def _default_metrics(self) -> MetricsSpec:
        """COCO mAP, the field's primary detection score.

        Class-count-agnostic, so unlike the classification/segmentation defaults it needs no
        ``num_classes`` (and `_metric_flow_dims` stays the empty default: detection metrics
        score box overlap, not a fixed-width head).
        """
        return {"map": LazyClass[Metric](MeanAveragePrecision, iou_type="bbox")}

    def _reportable(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """Drop MAP's ``classes`` key on EVERY reporting path.

        It is the OBSERVED class ids, not a metric: with one class seen it is a 0-dim int (a
        bogus scalar), with several a vector that passes the shape-based per-class rule (a
        bogus table).
        """
        return reportable_metrics(values)

    def _report_nonscalars(self, values: Mapping[str, Any], step: Optional[int], trackers: List[Tracker]) -> None:
        """Per-class VECTORS (``MeanAveragePrecision(class_metrics=True)``'s ``map_per_class``
        / ``mar_100_per_class``) to the log + every tracker's text (and figure) channel."""
        report_vectors(values, self.class_names or None, trackers, step=step, plot=self.plot_metrics)
