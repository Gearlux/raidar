"""LightningDetectionEvaluator: specialised object-detection evaluator.

Subclass of :class:`~marainer.evaluator.LightningEvaluator` that **overrides**
the test step so it consumes torchvision-native detection models and raw
``torchmetrics`` detection metrics directly — the eval-side counterpart of
:class:`~raidar.lightning_trainer.LightningDetectionTrainer`. There is no
adapter / ``DetectionStepOutput`` indirection (the marainer-generic adapter
path was removed).

Model contract (eval mode): ``model(images) → List[Dict[str, Tensor]]`` —
per-image ``{boxes, scores, labels}`` (a :class:`~raidar.detection.DetectionOutput`
list). Torchvision Faster R-CNN conforms raw; RF-DETR ships a native module at
:class:`raidar.rfdetr.RFDETRModel`; the ultralytics-family detectors ship
native modules in the ``raidar-ultralytics`` sibling package.

Metric contract: any ``torchmetrics.Metric`` / ``MetricCollection`` whose
``update(preds, targets)`` consumes the per-image prediction list directly.
Default: ``MeanAveragePrecision(iou_type="bbox")`` — its ``test/`` scalars are
logged at epoch end and (optionally) written to the metrics sidecar. Combine
MAP with :class:`~raidar.detection.DetectionConfusionMatrix` via a
``MetricCollection`` when you also want TP / FP / FN.

**``num_classes`` is model-driven (mirrors sonair) — NOT a ctor knob.** Like
:class:`sonair.evaluator.LightningClassificationEvaluator`, this evaluator
resolves the class count in an explicit :meth:`initialize` (called at the top of
:meth:`run`, BEFORE ``super().run()``): it **flows the model FIRST** and reads
``num_classes`` straight off the LIVE trained model (its head is already sized —
it loaded its own checkpoint — so it is the single source of truth, NOT the eval
dataset which may be a subset). There is **no** evaluator ``num_classes``
constructor argument. The read is *best-effort* (unlike sonair, which raises):
the default MAP metric is class-count-agnostic, so a detector that doesn't
expose its count (e.g. a single-stage torchvision model) simply leaves
``self.num_classes`` ``None`` without erroring.

Minimal YAML example (no ``num_classes`` — the trained model carries it)::

    evaluator: !class:raidar.evaluator.LightningDetectionEvaluator
      model: !class:raidar.rfdetr.RFDETRModel
        num_classes: 2                             # the model carries the count
        checkpoint_path: ./checkpoints/last.ckpt   # the model loads itself
      dataset: !ref:val_set
"""

from typing import Any, Dict, List, Optional, cast

import torch.nn as nn
from confluid import LazyClass, configurable, flow
from logflow import get_logger
from marainer.evaluator import LightningEvaluator
from marainer.trainer import ensure_sample_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import detection_collate_fn
from raidar.lightning_trainer import _log_scalar_dict, _model_num_classes

logger = get_logger(__name__)


@configurable(task="detection", role="evaluator")
class LightningDetectionEvaluator(LightningEvaluator):
    """Specialised object-detection evaluator with overridden test step.

    Mirrors :class:`~marainer.evaluator.LightningEvaluator`'s minimal-ctor
    discipline AND :class:`sonair.evaluator.LightningClassificationEvaluator`'s
    **model-driven** ``num_classes`` resolution: the constructor takes only
    ``model`` / ``dataset`` (+ identity ``experiment_name`` / ``run_name``) and
    does no functional work — there is **no** ``num_classes`` knob (it is read
    off the trained model in :meth:`initialize`). The detection defaults — the
    raw ``MeanAveragePrecision`` metric and the ``detection_collate_fn`` loader —
    are deferred body attributes (``LazyClass``); ``predictions_sink`` /
    ``lightning`` / ``metrics_sidecar_path`` inherit the base defaults and are
    configured post-construction (YAML / broadcasting; a deferred slot uses
    ``!lazy:``).

    **``num_classes`` resolution (:meth:`initialize`, mirroring sonair).** Called
    at the top of :meth:`run` (BEFORE ``super().run()``), it flows the model FIRST
    and reads ``num_classes`` straight off the live trained model
    (:func:`~raidar.lightning_trainer._model_num_classes` — an explicit
    ``num_classes`` attr, e.g. ``RFDETRModel`` / ``YOLO26Model``, or torchvision's
    ``roi_heads.box_predictor.cls_score.out_features``). The read is best-effort:
    the default MAP metric is class-count-agnostic, so ``self.num_classes`` may
    stay ``None`` without erroring.

    Args:
        model: Torchvision-native detection model whose eval-mode ``forward``
            returns a per-image ``List[Dict[str, Tensor]]``. The model loads
            its own weights — pass ``checkpoint_path`` to the model class. Its
            head is authoritative for ``num_classes`` (read in :meth:`initialize`).
        dataset: Evaluation dataset or Flux pipeline.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name. ``None`` (default) auto-generates a random slug
            like ``evaluate_funny_rabbit`` (via the base; see
            :func:`marainer.run_names.generate_run_name`).
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        experiment_name: str = "raidar_detection_eval",
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, dataset=dataset, experiment_name=experiment_name, run_name=run_name)
        # Store config ONLY — no functional work. There is NO num_classes knob: for
        # evaluation the trained model's head is authoritative, so initialize() reads
        # it off the LIVE model (best-effort — the MAP metric is class-agnostic).
        self.num_classes: Optional[int] = None
        # Raw torchmetrics MAP — same contract as the specialised trainer's
        # val/test metrics; no adapter. Deferred (LazyClass): run() flows it.
        self.metrics = LazyClass(MeanAveragePrecision, iou_type="bbox")
        self.loader = LazyClass(DataLoader, collate_fn=detection_collate_fn)

    def run(self) -> None:
        """Flow the model + read ``num_classes`` off it, then run the base eval loop."""
        self.initialize()
        super().run()

    def initialize(self) -> None:
        """Flow the model, read ``num_classes`` off it, and normalize the dataset (mirrors sonair).

        raidar OWNS this (no generic dimension-injection layer). Called explicitly
        from :meth:`run` (NOT confluid's auto ``solidify`` hook) so the model build +
        dataset normalization happen at run time, not when a config is merely
        materialized. Idempotent: re-running reuses the already-live model / dataset.

        For evaluation the model is already trained — it loads its own checkpoint, so
        its head is authoritative. We therefore **flow the model first** and read
        ``num_classes`` off the LIVE model (:func:`_model_num_classes`) — NOT from the
        eval dataset (which may be a subset and would mis-size a class-aware metric),
        and NOT from the model's deferred ``Fluid`` kwargs (filled in later). There is
        no evaluator ``num_classes`` knob: the model is the single source of truth. The
        read is best-effort (the default MAP metric is class-count-agnostic, unlike
        sonair's per-class metrics — so a detector that doesn't expose its count leaves
        ``self.num_classes`` ``None`` instead of erroring). The dataset is normalized to
        a ``Dataset[Sample]`` up front via the marainer ``ensure_sample_dataset`` helper.
        """
        # Flow the model first — it builds with its own num_classes (from config / the
        # checkpoint it loads). flow is idempotent on an already-live instance.
        self.model = flow(self.model)
        nc = _model_num_classes(cast(nn.Module, self.model))
        if nc:
            self.num_classes = int(nc)

        # Normalize the eval source to a Dataset[Sample] up front (marainer helper) so the
        # overridden test_step + collate see guaranteed Sample items.
        self.dataset = ensure_sample_dataset(flow(self.dataset))

    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:  # type: ignore[override]
        model = cast(nn.Module, self.model)
        return model(images)  # type: ignore[no-any-return]

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        # Unwrap the DetectionBatchInput into the native (images, targets) the
        # detector speaks — no DetectionStepOutput / adapter in between.
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        model = cast(nn.Module, self.model)
        predictions = model(images)
        if self.metrics is not None and targets is not None:
            self.metrics.update(predictions, targets)
        if self.predictions_sink is not None:
            # Reuse the base per-sample writer: predictions is a List[DetectionOutput].
            self._write_predictions(predictions, batch)
        return predictions

    def on_test_epoch_end(self) -> None:
        # Log the test/ scalars but DON'T reset — the base run() calls
        # _finalize_metrics() after the test loop, which reads metrics.compute()
        # for the (optional) sidecar; resetting here would zero it out.
        if self.metrics is not None:
            _log_scalar_dict(self, self.metrics, prefix="test/")


__all__ = ["LightningDetectionEvaluator"]
