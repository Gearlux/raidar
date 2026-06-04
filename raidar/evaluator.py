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

Minimal YAML example::

    evaluator: !class:raidar.evaluator.LightningDetectionEvaluator
      num_classes: 2
      model: !class:raidar.rfdetr.RFDETRModel
        num_classes: !ref:num_classes
        checkpoint_path: ./checkpoints/last.ckpt   # the model loads itself
      dataset: !ref:val_set
"""

from typing import Any, Dict, List, Optional, cast

import torch.nn as nn
from confluid import LazyClass, configurable
from marainer.evaluator import LightningEvaluator
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import detection_collate_fn
from raidar.lightning_trainer import _log_scalar_dict


@configurable(task="detection", role="evaluator")
class LightningDetectionEvaluator(LightningEvaluator):
    """Specialised object-detection evaluator with overridden test step.

    Mirrors :class:`~marainer.evaluator.LightningEvaluator`'s minimal-ctor
    discipline: only ``model`` / ``dataset`` / ``num_classes`` (+ identity
    ``experiment_name`` / ``run_name``). The detection defaults — the raw
    ``MeanAveragePrecision`` metric and the ``detection_collate_fn`` loader —
    are body attributes; ``predictions_sink`` / ``lightning`` /
    ``metrics_sidecar_path`` inherit the base defaults and are configured
    post-construction (YAML / broadcasting; a deferred slot uses ``!lazy:``).

    Args:
        model: Torchvision-native detection model whose eval-mode ``forward``
            returns a per-image ``List[Dict[str, Tensor]]``. The model loads
            its own weights — pass ``checkpoint_path`` to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        num_classes: Number of output classes. Surfaced for parity with
            sonair / deltaid evaluators (the default MAP metric is
            class-count-agnostic).
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        num_classes: int,
        experiment_name: str = "raidar_detection_eval",
        run_name: str = "evaluate",
    ) -> None:
        super().__init__(model=model, dataset=dataset, experiment_name=experiment_name, run_name=run_name)
        self.num_classes = int(num_classes)
        # Raw torchmetrics MAP — same contract as the specialised trainer's
        # val/test metrics; no adapter. Deferred (LazyClass): run() flows it.
        self.metrics = LazyClass(MeanAveragePrecision, iou_type="bbox")
        self.loader = LazyClass(DataLoader, collate_fn=detection_collate_fn)

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
