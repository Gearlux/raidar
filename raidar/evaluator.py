"""LightningDetectionEvaluator: specialised object-detection evaluator.

Subclass of :class:`~marainer.evaluator.LightningEvaluator` that bakes in
detection-specific defaults so a minimal YAML config only needs to specify
the dataset, model, and number of classes.

Baked-in defaults (all overridable via YAML ``@configurable`` attributes):

* ``loader`` ``collate_fn`` — :func:`~raidar.detection.detection_collate_fn`.
* ``metrics`` — :class:`~raidar.detection.DetectionMetricAdapter` wrapping
  :class:`torchmetrics.detection.MeanAveragePrecision` with ``iou_type=bbox``,
  prefixed ``test/``. Surfaces ``map``, ``map_50``, ``map_75``, ``mar_*``
  scalars to Lightning loggers and the metrics sidecar.

Minimal YAML example::

    evaluator: !class:raidar.evaluator.LightningDetectionEvaluator
      num_classes: 80
      model: !class:raidar.faster_rcnn.FasterRCNNModel
        num_classes: !ref:num_classes
        checkpoint_path: ./checkpoints/last.ckpt
      dataset: !ref:val_set
"""

from typing import Any, Optional

import pytorch_lightning as L
from torch.utils.data import DataLoader

from confluid import Class, Lazy, configurable
from marainer.evaluator import LightningEvaluator
from raidar.detection import detection_collate_fn
from raidar.trainer import _default_detection_metrics


@configurable
class LightningDetectionEvaluator(LightningEvaluator):
    """Specialised object-detection evaluator with baked-in defaults.

    Args:
        model: Detection model that returns a
            :class:`~raidar.detection.DetectionStepOutput`. The model is
            responsible for loading its own weights — pass ``checkpoint_path``
            to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        num_classes: Number of output classes. Surfaced for parity with
            sonair / deltaid evaluators.
        loader: Deferred DataLoader (default includes ``detection_collate_fn``).
        predictions_sink: Optional sink with ``.write(prediction, metadata)``
            and ``.close()``.
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        metrics_sidecar_path: Optional path; when set, writes a logger-agnostic
            JSON dump of the final metrics after evaluation completes.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        num_classes: int,
        loader: Lazy[Any] = Class(DataLoader, collate_fn=detection_collate_fn),
        predictions_sink: Any = None,
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "raidar_detection_eval",
        run_name: str = "evaluate",
        metrics_sidecar_path: Optional[str] = None,
    ) -> None:
        metrics = _default_detection_metrics(prefix="test/")
        super().__init__(
            model=model,
            dataset=dataset,
            loader=loader,
            metrics=metrics,
            predictions_sink=predictions_sink,
            lightning=lightning,
            experiment_name=experiment_name,
            run_name=run_name,
            metrics_sidecar_path=metrics_sidecar_path,
        )
        self.num_classes = int(num_classes)
