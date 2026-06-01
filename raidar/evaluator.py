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

from typing import Any

from confluid import LazyClass, configurable
from marainer.evaluator import LightningEvaluator
from torch.utils.data import DataLoader

from raidar.detection import _default_detection_metrics, detection_collate_fn


@configurable(task="detection", role="evaluator")
class LightningDetectionEvaluator(LightningEvaluator):
    """Specialised object-detection evaluator with baked-in defaults.

    Mirrors :class:`~marainer.evaluator.LightningEvaluator`'s minimal-ctor
    discipline: only ``model`` / ``dataset`` / ``num_classes`` (+ identity
    ``experiment_name`` / ``run_name``). The detection defaults — the ``test/``
    MAP metric set and the ``detection_collate_fn`` loader — are body attributes;
    ``predictions_sink`` / ``lightning`` / ``metrics_sidecar_path`` inherit the
    base defaults and are configured post-construction (YAML / broadcasting; a
    deferred slot uses ``!lazy:``).

    Args:
        model: Detection model that returns a
            :class:`~raidar.detection.DetectionStepOutput`. The model is
            responsible for loading its own weights — pass ``checkpoint_path``
            to the model class.
        dataset: Evaluation dataset or Flux pipeline.
        num_classes: Number of output classes. Surfaced for parity with
            sonair / deltaid evaluators.
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
        self.metrics = _default_detection_metrics(prefix="test/")
        self.loader = LazyClass(DataLoader, collate_fn=detection_collate_fn)
