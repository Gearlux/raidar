"""``KerasDetector`` — the same detection run, on Keras 3 instead of a torch loop.

Since P5 (2026-08-07) this is ``(DetectionKerasHooks, matrainer.keras.KerasRunnable)``: the
compile/fit machinery — build + compile, the ``History`` read-back, the ``RecordSequence``
batch boundary, the tracking callback — lives in
:class:`matrainer.keras.trainer.KerasRunnable`; the task half (``num_classes`` resolution, the
fit/eval initialization, the guarded no-train-metrics slot) in
:class:`raidar.detection_base.DetectionBase`; and detection's KERAS VOCABULARY (the dense-map
batch read, the spec's positional loss list, the numpy AP scoring path, the decode-and-stream
predict) in :class:`raidar.detection_keras.DetectionKerasHooks` — written once for this
concrete AND its native-TensorFlow sibling. What stays here is the FULL flat constructor (the
confluid introspection surface).

**Detection's batching problem, and the spec's answer.** The torch backends carry
variable-length per-image ``Boxes`` targets as per-record COLUMNS; this loop needs REGULAR
stacking for its dense maps, so a Keras detection config appends
:class:`raidar.fcos.targets.FcosTargets` to its ops chain — see the hooks module for the rest
(the CHW transpose, the loss design, the scoring).

Keras 3 picks its engine from ``KERAS_BACKEND``; this module never sets it.
"""

from typing import Iterable, List, Literal, Optional, Union

from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.keras.runnable import KerasOptimizer
from matrainer.keras.trainer import KerasRunnable
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource
from recordstream.keras import keras

from raidar.detection_keras import DetectionKerasHooks

logger = get_logger(__name__)

#: Which entry point :meth:`KerasDetector.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["DetectorTask", "KerasDetector", "KerasOptimizer"]


@configurable(task="detection", role="trainer", framework="keras")
class KerasDetector(DetectionKerasHooks, KerasRunnable):
    """Train, evaluate or predict the shared FCOS detection spec with Keras 3.

    Args:
        model: The spec model — wire ``!lazy:`` (e.g. ``KerasFcos``) so ``initialize()`` can
            inject the resolved ``num_classes``.
        train_set: Training source (``task="fit"``). Its ops chain MUST end in ``FcosTargets``.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Detection configs pin it explicitly — the ``FcosTargets`` op
            needs the same number.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the ground-truth ``Boxes``
            (``FcosTargets`` leaves it intact; ``batch_boxes`` reads it back for scoring).
        cls_key / box_key / ctr_key: Record keys of the dense maps ``FcosTargets`` wrote — the
            names must match that op's.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        learning_rate: Learning rate applied to a STRING-alias optimizer (an instance keeps its
            own).
        metric_names: How recorded metrics are named — the translation out of Keras's
            ``loss`` / ``val_loss`` vocabulary on the way to the trackers and the sidecar.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[keras.Model]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[keras.callbacks.Callback]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        cls_key: str = "fcos_cls",
        box_key: str = "fcos_box",
        ctr_key: str = "fcos_ctr",
        optimizer: Partial[KerasOptimizer] = "adam",
        learning_rate: float = 1e-3,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            optimizer=optimizer,
            metric_names=metric_names,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.cls_key = str(cls_key)
        self.box_key = str(box_key)
        self.ctr_key = str(ctr_key)
        self.learning_rate = float(learning_rate)
