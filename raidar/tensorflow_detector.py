"""``TensorFlowDetector`` — the shared FCOS spec on a native TensorFlow loop.

Since P5 (2026-08-07) this is ``(DetectionKerasHooks, matrainer.tensorflow.
TensorFlowRunnable)``: the loop machinery — the ``tf.data`` pipeline, the epoch driver, the
hand-driven ``CallbackList``, the engine guard, the SavedModel ``_export`` — lives in
:class:`matrainer.tensorflow.loop.TensorFlowRunnable`; the task half in
:class:`raidar.detection_base.DetectionBase`; and detection's keras vocabulary (the dense-map
batch read, the spec losses, the numpy AP scoring path, the decode-and-stream predict) in
:class:`raidar.detection_keras.DetectionKerasHooks` — shared verbatim with the compile/fit
sibling, which is the point.

What stays HERE beyond the flat constructor is exactly what a 4-ARITY step forces: the traced
train/eval steps (the spec's three losses summed inside the graph — ``keras.ops``, so the SAME
loss module serves both engines), the FLAT ``(x, cls, box, ctr)`` batch tuple those steps
splat over, and the traced batch predictor the scoring path uses. There are no metric objects
in this loop — detection quality is the spec's numpy AP, computed AFTER the epochs.

This loop is raw TensorFlow and REQUIRES the engine: run with ``KERAS_BACKEND=tensorflow``
(the ``--framework tensorflow`` CLI flag pins it for you).
"""

from typing import Any, Callable, Iterable, List, Literal, Optional, Union, cast

import numpy as np
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, batch_values

from raidar._tensorflow import keras, tf
from raidar.detection_keras import DetectionKerasHooks

# AFTER the `_tensorflow` shim import: the shim setdefaults KERAS_BACKEND before anything reads
# it (the matrainer module does the same itself — belt-and-braces, and it keeps this module's
# import story identical to every other raidar TF-touching module).
from matrainer.tensorflow.loop import TensorFlowNetwork, TensorFlowRunnable  # isort: skip

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = ["DetectorTask", "KerasOptimizer", "TensorFlowDetector", "TensorFlowNetwork"]


@configurable(task="detection", role="trainer", framework="keras")
class TensorFlowDetector(DetectionKerasHooks, TensorFlowRunnable):
    """Train, evaluate or predict the shared FCOS detection spec with a native TF loop.

    Args:
        model: The spec model in raw ``tf.nn`` ops (``TensorFlowFcos``) or any Keras
            three-output equivalent — wire ``!lazy:`` so ``initialize()`` injects the resolved
            ``num_classes``.
        train_set: Training source (``task="fit"``). Its ops chain MUST end in ``FcosTargets``.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Detection configs pin it explicitly.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        jit_compile: Compile the traced steps with XLA.
        prefetch: ``dataset.prefetch(AUTOTUNE)`` between the sequence and the loop.
        export_dir: Where ``fit()`` writes the SavedModel. ``None`` skips the export.
        trackers: matrainer trackers — one or a list.
        callbacks: Keras callbacks, driven by hand through a ``CallbackList``.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the ground-truth ``Boxes``.
        cls_key / box_key / ctr_key: Record keys of the dense maps ``FcosTargets`` wrote.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        learning_rate: Learning rate applied to a STRING-alias optimizer.
        metric_names: How recorded metrics are named.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[TensorFlowNetwork]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        jit_compile: bool = False,
        prefetch: bool = True,
        export_dir: Optional[str] = None,
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
            jit_compile=jit_compile,
            prefetch=prefetch,
            export_dir=export_dir,
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

    # ---- the batch boundary ------------------------------------------------------------------ #
    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """The FLAT ``(x, cls, box, ctr)`` tuple the 4-arity traced steps splat over.

        The hooks' nested shape is what ``keras.Model.fit`` wants; ``tf.data``'s
        ``from_generator`` signature and the loop's ``step(*batch)`` want one flat tuple.
        """
        x = np.transpose(np.asarray(batch_values(batch, self.input_key), dtype="float32"), (0, 2, 3, 1))
        if not with_targets:
            return (x,)
        return (
            x,
            np.asarray(batch_values(batch, self.cls_key), dtype="float32"),
            np.asarray(batch_values(batch, self.box_key), dtype="float32"),
            np.asarray(batch_values(batch, self.ctr_key), dtype="float32"),
        )

    # ---- live objects: the optimizer takes the task's learning rate -------------------------- #
    def _build_live_objects(self) -> None:
        super()._build_live_objects()
        # The task's string-alias + learning_rate coercion — the same seam the compile/fit
        # sibling routes through `_compiled_optimizer`.
        self._optimizer = self._compiled_optimizer()

    # ---- the graph steps (4-arity — the spec's three losses summed inside) ------------------- #
    def _train_step_fn(self) -> Callable[..., Any]:
        """The forward/backward pair, traced into a graph. Built ONCE and reused across epochs."""
        network, optimizer = self.model, self._optimizer
        cls_loss, box_loss, ctr_loss = self._spec_losses()

        @tf.function(jit_compile=True if self.jit_compile else None)
        def train_step(x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            with tf.GradientTape() as tape:
                cls_map, box_map, ctr_map = network(x, training=True)
                loss = (
                    tf.reduce_mean(cls_loss(y_cls, cls_map))
                    + tf.reduce_mean(box_loss(y_box, box_map))
                    + tf.reduce_mean(ctr_loss(y_ctr, ctr_map))
                )
            variables = network.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return loss

        return cast(Callable[..., Any], train_step)

    def _eval_step_fn(self, split: str = "val") -> Callable[..., Any]:
        """The forward pass plus the summed spec loss — no tape, and ``training=False``.

        ``split`` is the machinery base's signature (its metric objects are per split); this
        loop has none — detection quality is the numpy AP, computed after the epochs.
        """
        network = self.model
        cls_loss, box_loss, ctr_loss = self._spec_losses()

        @tf.function(jit_compile=True if self.jit_compile else None)
        def eval_step(x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            cls_map, box_map, ctr_map = network(x, training=False)
            return (
                tf.reduce_mean(cls_loss(y_cls, cls_map))
                + tf.reduce_mean(box_loss(y_box, box_map))
                + tf.reduce_mean(ctr_loss(y_ctr, ctr_map))
            )

        return cast(Callable[..., Any], eval_step)

    # ---- the engine seam of the scoring path ------------------------------------------------- #
    def _batch_predictor(self) -> Callable[[np.ndarray], Any]:
        """The traced predict step, built ONCE — a fresh trace per batch would retrace."""
        predict_step = self._predict_step_fn()
        return lambda x: predict_step(tf.convert_to_tensor(x))
