"""``TensorFlowDetector`` — the same detection run on a native ``tf.data`` + ``GradientTape`` loop.

The sibling of :class:`~raidar.keras_detector.KerasDetector`, and the one that answers a fair
question: *does this need its own runnable, when ``KERAS_BACKEND=tensorflow raidar train
--framework keras`` already runs the identical config?* Yes — but ONLY for what Keras 3 does not
expose:

* ``_dataset`` — a ``tf.data.Dataset`` over recordstream batches, with ``prefetch(AUTOTUNE)``;
* ``_train_step_fn`` / ``_eval_step_fn`` / ``_predict_step_fn`` — ``tf.GradientTape`` under
  ``@tf.function``, which is what makes ``jit_compile`` (XLA) a knob at all;
* ``_run_epoch`` — the loop ``fit()`` would have run;
* ``_export`` — the SavedModel, the one artifact that is TensorFlow's and not Keras's.

It is a SIBLING, not a subclass: it **narrows a precondition** its would-be parent does not have
(``_require_tensorflow_backend()`` raises where a ``KerasDetector`` runs fine), so a picker
offering "any KerasDetector" would hand out something that cannot run. It still declares
``framework="keras"`` — the axis names the API a class is wired to, and its slots are
Keras-shaped. The LOSS is the spec's own three functions from :mod:`raidar.fcos.keras_fcos` —
``keras.ops`` on the TensorFlow engine traces cleanly under ``tf.function`` — and detection
quality is the spec's numpy AP, exactly as on the Keras backend. Batching needs the same
``FcosTargets`` dense-map chain: this loop, too, stacks through ``collate_records``.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from recordstream import (
    PredictionsSink,
    RecordSource,
    batch_metadata,
    batch_regions,
    batch_values,
    ensure_materialized,
    ensure_record_dataset,
)
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

# Everything TF- and Keras-shaped comes from ONE module, which pins KERAS_BACKEND=tensorflow
# before importing keras — see raidar/_tensorflow.py.
from raidar._tensorflow import RecordSequence, keras, keras_backend, tf
from raidar.fcos import spec
from raidar.fcos.keras_fcos import fcos_box_loss, fcos_cls_loss, fcos_ctr_loss

logger = get_logger(__name__)

#: Which entry point :meth:`TensorFlowDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

KerasOptimizer = Union[str, keras.optimizers.Optimizer]

#: The model slot takes two kinds of thing sharing no useful ancestor: a ``keras.Model`` is NOT
#: a ``tf.Module``. Spelled as a union rather than `Any` so `to_pydantic` still has a widget.
TensorFlowNetwork = Union[keras.Model, tf.Module]

__all__ = ["DetectorTask", "KerasOptimizer", "TensorFlowDetector", "TensorFlowNetwork"]


@configurable(task="detection", role="trainer", framework="keras")
class TensorFlowDetector(ProgressReporting):
    """Train, evaluate or predict the shared FCOS detection spec with a native TensorFlow loop.

    Args:
        model: The spec model — a ``keras.Model`` (``KerasFcos``) or a raw ``tf.Module``
            (``TensorFlowFcos``). Wire ``!lazy:`` so ``initialize()`` injects ``num_classes``.
        train_set: Training source (``task="fit"``). Its ops chain MUST end in ``FcosTargets``.
        test_set: Evaluation / prediction source.
        num_classes: Class count — pinned in detection configs (``FcosTargets`` needs it too).
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size: the shared knobs.
        jit_compile: Trace the steps with XLA. Off by default — a slow first step and a harder
            traceback is the wrong trade for a smoke run.
        prefetch: Overlap input preparation with the step (``tf.data.AUTOTUNE``).
        export_dir: Where to write the SavedModel after a fit. ``None`` writes none.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances, driven through a ``CallbackList``.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the ground-truth ``Regions``
            (``FcosTargets`` leaves it intact; ``batch_regions`` reads it back for scoring).
        cls_key / box_key / ctr_key: Record keys of the dense maps ``FcosTargets`` wrote.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        learning_rate: Learning rate applied to a STRING-alias optimizer.
        metric_names: How recorded metrics are named on the way to the trackers and the sidecar.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[TensorFlowNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        jit_compile: bool = False,
        prefetch: bool = True,
        export_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        cls_key: str = "fcos_cls",
        box_key: str = "fcos_box",
        ctr_key: str = "fcos_ctr",
        optimizer: Lazy[KerasOptimizer] = "adam",
        learning_rate: float = 1e-3,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        self.model: TensorFlowNetwork = cast(TensorFlowNetwork, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: DetectorTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.jit_compile = jit_compile
        self.prefetch = prefetch
        self.export_dir = export_dir
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.cls_key = str(cls_key)
        self.box_key = str(box_key)
        self.ctr_key = str(ctr_key)
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        self.learning_rate = float(learning_rate)
        self.callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._optimizer: Any = None
        self._losses: Any = None
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Check the engine, resolve ``num_classes``, then build the model and the live objects."""
        self._require_tensorflow_backend()
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        nc = self._resolve_num_classes()
        if nc is None:
            raise RuntimeError(
                "could not determine num_classes — set it on the detector, the model, or the "
                "FcosTargets op (a detection config pins it)."
            )
        self.num_classes = int(nc)
        self.model = flow(self.model, num_classes=self.num_classes)
        box_loss = fcos_box_loss(float(getattr(self.model, "spec_stride", spec.stride())))
        self._losses = (fcos_cls_loss, box_loss, fcos_ctr_loss)
        optimizer = flow(self.optimizer)
        if isinstance(optimizer, str):
            optimizer = keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": self.learning_rate}})
        self._optimizer = optimizer

    @staticmethod
    def _require_tensorflow_backend() -> None:
        """Fail loudly when keras loaded some OTHER engine — the one way this backend goes wrong.

        The spec losses are ``keras.ops``, and on another engine ``tf.GradientTape`` cannot
        differentiate through them; the failure deep inside the tape says nothing useful.
        ``recordstream.keras`` defaults KERAS_BACKEND to the first engine INSTALLED (torch wins
        on this workspace) unless something set the variable first (:mod:`raidar._tensorflow`
        does, for anything imported through it).
        """
        backend = keras_backend()
        if backend != "tensorflow":
            raise RuntimeError(
                f"TensorFlowDetector needs Keras on the TensorFlow engine, but keras loaded the "
                f"{backend!r} one. Keras reads KERAS_BACKEND ONCE, at import time. Export "
                f"KERAS_BACKEND=tensorflow before running, or use the `keras` backend."
            )

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_num_classes(self) -> Optional[int]:
        """Model head → explicit arg → a lazy model's pinned kwarg. No data walk."""
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    # ---- the batch boundary -------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape."""
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=lambda batch: self._to_arrays(batch, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → FLAT numpy ``(x, cls, box, ctr)`` for the traced step.

        Flat rather than nested — ``tf.data.from_generator``'s signature stays a plain tuple.
        The image comes through the shared ``ToTensor`` chain as ``[N, C, H, W]``; the engine's
        channels-last layout is the one transpose here (same as the Keras backend).
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

    def _dataset(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> Tuple[Any, Any]:
        """A ``tf.data.Dataset`` over recordstream batches, plus the sequence that feeds it.

        The sequence comes back too because the caller owns one thing ``fit()`` would have done:
        ``on_epoch_end``, the per-epoch reshuffle. The ``output_signature`` is derived from a
        real first batch so the batch shape is stated in exactly one place (``_to_arrays``).
        """
        sequence = self._sequence(source, shuffle=shuffle, with_targets=with_targets)
        dataset = tf.data.Dataset.from_generator(
            lambda: (sequence[i] for i in range(len(sequence))),
            output_signature=_output_signature(sequence[0]),
        )
        return (dataset.prefetch(tf.data.AUTOTUNE) if self.prefetch else dataset), sequence

    # ---- the graph steps ----------------------------------------------------- #
    def _train_step_fn(self) -> Any:
        """The forward/backward pair, traced into a graph. Built ONCE and reused across epochs."""
        network, optimizer = self.model, self._optimizer
        cls_loss, box_loss, ctr_loss = self._losses

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

        return train_step

    def _eval_step_fn(self) -> Any:
        """The forward pass plus the summed spec loss — no tape, and ``training=False``."""
        network = self.model
        cls_loss, box_loss, ctr_loss = self._losses

        @tf.function(jit_compile=True if self.jit_compile else None)
        def eval_step(x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            cls_map, box_map, ctr_map = network(x, training=False)
            return (
                tf.reduce_mean(cls_loss(y_cls, cls_map))
                + tf.reduce_mean(box_loss(y_box, box_map))
                + tf.reduce_mean(ctr_loss(y_ctr, ctr_map))
            )

        return eval_step

    def _predict_step_fn(self) -> Any:
        """The three raw head maps — the graph half of the decode path."""
        network = self.model

        @tf.function(jit_compile=True if self.jit_compile else None)
        def predict_step(x: Any) -> Any:
            return network(x, training=False)

        return predict_step

    def _run_epoch(self, dataset: Any, step: Any, callbacks: Any = None) -> float:
        """One pass over ``dataset``, returning the mean loss."""
        total, batches = 0.0, 0
        for index, batch in enumerate(dataset):
            total += float(step(*batch))
            batches += 1
            if callbacks is not None:
                callbacks.on_train_batch_end(index, {"loss": total / batches})
        return total / batches if batches else float("nan")

    # ---- callbacks + tracking ------------------------------------------------ #
    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL matrainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = self._live_trackers()
        if trackers:
            from matrainer.keras import keras_tracking_callback

            callbacks.append(
                keras_tracking_callback(trackers, runnable=self, metric_names=as_metric_names(self.metric_names))
            )
        return callbacks

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """The custom loop: epochs, tape steps, callbacks, the numpy AP pass, the SavedModel."""
        logger.info(f"raidar tensorflow training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TensorFlowDetector.fit() requires a 'train_set'.")
        train_data, sequence = self._dataset(self.train_set, shuffle=True)
        val_data = self._dataset(self.val_set)[0] if self.val_set is not None else None

        # Slot variables OUTSIDE the graph: Keras builds them on the first `apply_gradients`,
        # and creating variables during a retrace (a short final batch triggers one) is an error.
        self._optimizer.build(list(self.model.trainable_variables))
        train_step, eval_step = self._train_step_fn(), self._eval_step_fn()

        # A raw tf.Module has no Keras callback surface, so the CallbackList takes no model then.
        callback_model = self.model if isinstance(self.model, keras.Model) else None
        callbacks = keras.callbacks.CallbackList(self._callbacks(), add_history=False, model=callback_model)
        callbacks.set_params({"epochs": self.max_epochs, "steps": len(sequence), "verbose": 0})
        callbacks.on_train_begin()

        history: Dict[str, List[float]] = {}
        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            callbacks.on_epoch_begin(epoch)
            logs = {"loss": self._run_epoch(train_data, train_step, callbacks)}
            if val_data is not None:
                logs["val_loss"] = self._run_epoch(val_data, eval_step)
            callbacks.on_epoch_end(epoch, logs)
            sequence.on_epoch_end()  # Keras's `fit()` calls this between epochs; this loop must.
            for key, value in logs.items():
                history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        callbacks.on_train_end(logs)

        self._history = {k: list(v) for k, v in history.items()}
        final: Dict[str, float] = {k: v[-1] for k, v in history.items()}
        if self.val_set is not None:
            final.update({f"val_{k}": v for k, v in self._score(self.val_set).items()})
        self._computed_metrics = as_metric_names(self.metric_names)(final, source="keras")
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        self._export()

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("TensorFlowDetector.test() requires a 'test_set'.")
        self._computed_metrics = as_metric_names(self.metric_names)(
            self._score(self.test_set), source="keras", split="test"
        )
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("TensorFlowDetector.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        predict_step = self._predict_step_fn()
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            decoded = self._decoded_batch(sequence[index][0], predict_step)
            if sink is not None:
                meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
                for i, prediction in enumerate(decoded):
                    sink.write(prediction, meta[i] if meta and i < len(meta) else {})
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- the spec's eval path ------------------------------------------------ #
    def _decode_config(self) -> Dict[str, Any]:
        """The decode knobs, read off the model wrapper (spec defaults for a bare model)."""
        return {
            "input_shape": tuple(getattr(self.model, "input_shape", spec.INPUT_SHAPE)),
            "widths": tuple(getattr(self.model, "widths", spec.BACKBONE_WIDTHS)),
            "score_threshold": float(getattr(self.model, "score_threshold", 0.05)),
            "iou_threshold": float(getattr(self.model, "iou_threshold", 0.5)),
            "max_detections": int(getattr(self.model, "max_detections", 100)),
        }

    def _decoded_batch(self, x: np.ndarray, predict_step: Any) -> List[Dict[str, np.ndarray]]:
        """One image batch → per-image scored boxes, through the spec's numpy decode."""
        cls_map, box_map, ctr_map = predict_step(tf.convert_to_tensor(x))
        cls_np, box_np, ctr_np = np.asarray(cls_map), np.asarray(box_map), np.asarray(ctr_map)
        config = self._decode_config()
        return [spec.decode_predictions(cls_np[i], box_np[i], ctr_np[i][..., 0], **config) for i in range(len(cls_np))]

    def _score(self, source: Any) -> Dict[str, float]:
        """Decode a split and score it with the spec's AP50 — the numpy scoring path."""
        predict_step = self._predict_step_fn()
        sequence = self._sequence(source, with_targets=False)
        predictions: List[Dict[str, np.ndarray]] = []
        targets: List[Dict[str, np.ndarray]] = []
        for index, batch in enumerate(sequence.batches()):
            predictions.extend(self._decoded_batch(sequence[index][0], predict_step))
            targets.extend(batch_regions(batch, self.target_key))
        return spec.average_precision_50(predictions, targets)

    # ---- the SavedModel ------------------------------------------------------ #
    def _export(self) -> None:
        """Write the SavedModel — the one run artifact that is TensorFlow's and not Keras's.

        A ``keras.Model`` has ``export``; a raw ``tf.Module`` does not, so it goes through
        ``tf.saved_model.save`` with an explicitly traced signature (a bare ``__call__`` is
        untraced Python). Branching on the CAPABILITY keeps anything else callable working too.
        """
        if not self.export_dir:
            return
        path = Path(self.export_dir).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "export"):
            self.model.export(str(path))
        else:
            height, width, channels = self._decode_config()["input_shape"]
            signature = tf.function(
                lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec(shape=(None, height, width, channels), dtype=tf.float32)],
            )
            tf.saved_model.save(self.model, str(path), signatures={"serving_default": signature})
        logger.info(f"Exported SavedModel to {path}")

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
        return self.model

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)

    @property
    @output
    def history(self) -> Dict[str, List[float]]:
        """Per-epoch loss history — what a ``fit()``'s ``History`` would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}


def _output_signature(batch: Any) -> Tuple[Any, ...]:
    """``tf.TensorSpec``s matching a real batch, with the row count left dynamic."""
    arrays = [np.asarray(item) for item in batch]
    return tuple(tf.TensorSpec(shape=(None, *a.shape[1:]), dtype=tf.as_dtype(a.dtype)) for a in arrays)
