"""``KerasDetector`` — the same detection run, on Keras 3 instead of a torch loop.

The first non-torch backend, and its job is to make one claim checkable: **the config is the
program**. A Keras run and a Lightning run share the dataset, the batch/epoch knobs, the run
identity, the metrics sidecar and the tracking. What differs is the training loop, the model —
and, uniquely for detection, HOW THE TARGETS BATCH.

**Detection's batching problem, and the spec's answer.** The torch backends carry variable-length
per-image ``Regions`` targets as per-record COLUMNS — recordstream's own ``collate_records``
handles that shape natively. This loop needs REGULAR stacking for its dense maps, so a Keras
detection config appends
:class:`raidar.fcos.targets.FcosTargets` to its ops chain, which encodes each record's boxes
through the spec's numpy assignment into FIXED-SHAPE dense maps before collation. The batch is
then ``x [N, H, W, C]`` against three dense target maps, and :meth:`_to_arrays` only reads them.

**Three engine differences a reader should know:**

1. **Keras is channels-LAST**, but the detection preprocess KEEPS the shared ``ToTensor`` (the
   RGB coercion for cppe-5's mixed RGB/RGBA/grayscale rows lives there, and re-deriving it here
   would be a second copy) — the layout difference is ONE transpose in :meth:`_to_arrays`.
2. **The loss is the spec's, not a compile-slot vocabulary**: three per-output functions
   (:mod:`raidar.fcos.keras_fcos`) matched to the model's three raw head maps. There is no
   ``loss:`` slot — a detection spec model and its loss are one design, exactly as an HF
   detector's loss lives inside its forward.
3. **Detection quality is scored by the SPEC's numpy AP, not by Keras metrics** — torchmetrics
   is torch-only and Keras ships no variable-size detection metric, so after ``fit`` (and in
   ``test()``) the backend decodes the raw maps through :func:`raidar.fcos.spec.decode_predictions`
   and scores :func:`raidar.fcos.spec.average_precision_50`, recording the canonical
   ``<split>/map_50`` / ``<split>/precision`` / ``<split>/recall``.

Keras 3 picks its engine from ``KERAS_BACKEND``; this module never sets it.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

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
from recordstream.keras import RecordSequence
from recordstream.runnable import ProgressReporting, entrypoint, run_entrypoint

from raidar._keras import keras, keras_backend
from raidar.fcos import spec
from raidar.fcos.keras_fcos import fcos_box_loss, fcos_cls_loss, fcos_ctr_loss

logger = get_logger(__name__)

#: Which entry point :meth:`KerasDetector.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

KerasOptimizer = Union[str, keras.optimizers.Optimizer]

__all__ = ["DetectorTask", "KerasDetector", "KerasOptimizer"]


@configurable(task="detection", role="trainer", framework="keras")
class KerasDetector(ProgressReporting):
    """Train, evaluate or predict the shared FCOS detection spec with Keras 3.

    Args:
        model: The spec model — wire ``!lazy:`` (e.g. ``KerasFcos``) so ``initialize()`` can
            inject the resolved ``num_classes``.
        train_set: Training source (``task="fit"``). Its ops chain MUST end in ``FcosTargets``.
        test_set: Evaluation / prediction source.
        num_classes: Class count. Detection configs pin it explicitly — the ``FcosTargets`` op
            needs the same number, so there is no derivation walk here.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: ``keras.callbacks.Callback`` instances.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the ground-truth ``Regions``
            (``FcosTargets`` leaves it intact; ``batch_regions`` reads it back for scoring).
        cls_key / box_key / ctr_key: Record keys of the dense maps ``FcosTargets`` wrote — the
            names must match that op's.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        learning_rate: Learning rate applied to a STRING-alias optimizer (an instance keeps its
            own; YAML floats like ``1e-3`` may arrive as strings, hence the coercion).
        metric_names: How recorded metrics are named — the translation out of Keras's
            ``loss`` / ``val_loss`` vocabulary on the way to the trackers and the sidecar.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``predictions_sink``,
    ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[keras.Model]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
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
        # `keras.Model`, not `Lazy[keras.Model]`: the DEFERRED-ness is declared once, on the
        # ctor param above, which is what `lazy_param_names` / `to_pydantic` read.
        self.model: keras.Model = cast(keras.Model, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: DetectorTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.cls_key = str(cls_key)
        self.box_key = str(box_key)
        self.ctr_key = str(ctr_key)
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        self.learning_rate = float(learning_rate)
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy[keras.callbacks.Callback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then build and compile the model with the spec loss."""
        logger.info(f"Keras backend in effect: {keras_backend()}")
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        nc = self._resolve_num_classes()
        if nc is None:
            raise RuntimeError(
                "could not determine num_classes — set it on the detector, the model, or the "
                "FcosTargets op (a detection config pins it: the dense-target op needs the same number)."
            )
        self.num_classes = int(nc)
        self._compile()

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _resolve_num_classes(self) -> Optional[int]:
        """Model head → explicit arg → a lazy model's pinned kwarg. No data walk — see Args."""
        if not isinstance(self.model, Fluid):
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(
                        f"Detector num_classes={self.num_classes} != model num_classes={model_nc}; using model's."
                    )
                return int(model_nc)
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return None

    def _compile(self) -> None:
        """Build the model at the resolved width, then compile it with the spec's three losses.

        The loss list matches the model's three raw outputs positionally — cls, box, ctr. There
        is deliberately NO ``loss:`` slot to override: the spec model and its loss are one design.
        The box loss is a closure (the raw-to-distance scale closes over the model's stride), so
        it is renamed after itself — Keras names the history columns from the FUNCTION name, and
        a bare ``loss`` there would log the box component as the ambiguous ``loss_loss``.
        """
        self.model = flow(self.model, num_classes=self.num_classes)
        box_loss = fcos_box_loss(float(getattr(self.model, "spec_stride", spec.stride())))
        box_loss.__name__ = "fcos_box_loss"
        optimizer = flow(self.optimizer)
        if isinstance(optimizer, str):
            optimizer = keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": self.learning_rate}})
        self.model.compile(optimizer=optimizer, loss=[fcos_cls_loss, box_loss, fcos_ctr_loss])

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
        """One collated record batch → the numpy ``(x, y)`` Keras consumes.

        The image comes through the SHARED ``ToTensor`` chain as ``[N, C, H, W]`` — the RGB
        coercion for mixed-mode rows lives there — so the engine's channels-last layout is ONE
        transpose here. The targets are the dense maps ``FcosTargets`` stacked regularly.
        """
        x = np.transpose(np.asarray(batch_values(batch, self.input_key), dtype="float32"), (0, 2, 3, 1))
        if not with_targets:
            return (x,)
        y = (
            np.asarray(batch_values(batch, self.cls_key), dtype="float32"),
            np.asarray(batch_values(batch, self.box_key), dtype="float32"),
            np.asarray(batch_values(batch, self.ctr_key), dtype="float32"),
        )
        return x, y

    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL matrainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            # Keras's file-writing callbacks open their path without creating its parent.
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

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"raidar keras training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("KerasDetector.fit() requires a 'train_set'.")
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None
        history = self.model.fit(
            train_seq, validation_data=val_seq, epochs=self.max_epochs, callbacks=self._callbacks()
        )
        self._history = {k: [float(x) for x in v] for k, v in history.history.items()}
        final = {k: v[-1] for k, v in self._history.items()}
        # Detection quality: the spec's numpy AP over the validation split — Keras metrics
        # cannot score variable-size detections, so the loop's own history carries losses only.
        if self.val_set is not None:
            final.update({f"val_{k}": v for k, v in self._score(self.val_set).items()})
        self._computed_metrics = as_metric_names(self.metric_names)(final, source="keras")
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasDetector.test() requires a 'test_set'.")
        self._computed_metrics = as_metric_names(self.metric_names)(
            self._score(self.test_set), source="keras", split="test"
        )
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("KerasDetector.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            decoded = self._decoded_batch(sequence[index][0])
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

    def _decoded_batch(self, x: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """One image batch → per-image scored boxes, through the spec's numpy decode."""
        cls_map, box_map, ctr_map = self.model.predict(x, verbose=0)
        config = self._decode_config()
        return [
            spec.decode_predictions(
                np.asarray(cls_map[i]), np.asarray(box_map[i]), np.asarray(ctr_map[i][..., 0]), **config
            )
            for i in range(len(cls_map))
        ]

    def _score(self, source: Any) -> Dict[str, float]:
        """Decode a split and score it with the spec's AP50 — the numpy scoring path.

        The ground truth comes from the batched record's ``target_key`` — the plain
        ``{"boxes", "labels"}`` dicts ``FcosTargets`` left beside the dense maps, batched as a
        per-record list.
        """
        sequence = self._sequence(source, with_targets=False)
        predictions: List[Dict[str, np.ndarray]] = []
        targets: List[Dict[str, np.ndarray]] = []
        for index, batch in enumerate(sequence.batches()):
            decoded = self._decoded_batch(sequence[index][0])
            predictions.extend(decoded)
            targets.extend(batch_regions(batch, self.target_key))
        return spec.average_precision_50(predictions, targets)

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

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
        """Per-epoch metric history, as Keras's ``History`` recorded it."""
        return {key: list(values) for key, values in self._history.items()}
