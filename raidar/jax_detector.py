"""``JaxDetector`` — the same detection run, on a jitted ``optax`` loop.

The sibling of the TensorFlow backend — both replace a framework's ``fit()`` with a traced step —
where the differences between them are exactly the differences between the two programming
models: the step is a PURE FUNCTION of ``(model, batch)`` that ``nnx.value_and_grad``
differentiates and ``nnx.Optimizer`` applies in place, it is jitted with ``nnx.jit`` ONCE and
reused across epochs (a fresh jit per epoch retraces the whole graph — the same trap
``tf.function`` has), and randomness is explicit (``seed:`` on ``FlaxFcos`` is a real knob).

**The loss is the spec's, written in ``jnp``** — the differentiable half of
:mod:`raidar.fcos.spec`'s recipe, the third spelling after torch and ``keras.ops``. Focal uses
``optax.sigmoid_binary_cross_entropy`` for the stable BCE core; the IoU and centerness terms
mask on the positive cells their own dense targets identify. There is no ``loss:`` slot — the
spec model and its loss are one design.

**Batching and scoring are the Keras backend's**: ``FcosTargets`` makes the batches regular
(dense fixed-shape maps; this loop too stacks through ``collate_records``), ``RecordSequence``
is reused verbatim (batching a record stream is not a JAX problem), and detection quality is the
spec's numpy AP over decoded head maps — recorded as the canonical ``<split>/map_50`` /
``<split>/precision`` / ``<split>/recall``.
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from confluid import Fluid, Lazy, Mandatory, configurable, flow, output
from flax import nnx
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
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

from raidar.fcos import spec

logger = get_logger(__name__)

#: Which entry point :meth:`JaxDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

#: An ``optax`` optimizer NAME or a callable returning a ``GradientTransformation``.
JaxOptimizer = Union[str, Callable[..., Any]]

#: A plain callable invoked once per epoch with the epoch index and that epoch's logs — JAX
#: ships no callback protocol, so this is the plainest thing that works.
EpochCallback = Callable[[int, Dict[str, float]], None]

__all__ = ["DetectorTask", "EpochCallback", "JaxDetector", "JaxOptimizer"]


def _spec_loss(box_stride: float) -> Callable[..., Any]:
    """The spec's loss recipe in ``jnp`` — focal + (1 − IoU) + centerness BCE, per image.

    A factory because the raw-to-distance rule (``exp(raw) * stride``) closes over the model's
    stride. Each term derives the positive-cell mask from its OWN dense target and normalizes by
    that image's positive count, exactly as the torch and ``keras.ops`` spellings do.
    """

    def compute(maps: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
        cls_map, box_map, ctr_map = maps
        num_fg = jnp.maximum(jnp.sum(jnp.max(y_cls, axis=-1), axis=(1, 2)), 1.0)  # [B]

        # Focal (alpha=0.25, gamma=2) over ALL cells.
        p = jax.nn.sigmoid(cls_map)
        bce = optax.sigmoid_binary_cross_entropy(cls_map, y_cls)
        p_t = y_cls * p + (1.0 - y_cls) * (1.0 - p)
        alpha_t = y_cls * 0.25 + (1.0 - y_cls) * 0.75
        loss_cls = jnp.sum(alpha_t * (1.0 - p_t) ** 2 * bce, axis=(1, 2, 3)) / num_fg

        # 1 - IoU of the (l, t, r, b) pairs on the positive cells.
        fg = (jnp.min(y_box, axis=-1) > 0.0).astype(box_map.dtype)  # [B, rows, cols]
        predicted = jnp.exp(box_map) * box_stride
        lr = jnp.minimum(predicted[..., 0], y_box[..., 0]) + jnp.minimum(predicted[..., 2], y_box[..., 2])
        tb = jnp.minimum(predicted[..., 1], y_box[..., 1]) + jnp.minimum(predicted[..., 3], y_box[..., 3])
        intersection = jnp.maximum(lr, 0.0) * jnp.maximum(tb, 0.0)
        area_p = (predicted[..., 0] + predicted[..., 2]) * (predicted[..., 1] + predicted[..., 3])
        area_t = (y_box[..., 0] + y_box[..., 2]) * (y_box[..., 1] + y_box[..., 3])
        iou = intersection / jnp.maximum(area_p + area_t - intersection, 1e-9)
        loss_box = jnp.sum((1.0 - iou) * fg, axis=(1, 2)) / num_fg

        # Centerness BCE on the positive cells (a positive cell's target is strictly positive).
        ctr_logit = ctr_map[..., 0]
        ctr_fg = (y_ctr > 0.0).astype(ctr_logit.dtype)
        ctr_bce = optax.sigmoid_binary_cross_entropy(ctr_logit, y_ctr)
        loss_ctr = jnp.sum(ctr_bce * ctr_fg, axis=(1, 2)) / num_fg

        return jnp.mean(loss_cls + loss_box + loss_ctr)

    return compute


def _as_optimizer(value: JaxOptimizer, learning_rate: float) -> Any:
    """A live ``GradientTransformation`` from an optax factory NAME or a callable."""
    if callable(value):
        return value(learning_rate)
    factory = getattr(optax, str(value), None)
    if factory is None:
        raise ValueError(f"unknown optax optimizer {value!r} — use an optax factory name (e.g. 'adam', 'adamw').")
    return factory(learning_rate)


@configurable(task="detection", role="trainer", framework="jax")
class JaxDetector(ProgressReporting):
    """Train, evaluate or predict the shared FCOS detection spec with a jitted optax loop.

    Args:
        model: The spec model — wire ``!lazy:raidar.fcos.flax_fcos.FlaxFcos`` so ``initialize()``
            can inject the resolved ``num_classes``.
        train_set: Training source (``task="fit"``). Its ops chain MUST end in ``FcosTargets``.
        val_set: Validation source, loss-scored per epoch and AP-scored after the fit.
        test_set: Evaluation / prediction source.
        num_classes: Class count — pinned in detection configs (``FcosTargets`` needs it too).
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        trackers: matrainer trackers — one or a list. This run's ONLY output wiring.
        callbacks: Callables invoked as ``callback(epoch, logs)`` after each epoch.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the ground-truth ``Regions``
            (``FcosTargets`` leaves it intact; ``batch_regions`` reads it back for scoring).
        cls_key / box_key / ctr_key: Record keys of the dense maps ``FcosTargets`` wrote.
        learning_rate: Peak LR for the optimizer.
        optimizer: An ``optax`` factory NAME (``"adam"`` / ``"adamw"`` / ``"sgd"``) or a callable
            returning a ``GradientTransformation``.
        metric_names: How recorded metrics are named. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing.
        jit: Trace the step functions with ``nnx.jit``. On by default — that IS the JAX story.
            Turn it off to read a Python traceback from inside a step.

    Post-construction slots (YAML / broadcasting): ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nnx.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        val_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[EpochCallback]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        cls_key: str = "fcos_cls",
        box_key: str = "fcos_box",
        ctr_key: str = "fcos_ctr",
        learning_rate: float = 1e-3,
        optimizer: JaxOptimizer = "adam",
        metric_names: Optional[Lazy[MetricNames]] = None,
        jit: bool = True,
    ) -> None:
        super().__init__()
        # `nnx.Module`, not `Lazy[...]`: the deferred-ness is declared once, on the ctor param.
        self.model: nnx.Module = cast(nnx.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = val_set
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
        self.learning_rate = float(learning_rate)
        self.optimizer: JaxOptimizer = optimizer
        self.jit = jit
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy[EpochCallback]]] = callbacks
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
        """Resolve ``num_classes``, then build the model at that width."""
        logger.info(f"JAX devices: {jax.devices()}")
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
        # `flow()` on a LIVE object is a pass-through — force the lazy build so `nnx.Optimizer`
        # reads a populated parameter tree rather than deferring the build into the first trace.
        if hasattr(self.model, "solidify"):
            self.model.solidify()

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
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → flat numpy ``(x, cls, box, ctr)`` for the traced step.

        The image comes through the shared ``ToTensor`` chain as ``[N, C, H, W]``; the engine's
        channels-last layout is the one transpose here (same as the Keras/TensorFlow backends).
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

    # ---- the traced steps ---------------------------------------------------- #
    def _make_steps(self) -> Any:
        """Build the train and eval steps once, and jit them once.

        Built once and reused across epochs on purpose: a fresh ``nnx.jit`` each epoch retraces
        the whole graph every time, which is slower than the eager loop it replaced.
        """
        loss_fn = _spec_loss(float(getattr(self.model, "spec_stride", spec.stride())))

        def train_step(model: Any, optimizer: Any, x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            def forward(m: Any) -> Any:
                return loss_fn(m(x, training=True), y_cls, y_box, y_ctr)

            loss, grads = nnx.value_and_grad(forward)(model)
            optimizer.update(model, grads)
            return loss

        def eval_step(model: Any, x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            return loss_fn(model(x, training=False), y_cls, y_box, y_ctr)

        def predict_step(model: Any, x: Any) -> Any:
            return model(x, training=False)

        if self.jit:
            return nnx.jit(train_step), nnx.jit(eval_step), nnx.jit(predict_step)
        return train_step, eval_step, predict_step

    # ---- the four entry points ----------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"raidar jax training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("JaxDetector.fit() requires a 'train_set'.")
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        # `runnable=self`: matrainer owns the ONE hyperparameter derivation.
        manifest = log_run_context(trackers, runnable=self)
        model = self.model
        optimizer = nnx.Optimizer(model, _as_optimizer(self.optimizer, self.learning_rate), wrt=nnx.Param)
        train_step, eval_step, predict_step = self._make_steps()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        scalars: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            total, batches = 0.0, 0
            for index in range(len(train_seq)):
                x, y_cls, y_box, y_ctr = train_seq[index]
                total += float(
                    train_step(
                        model, optimizer, jnp.asarray(x), jnp.asarray(y_cls), jnp.asarray(y_box), jnp.asarray(y_ctr)
                    )
                )
                batches += 1
            scalars = {"train/loss": total / batches if batches else float("nan")}
            if val_seq is not None:
                scalars["val/loss"] = self._eval_loss(val_seq, eval_step, model)
            # A `fit()` would have reshuffled for us; a hand-written loop has to. Forgetting it
            # is INVISIBLE — training just runs on one fixed row order forever.
            train_seq.on_epoch_end()
            for callback in self.callbacks or []:
                flow(callback)(epoch, scalars)
            for tracker in trackers:
                tracker.log_metrics(names(scalars), step=epoch)
            for key, value in scalars.items():
                self._history.setdefault(key, []).append(value)
            logger.info(
                f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in scalars.items())
            )

        # Detection quality: the spec's numpy AP over the validation split, once after the fit.
        if self.val_set is not None:
            scalars.update({f"val/{k}": v for k, v in self._score(self.val_set, predict_step).items()})
        self._computed_metrics = names(scalars)
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Stamps the manifest's terminal status — left unfinished it stays `running`, and
        # matrainer's atexit hook then calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self.test()

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("JaxDetector.test() requires a 'test_set'.")
        _, _, predict_step = self._make_steps()
        self._computed_metrics = as_metric_names(self.metric_names)(
            self._score(self.test_set, predict_step), split="test"
        )
        for tracker in self._live_trackers():
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        if self.test_set is None:
            raise ValueError("JaxDetector.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        _, _, predict_step = self._make_steps()
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
    def _eval_loss(self, sequence: RecordSequence, eval_step: Any, model: Any) -> float:
        total, batches = 0.0, 0
        for index in range(len(sequence)):
            x, y_cls, y_box, y_ctr = sequence[index]
            total += float(eval_step(model, jnp.asarray(x), jnp.asarray(y_cls), jnp.asarray(y_box), jnp.asarray(y_ctr)))
            batches += 1
        return total / batches if batches else float("nan")

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
        cls_map, box_map, ctr_map = predict_step(self.model, jnp.asarray(x))
        cls_np, box_np, ctr_np = np.asarray(cls_map), np.asarray(box_map), np.asarray(ctr_map)
        config = self._decode_config()
        return [spec.decode_predictions(cls_np[i], box_np[i], ctr_np[i][..., 0], **config) for i in range(len(cls_np))]

    def _score(self, source: Any, predict_step: Any) -> Dict[str, float]:
        """Decode a split and score it with the spec's AP50 — the numpy scoring path."""
        sequence = self._sequence(source, with_targets=False)
        predictions: List[Dict[str, np.ndarray]] = []
        targets: List[Dict[str, np.ndarray]] = []
        for index, batch in enumerate(sequence.batches()):
            predictions.extend(self._decoded_batch(sequence[index][0], predict_step))
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
        """Per-epoch loss history — what a framework's logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}
