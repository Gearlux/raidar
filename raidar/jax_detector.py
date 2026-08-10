"""``JaxDetector`` — the same detection run, on a jitted ``optax`` loop.

Since P6 (2026-08-08) this is ``(DetectionBase, matrainer.jax.JaxLoopRunnable)``: the loop
machinery — the jit-ONCE step builders, the SPLAT-tolerant epoch driver (this task's 4-arity
steps drive the same loop), the per-epoch reshuffle, the plain-function tracking hookup, the
record-batch predict walk — lives in :class:`matrainer.jax.loop.JaxLoopRunnable`; the
``num_classes`` resolution in :class:`~raidar.detection_base.DetectionBase`. What stays HERE
is detection's jax vocabulary:

* **The loss is the spec's, written in ``jnp``** (:func:`_spec_loss`) — the differentiable
  half of :mod:`raidar.fcos.spec`'s recipe, the third spelling after torch and ``keras.ops``.
  There is no ``loss:`` slot — the spec model and its loss are one design.
* **The batch is the FLAT dense-map tuple** ``(x, cls, box, ctr)`` the 4-arity traced steps
  splat over (``FcosTargets`` makes the batches regular).
* **Detection quality is the spec's numpy AP** over decoded head maps — appended to the fit's
  logs by :meth:`_fit_scores` and serving both eval seams; recorded as the canonical
  ``<split>/map_50`` / ``<split>/precision`` / ``<split>/recall``.
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from confluid import Lazy, Mandatory, configurable
from flax import nnx
from loggair import get_logger
from matrainer.jax.loop import EpochCallback, JaxLoopRunnable, JaxOptimizer
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, batch_boxes, batch_values

from raidar.detection_base import DetectionBase
from raidar.fcos import spec

logger = get_logger(__name__)

#: Which entry point :meth:`JaxDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

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


@configurable(task="detection", role="trainer", framework="jax")
class JaxDetector(DetectionBase, JaxLoopRunnable):
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
        input_key / target_key: Record keys for the image and the ground-truth ``Boxes``
            (``FcosTargets`` leaves it intact; ``batch_boxes`` reads it back for scoring).
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
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            jit=jit,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metric_names=metric_names,
        )
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.cls_key = str(cls_key)
        self.box_key = str(box_key)
        self.ctr_key = str(ctr_key)

    # ---- setup: num_classes is the task base's; the forced build is ours --------------------- #
    def initialize(self) -> None:
        super().initialize()
        # `flow()` on a LIVE object is a pass-through — force the lazy build so `nnx.Optimizer`
        # reads a populated parameter tree rather than deferring the build into the first trace.
        if hasattr(self.model, "solidify"):
            self.model.solidify()

    def _resolve_num_classes(self) -> Optional[int]:
        """The task base's ladder, made LOUD: the spec model builds its head at this width."""
        nc = super()._resolve_num_classes()
        if nc is None:
            raise RuntimeError(
                "could not determine num_classes — set it on the detector, the model, or the "
                "FcosTargets op (a detection config pins it)."
            )
        return nc

    # ---- the batch boundary -------------------------------------------------------------------- #
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

    # ---- the traced steps (4-arity — the spec's three losses summed inside) ------------------- #
    def _make_steps(self) -> Tuple[Callable[..., Any], Callable[..., Any]]:
        """The spec's steps, built once and jitted once — bare-loss returns, which the
        machinery's splat-tolerant driver unpacks."""
        loss_fn = _spec_loss(float(getattr(self.model, "spec_stride", spec.stride())))

        def train_step(model: Any, optimizer: Any, x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            def forward(m: Any) -> Any:
                return loss_fn(m(x, training=True), y_cls, y_box, y_ctr)

            loss, grads = nnx.value_and_grad(forward)(model)
            optimizer.update(model, grads)
            return loss

        def eval_step(model: Any, x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            return loss_fn(model(x, training=False), y_cls, y_box, y_ctr)

        if self.jit:
            return nnx.jit(train_step), nnx.jit(eval_step)
        return train_step, eval_step

    # ---- the spec's eval path ------------------------------------------------------------------ #
    def _fit_scores(self, logs: Dict[str, float]) -> Dict[str, float]:
        """Append the spec's numpy AP over the validation split, once after the fit."""
        if self.val_set is not None:
            predict_step = self._predict_step()
            logs.update({f"val/{k}": v for k, v in self._score_ap(self.val_set, predict_step).items()})
        return logs

    def _evaluate_split(self, source: Any, split: Literal["val", "test"]) -> Dict[str, float]:
        """Score a split with the spec's AP — neither engine loss pass runs, because a loss
        over dense maps is not a detection score."""
        predict_step = self._predict_step()
        return as_metric_names(self.metric_names)(self._score_ap(source, predict_step), split=split)

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

    def _score_ap(self, source: Any, predict_step: Any) -> Dict[str, float]:
        """Decode a split and score it with the spec's AP50 — the numpy scoring path."""
        sequence = self._sequence(source, with_targets=False)
        predictions: List[Dict[str, np.ndarray]] = []
        targets: List[Dict[str, np.ndarray]] = []
        for index, batch in enumerate(sequence.batches()):
            predictions.extend(self._decoded_batch(sequence[index][0], predict_step))
            targets.extend(batch_boxes(batch, self.target_key))
        return spec.average_precision_50(predictions, targets)

    # ---- predictions --------------------------------------------------------------------------- #
    def _predict_batch(self, predict_step: Any, x: Any) -> Any:
        """Per-image DECODED detections — the spec's decode IS the task's prediction contract,
        and a LIST is already per-record, so the machinery writes it as-is."""
        return self._decoded_batch(np.asarray(x), predict_step)
