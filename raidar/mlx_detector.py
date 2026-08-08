"""``MlxDetector`` — the same detection run on Apple's MLX, completing the engine set.

Since P6 (2026-08-08) this is ``(DetectionBase, matrainer.mlx.MlxLoopRunnable)``: the loop
machinery — the ``nn.value_and_grad`` step with the per-step ``mx.eval``, the train/eval MODE
flips at the loop boundaries, the epoch driver with the per-epoch reshuffle, the
plain-function tracking hookup — lives in :class:`matrainer.mlx.loop.MlxLoopRunnable`; the
``num_classes`` resolution in :class:`~raidar.detection_base.DetectionBase`. What stays HERE
is detection's mlx vocabulary:

* **The loss is the spec's, written in ``mx``** (:func:`_spec_loss`) — the fourth spelling of
  :mod:`raidar.fcos.spec`'s recipe (after torch, ``keras.ops`` and ``jnp``), with the stable
  BCE written out because ``mlx.nn.losses`` ships no sigmoid-BCE-with-logits over dense maps
  in the shape this needs. There is no ``loss:`` slot — the spec model and its loss are one
  design.
* **The batch is the FLAT dense-map tuple** ``(x, cls, box, ctr)`` — the 4-arity objective
  and eval batch override the machinery's pair-shaped defaults; the driver splats.
* **Detection quality is the spec's numpy AP** over decoded head maps — appended to the
  fit's logs by :meth:`_fit_scores` and serving both eval seams.

**Apple silicon ONLY** — the extra's platform marker keeps ``pip install raidar[mlx]``
succeeding (and empty) elsewhere.
"""

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.mlx.loop import EpochCallback, MlxLoopRunnable, MlxOptimizer
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, batch_regions, batch_values

from raidar.detection_base import DetectionBase
from raidar.fcos import spec

logger = get_logger(__name__)

#: Which entry point :meth:`MlxDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

__all__ = ["DetectorTask", "EpochCallback", "MlxDetector", "MlxOptimizer"]


def _sigmoid_bce(logits: Any, labels: Any) -> Any:
    """Numerically stable sigmoid BCE with logits: ``max(l,0) - l*y + log(1 + exp(-|l|))``."""
    return mx.maximum(logits, 0.0) - logits * labels + mx.log1p(mx.exp(-mx.abs(logits)))


def _spec_loss(box_stride: float) -> Callable[..., Any]:
    """The spec's loss recipe in ``mx`` — focal + (1 − IoU) + centerness BCE, per image."""

    def compute(maps: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
        cls_map, box_map, ctr_map = maps
        num_fg = mx.maximum(mx.max(y_cls, axis=-1).sum(axis=(1, 2)), 1.0)  # [B]

        # Focal (alpha=0.25, gamma=2) over ALL cells.
        p = mx.sigmoid(cls_map)
        bce = _sigmoid_bce(cls_map, y_cls)
        p_t = y_cls * p + (1.0 - y_cls) * (1.0 - p)
        alpha_t = y_cls * 0.25 + (1.0 - y_cls) * 0.75
        loss_cls = (alpha_t * (1.0 - p_t) ** 2 * bce).sum(axis=(1, 2, 3)) / num_fg

        # 1 - IoU of the (l, t, r, b) pairs on the positive cells.
        fg = (mx.min(y_box, axis=-1) > 0.0).astype(box_map.dtype)  # [B, rows, cols]
        predicted = mx.exp(box_map) * box_stride
        lr = mx.minimum(predicted[..., 0], y_box[..., 0]) + mx.minimum(predicted[..., 2], y_box[..., 2])
        tb = mx.minimum(predicted[..., 1], y_box[..., 1]) + mx.minimum(predicted[..., 3], y_box[..., 3])
        intersection = mx.maximum(lr, 0.0) * mx.maximum(tb, 0.0)
        area_p = (predicted[..., 0] + predicted[..., 2]) * (predicted[..., 1] + predicted[..., 3])
        area_t = (y_box[..., 0] + y_box[..., 2]) * (y_box[..., 1] + y_box[..., 3])
        iou = intersection / mx.maximum(area_p + area_t - intersection, 1e-9)
        loss_box = ((1.0 - iou) * fg).sum(axis=(1, 2)) / num_fg

        # Centerness BCE on the positive cells.
        ctr_logit = ctr_map[..., 0]
        ctr_fg = (y_ctr > 0.0).astype(ctr_logit.dtype)
        loss_ctr = (_sigmoid_bce(ctr_logit, y_ctr) * ctr_fg).sum(axis=(1, 2)) / num_fg

        return (loss_cls + loss_box + loss_ctr).mean()

    return compute


@configurable(task="detection", role="trainer", framework="mlx")
class MlxDetector(DetectionBase, MlxLoopRunnable):
    """Train, evaluate or predict the shared FCOS detection spec with Apple's MLX.

    Args:
        model: The spec model — wire ``!lazy:raidar.fcos.mlx_fcos.MlxFcos`` so ``initialize()``
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
        learning_rate: Learning rate for the optimizer.
        optimizer: An ``mlx.optimizers`` class NAME (``"adam"`` / ``"adamw"`` / ``"sgd"``) or the
            class itself.
        metric_names: How recorded metrics are named. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing.

    Post-construction slots (YAML / broadcasting): ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
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
        optimizer: MlxOptimizer = "adam",
        metric_names: Optional[Lazy[MetricNames]] = None,
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
        # `flow()` on a LIVE object is a pass-through — it does not call the lazy-build hook the
        # `!lazy:` path fires. Unbuilt, the wrapper's parameter tree is EMPTY, and
        # `nn.value_and_grad` fails with "[grad] Must specify at least one argument".
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
        """One collated record batch → flat numpy ``(x, cls, box, ctr)`` for the step.

        The image comes through the shared ``ToTensor`` chain as ``[N, C, H, W]``; MLX is
        channels-last native, so the one transpose lives here — unified memory means the arrays
        then reach the GPU with no transfer at all.
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

    # ---- the step (4-arity — the spec's three losses summed inside) --------------------------- #
    def _loss_fn(self) -> Callable[..., Any]:
        """The spec's summed loss over ``(maps, cls, box, ctr)`` — there is no ``loss:`` slot."""
        return _spec_loss(float(getattr(self.model, "spec_stride", spec.stride())))

    def _make_step(self) -> Callable[..., Any]:
        """The 4-arity objective, bare-loss return — the driver splats and unpacks either shape."""
        loss_fn = self._loss_fn()

        def compute(model: Any, x: Any, y_cls: Any, y_box: Any, y_ctr: Any) -> Any:
            return loss_fn(model(x), y_cls, y_box, y_ctr)

        return cast(Callable[..., Any], nn.value_and_grad(cast(nn.Module, self.model), compute))

    def _eval_batch(self, loss_fn: Callable[..., Any], network: Any, arrays: Tuple[Any, ...]) -> Any:
        x, y_cls, y_box, y_ctr = arrays
        outputs = network(x)
        return loss_fn(outputs, y_cls, y_box, y_ctr), outputs

    # ---- the spec's eval path ------------------------------------------------------------------ #
    def _fit_scores(self, logs: Dict[str, float]) -> Dict[str, float]:
        """Append the spec's numpy AP over the validation split, once after the fit."""
        if self.val_set is not None:
            logs.update({f"val/{k}": v for k, v in self._score_ap(self.val_set).items()})
        return logs

    def _evaluate_split(self, source: Any, split: Literal["val", "test"]) -> Dict[str, float]:
        """Score a split with the spec's AP — neither engine loss pass runs, because a loss
        over dense maps is not a detection score."""
        return as_metric_names(self.metric_names)(self._score_ap(source), split=split)

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
        network = cast(nn.Module, self.model)
        network.train(False)
        cls_map, box_map, ctr_map = network(mx.array(x))
        mx.eval(cls_map, box_map, ctr_map)
        cls_np, box_np, ctr_np = np.array(cls_map), np.array(box_map), np.array(ctr_map)
        config = self._decode_config()
        return [spec.decode_predictions(cls_np[i], box_np[i], ctr_np[i][..., 0], **config) for i in range(len(cls_np))]

    def _score_ap(self, source: Any) -> Dict[str, float]:
        """Decode a split and score it with the spec's AP50 — the numpy scoring path."""
        sequence = self._sequence(source, with_targets=False)
        predictions: List[Dict[str, np.ndarray]] = []
        targets: List[Dict[str, np.ndarray]] = []
        for index, batch in enumerate(sequence.batches()):
            predictions.extend(self._decoded_batch(sequence[index][0]))
            targets.extend(batch_regions(batch, self.target_key))
        return spec.average_precision_50(predictions, targets)

    # ---- predictions --------------------------------------------------------------------------- #
    def _predict_batch(self, x: Any) -> Any:
        """Per-image DECODED detections — the spec's decode IS the task's prediction contract,
        and a LIST is already per-record, so the machinery writes it as-is."""
        return self._decoded_batch(np.asarray(x))
