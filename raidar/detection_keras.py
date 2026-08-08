"""``DetectionKerasHooks`` — the detection task's KERAS-family hook implementations.

The keras twin of :mod:`raidar.detection_torch`: the task hooks written once for BOTH
Keras-family machinery bases — the compile/fit member (:class:`matrainer.keras.KerasRunnable`)
and the native-TensorFlow loop (:class:`matrainer.tensorflow.TensorFlowRunnable`) — so the two
spec-engine concretes cannot drift apart. A concrete composes
``(DetectionKerasHooks, <Machinery>Runnable)``.

Detection is the family's odd member, and every oddity is a task fact:

* **The batch stacks DENSE MAPS, not ragged boxes** — a detection config's ops chain ends in
  :class:`raidar.fcos.targets.FcosTargets`, which encodes each record's boxes through the
  spec's numpy assignment into fixed-shape per-cell maps. :meth:`_to_arrays` only READS them
  (and transposes the image: the chain keeps the shared CHW ``ToTensor`` — the RGB coercion
  lives there — so the engine's channels-last layout is ONE transpose here).
* **There is no ``loss:`` slot** — the spec model and its loss are one design, exactly as an
  HF detector's loss lives inside its forward. :meth:`_compiled_loss` hands ``compile()`` the
  spec's three per-output functions positionally, and the TensorFlow concrete's traced step
  sums the same three (from :mod:`raidar.fcos.keras_fcos` — ``keras.ops``, so ONE loss module
  serves both engines).
* **Detection quality is the SPEC's numpy AP, not compile-slot metrics** — torchmetrics is
  torch-only and Keras ships no variable-size detection metric, so :meth:`_score` decodes the
  raw head maps (:func:`raidar.fcos.spec.decode_predictions`) and scores
  :func:`raidar.fcos.spec.average_precision_50` against the ``Regions`` ground truth
  ``batch_regions`` reads back. The engines differ only in HOW a batch is predicted, which is
  the one seam (:meth:`_batch_predictor`).
* **``evaluate()`` scores the VAL split** (the fleet harmonization) and ``test()`` the
  held-out one — both through :meth:`_score`; neither engine's own eval pass runs, because a
  loss over dense maps is not a detection score.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from confluid import flow
from loggair import get_logger
from matrainer.metric_names import as_metric_names
from recordstream import PredictionsSink, batch_metadata, batch_regions, batch_values
from recordstream.keras import keras

from raidar.detection_base import DetectionBase
from raidar.fcos import spec
from raidar.fcos.keras_fcos import fcos_box_loss, fcos_cls_loss, fcos_ctr_loss

logger = get_logger(__name__)

__all__ = ["DetectionKerasHooks"]


class DetectionKerasHooks(DetectionBase):
    """The keras-typed detection hooks, shared by the Keras and TensorFlow concretes."""

    #: Record keys of the dense maps ``FcosTargets`` wrote — stored by the concrete's ctor.
    cls_key: str
    box_key: str
    ctr_key: str
    #: Applied to a STRING-alias optimizer (an instance keeps its own).
    learning_rate: float
    #: The keras family base's members these hooks read/call — declared for the type checker
    #: (the hooks class is always composed with a `<Machinery>Runnable` that provides them).
    optimizer: Any
    _sequence: Callable[..., Any]

    # ---- the batch boundary ----------------------------------------------------------------- #
    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """Image transposed to channels-last; targets = the three dense maps, NESTED.

        The nested ``(x, (cls, box, ctr))`` is what ``keras.Model.fit`` consumes against a
        three-output model; the TensorFlow concrete overrides with the FLAT tuple its traced
        step's arity wants.
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

    # ---- the compile seams (the compile/fit member's) ---------------------------------------- #
    def _spec_losses(self) -> Any:
        """The spec's three per-output losses, positionally matched to the raw head maps.

        The box loss is a closure (the raw-to-distance scale closes over the model's stride),
        renamed after itself — Keras names history columns from the FUNCTION name, and a bare
        ``loss`` there would log the box component as the ambiguous ``loss_loss``.
        """
        box_loss = fcos_box_loss(float(getattr(self.model, "spec_stride", spec.stride())))
        box_loss.__name__ = "fcos_box_loss"
        return [fcos_cls_loss, box_loss, fcos_ctr_loss]

    def _compiled_loss(self) -> Any:
        return self._spec_losses()

    def _compiled_optimizer(self) -> Any:
        """A string alias gets the task's ``learning_rate``; an instance keeps its own.

        YAML floats like ``1e-3`` may arrive as strings, hence the coercion.
        """
        optimizer = flow(self.optimizer)
        if isinstance(optimizer, str):
            return keras.optimizers.get(
                {"class_name": optimizer, "config": {"learning_rate": float(self.learning_rate)}}
            )
        return optimizer

    # ---- the spec's eval path ---------------------------------------------------------------- #
    def _batch_predictor(self) -> Callable[[np.ndarray], Any]:
        """One callable mapping an image batch to the three raw head maps — THE engine seam.

        The compile/fit member predicts through the compiled model; the TensorFlow concrete
        overrides with its traced predict step (built ONCE — a fresh trace per batch retraces).
        """
        return lambda x: self.model.predict(x, verbose=0)

    def _decode_config(self) -> Dict[str, Any]:
        """The decode knobs, read off the model wrapper (spec defaults for a bare model)."""
        return {
            "input_shape": tuple(getattr(self.model, "input_shape", spec.INPUT_SHAPE)),
            "widths": tuple(getattr(self.model, "widths", spec.BACKBONE_WIDTHS)),
            "score_threshold": float(getattr(self.model, "score_threshold", 0.05)),
            "iou_threshold": float(getattr(self.model, "iou_threshold", 0.5)),
            "max_detections": int(getattr(self.model, "max_detections", 100)),
        }

    def _decoded_maps(self, maps: Any) -> List[Dict[str, np.ndarray]]:
        """Raw head maps → per-image scored boxes, through the spec's numpy decode."""
        cls_map, box_map, ctr_map = maps
        config = self._decode_config()
        return [
            spec.decode_predictions(
                np.asarray(cls_map[i]), np.asarray(box_map[i]), np.asarray(np.asarray(ctr_map[i])[..., 0]), **config
            )
            for i in range(len(cls_map))
        ]

    def _score(self, source: Any) -> Dict[str, float]:
        """Decode a split and score it with the spec's AP50 — the numpy scoring path.

        The ground truth comes from the batched record's ``target_key`` — the plain
        ``{"boxes", "labels"}`` dicts ``FcosTargets`` left beside the dense maps, batched as a
        per-record list and read back by ``batch_regions``.
        """
        predict = self._batch_predictor()
        sequence = self._sequence(source, with_targets=False)
        predictions: List[Dict[str, np.ndarray]] = []
        targets: List[Dict[str, np.ndarray]] = []
        for index, batch in enumerate(sequence.batches()):
            predictions.extend(self._decoded_maps(predict(sequence[index][0])))
            targets.extend(batch_regions(batch, self.target_key))
        return spec.average_precision_50(predictions, targets)

    # ---- the fit epilogue + the eval/predict entry points ------------------------------------ #
    def _fit_scores(self, final: Dict[str, float]) -> Dict[str, float]:
        """Append the spec's AP over the validation split — the engine's own history carries
        losses only (no compile-slot metric can score variable-size detections)."""
        if self.val_set is not None:
            final.update({f"val_{k}": v for k, v in self._score(self.val_set).items()})
        return final

    def evaluate(self) -> None:
        """Score the VALIDATION split with the spec's AP (the fleet harmonization)."""
        if self.val_set is None:
            raise ValueError(f"{type(self).__name__}.evaluate() requires a 'val_set'.")
        self._computed_metrics = as_metric_names(self.metric_names)(
            self._score(self.val_set), source="keras", split="val"
        )
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        self._write_eval_sidecar()

    def test(self) -> None:
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.test() requires a 'test_set'.")
        self._computed_metrics = as_metric_names(self.metric_names)(
            self._score(self.test_set), source="keras", split="test"
        )
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        self._write_eval_sidecar()

    def _write_eval_sidecar(self) -> None:
        from matrainer.artifacts import write_metrics_sidecar

        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def predict(self) -> None:
        """Stream per-image DECODED detections to the sink — the spec's decode IS the task's
        prediction contract, so there is nothing further to wrap."""
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        sink: Optional[PredictionsSink] = flow(self.predictions_sink)
        predict = self._batch_predictor()
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            decoded = self._decoded_maps(predict(sequence[index][0]))
            if sink is not None:
                meta = batch_metadata(batch, exclude=(self.input_key, self.target_key))
                for i, prediction in enumerate(decoded):
                    sink.write(prediction, meta[i] if meta and i < len(meta) else {})
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")
