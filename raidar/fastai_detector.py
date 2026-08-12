"""``FastaiDetector`` — the same detection run with a fastai ``Learner``.

The fastai member of the detection fleet, assembled from the workspace trainer bases
(2026-08-07): the Learner machinery — the ``DataLoaders``/``Learner`` assembly, the Recorder
read-back, the plain-function tracking hookup — is ``matrainer.fastai.FastaiRunnable``, and the
detection task is the SAME :mod:`raidar.detection_base` + :mod:`raidar.detection_torch` pair
every torch-family backend runs on. What remains HERE is the config surface (the full flat
constructor) plus the NO-OP-LOSS BRIDGE the loss-inside-forward contract needs:

* a fastai ``Learner`` computes ``loss_func(model(*xb), *yb)``, but the native detection
  contract computes its loss INSIDE the train-mode forward — so :func:`fastai_detection_collate`
  packs ``((images, targets), dummy_y)`` (both ride in ``xb``),
  :class:`NativeDetectionAdapter`'s training forward returns the SUMMED loss dict as the
  "prediction", and :func:`passthrough_loss` hands it to backprop unchanged — ONE forward, the
  fit loop untouched;
* the eval forward returns the per-image prediction list and the pass-through loss returns a
  ZERO scalar for fastai's bookkeeping — the recorded ``valid_loss`` is MEANINGLESS under this
  contract and is DROPPED from what the run records rather than smoothed over;
* validation quality comes from the SAME torchmetrics collections every other backend builds
  (the family builders are RESTORED here — the base leaves the metric slots as fastai name
  lists, which this contract does not use), fed by :class:`DetectionMetricsCallback` because no
  fastai metric consumes a per-image prediction list.

Run it with ``raidar train cfg.yaml --framework fastai``.
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from fastai.callback.core import Callback as FastaiCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from loggair import get_logger
from matrainer.fastai import FastaiOptimizer, FastaiRunnable
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.torch import Device, resolve_device, scalar_metrics
from matrainer.torch.metrics import MetricsSpec
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, collate_list
from recordstream.items import Record
from recordstream.runnable import RunnableTask
from torchmetrics import Metric, MetricCollection

from raidar.detection_torch import DetectionTorchHooks
from raidar.torch_detection import native_batch

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiDetector.run` dispatches to (the standard four).
DetectorTask = RunnableTask

__all__ = ["DetectorTask", "FastaiDetector", "MetricsSpec", "NativeDetectionAdapter", "passthrough_loss"]


def fastai_detection_collate(
    records: Sequence[Any], input_key: str = "image", target_key: str = "target"
) -> Tuple[Any, torch.Tensor]:
    """The fastai twin of the shared detection collate — ``((images, targets), dummy_y)``.

    fastai unpacks a batch positionally into ``(xb, yb)`` and calls ``model(*xb)``, but the
    native train-mode forward needs the targets IN the call — so BOTH ride in ``xb`` and ``yb``
    is a dummy the pass-through loss ignores. Built ON the shared collate, so the record keys and
    the never-pad-never-drop target rule stay one implementation.
    """
    batch = collate_list(list(records))
    images, targets = native_batch(batch, input_key, target_key)
    return (images, targets), torch.zeros(len(images))


class NativeDetectionAdapter(nn.Module):
    """The no-op-loss bridge: training forward returns the SUMMED loss dict as the "prediction".

    In eval mode it returns the model's per-image prediction list — the same objects the fleet's
    torchmetrics consume — and the pass-through loss then reports a zero ``valid_loss``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, payload: Tuple[Any, Any]) -> Any:
        images, targets = payload
        if self.training:
            if targets is None:
                raise ValueError("FastaiDetector training requires targets on the batch.")
            loss_dict = self.model(images, targets)
            if not isinstance(loss_dict, dict):
                raise TypeError(
                    f"Detection model in train mode must return Dict[str, Tensor] (a loss dict); "
                    f"got {type(loss_dict).__name__}."
                )
            return cast(torch.Tensor, sum(loss_dict.values()))
        return self.model(images)


def passthrough_loss(prediction: Any, *_: Any) -> torch.Tensor:
    """The no-op loss: the model already computed it — pass the scalar through unchanged.

    On the validation pass the "prediction" is the per-image list, for which no loss exists
    under this contract — a ZERO scalar keeps fastai's bookkeeping running, and the recorded
    ``valid_loss`` of 0 is documented as meaningless rather than smoothed over.
    """
    if isinstance(prediction, torch.Tensor) and prediction.ndim == 0:
        return prediction
    return torch.zeros((), requires_grad=False)


class DetectionMetricsCallback(FastaiCallback):
    """Feed the validation pass's predictions to the fleet's torchmetrics collection.

    ``Learner.metrics`` cannot serve here (no fastai metric consumes a per-image prediction
    list), so the SAME ``MetricCollection`` every other backend builds is updated from a
    callback: on each non-training batch the adapter's prediction list and the targets packed in
    ``xb`` are handed to ``update()``. The runnable computes/reports at the end.
    """

    def __init__(self, collection: Optional[MetricCollection]) -> None:
        self.collection = collection

    def after_pred(self) -> None:
        if self.training or self.collection is None:
            return
        predictions = self.learn.pred
        targets = self.learn.xb[0][1]
        if isinstance(predictions, list) and targets is not None:
            self.collection.update(predictions, targets)


@configurable(task="detection", role="trainer", framework="torch")
class FastaiDetector(DetectionTorchHooks, FastaiRunnable):
    """Train, evaluate or predict an object detector with a fastai ``Learner``.

    Args:
        model: Torchvision-native detection model — wire ``!lazy:`` so :meth:`initialize` can
            inject the resolved ``num_classes``. Wrapped in :class:`NativeDetectionAdapter` for
            the Learner (the no-op-loss design; see the module docstring).
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``).
        num_classes: Class count (torchvision-style incl. background). Omitted, derived from the
            model head or the train targets.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size / num_workers: the shared knobs.
        learning_rate: Peak LR — ``lr_max`` for one-cycle, ``lr`` otherwise.
        weight_decay: fastai's ``wd``. ``None`` is fastai's own default.
        one_cycle: Use ``fit_one_cycle`` (fastai's headline scheduler) instead of plain ``fit``.
        opt_func: A ``fastai.optimizer`` function or its name (``"Adam"``, ``"SGD"``, …).
        metrics: torchmetrics templates for the EVAL splits — the fleet vocabulary, fed through
            :class:`DetectionMetricsCallback`. ``val_metrics`` / ``test_metrics`` merge over it.
            No ``train_metrics``: the training pass computes the loss dict and nothing else
            (wiring one raises loudly, via the task base's guarding property).
        device: Where to run — see :data:`Device`. Carried onto the ``DataLoaders`` so fastai's
            own placement callback honours it (see :meth:`_place`).
        class_names: Labels for the per-class metric report, in class-id order.
        plot_metrics: ALSO send each per-class metric to the trackers as a bar chart.
        seed: Seed for torch's global RNG, applied when `initialize()` builds the model —
            the torchvision builders randomly reinitialize their `num_classes` head, and no
            framework covers that draw. Applied without restoring the RNG state (the
            workspace wrapper-seeding idiom), so the loader shuffle is seeded too.
        metric_names: How recorded metrics are named. This backend reports through its OWN
            torchmetrics pass under canonical names; the Recorder's ``train_loss`` is translated
            from fastai's vocabulary.
        callbacks: fastai callbacks.
        trackers: matrainer trackers — one or a list.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the detection target ``Boxes``.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__``, so nothing here NEEDS wiring; a same-named top-level YAML key replaces one
    via broadcasting when you want to: ``val_set``, ``train/val/test_loader``, ``dataloaders``,
    ``learner``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Partial[nn.Module]]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        num_workers: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: Optional[float] = None,
        one_cycle: bool = True,
        opt_func: FastaiOptimizer = "Adam",
        metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        device: Device = "auto",
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
        seed: int = 0,
        metric_names: Optional[Partial[MetricNames]] = None,
        callbacks: Optional[List[Partial[FastaiCallback]]] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            one_cycle=one_cycle,
            opt_func=opt_func,
            callbacks=callbacks,
            # The batch SHAPE is DECLARED: the no-op-loss packing over the shared list collate.
            collate_fn=partial(fastai_detection_collate, input_key=str(input_key), target_key=str(target_key)),
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.device_choice: Device = device
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self.seed = int(seed)

    # ---- the torchmetrics half the base leaves open ----------------------------------------- #
    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """RESTORED family builders: this contract scores through torchmetrics collections
        (fed by :class:`DetectionMetricsCallback`), not through ``Learner.metrics`` names —
        the case the base's no-op documents."""
        TorchTrainerRunnable._build_fit_metrics(self, include_train=False)

    def _build_eval_metrics(self) -> None:
        TorchTrainerRunnable._build_eval_metrics(self)

    def _active_collection(self, split: str) -> Optional[MetricCollection]:
        if split == "val" and self.task == "fit":
            return cast(Optional[MetricCollection], self.val_metrics if self.val_set is not None else None)
        return cast(Optional[MetricCollection], getattr(self, f"{split}_metrics"))

    def _reduce_collection(self, collection: Optional[MetricCollection]) -> Dict[str, float]:
        """Compute + report + RESET the collection — scalars returned, vectors through the
        task hooks to the text channel (the same reporting every other backend does)."""
        if not isinstance(collection, (Metric, MetricCollection)):
            return {}
        computed = self._reportable(collection.compute())
        self._report_nonscalars(computed, None, self._live_trackers())
        values = scalar_metrics(computed)
        collection.reset()
        return values

    # ---- the no-op-loss bridge, hung on the base's Learner seams ---------------------------- #
    def _learner_model(self) -> nn.Module:
        inner = super()._learner_model()  # the nn.Module validation
        self._device = resolve_device(self.device_choice)
        inner.to(self._device)
        return NativeDetectionAdapter(inner)

    def _learner_loss(self) -> Any:
        return passthrough_loss

    def _learner_metric_objects(self, split: str) -> List[Any]:
        # `Learner.metrics` stays EMPTY — no fastai metric consumes a per-image prediction
        # list; the torchmetrics collection is fed from the callback below instead.
        return []

    def _learner_callbacks(self, split: str) -> List[Any]:
        return [DetectionMetricsCallback(self._active_collection(split))] + self._callbacks()

    def _place(self, dataloaders: DataLoaders) -> DataLoaders:
        """Carry this run's device onto the ``DataLoaders`` so fastai's own callback honours it.

        ``TrainEvalCallback.before_fit`` runs ``self.model.to(getattr(dls, 'device',
        default_device()))`` — and over PLAIN torch loaders ``dls.device`` raises (fastcore's
        attribute delegation reaches the first loader, which has no ``device``), so the fallback
        is ``default_device()``: on Apple silicon the cpu model was silently moved to MPS and the
        measured step went 1.0 s → **32.9 s** (torchvision detection's ~30x MPS fallback, hit for
        real on the first cppe-5 run). The PUBLIC setter cannot carry the pin — it recurses
        ``dl.to(d)`` into the plain loaders and raises — so the private ``_device`` field the
        getter reads is the working route (the workspace's known-private-API precedent: used
        knowingly because the public alternative is broken, with the measurement recorded).
        """
        self._device = resolve_device(self.device_choice)
        dataloaders._device = self._device
        return dataloaders

    # ---- what a run RECORDS (the Recorder holds only train_loss here) ----------------------- #
    def _fit_results(self, learner: Learner, names: MetricNames) -> Dict[str, float]:
        logs = _recorded_train_loss(learner, names)
        logs.update(names(self._reduce_collection(self._active_collection("val"))))
        return logs

    def _log_epoch_history(self, learner: Learner, trackers: Sequence[Tracker], names: MetricNames) -> None:
        """Every epoch's ``train_loss`` — the only Recorder column that means anything here —
        plus the final torchmetrics scores, which live outside the Recorder entirely."""
        for epoch, row in enumerate(list(getattr(learner.recorder, "values", []) or [])):
            metrics = names({"train_loss": float(row[0])}, source="fastai")
            for tracker in trackers:
                tracker.log_metrics(metrics, step=epoch)
        for tracker in trackers:
            tracker.log_metrics(self._computed_metrics)

    def _validate_results(self, learner: Learner, values: Sequence[Any], split: str) -> Dict[str, float]:
        """The torchmetrics scores — the Learner's own row is a meaningless zero loss."""
        results = as_metric_names(self.metric_names)(self._reduce_collection(self._active_collection(split)))
        for tracker in self._live_trackers():
            tracker.log_metrics(results)
        return results

    # ---- predict: a plain forward over the record-shaped batch ------------------------------ #
    def _record_collate(self) -> Callable[[List[Record]], Any]:
        return collate_list

    @torch.no_grad()
    def _predict_batch(self, batch: Record, device: torch.device) -> Any:
        images, _ = native_batch(batch, self.input_key, self.target_key)
        return cast(nn.Module, self.model)([image.to(device) for image in images])


def _recorded_train_loss(learner: Learner, names: MetricNames) -> Dict[str, float]:
    """The last epoch's ``train_loss`` in canonical spelling. ``valid_loss`` is DROPPED — under
    the no-op-loss contract it is a recorded zero, and a meaningless number under a real-looking
    name is worse than a gap."""
    rows = list(getattr(learner.recorder, "values", []) or [])
    if not rows:
        return {}
    return names({"train_loss": float(rows[-1][0])}, source="fastai")
