"""LightningDetector: a standard PyTorch-Lightning object detector (train + eval in one class).

The Lightning member of the detection fleet, assembled from the workspace trainer bases
(2026-08-06): the Lightning machinery — loaders, the deferred ``pl.Trainer`` slot, tracking —
is ``matrainer.lightning.LightningRunnable``, and the detection task — ``num_classes``
resolution, the MAP default + its ``classes``-key strip, per-class reporting — is
:mod:`raidar.detection_base` + :mod:`raidar.detection_torch`. What remains HERE is the config
surface (the full flat constructor — confluid introspection reads THIS signature) and the ONE
thing detection genuinely does differently: the native torchvision step functions
(``forward(images, targets) → loss dict`` in train, per-image ``{boxes, scores, labels}`` in
eval), which OVERRIDE the base's ``batch_tensor``-shaped steps and read the batch through
``native_batch`` instead.

Minimal flat YAML (top-level keys wired by Confluid broadcasting)::

    runnable: !class:raidar.LightningDetector
    model: !lazy:torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
      weights_backbone: DEFAULT
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:detection_ops }
    val_set:   !class:recordstream.Stream { source: ..., ops: !ref:detection_ops }
    batch_size: 2
    max_epochs: 10

    raidar train train.yaml        # or: recordstream run train.yaml
"""

from typing import Any, Dict, Iterable, List, Optional, Union, cast

import torch.nn as nn
from confluid import Mandatory, Partial, configurable
from loggair import get_logger
from matrainer.lightning import LightningRunnable
from matrainer.metric_names import MetricNames
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, collate_list
from recordstream.runnable import RunnableTask
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from raidar.detection_base import _model_num_classes  # noqa: F401  — re-exported (tests + consumers)
from raidar.detection_torch import DetectionTorchHooks
from raidar.torch_detection import native_batch

logger = get_logger(__name__)

#: Which entry point :meth:`LightningDetector.run` dispatches to (the standard four).
DetectorTask = RunnableTask


@configurable(task="detection", role="trainer", framework="torch")
class LightningDetector(DetectionTorchHooks, LightningRunnable):
    """A standard Lightning object detector — train, evaluate, test, or predict via ``task``.

    Args:
        model: Torchvision-native detection model (loss dict in train, per-image predictions in
            eval). Wire ``!lazy:`` to have :meth:`initialize` flow it with the resolved
            ``num_classes``.
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``; also tested
            after ``fit`` when set).
        num_classes: Output class count (torchvision-style incl. background). Omit to resolve from
            the model head or the train targets.
        task: Entry point :meth:`run` dispatches to.
        max_epochs / batch_size / num_workers: ergonomic flat-config knobs.
        persistent_workers: Override the pairing derived from ``num_workers`` (``None`` derives
            ``num_workers != 0``). An explicit value is for a HOST fact no derivation can see —
            macOS terminates persistent workers slowly, so a config there sets ``false`` while
            keeping its workers. ``True`` with ``num_workers=0`` is refused.
        trackers: matrainer trackers — one or a list. They ARE this run's metric sinks: each is
            adapted to Lightning's logger interface, so there is no separate ``logger:`` slot of
            ``pytorch_lightning.loggers``.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the detection target ``Boxes`` (default ``"target"``).
        metrics: Single-metric templates flowed per EVAL split (default ``MeanAveragePrecision``).
            There is deliberately NO ``train_metrics`` slot: a torchvision-native model in train
            mode returns ONLY the loss dict — there are no train-split predictions to score, so a
            slot would accept a wired set and silently never call it.
        val_metrics / test_metrics: That split's OWN templates, merged over ``metrics`` — how a
            metric is added to one split only (e.g. a ``DetectionConfusionMatrix`` on validation).
            Each slot holds the spec until :meth:`initialize` builds it, and the live prefixed
            ``MetricCollection`` after.
        class_names: Labels for the per-class metric report, in class-id order (used when a
            ``MeanAveragePrecision(class_metrics=True)`` emits per-class vectors). Omitted, the
            rows are labelled by the OBSERVED class ids' positions, which is readable but not
            informative. There is no label map to derive them from: a detection target carries
            ids, not names.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart,
            not only as a table. Off by default: it makes matplotlib (``matrainer[plot]``) a cost
            of every eval epoch, where the table costs nothing.
        seed: Seed for torch's global RNG, applied when ``initialize()`` builds the model —
            the torchvision builders randomly reinitialize their ``num_classes`` head, and no
            framework covers that draw. Applied without restoring the RNG state (the
            workspace wrapper-seeding idiom), so the loader shuffle is seeded too.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``self.log`` calls already write
            the canonical ``<split>/<name>``, so the default translates nothing; it is handed to
            each ``TrackerLogger``, where a wired ``target:`` records the same run in another
            engine's spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the metric slots resolve to detection's COCO-mAP default in ``initialize()``;
    there is deliberately no ``loss_fn`` — a native model computes its loss inside ``forward``),
    so nothing here NEEDS wiring; a same-named top-level YAML key replaces one via broadcasting
    when you want to. ``val_set`` is the one a config typically does wire: ``val_set``,
    ``optimizer``, ``train/val/test_loader``, ``lightning``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
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
        persistent_workers: Optional[bool] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
        seed: int = 0,
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
            # The batch SHAPE is DECLARED, not inherited from whether `ToTensor` ran: the native
            # contract takes `List[Tensor]`, so this asks recordstream for the unstacked batch.
            # Override `collate_fn` in the config (per loader slot) for a model wanting one tensor.
            collate_fn=collate_list,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self.seed = int(seed)

    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:  # type: ignore[override]
        return cast(List[Dict[str, Tensor]], cast(nn.Module, self.model)(images))

    # ---- detection step functions (OVERRIDING the base's batch_tensor-shaped ones) ---------- #
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        images, targets = native_batch(batch, self.input_key, self.target_key)
        if targets is None:
            raise ValueError("LightningDetector.training_step requires targets on the batch.")
        loss_dict = cast(nn.Module, self.model)(images, targets)
        if not isinstance(loss_dict, dict):
            raise TypeError(
                f"Detection model in train mode must return Dict[str, Tensor] (a loss dict); "
                f"got {type(loss_dict).__name__}."
            )
        total: Tensor = sum(loss_dict.values())  # type: ignore[assignment]
        bs = len(images)
        # TWO calls, one series each: Lightning appends `_step` / `_epoch` when ONE name carries
        # both, and `train/loss_epoch` is a name no other backend writes. The per-step series is
        # kept for the progress bar and to show the loss moving WITHIN an epoch.
        self.log("train/loss", total, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss_step", total, on_step=True, on_epoch=False, prog_bar=True, batch_size=bs)
        if not (len(loss_dict) == 1 and "loss" in loss_dict):
            for k, v in loss_dict.items():
                if isinstance(v, Tensor) and v.numel() == 1:
                    self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        images, targets = native_batch(batch, self.input_key, self.target_key)
        predictions = cast(nn.Module, self.model)(images)
        if isinstance(self.val_metrics, (Metric, MetricCollection)) and targets is not None:
            self.val_metrics.update(predictions, targets)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        images, targets = native_batch(batch, self.input_key, self.target_key)
        predictions = cast(nn.Module, self.model)(images)
        if isinstance(self.test_metrics, (Metric, MetricCollection)) and targets is not None:
            self.test_metrics.update(predictions, targets)
        if self.predictions_sink is not None:
            self._write_predictions(predictions, batch)
        return predictions

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        images, _ = native_batch(batch, self.input_key, self.target_key)
        predictions = cast(nn.Module, self.model)(images)
        if self.predictions_sink is not None:
            self._write_predictions(predictions, batch)
        return predictions


__all__ = ["DetectorTask", "LightningDetector", "MetricsSpec", "_model_num_classes"]
