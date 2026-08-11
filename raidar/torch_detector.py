"""``TorchDetector`` — the same detection run as a PLAIN PyTorch training loop.

The control-experiment member of the detection fleet, assembled from the workspace trainer
bases (2026-08-06): the hand-written loop machinery — device bracketing, the epoch driver,
scoring, checkpointing, the plain-function tracking hookup — is
``matrainer.torch.TorchLoopRunnable``, and the detection task is the SAME
:mod:`raidar.detection_base` + :mod:`raidar.detection_torch` pair the Lightning backend runs
on — which is the control experiment executed structurally: the two backends now share the
task by construction, so "swap the runnable, keep the config" cannot drift. What remains HERE
is the config surface (the full flat constructor — the confluid introspection reads THIS
signature) plus the NATIVE batch hooks: the torchvision contract computes its loss INSIDE the
train-mode forward and returns per-image predictions in eval mode, so this module overrides
the loop base's three batch hooks (``_train_batch_loss`` / ``_eval_batch`` /
``_predict_batch``) around ``native_batch`` — exactly as the Lightning concrete overrides its
four step functions.

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:raidar.TorchDetector
    model: !lazy:torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:detection_ops }
    num_classes: 6
    max_epochs: 5

Run it with ``raidar train cfg.yaml --framework torch``.
"""

from typing import Any, Iterable, List, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import Lazy, Mandatory, configurable
from loggair import get_logger
from matrainer.metric_names import MetricNames
from matrainer.torch import Device, TorchLoopRunnable
from matrainer.torch.metrics import MetricsSpec
from matrainer.tracking.base import Tracker
from recordstream import RecordSource, collate_list
from recordstream.items import Record
from recordstream.runnable import RunnableTask
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from raidar.detection_torch import DetectionTorchHooks
from raidar.torch_detection import native_batch

logger = get_logger(__name__)

#: Which entry point :meth:`TorchDetector.run` dispatches to (the standard four).
DetectorTask = RunnableTask

__all__ = ["Device", "DetectorTask", "MetricsSpec", "TorchDetector"]


@configurable(task="detection", role="trainer", framework="torch")
class TorchDetector(DetectionTorchHooks, TorchLoopRunnable):
    """Train, evaluate or predict an object detector with a hand-written PyTorch loop.

    Args:
        model: Torchvision-native detection model (loss dict in train, per-image predictions in
            eval) — wire ``!lazy:`` so :meth:`initialize` can inject the resolved ``num_classes``.
        train_set: Training source (``task="fit"``).
        test_set: Held-out source (``task="test"`` / ``"predict"``; also tested after a fit).
        num_classes: Class count (torchvision-style incl. background). Omitted, it is derived
            from the model head or the train targets.
        task: Which entry point ``run()`` dispatches to.
        max_epochs: Training epochs.
        batch_size: Rows per batch.
        num_workers: DataLoader worker processes. Persistent workers follow from this, exactly as
            in the Lightning backend — there is no separate knob.
        experiment_name: Run identity, recorded in artifacts and tracking.
        run_name: Run identity; a random ``adjective_noun`` when omitted.
        input_key: Record key holding the image.
        target_key: Record key holding the detection target ``Boxes``.
        metrics: torchmetrics templates, flowed per EVAL split. ``None`` picks COCO mAP. There is
            deliberately NO ``train_metrics`` slot: a native model in train mode returns ONLY the
            loss dict — no train-split predictions exist to score (wiring one raises loudly, via
            the task base's guarding property).
        val_metrics / test_metrics: That split's OWN templates, merged over ``metrics``.
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``). ``None`` writes no
            weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        trackers: matrainer trackers — one or a list.
        class_names: Labels for the per-class metric report, in class-id order; indices otherwise.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart.
        seed: Seed for torch's global RNG, applied when `initialize()` builds the model —
            the torchvision builders randomly reinitialize their `num_classes` head, and no
            framework covers that draw. Applied without restoring the RNG state (the
            workspace wrapper-seeding idiom), so the loader shuffle is seeded too.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing; wiring ``target:`` is what
            makes it record in another engine's spelling.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ``metrics`` sentinel resolves to COCO mAP in ``initialize()``),
    so nothing here NEEDS wiring; a same-named top-level YAML key replaces one via broadcasting
    when you want to: ``val_set``, ``optimizer``, ``train/val/test_loader``,
    ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Optional[Mandatory[Lazy[nn.Module]]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        num_workers: int = 0,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        device: Device = "auto",
        checkpoint_dir: Optional[str] = None,
        grad_clip: Optional[float] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
        seed: int = 0,
        metric_names: Optional[Lazy[MetricNames]] = None,
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
            device=device,
            checkpoint_dir=checkpoint_dir,
            grad_clip=grad_clip,
            # The batch SHAPE is DECLARED, not inherited from whether `ToTensor` ran: the native
            # contract takes `List[Tensor]`, so this asks recordstream for the unstacked batch.
            # Override `collate_fn` in the config for a model wanting one stacked tensor.
            collate_fn=collate_list,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self.seed = int(seed)

    # ---- the batch boundary --------------------------------------------------------------- #
    def _to_device(self, batch: Any) -> Any:
        """Unwrap the detection batch and move images + targets to this run's device."""
        batch_images, batch_targets = native_batch(batch, self.input_key, self.target_key)
        images = [image.to(self.device) for image in batch_images]
        targets = (
            [{k: v.to(self.device) for k, v in t.items()} for t in batch_targets] if batch_targets is not None else None
        )
        return images, targets

    # ---- the native batch hooks (the loss-inside-forward contract) ------------------------- #
    def _train_batch_loss(self, batch: Record) -> Tensor:
        """ONE training batch → the summed loss dict the train-mode forward returns.

        The native contract computes loss INSIDE the forward, so the loop's job is the sum —
        there is no ``loss_fn`` to call (and no train metrics to update: train mode returns
        only the loss dict).
        """
        images, targets = self._to_device(batch)
        if targets is None:
            raise ValueError(f"{type(self).__name__} training requires targets on the batch.")
        loss_dict = cast(nn.Module, self.model)(images, targets)
        if not isinstance(loss_dict, dict):
            raise TypeError(
                f"Detection model in train mode must return Dict[str, Tensor] (a loss dict); "
                f"got {type(loss_dict).__name__}."
            )
        return cast(Tensor, sum(loss_dict.values()))

    def _eval_batch(self, batch: Record, collection: MetricsSpec) -> Optional[float]:
        """ONE eval batch → ``None``: the native contract returns predictions in eval mode,
        not losses — the split is scored by its metrics alone (the same honesty as the
        Lightning backend's loss-less eval splits)."""
        images, targets = self._to_device(batch)
        predictions = cast(nn.Module, self.model)(images)
        if isinstance(collection, (Metric, MetricCollection)) and targets is not None:
            collection.update(predictions, targets)
        return None

    @torch.no_grad()
    def _predict_batch(self, batch: Record) -> Any:
        """ONE predict batch → the model's own per-image prediction list."""
        images, _ = self._to_device(batch)
        return cast(nn.Module, self.model)(images)
