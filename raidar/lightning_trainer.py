"""LightningDetectionTrainer: specialised object-detection trainer.

Subclass of :class:`~marainer.trainer.LightningTrainer` that **overrides** the
step functions so it consumes torchvision-native detection models and raw
``torchmetrics`` detection metrics (``MeanAveragePrecision`` and friends)
directly. There is no adapter / indirection layer — detection runs exclusively
through this specialised trainer (the marainer-generic ``DetectionStepOutput``
+ ``TorchvisionDetectionLoss`` + ``DetectionMetricAdapter`` path was removed).

Model contract (the same shape torchvision detection models already speak):

* ``model.train(); model(images, targets) → Dict[str, Tensor]`` — loss dict.
  The trainer sums it to get the optimisation scalar.
* ``model.eval();  model(images)         → List[Dict[str, Tensor]]`` — per-image
  ``{boxes, scores, labels}`` in xyxy-pixel space.

Torchvision Faster R-CNN already conforms natively; pass it raw without a
raidar wrapper. RF-DETR ships a native module at
:class:`raidar.rfdetr.RFDETRModel`. The ultralytics-family detectors ship
native modules in the sibling ``raidar-ultralytics`` package, at
:class:`raidar_ultralytics.yolo26.YOLO26Model` and
:class:`raidar_ultralytics.rtdetr.RTDETRModel`.

Metric contract: any ``torchmetrics.Metric`` or
``torchmetrics.MetricCollection`` whose ``update(preds, targets)`` consumes
the per-image prediction list directly. Default:
``MeanAveragePrecision(iou_type="bbox")``. Use a ``MetricCollection`` to
combine MAP with e.g. :class:`~raidar.detection.DetectionConfusionMatrix`.

Minimal YAML (native Faster R-CNN — no raidar model wrapper)::

    trainer: !class:raidar.lightning_trainer.LightningDetectionTrainer
      num_classes: !ref:num_classes
      model: !class:torchvision.models.detection.fasterrcnn_resnet50_fpn
        num_classes: !ref:num_classes
        weights_backbone: DEFAULT
      train_set: !ref:train_set
      val_set: !ref:val_set

Custom metric collection::

    val_metrics: !class:torchmetrics.MetricCollection
      metrics:
        - !class:torchmetrics.detection.MeanAveragePrecision
          iou_type: bbox
        - !class:raidar.detection.DetectionConfusionMatrix
          iou_threshold: 0.5
"""

from typing import Any, Dict, Iterable, List, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from marainer.trainer import LightningTrainer, ensure_sample_dataset
from pytorch_lightning.loggers import Logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from raidar.detection import detection_collate_fn

logger = get_logger(__name__)


def _log_scalar_dict(module: L.LightningModule, metric: Any, prefix: str) -> None:
    """Compute *metric* and log every 0-dim Tensor value under *prefix*.

    Torchmetrics detection metrics typically return a dict (e.g. ``MAP``'s
    ``{map, map_50, map_75, ...}``) — we log every scalar entry. Non-scalar
    entries (``classes``, ``map_per_class`` when ``class_metrics=True``) are
    skipped because Lightning's ``self.log`` only accepts scalars.
    """
    result = metric.compute()
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, Tensor) and v.ndim == 0 and v.is_floating_point():
                module.log(f"{prefix}{k}", v)
    elif isinstance(result, Tensor) and result.ndim == 0 and result.is_floating_point():
        name = type(metric).__name__
        module.log(f"{prefix}{name}", result)


def _model_num_classes(model: nn.Module) -> Optional[int]:
    """Best-effort read of a BUILT detector's class count, or ``None``.

    A concrete model's head is authoritative (the count MUST come from it, not a
    dataset walk that could mis-size against an already-built head). Tries, in
    order: an explicit ``num_classes`` attribute (raidar's ``RFDETRModel`` /
    raidar-ultralytics' ``YOLO26Model`` / ``RTDETRModel`` wrappers expose it),
    then torchvision's two-stage box-predictor head
    (``roi_heads.box_predictor.cls_score.out_features`` — Faster R-CNN). Returns
    ``None`` when the architecture exposes neither (the count then comes from the
    config or a target walk).
    """
    nc = getattr(model, "num_classes", None)
    if nc:
        return int(nc)
    box_predictor = getattr(getattr(model, "roi_heads", None), "box_predictor", None)
    cls_score = getattr(box_predictor, "cls_score", None)
    out_features = getattr(cls_score, "out_features", None)
    if out_features:
        return int(out_features)
    return None


@configurable(task="detection", role="trainer")
class LightningDetectionTrainer(LightningTrainer):
    """Specialised object-detection trainer with overridden step functions.

    Mirrors :class:`~marainer.trainer.LightningTrainer`'s constructor discipline
    (confluid "Lazy Initialization & Zero-Arg Construction"): the constructor
    takes only ``model`` / ``train_set`` (+ identity ``num_classes`` /
    ``experiment_name`` / ``run_name``) and does **no functional work** — every
    other slot is a defaulted attribute set in the body and configured
    post-construction (by YAML, or by Confluid broadcasting of flat top-level
    keys). Detection baked-in attributes (all overridable post-construction):

    * ``loss_fn`` — ``None``; native detection models compute loss inside
      ``forward`` (no separate loss callable).
    * ``optimizer`` — deferred ``AdamW(lr=1e-4, weight_decay=0.05)``.
    * ``train/val/test_loader`` — deferred ``DataLoader`` with
      :func:`~raidar.detection.detection_collate_fn`.
    * ``val_metrics`` / ``test_metrics`` — deferred
      ``MeanAveragePrecision(iou_type="bbox")`` (held separately so val and test
      accumulators don't share state); ``train_metrics`` is unset because most
      detection models return a loss dict (not predictions) in ``training_step``.

    **num_classes resolution (``initialize()``, mirroring sonair).** Like
    :class:`sonair.LightningClassificationTrainer`, this trainer resolves
    ``num_classes`` at run time in an explicit :meth:`initialize` (called at the
    top of :meth:`run`, BEFORE ``super().run()``) and flows the model with it —
    so a ``!lazy:`` model can omit ``num_classes`` and have it derived from the
    data. Order: a **concrete (non-``Fluid``) model is authoritative**
    (:func:`_model_num_classes` — an explicit ``num_classes`` attr, or
    torchvision's box-predictor head) → explicit trainer arg → a lazy model's
    pinned ``num_classes`` kwarg → derived from the train targets
    (``max(label) + 1``, the last resort — walks the set, logs progress). The
    resolved count is then injected into a deferred model via
    ``flow(self.model, num_classes=…)``. The default MAP metric is
    class-count-agnostic, so ``num_classes`` only sizes the model head.

    **Ergonomic flat-config knobs** — ``max_epochs`` / ``batch_size`` /
    ``num_workers`` / ``logger`` are constructor params, so Confluid broadcasting
    injects matching flat top-level YAML keys. ``batch_size`` / ``num_workers``
    are baked into the deferred ``DataLoader`` stubs in ``__init__``;
    ``max_epochs`` / ``logger`` are forwarded to the base, which bakes them into
    ``LazyClass(pytorch_lightning.Trainer, max_epochs=…, logger=…)``. An explicit
    ``train_loader`` / ``val_loader`` / ``lightning`` overrides the knob-built default.

    Args:
        model: Torchvision-native detection model emitting a loss dict in train
            mode and a per-image prediction list in eval. Wire it ``!lazy:`` to
            have :meth:`initialize` flow it with the resolved ``num_classes`` —
            e.g. ``!lazy:torchvision.models.detection.fasterrcnn_resnet50_fpn``
            (raw torchvision, no raidar wrapper), or the native modules
            :class:`raidar.rfdetr.RFDETRModel` /
            :class:`raidar_ultralytics.yolo26.YOLO26Model`.
        train_set: Training dataset or Flux pipeline. Each sample carries
            ``input=[3,H,W]`` image, ``target={"boxes":[N,4], "labels":[N]}``.
        num_classes: Output class count (torchvision-style: foreground + 1 for
            background). Omit to resolve it from the model head or
            ``max(label)+1`` over the train targets in :meth:`initialize`.
        max_epochs: ``pytorch_lightning.Trainer(max_epochs=...)`` (forwarded to the base).
        batch_size: DataLoader batch size baked into the default loaders in ``__init__``.
        num_workers: DataLoader worker-process count baked into the default loaders.
        logger: Optional list of Lightning loggers (``List``-typed so a flat
            ``logger: [...]`` broadcasts in); forwarded to the base Trainer.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name. ``None`` (default) auto-generates a random slug
            like ``train_funny_rabbit`` (via the base; see
            :func:`marainer.run_names.generate_run_name`).
    """

    def __init__(
        self,
        model: Mandatory[Union[nn.Module, Fluid]],
        train_set: Mandatory[Union[Dataset[Any], Fluid]],
        num_classes: Optional[int] = None,
        max_epochs: int = 10,
        batch_size: int = 2,
        num_workers: int = 0,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            experiment_name=experiment_name,
            run_name=run_name,
            logger=logger,
            max_epochs=max_epochs,
        )
        # Store config ONLY — no functional work. The loaders/optimizer are deferred
        # (LazyClass): the loaders bake in the ergonomic batch_size/num_workers knobs
        # here (cheap — LazyClass defers construction), and the base run() flows them
        # with the dataset (the optimizer with model params). max_epochs / logger are
        # forwarded to the base (baked into LazyClass(L.Trainer, ...)). The model is
        # flowed with the resolved num_classes in initialize() (run time).
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        # Native detection models compute loss inside forward — no separate
        # loss_fn callable is involved. Set to None to make that explicit (and
        # to detach the CrossEntropyLoss default the base trainer sets).
        self.loss_fn = None  # type: ignore[assignment]
        self.optimizer = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        self.train_loader = LazyClass(
            DataLoader,
            collate_fn=detection_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            shuffle=True,
        )
        self.val_loader = LazyClass(
            DataLoader,
            collate_fn=detection_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            shuffle=False,
        )
        self.test_loader = LazyClass(
            DataLoader,
            collate_fn=detection_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
            shuffle=False,
        )
        # Predictions are unavailable during training_step for most detection
        # models (the model returns a loss dict, not predictions). Leave train
        # metrics unset; val/test get a fresh deferred MAP each.
        self.train_metrics = None
        self.val_metrics = LazyClass(MeanAveragePrecision, iou_type="bbox")
        self.test_metrics = LazyClass(MeanAveragePrecision, iou_type="bbox")

    def run(self) -> None:
        """Resolve num_classes + flow the model, then run the base training loop."""
        self.initialize()
        super().run()

    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained detection model (a live ``nn.Module``) produced by :meth:`run`.

        The trainer's declared confluid ``@output`` (the I/O contract FluxStudio /
        navigaitor read via :func:`confluid.output_specs`): the post-run artifact is
        the trained model, wired downstream into an evaluator's ``model`` slot. It is
        the live ``self.model`` that :meth:`initialize` flowed and the Lightning ``fit``
        loop trained in place — NOT the ``model`` *input* (the ``!lazy:`` arch spec).
        Accessing it before :meth:`run` raises, because the model is still a deferred
        :class:`~confluid.fluid.Fluid`: it is a run OUTPUT, not a config knob.
        """
        if isinstance(self.model, Fluid):
            raise RuntimeError(
                "LightningDetectionTrainer.trained_model is only available after run(): "
                "the model is still a deferred Fluid. Wire this output downstream of a Run node."
            )
        return cast(nn.Module, self.model)

    def initialize(self) -> None:
        """Resolve ``num_classes`` and flow the model with it (mirrors sonair).

        raidar OWNS this (no generic dimension-injection layer). Called explicitly
        from :meth:`run` (NOT confluid's auto ``solidify`` hook) so the optional
        target walk + model build happen at run time, not when a config is merely
        materialized. Idempotent: re-running reuses the resolved count and the
        already-live model.

        Steps: normalize the wired sources to ``Dataset[Sample]`` (the marainer
        ``ensure_sample_dataset`` helper, idempotent on a ``Flux``); resolve
        ``num_classes`` (see :meth:`_resolve_num_classes`); then, when the model is
        still deferred, ``flow`` it injecting the resolved count
        (``flow(self.model, num_classes=…)`` — confluid passes the kwarg to the
        builder, e.g. ``torchvision.models.detection.fasterrcnn_resnet50_fpn``). A
        concrete model is left untouched (the base ``run()``'s ``flow`` is then
        idempotent on it).
        """
        self.train_set = ensure_sample_dataset(flow(self.train_set))
        if self.val_set is not None:
            self.val_set = ensure_sample_dataset(flow(self.val_set))
        if self.test_set is not None:
            self.test_set = ensure_sample_dataset(flow(self.test_set))

        nc = self._resolve_num_classes()
        if nc is not None:
            self.num_classes = int(nc)

        # Inject the resolved count into a STILL-DEFERRED model. A model that
        # doesn't take a num_classes kwarg is flowed plainly. A concrete model is
        # left as-is (the base run()'s flow() is idempotent on a live instance).
        if isinstance(self.model, Fluid):
            if self.num_classes is not None and self._model_accepts_num_classes():
                self.model = flow(self.model, num_classes=self.num_classes)
            else:
                self.model = flow(self.model)

    def _model_accepts_num_classes(self) -> bool:
        """True when the deferred model's target callable declares a ``num_classes`` param.

        Guards the ``flow(self.model, num_classes=…)`` injection so a model whose
        builder doesn't take ``num_classes`` (it sizes its head another way) is flowed
        plainly instead of erroring on an unexpected kwarg.
        """
        import inspect

        from confluid.registry import resolve_class

        if not isinstance(self.model, Fluid):
            return False
        target = self.model.target
        resolved = resolve_class(target) if isinstance(target, str) else target
        if resolved is None or not callable(resolved):
            return False
        try:
            sig = inspect.signature(resolved)
        except (ValueError, TypeError):
            return False
        params = sig.parameters
        return "num_classes" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _resolve_num_classes(self) -> Optional[int]:
        """Resolve the detector's class count; ``None`` when undeterminable.

        Order: a concrete model's head (:func:`_model_num_classes`) is authoritative;
        then an explicit trainer arg; then a deferred model's pinned ``num_classes``
        kwarg; then a target walk (``max(label)+1``).
        """
        if not isinstance(self.model, Fluid):
            model_nc = _model_num_classes(cast(nn.Module, self.model))
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != model_nc:
                    logger.warning(
                        f"Trainer num_classes={self.num_classes} != concrete model num_classes={model_nc}; "
                        "using the model's (its head is already sized)."
                    )
                return model_nc
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return self._num_classes_from_targets(self.train_set)

    def _num_classes_from_targets(self, source: Any) -> Optional[int]:
        """Derive ``num_classes`` by walking the per-image detection targets.

        torchvision-style detectors reserve class 0 for background and number
        foreground classes ``1..K``, so the head size is ``max(label) + 1``. This is
        the LAST-resort path (every other source needs no walk); it logs progress and
        a hint to pin ``num_classes`` to skip it. Returns ``None`` for an empty /
        label-less set (the caller then errors or leaves the model's own default).
        """
        from dataflux import iter_targets

        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        logger.info(
            "raidar: deriving num_classes from detection targets (pin num_classes on the model/trainer to skip)"
        )
        highest = -1
        for target in tqdm(iter_targets(src), total=total, desc="raidar: deriving num_classes", unit="sample"):
            if not isinstance(target, dict):
                continue
            labels = target.get("labels")
            if labels is None:
                continue
            if hasattr(labels, "numel"):
                if int(labels.numel()) == 0:
                    continue
                highest = max(highest, int(labels.max()))
            else:
                for x in labels:
                    highest = max(highest, int(x))
        return highest + 1 if highest >= 0 else None

    # --- inference helper ----------------------------------------------------

    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:  # type: ignore[override]
        model = cast(nn.Module, self.model)
        return model(images)  # type: ignore[no-any-return]

    # --- step functions ------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        if targets is None:
            raise ValueError("LightningDetectionTrainer.training_step requires targets on the batch.")

        model = cast(nn.Module, self.model)
        loss_dict = model(images, targets)
        if not isinstance(loss_dict, dict):
            raise TypeError(
                f"Detection model in train mode must return Dict[str, Tensor] (a loss dict); "
                f"got {type(loss_dict).__name__}. The native detection path expects the "
                f"torchvision-native forward contract: forward(images, targets) → loss dict."
            )
        total: Tensor = sum(loss_dict.values())  # type: ignore[assignment]
        bs = len(images)
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        # Per-component breakdown — skip if the model returned a single "loss"
        # key (otherwise we'd double-log under the same name).
        if not (len(loss_dict) == 1 and "loss" in loss_dict):
            for k, v in loss_dict.items():
                if isinstance(v, Tensor) and v.numel() == 1:
                    self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        model = cast(nn.Module, self.model)
        predictions = model(images)
        if self.val_metrics is not None and targets is not None:
            self.val_metrics.update(predictions, targets)

    def test_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        images: List[Tensor] = batch.input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch.input.targets
        model = cast(nn.Module, self.model)
        predictions = model(images)
        if self.test_metrics is not None and targets is not None:
            self.test_metrics.update(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is not None:
            _log_scalar_dict(self, self.val_metrics, prefix="val/")
            self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        if self.test_metrics is not None:
            _log_scalar_dict(self, self.test_metrics, prefix="test/")
            self.test_metrics.reset()


__all__ = ["LightningDetectionTrainer"]
