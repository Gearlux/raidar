"""LightningDetector: a standard PyTorch-Lightning object detector (train + eval in one class).

A self-contained ``pytorch_lightning.LightningModule`` — its own detection step functions
(``forward(images, targets) → loss dict`` in train, ``forward(images) → per-image predictions``
in eval), ``configure_optimizers``, and a ``run()`` that builds the loaders + ``pl.Trainer`` and
calls ``fit`` / ``validate`` / ``test`` / ``predict`` per the ``task`` knob. matrainer is used ONLY
for optional experiment tracking and a couple of small utilities — remove it and this class still
trains; you lose the rich tracking, not the training.

Detection specifics it owns: the native torchvision model contract (loss dict summed to the
optimisation scalar; per-image ``{boxes, scores, labels}`` fed to the metrics), and ``num_classes``
resolution (concrete model head → explicit arg → lazy kwarg → target walk).

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

import inspect
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from confluid.registry import resolve_class
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.lightning import (
    TrackerLogger,
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_metric_collection,
    scalar_metrics,
)
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table
from recordstream import (
    PredictionsSink,
    RecordSource,
    Regions,
    batch_metadata,
    collate_list,
    ensure_materialized,
    ensure_record_dataset,
)
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from raidar.torch_detection import native_batch, reportable_metrics

logger = get_logger(__name__)

#: Which entry point :meth:`LightningDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits: a ``{name: template}`` mapping, a ready-made collection, or ``None``
#: for "not chosen". Named because THREE slots share it (the shared set + val/test) and
#: `to_pydantic` reads the annotation for navigaitor's form-spec / StreamStudio widgets.
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]


def _model_num_classes(model: nn.Module) -> Optional[int]:
    """Best-effort read of a BUILT detector's class count (explicit attr or torchvision head), or ``None``."""
    nc = getattr(model, "num_classes", None)
    if nc:
        return int(nc)
    box_predictor = getattr(getattr(model, "roi_heads", None), "box_predictor", None)
    out_features = getattr(getattr(box_predictor, "cls_score", None), "out_features", None)
    if out_features:
        return int(out_features)
    return None


@configurable(task="detection", role="trainer", framework="torch")
class LightningDetector(TorchRunner, ProgressReporting, L.LightningModule):
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
        trackers: matrainer trackers — one or a list. They ARE this run's metric sinks: each is
            adapted to Lightning's logger interface, so there is no separate ``logger:`` slot of
            ``pytorch_lightning.loggers``.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the detection target ``Regions`` (default ``"target"``).
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
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. The ``self.log`` calls already write
            the canonical ``<split>/<name>``, so the default translates nothing; it is handed to
            each ``TrackerLogger``, where a wired ``target:`` records the same run in another
            engine's spelling.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``predictions_sink``, ``metrics_sidecar_path``,
    ``log_system_metrics``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[nn.Module]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        class_names: Optional[List[str]] = None,
        plot_metrics: bool = False,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        self.model: Lazy[nn.Module] = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.task: DetectorTask = task
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # Native detection models compute loss inside forward — no separate loss_fn callable.
        self.loss_fn = None
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            # The batch SHAPE is DECLARED, not inherited from whether `ToTensor` ran: the native
            # contract takes `List[Tensor]`, so this asks recordstream for the unstacked
            # batch. Override `collate_fn` in the config for a model wanting one tensor.
            collate_fn=collate_list,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # No `logger=` here: the trackers ARE the metric sinks, adapted to Lightning's logger
        # interface in `_build_trainer` (a Tracker may still be a deferred Fluid at this point, so
        # the adapter is built at run time, not now).
        self.lightning: Lazy[L.Trainer] = LazyClass(L.Trainer, max_epochs=max_epochs)
        # Sentinels: `None` means "not chosen" — a live default here could not be told apart from a
        # user's explicit choice. `_resolve_task_defaults()` fills them in `initialize()`.
        self.metrics: Any = metrics
        # Per-split slots, each dual-purpose the way `model` is: they hold that split's metrics
        # SPEC until `initialize()` builds it, and the live prefixed `MetricCollection` after.
        # `_metrics_built` is what tells the two states apart — `is None` cannot, once a slot may
        # legitimately arrive carrying a spec.
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work). It may hold a single
        # tracker, a list, or deferred `Fluid` markers; `_live_trackers()` normalizes and flows it
        # at run time.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # Stored VERBATIM for the same reason `trackers` is: wired from YAML it arrives as a
        # deferred marker, and `as_metric_names()` builds it where the metrics are reported.
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.log_system_metrics: bool = True
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table — a hand-written
        # {task: method} dict restates them and drifts (recordstream `docs/runnable.md`).
        run_entrypoint(self, self.task)

    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:  # type: ignore[override]
        return cast(List[Dict[str, Tensor]], cast(nn.Module, self.model)(images))

    # ---- detection specifics ------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, flow the model, and build the per-split metric collections."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

    def _initialize_fit(self) -> None:
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        nc = self._resolve_num_classes()
        if nc is not None:
            self.num_classes = int(nc)
        if isinstance(self.model, Fluid):
            # Injected ONLY when the deferred builder accepts it — torchvision ships its detectors
            # as builder FUNCTIONS with differing signatures, so a builder without the param is
            # flowed plainly rather than handed a kwarg it would reject.
            if self.num_classes is not None and self._model_accepts_num_classes():
                self.model = flow(self.model, num_classes=self.num_classes)
            else:
                self.model = flow(self.model)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _initialize_eval(self) -> None:
        # The trained model's head is authoritative for eval — flow it first, read num_classes off
        # it. NOT the eval dataset, which may be a subset whose targets never show the last class.
        # Best-effort: the default MAP metric is class-count-agnostic, so an undeterminable count
        # leaves `num_classes` as-is instead of raising.
        self.model = flow(self.model)
        nc = _model_num_classes(cast(nn.Module, self.model))
        if nc:
            self.num_classes = int(nc)
        self._resolve_task_defaults()
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        setattr(self, eval_slot, self._prepare(getattr(self, eval_slot)))
        metrics_slot = "val_metrics" if self.task == "evaluate" else "test_metrics"
        prefix = "val/" if self.task == "evaluate" else "test/"
        if not self._metrics_built:
            setattr(self, metrics_slot, self._build_split_metrics(getattr(self, metrics_slot), prefix))
            self._metrics_built = True

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        ``ensure_record_dataset`` does the first; ``ensure_materialized`` does the second, and it
        matters as soon as ``num_workers`` is non-zero: a FORKED DataLoader worker that is first
        to touch a lazy source builds it through ``_scproxy`` / CoreFoundation on macOS, which is
        not fork-safe — a SIGSEGV with no Python traceback. Reading one record here, in the
        parent, leaves the child nothing to build.
        """
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _model_accepts_num_classes(self) -> bool:
        if not isinstance(self.model, Fluid):
            return False
        target = self.model.target
        resolved = resolve_class(target) if isinstance(target, str) else target
        if resolved is None or not callable(resolved):
            return False
        try:
            params = inspect.signature(resolved).parameters
        except (ValueError, TypeError):
            return False
        return "num_classes" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _resolve_num_classes(self) -> Optional[int]:
        if not isinstance(self.model, Fluid):
            model_nc = _model_num_classes(cast(nn.Module, self.model))
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != model_nc:
                    logger.warning(f"Trainer num_classes={self.num_classes} != model {model_nc}; using the model's.")
                return model_nc
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return self._num_classes_from_targets(self.train_set)

    def _num_classes_from_targets(self, source: Any) -> Optional[int]:
        from recordstream import iter_key

        if source is None:
            return None
        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        logger.info("raidar: deriving num_classes from detection targets (pin num_classes to skip)")
        highest = -1
        for target in tqdm(
            iter_key(src, self.target_key), total=total, desc="raidar: deriving num_classes", unit="record"
        ):
            labels: Any
            if isinstance(target, Regions):
                labels = target.labels
            elif isinstance(target, dict):
                labels = target.get("labels")
            else:
                continue
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

    def _resolve_task_defaults(self) -> None:
        """Fill the ``metrics`` sentinel — COCO mAP, the field's primary detection score.

        A value the user DID wire is left untouched (that is what the ``None`` sentinel buys:
        "not chosen" is distinguishable from "chosen to be the default"), and this is idempotent
        so a directly-invoked step function works without :meth:`initialize`. Class-count-
        agnostic, so unlike the classification/segmentation defaults it needs no ``num_classes``.
        """
        if self.metrics is None:
            self.metrics = {"map": LazyClass[Metric](MeanAveragePrecision, iou_type="bbox")}

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`matrainer.metric_slots.merge_metric_specs`, shared with the
        classification and segmentation tasks, so a config moving between them cannot change
        shape. No ``num_classes`` flow dimension: detection metrics score box overlap, not a
        fixed-width head (``MeanAveragePrecision`` and ``DetectionConfusionMatrix`` take none).
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec)).clone(prefix=prefix)

    # ---- detection step functions ------------------------------------------- #
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

    # ---- epoch-end metric reporting ----------------------------------------- #
    # Reset happens at epoch START, not end, so the LAST epoch's state survives the run —
    # `_write_metrics_sidecar` and the `computed_metrics` @output both `compute()` after
    # `trainer.validate()` / `.test()` has returned.
    def on_validation_epoch_start(self) -> None:
        if isinstance(self.val_metrics, (Metric, MetricCollection)):
            self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._report_epoch_metrics(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        if isinstance(self.test_metrics, (Metric, MetricCollection)):
            self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self._report_epoch_metrics(self.test_metrics)

    def _report_epoch_metrics(self, collection: Any) -> None:
        """Compute one split's metrics once at epoch end: scalars logged, per-class vectors reported.

        Computing here rather than handing the whole collection to ``log_dict`` is what lets a
        non-scalar metric be wired at all: ``log_dict`` reduces every value through Lightning's
        ``convert_tensors_to_scalars``, which RAISES on anything that is not one number — and a
        ``MeanAveragePrecision(class_metrics=True)`` emits per-class ``[C]`` vectors
        (``map_per_class`` / ``mar_100_per_class``), exactly the report a detection run wants on
        validation.
        """
        if not isinstance(collection, (Metric, MetricCollection)):
            return
        values = reportable_metrics(collection.compute())
        self.log_dict(scalar_metrics(values))
        self._report_per_class(values)

    def _report_per_class(self, values: Any) -> None:
        """Send every per-class VECTOR to the log and to each tracker's text (and figure) channel.

        Which computed values ARE per-class vectors is matrainer's rule
        (:func:`~matrainer.tracking.render.per_class_vectors` — a flat numeric sequence of length
        >= 2, by SHAPE not by metric name), and so are the renderers; this method only decides
        WHEN and WHERE. Skipped during Lightning's sanity check.
        """
        if self._trainer is not None and self.trainer.sanity_checking:
            return
        step = self.current_epoch if self._trainer is not None else None
        trackers = self._live_trackers()
        for payload in per_class_vectors(values, self.class_names or None):
            table = render_vector_as_table(payload["name"], payload["values"], payload["class_names"])
            logger.info(table)
            figure = (
                render_vector_as_figure(payload["name"], payload["values"], payload["class_names"])
                if self.plot_metrics and trackers
                else None
            )
            for tracker in trackers:
                tracker.log_text(payload["name"], table, step=step)
                if figure is not None:
                    tracker.log_figure(payload["name"], figure, step=step)

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"raidar training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningDetector.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(self.train_set)
        val_ds = ensure_record_dataset(self.val_set) if self.val_set is not None else None
        test_ds = ensure_record_dataset(self.test_set) if self.test_set is not None else None
        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None
        trainer = self._build_trainer()
        trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")
        if test_ds is not None:
            trainer.test(model=self, dataloaders=flow(self.test_loader, dataset=test_ds))

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self._run_eval("validate", "val_set", "val_loader", "val_metrics")

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        self._run_eval("test", "test_set", "test_loader", "test_metrics")

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        logger.info(f"raidar predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("LightningDetector.predict() requires a 'test_set'.")
        dataset = ensure_record_dataset(self.test_set)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(self.test_loader, dataset=dataset)
        trainer = self._build_trainer()
        trainer.predict(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()

    def _run_eval(self, stage: str, dataset_slot: str, loader_slot: str, metrics_slot: str) -> None:
        logger.info(f"raidar evaluating ({stage}): {self.experiment_name} ({self.run_name})")
        source = getattr(self, dataset_slot)
        if source is None:
            raise ValueError(f"LightningDetector.{stage}() requires a '{dataset_slot}'.")
        dataset = ensure_record_dataset(source)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._write_metrics_sidecar(getattr(self, metrics_slot))

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object returns
        that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on every call —
        two MLflow runs for one training run.
        """
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL matrainer tracking (a few lines; removable).

        The trackers ARE the Trainer's loggers: each is flowed (it may still be a deferred
        ``Fluid``) and adapted through ``TrackerLogger``, so Lightning's per-step metrics land in
        the same backends as the provenance and the manifest. There is no separate ``logger:``
        slot — one wiring, one place a run's output goes.
        """
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        adapters = [TrackerLogger(t, metric_names=names) for t in trackers]
        # A runtime kwarg wins over the stored one, so an explicitly wired `lightning:` block keeps
        # every knob it set and only gains the loggers.
        trainer = cast(L.Trainer, flow(self.lightning, logger=adapters or False))
        if trackers:
            # `runnable=self` rather than a params dict: matrainer derives the hyperparameters from
            # the flowed object graph AND dumps the resolved config, so no two backends can answer
            # "what were the hyperparameters?" differently.
            cb = TrackingCallback(trackers, runnable=self)
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    # ---- predictions + metrics ---------------------------------------------- #
    def _write_predictions(self, preds: Any, batch: Any) -> None:
        # recordstream's transpose: the columns the model did not consume, back into per-record
        # dicts — the same call every non-torch backend here makes.
        per_record_meta = (
            batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        )
        items = (
            preds["predictions"]
            if isinstance(preds, dict) and "predictions" in preds
            else (preds if isinstance(preds, list) else [preds])
        )
        for i in range(len(items)):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(items[i], m)

    def _write_metrics_sidecar(self, metrics: Any) -> None:
        """Hand the run's final scalars to matrainer, which owns where run artifacts land."""
        if not isinstance(metrics, (Metric, MetricCollection)):
            return
        try:
            payload = scalar_metrics(reportable_metrics(metrics.compute()))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for the sidecar: {exc}")
            return
        write_metrics_sidecar(self.metrics_sidecar_path, payload, self.experiment_name, self.run_name)

    # ---- @outputs ----------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained/live detection model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run(): the model is still a deferred Fluid.")
        return cast(nn.Module, self.model)

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if self.task == "fit":
            try:
                return scalar_metrics(dict(self.trainer.callback_metrics))
            except RuntimeError as exc:
                raise RuntimeError("computed_metrics is only available after run().") from exc
        metrics = self.val_metrics if self.task == "evaluate" else self.test_metrics
        if not isinstance(metrics, (Metric, MetricCollection)):
            raise RuntimeError("computed_metrics is only available after run().")
        return scalar_metrics(reportable_metrics(metrics.compute()))


__all__ = ["DetectorTask", "LightningDetector", "MetricsSpec", "_model_num_classes"]
