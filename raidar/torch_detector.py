"""``TorchDetector`` — the same detection run as a PLAIN PyTorch training loop.

The control experiment, exactly as in the workspace's classification and segmentation projects:
the Lightning backend hands the epoch to a ``Trainer``; this one writes the loop out —
``zero_grad`` / ``backward`` / ``step``, an explicit device, an explicit ``model.train()`` /
``model.eval()``, an explicit ``torch.no_grad()`` on the eval pass. The claim every other backend
makes is "swap the runnable, keep the config"; this one removes the training *framework* entirely
and the config still does not change — same dataset, same ``detection_ops`` chain, same native
models (torchvision builders, ``RFDETRModel``, ``TorchFcos``), same registered torchmetrics
detection metrics, same artifacts.

**What it shares with the Lightning backend, and what it does not.** The model contract (loss
dict inside the train-mode forward — so there is NO ``loss_fn`` slot on either), the collate
(``recordstream.collate_list``), the metric set
(:mod:`raidar.torch_detection`), and the run artifacts are the SAME objects. What differs is the
loop and what a loop implies: no ``lightning:`` slot, no framework loggers (tracking goes through
``trackers:``), no callback protocol, no distributed strategy — multi-GPU is deliberately out of
scope; that is what Lightning is for.

The ``num_classes`` resolution and the metric slots are DUPLICATED from the Lightning backend
rather than shared, deliberately and temporarily: the engine-neutral base is deferred until it can
be designed against several tasks at once (the workspace's standing decision — see the
segmentation project's identical note).

Minimal flat YAML — the Lightning block with one key changed::

    runnable: !class:raidar.TorchDetector
    model: !lazy:torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
    train_set: !class:recordstream.Stream { source: ..., ops: !ref:detection_ops }
    num_classes: 6
    max_epochs: 5

Run it with ``raidar train cfg.yaml --framework torch``.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from confluid.registry import resolve_class
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.torch import Device, build_metric_collection, resolve_device, scalar_metrics
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file
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
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from raidar.torch_detection import native_batch, reportable_metrics

logger = get_logger(__name__)

#: Which entry point :meth:`TorchDetector.run` dispatches to — the same vocabulary every other
#: backend uses, because the verb is a property of the TASK, not of the engine.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits. Declared here rather than imported from the Lightning backend for
#: the reason this whole module exists — that import would pull ``pytorch_lightning`` into a
#: deliberately framework-free one (``test_it_needs_no_training_framework`` scans for exactly it).
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]

#: Re-exported so ``TorchDetector``'s ``device`` knob can be read from this module the way every
#: other name in its signature can. It is matrainer's: a ``Trainer`` picks an accelerator for you,
#: a hand-written loop has to pick one itself, and every such loop picks identically.
__all__ = ["Device", "DetectorTask", "MetricsSpec", "TorchDetector"]


@configurable(task="detection", role="trainer", framework="torch")
class TorchDetector(TorchRunner, ProgressReporting):
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
        experiment_name / run_name: Run identity, recorded in artifacts and tracking.
        input_key: Record key holding the image.
        target_key: Record key holding the detection target ``Regions``.
        metrics: torchmetrics templates, flowed per EVAL split. ``None`` picks COCO mAP. There is
            deliberately NO ``train_metrics`` slot: a native model in train mode returns ONLY the
            loss dict — no train-split predictions exist to score.
        val_metrics / test_metrics: That split's OWN templates, merged over ``metrics``.
        device: Where to run — see :data:`Device`.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict``). ``None`` writes no
            weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        trackers: matrainer trackers — one or a list.
        class_names: Labels for the per-class metric report, in class-id order; indices otherwise.
        plot_metrics: ALSO send each per-class metric to the trackers as a matplotlib bar chart.
        metric_names: How recorded metrics are named — see
            :class:`~matrainer.metric_names.MetricNames`. This loop already writes the canonical
            ``<split>/<name>``, so the default translates nothing.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__()
        # `nn.Module`, not `Lazy[nn.Module]`: the DEFERRED-ness is declared once, on the ctor
        # param above, which is what `lazy_param_names` / `to_pydantic` read.
        self.model: nn.Module = cast(nn.Module, model)
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
        self.device_choice: Device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        # Deferred runtime slots, built as `LazyClass` so a config can override any DataLoader
        # knob without this class enumerating them — the same construction the Lightning backend
        # uses, divergent in NOTHING (the same recordstream collate, the same keys).
        loader_kw: Dict[str, Any] = dict(
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
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        # Native detection models compute loss inside forward — no separate loss_fn slot.
        self.loss_fn = None
        # Sentinels: `None` means "not chosen" — resolved in `initialize()`, so an explicitly
        # wired metric is never silently replaced.
        self.metrics: Any = metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM (lazy-init mandate: the constructor does no work).
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}
        self._device: torch.device = torch.device("cpu")

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table (recordstream docs/runnable.md).
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, build the model, and place everything on a device."""
        self._device = resolve_device(self.device_choice)
        logger.info(f"raidar torch backend on {self._device}.")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"TorchDetector.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire a torchvision builder (or any native-contract nn.Module) into the model slot."
            )
        self.model.to(self._device)

    def _initialize_fit(self) -> None:
        """Resolve ``num_classes``, then build the model and the metrics with it.

        Deliberately the same sequence as the Lightning backend's method of this name — none of it
        is engine-specific, and it is duplicated rather than shared until the engine-neutral base
        can be designed against several tasks (see the module docstring).
        """
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        nc = self._resolve_num_classes()
        if nc is not None:
            self.num_classes = int(nc)
        if isinstance(self.model, Fluid):
            if self.num_classes is not None and self._model_accepts_num_classes():
                self.model = flow(self.model, num_classes=self.num_classes)
            else:
                self.model = flow(self.model)
        else:
            self.model = flow(self.model)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _initialize_eval(self) -> None:
        """Model-driven setup: the trained head's width is authoritative, best-effort.

        The default MAP metric is class-count-agnostic, so an undeterminable count leaves
        ``num_classes`` as-is instead of raising.
        """
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if nc:
            self.num_classes = int(nc)
        self.test_set = self._prepare(self.test_set)
        self.val_set = self._prepare(self.val_set)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self._metrics_built = True

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        ``ensure_record_dataset`` does the first; ``ensure_materialized`` does the second — the
        fork-safety guard: a lazy source first built in a FORKED DataLoader worker goes through
        ``_scproxy`` on macOS, a SIGSEGV with no Python traceback.
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
            model_nc = getattr(self.model, "num_classes", None)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != int(model_nc):
                    logger.warning(f"Detector num_classes={self.num_classes} != model {model_nc}; using the model's.")
                return int(model_nc)
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
        """Fill the ``metrics`` sentinel — COCO mAP, identical to the Lightning backend's choice.

        Identical ON PURPOSE: a config moved between the two backends must not silently change
        what it measures. Idempotent, so an explicitly wired value is never replaced.
        """
        if self.metrics is None:
            self.metrics = {"map": LazyClass[Metric](MeanAveragePrecision, iou_type="bbox")}

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`~matrainer.metric_slots.merge_metric_specs`, shared with
        every other backend. No ``num_classes`` flow dimension: detection metrics score box
        overlap, not a fixed-width head.
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec)).clone(prefix=prefix)

    # ---- the batch boundary -------------------------------------------------- #
    def _to_device(self, batch: Any) -> Any:
        """Unwrap the detection batch and move images + targets to this run's device."""
        batch_images, batch_targets = native_batch(batch, self.input_key, self.target_key)
        images = [image.to(self._device) for image in batch_images]
        targets = (
            [{k: v.to(self._device) for k, v in t.items()} for t in batch_targets]
            if batch_targets is not None
            else None
        )
        return images, targets

    def _loaders(self) -> Dict[str, Optional[DataLoader[Any]]]:
        """The deferred loader slots, flowed with the datasets this run resolved."""
        return {
            "train": flow(self.train_loader, dataset=self.train_set) if self.train_set is not None else None,
            "val": flow(self.val_loader, dataset=self.val_set) if self.val_set is not None else None,
            "test": flow(self.test_loader, dataset=self.test_set) if self.test_set is not None else None,
        }

    # ---- the loop ------------------------------------------------------------ #
    def _train_epoch(self, loader: DataLoader[Any], optimizer: Optimizer) -> float:
        """One training pass, returning the mean loss. The whole reason this backend exists.

        The native contract computes loss INSIDE the train-mode forward (a loss dict), so the
        loop's job is the sum, the backward and the step — there is no loss_fn to call.
        """
        model = self.model
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            images, targets = self._to_device(batch)
            if targets is None:
                raise ValueError("TorchDetector training requires targets on the batch.")
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            if not isinstance(loss_dict, dict):
                raise TypeError(
                    f"Detection model in train mode must return Dict[str, Tensor] (a loss dict); "
                    f"got {type(loss_dict).__name__}."
                )
            loss = cast(torch.Tensor, sum(loss_dict.values()))
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        return total / batches if batches else float("nan")

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader[Any], metrics: Optional[MetricCollection]) -> None:
        """One evaluation pass. NO eval loss: the native contract returns predictions in eval
        mode, not losses — the split is scored by its metrics alone (the same honesty as the
        Lightning backend's loss-less eval splits)."""
        model = self.model
        model.eval()
        for batch in loader:
            images, targets = self._to_device(batch)
            predictions = model(images)
            if metrics is not None and targets is not None:
                metrics.update(predictions, targets)

    def _epoch_metrics(self, collection: Optional[MetricCollection], step: Optional[int] = None) -> Dict[str, float]:
        """Compute and RESET a split's metrics — the bookkeeping a ``Trainer`` hides.

        Only the SCALARS are RETURNED (`Tracker.log_metrics` is a `{name: float}` channel); the
        per-class VECTORS (`MeanAveragePrecision(class_metrics=True)`) go out through the
        trackers' TEXT channel from :meth:`_report_per_class` — identical reporting to the
        Lightning backend, which matters because the two backends exist to be compared.
        """
        if collection is None:
            return {}
        computed = reportable_metrics(collection.compute())
        self._report_per_class(computed, step=step)
        values = scalar_metrics(computed)
        collection.reset()
        return values

    def _report_per_class(self, values: Any, step: Optional[int] = None) -> None:
        """Send every per-class VECTOR to the log and to each tracker's text (and figure) channel."""
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

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        """Epochs, in the open: forward, backward, step, validate, log, checkpoint."""
        logger.info(f"raidar torch training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("TorchDetector.fit() requires a 'train_set'.")
        loaders = self._loaders()
        train_loader = cast(DataLoader[Any], loaders["train"])
        optimizer = cast(Optimizer, flow(self.optimizer, params=self.model.parameters()))
        trackers = self._live_trackers()
        manifest = self._log_run_context(trackers)
        self._move_metrics()

        for epoch in range(self.max_epochs):
            logs: Dict[str, float] = {"train/loss": self._train_epoch(train_loader, optimizer)}
            if loaders["val"] is not None:
                self._eval_epoch(cast(DataLoader[Any], loaders["val"]), self.val_metrics)
                logs.update(self._epoch_metrics(self.val_metrics, step=epoch))
            logs = as_metric_names(self.metric_names)(logs)
            for tracker in trackers:
                tracker.log_metrics(logs, step=epoch)
            for key, value in logs.items():
                self._history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            self._computed_metrics = dict(logs)

        logger.info("Training complete.")
        self._save_checkpoint()
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        if loaders["test"] is not None:
            self._computed_metrics.update(self._score(cast(DataLoader[Any], loaders["test"])))
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        """Score the VALIDATION split — the `evaluate` / `test` split is the only difference."""
        if self.val_set is None:
            raise ValueError("TorchDetector.evaluate() requires a 'val_set'.")
        self._move_metrics()
        loader = cast(DataLoader[Any], self._loaders()["val"])
        self._computed_metrics = self._score(loader, split="val", metrics=self.val_metrics)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("TorchDetector.test() requires a 'test_set'.")
        self._move_metrics()
        self._computed_metrics = self._score(cast(DataLoader[Any], self._loaders()["test"]))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _score(
        self, loader: DataLoader[Any], split: str = "test", metrics: Optional[MetricCollection] = None
    ) -> Dict[str, float]:
        """Evaluate over ``loader`` and report — shared by ``test``, ``evaluate`` and the post-fit pass."""
        trackers = self._live_trackers()
        collection = metrics if metrics is not None else self.test_metrics
        self._eval_epoch(loader, collection)
        results = self._epoch_metrics(collection)
        results = as_metric_names(self.metric_names)(results)
        for tracker in trackers:
            tracker.log_metrics(results)
        logger.info(f"Evaluation complete: {results}")
        return results

    @entrypoint("predict", role="predictor", primary=True)
    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each with its metadata."""
        logger.info(f"raidar torch predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError("TorchDetector.predict() requires a 'test_set'.")
        self.model.eval()
        sink = flow(self.predictions_sink)
        loader = cast(DataLoader[Any], self._loaders()["test"])
        for batch in loader:
            images, _ = self._to_device(batch)
            predictions = self.model(images)
            if sink is not None:
                self._write_predictions(predictions, batch, cast(PredictionsSink, sink))
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- predictions -------------------------------------------------------- #
    def _write_predictions(self, predictions: Any, batch: Any, sink: PredictionsSink) -> None:
        """One prediction per record — the model already emits a per-image list."""
        metas = batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        items = predictions if isinstance(predictions, list) else [predictions]
        for i, item in enumerate(items):
            meta = metas[i] if metas and i < len(metas) else {}
            sink.write(item, meta)

    # ---- artifacts + tracking ----------------------------------------------- #
    def _save_checkpoint(self) -> None:
        """A plain ``state_dict`` — no framework format, because there is no framework."""
        if not self.checkpoint_dir:
            return
        directory = Path(self.checkpoint_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved weights to {path}")

    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _log_run_context(self, trackers: List[Tracker]) -> Any:
        """Provenance through matrainer's plain FUNCTIONS — there is no callback protocol here.

        ``runnable=self`` rather than a hand-written params dict: matrainer owns the ONE
        hyperparameter derivation, so this backend and the Lightning one cannot answer "what were
        the hyperparameters?" differently. Returns the run manifest so ``fit()`` can finish it.
        """
        if not trackers:
            return None
        from matrainer.tracking.provenance import log_run_context

        return log_run_context(trackers, {"device": str(self._device)}, runnable=self)

    def _move_metrics(self) -> None:
        """Metric state lives on the same device as the predictions it consumes."""
        for collection in (self.val_metrics, self.test_metrics):
            if isinstance(collection, MetricCollection):
                collection.to(self._device)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> nn.Module:
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
        """Per-epoch metric history — what a ``Trainer``'s logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}
