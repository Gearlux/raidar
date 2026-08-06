"""``FastaiDetector`` — the same detection run with a fastai ``Learner``.

fastai runs on torch, so it reuses the same native models, the same collate and the same
torchmetrics detection set — what it brings is one-cycle scheduling, a ``Recorder`` that reports
every epoch, and fastai's callback ecosystem.

**The no-op-loss design — how a loss-inside-forward model meets fastai's loop.** A fastai
``Learner`` computes ``loss_func(model(*xb), *yb)``, but the native detection contract computes
its loss INSIDE the train-mode forward (``model(images, targets) → loss dict``) — a shape that is
COMMON for detectors, so the engine accommodates it rather than being excluded. Three designs
were weighed and this one chosen:

* computing the loss in ``loss_func`` — rejected: the targets must be IN the train-mode forward,
  so the loss would re-run the model (a double forward) or smuggle it in;
* a callback owning its own backward via ``CancelBatchException`` — rejected: it replaces the fit
  loop fastai was chosen for;
* **the no-op loss** — :func:`fastai_detection_collate` packs ``((images, targets), dummy_y)``
  so BOTH reach the model call, :class:`NativeDetectionAdapter`'s training forward returns the
  SUMMED loss dict as the "prediction", and :func:`passthrough_loss` returns that prediction
  unchanged — fastai backprops it, one forward, the loop untouched.

**The eval branch is explicit.** The native contract yields NO eval loss, so the adapter's eval
forward returns the prediction list, the pass-through loss returns a ZERO scalar for fastai's
bookkeeping (``valid_loss`` is recorded as 0 and is MEANINGLESS for this contract — documented,
not hidden), and validation quality comes from the SAME torchmetrics path as the torch backend,
fed by :class:`DetectionMetricsCallback` rather than ``Learner.metrics`` names (no fastai metric
consumes a per-image prediction list). Once a detector with a SEPARATE loss is wired (the shared
spec's), a config gets a real ``valid_loss`` with no adapter at all.

``from fastai.callback import schedule`` is load-bearing: ``Learner.fit_one_cycle`` is a
``@patch`` installed as that module's IMPORT SIDE EFFECT (fastai's ``@patch`` returns ``None``,
so importing the name binds None and the call would fall through to the MODEL).

Run it with ``raidar train cfg.yaml --framework fastai``.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from fastai import optimizer as fastai_optimizer

# `Learner.fit_one_cycle` is a `@patch` installed as this module's import SIDE EFFECT — see the
# module docstring. Do not "tidy" it away.
from fastai.callback import schedule  # noqa: F401
from fastai.callback.core import Callback as FastaiCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.torch import Device, build_metric_collection, resolve_device, scalar_metrics
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
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
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from raidar.torch_detection import native_batch, reportable_metrics

logger = get_logger(__name__)

#: Which entry point :meth:`FastaiDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits — the SAME torchmetrics vocabulary as every torch-family backend
#: (no fastai metric consumes a per-image prediction list, so the fleet set is what applies).
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]

#: A fastai optimizer function, or its name in ``fastai.optimizer``.
FastaiOptimizer = Union[str, Any]

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
class FastaiDetector(TorchRunner, ProgressReporting):
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
            No ``train_metrics``: the training pass computes the loss dict and nothing else.
        class_names: Labels for the per-class metric report, in class-id order.
        plot_metrics: ALSO send each per-class metric to the trackers as a bar chart.
        metric_names: How recorded metrics are named. This backend reports through its OWN
            torchmetrics pass under canonical names; the Recorder's ``train_loss`` is translated
            from fastai's vocabulary.
        callbacks: fastai callbacks.
        trackers: matrainer trackers — one or a list.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the detection ``Regions``.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``train/val/test_loader``,
    ``dataloaders``, ``learner``, ``predictions_sink``, ``metrics_sidecar_path``.
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
        metric_names: Optional[Lazy[MetricNames]] = None,
        callbacks: Optional[List[Lazy[FastaiCallback]]] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.model: nn.Module = cast(nn.Module, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: DetectorTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Coerced EXPLICITLY: YAML 1.1 parses `1e-4` (no decimal point) as a STRING, and fastai's
        # scheduler then dies in a numpy divide three frames from the config line (hit for real).
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay) if weight_decay is not None else None
        self.one_cycle = one_cycle
        self.opt_func: FastaiOptimizer = opt_func
        self.device_choice: Device = device
        self._device: torch.device = torch.device("cpu")
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # Stored VERBATIM (lazy-init): entries may be deferred `Fluid` markers, flowed where used.
        self.callbacks: Optional[List[Lazy[FastaiCallback]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        # The loaders are the torch backend's construction with exactly ONE kwarg changed:
        # `collate_fn=fastai_detection_collate` (positional unpacking) where the others bake the
        # shared record-dict collate.
        from functools import partial

        loader_kw: Dict[str, Any] = dict(
            collate_fn=partial(fastai_detection_collate, input_key=self.input_key, target_key=self.target_key),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        # fastai's own two objects, deferred so every knob they take is reachable from config.
        # `DataLoaders(*loaders, …)` takes its loaders POSITIONALLY — `_dataloaders()` supplies
        # them as `flow()`'s positional runtime args.
        self.dataloaders: Lazy[DataLoaders] = LazyClass(DataLoaders)
        self.learner: Lazy[Learner] = LazyClass(Learner, lr=learning_rate, wd=weight_decay)
        # Sentinels, resolved in `initialize()` — the same COCO-mAP default as every backend.
        self.metrics: Any = metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._computed_metrics: Dict[str, float] = {}
        self._learner: Optional[Learner] = None

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then build the model and the metric collections."""
        self._device = resolve_device(self.device_choice)
        logger.info(f"raidar fastai backend on {self._device}.")
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"FastaiDetector.model must be a torch.nn.Module, got {type(self.model).__name__}.")
        self.model.to(self._device)
        self._resolve_task_defaults()
        if not self._metrics_built:
            self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
            self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
            self._metrics_built = True

    def _initialize_fit(self) -> None:
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        nc = self._num_classes_without_data()
        if nc is None:
            if self.train_set is None:
                raise RuntimeError(
                    "could not determine num_classes — set it on the detector or the model, or wire a train_set."
                )
            nc = self._num_classes_from_targets(self.train_set)
        if nc is not None:
            self.num_classes = int(nc)
        if isinstance(self.model, Fluid):
            self.model = flow(self.model, num_classes=self.num_classes)
        else:
            self.model = flow(self.model)

    def _initialize_eval(self) -> None:
        """Model-driven setup, best-effort: the default MAP metric is class-count-agnostic."""
        self.model = flow(self.model)
        nc = getattr(self.model, "num_classes", None)
        if nc:
            self.num_classes = int(nc)
        self.test_set = self._prepare(self.test_set)
        self.val_set = self._prepare(self.val_set)

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process.

        For THIS backend the materialization is not optional: **fastai sets the multiprocessing
        start method to ``fork`` at import**, so a DataLoader worker is a forked child — a lazy
        source first built there goes through ``_scproxy`` on macOS, a SIGSEGV with no traceback.
        """
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _num_classes_without_data(self) -> Optional[int]:
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

    def _num_classes_from_targets(self, source: Any) -> Optional[int]:
        from recordstream import iter_key

        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        logger.info("raidar: deriving num_classes from detection targets (pin num_classes to skip)")
        highest = -1
        for target in tqdm(iter_key(src, self.target_key), total=total, desc="deriving num_classes", unit="record"):
            labels = target.labels if isinstance(target, Regions) else target.get("labels")
            if labels is None:
                continue
            if hasattr(labels, "numel"):
                if int(labels.numel()) == 0:
                    continue
                highest = max(highest, int(labels.max()))
            else:
                for value in labels:
                    highest = max(highest, int(value))
        return highest + 1 if highest >= 0 else None

    def _resolve_task_defaults(self) -> None:
        """Fill the ``metrics`` sentinel — COCO mAP, identical to every other backend's choice."""
        if self.metrics is None:
            self.metrics = {"map": LazyClass[Metric](MeanAveragePrecision, iou_type="bbox")}

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — merged by the fleet-shared rule."""
        return build_metric_collection(merge_metric_specs(self.metrics, spec)).clone(prefix=prefix)

    # ---- the learner --------------------------------------------------------- #
    def _dataloaders(self) -> DataLoaders:
        """fastai's ``DataLoaders`` — a train loader and a validation loader, always both.

        A ``Learner`` validates at the end of every epoch, with no "no validation" mode. When no
        ``val_set`` is wired the training data is reused, which keeps a run working but makes the
        reported metrics training metrics — said out loud, hence the warning.
        """
        train = flow(self.train_loader, dataset=self.train_set)
        if self.val_set is not None:
            valid = flow(self.val_loader, dataset=self.val_set)
            return self._pin_device(cast(DataLoaders, flow(self.dataloaders, train, valid)))
        logger.warning(
            "No 'val_set' wired: fastai's Learner always validates, so the TRAINING data is being "
            "used for validation and the reported metrics measure fit, not generalization."
        )
        return self._pin_device(
            cast(DataLoaders, flow(self.dataloaders, train, flow(self.val_loader, dataset=self.train_set)))
        )

    def _pin_device(self, dataloaders: DataLoaders) -> DataLoaders:
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
        dataloaders._device = self._device
        return dataloaders

    def _build_learner(self, dataloaders: DataLoaders, collection: Optional[MetricCollection]) -> Learner:
        """Assemble the ``Learner``: the adapter, the pass-through loss, the metrics callback.

        ``Learner.metrics`` stays EMPTY — validation quality goes through the fleet's
        torchmetrics collection via :class:`DetectionMetricsCallback` (module docstring).
        """
        path = self._artifact_dir()
        learner = cast(
            Learner,
            flow(
                self.learner,
                dls=dataloaders,
                model=NativeDetectionAdapter(self.model),
                loss_func=passthrough_loss,
                opt_func=_as_opt_func(self.opt_func),
                cbs=[DetectionMetricsCallback(collection)] + [flow(cb) for cb in (self.callbacks or [])],
                path=Path(path) if path else None,
            ),
        )
        self._learner = learner
        return learner

    def _artifact_dir(self) -> Optional[str]:
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"raidar fastai training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("FastaiDetector.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        collection = cast(Optional[MetricCollection], self.val_metrics if self.val_set is not None else None)
        learner = self._build_learner(self._dataloaders(), collection)
        if self.one_cycle:
            learner.fit_one_cycle(self.max_epochs, lr_max=self.learning_rate)
        else:
            learner.fit(self.max_epochs, lr=self.learning_rate)
        names = as_metric_names(self.metric_names)
        logs = _recorded_train_loss(learner, names)
        logs.update(self._epoch_metrics(collection))
        self._computed_metrics = logs
        _log_recorded_epochs(learner, trackers, names)
        for tracker in trackers:
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        """Score the VALIDATION split — the `evaluate` / `test` split is the only difference."""
        if self.val_set is None:
            raise ValueError("FastaiDetector.evaluate() requires a 'val_set'.")
        self._computed_metrics = self._score(cast(Any, self.val_set), cast(MetricCollection, self.val_metrics))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("FastaiDetector.test() requires a 'test_set'.")
        self._computed_metrics = self._score(cast(Any, self.test_set), cast(MetricCollection, self.test_metrics))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _score(self, dataset: Any, collection: MetricCollection) -> Dict[str, float]:
        """Run the Learner's validation loop over ``dataset`` and report from the collection."""
        loader = flow(self.test_loader, dataset=dataset)
        learner = self._build_learner(
            self._pin_device(cast(DataLoaders, flow(self.dataloaders, loader, loader))), collection
        )
        learner.validate()
        results = self._epoch_metrics(collection)
        for tracker in self._live_trackers():
            tracker.log_metrics(results)
        logger.info(f"Evaluation complete: {results}")
        return results

    def _epoch_metrics(self, collection: Optional[MetricCollection]) -> Dict[str, float]:
        """Compute + report + RESET the collection — scalars returned, vectors to the text channel."""
        if collection is None:
            return {}
        computed = reportable_metrics(collection.compute())
        self._report_per_class(computed)
        values = scalar_metrics(computed)
        collection.reset()
        return values

    def _report_per_class(self, values: Any) -> None:
        from matrainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table

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
                tracker.log_text(payload["name"], table)
                if figure is not None:
                    tracker.log_figure(payload["name"], figure)

    @entrypoint("predict", role="predictor", primary=True)
    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record predictions to the sink — a plain forward over a RECORD-dict loader.

        The fastai collate has already dropped the metadata a sink needs, so predict flows the
        SAME ``test_loader`` slot with a runtime ``collate_fn`` override back to the shared
        record collate (a runtime kwarg wins over the stored one).
        """
        if self.test_set is None:
            raise ValueError("FastaiDetector.predict() requires a 'test_set'.")
        self.model.eval()
        sink = flow(self.predictions_sink)
        record_collate = collate_list
        loader = flow(self.test_loader, dataset=self.test_set, collate_fn=record_collate)
        for batch in loader:
            images, _ = native_batch(batch, self.input_key, self.target_key)
            predictions = self.model([image.to(self._device) for image in images])
            if sink is not None:
                metas = batch_metadata(batch, exclude=(self.input_key, self.target_key))
                for i, item in enumerate(predictions if isinstance(predictions, list) else [predictions]):
                    meta = metas[i] if metas and i < len(metas) else {}
                    cast(PredictionsSink, sink).write(item, meta)
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

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


def _as_opt_func(value: FastaiOptimizer) -> Any:
    """A fastai optimizer function from a callable or its name (``"Adam"``, ``"SGD"``, …)."""
    built = flow(value)
    if not isinstance(built, str):
        return built
    opt = getattr(fastai_optimizer, built, None)
    if opt is None:
        raise ValueError(
            f"Unknown fastai optimizer {built!r}. Names come from `fastai.optimizer` — e.g. Adam, "
            f"SGD, RMSProp, RAdam, QHAdam, Lamb, ranger."
        )
    return opt


def _recorded_train_loss(learner: Learner, names: MetricNames) -> Dict[str, float]:
    """The last epoch's ``train_loss`` in canonical spelling. ``valid_loss`` is DROPPED — under
    the no-op-loss contract it is a recorded zero, and a meaningless number under a real-looking
    name is worse than a gap."""
    rows = list(getattr(learner.recorder, "values", []) or [])
    if not rows:
        return {}
    return names({"train_loss": float(rows[-1][0])}, source="fastai")


def _log_recorded_epochs(learner: Learner, trackers: Sequence[Tracker], names: MetricNames) -> None:
    """Forward every recorded epoch's ``train_loss`` to the trackers — the Recorder's rows are
    otherwise invisible, and a tracked run would show one point where other backends show a curve."""
    if not trackers:
        return
    for epoch, row in enumerate(list(getattr(learner.recorder, "values", []) or [])):
        metrics = names({"train_loss": float(row[0])}, source="fastai")
        for tracker in trackers:
            tracker.log_metrics(metrics, step=epoch)
