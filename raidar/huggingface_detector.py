"""``HuggingFaceDetector`` — the detection run with ``transformers.Trainer`` driving it.

The backend that writes least while swapping most: ``Trainer`` owns the epoch, the optimizer, the
scheduler, gradient accumulation, mixed precision and distribution (via Accelerate). What raidar
supplies is the data, the run identity and the artifacts.

**What it owns is three things, and each exists because detection is not classification:**

1. ``_to_hf_batch`` — a list of RECORDS to the mapping ``Trainer`` feeds the model, through the
   checkpoint's OWN image processor: raidar hands over raw images + COCO-style annotations, and
   the processor does the resize / pad / normalize AND the box conversion to the normalized
   cxcywh the DETR-family loss consumes. Shaping belongs to the checkpoint, not to raidar.
2. **Evaluation is OUR loop, with the SAME torchmetrics the torch-family backends use.** A
   ``Trainer``'s ``compute_metrics`` hook receives per-batch stacked logits, but detection mAP
   needs decoded boxes in the ORIGINAL pixel space — the processor's
   ``post_process_object_detection`` does that decode, and the scores then flow through the same
   ``MeanAveragePrecision`` / ``DetectionConfusionMatrix`` templates as every other backend, so
   the ``metrics:`` slots keep ONE vocabulary across the fleet. The Trainer still reports its own
   per-epoch ``eval_loss`` when a val split is wired.
3. **The model computes its own loss** (the DETR-family Hungarian matcher, from ``labels=``), so
   there is NO loss slot and NOTHING to register — ``FRAMEWORKS["huggingface"].registrations`` is
   ``None``.

Labels are 0-indexed ``0..K-1`` with no background slot (wire
``CocoToTorchVisionDetection(label_offset=0)``); images stay RAW (no ToTensor) — the processor
normalizes.

``report_to`` defaults to ``"none"`` ON PURPOSE: Trainer auto-enables every tracker it can
import, and this workspace has wandb AND mlflow installed. Tracking goes through ``trackers:``.

Run it with ``raidar train cfg.yaml --framework huggingface``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.torch import build_metric_collection, scalar_metrics
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
from matrainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table
from recordstream import PredictionsSink, RecordSource, Regions, ensure_materialized, ensure_record_dataset, item_data
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from raidar.torch_detection import reportable_metrics

if TYPE_CHECKING:  # pragma: no cover - typing only; the module imports transformers lazily
    from transformers import Trainer, TrainingArguments

logger = get_logger(__name__)

#: Which entry point :meth:`HuggingFaceDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]

#: What a metrics slot admits — the SAME torchmetrics vocabulary as the torch-family backends.
MetricsSpec = Optional[Union[Dict[str, Union[Metric, LazyClass[Metric]]], MetricCollection]]

#: The model slot: the raidar Hub wrapper, or any torch module carrying a compatible
#: ``transformers`` detection model. ``Any`` in the first arm keeps the module-level
#: ``import transformers`` deferred.
HuggingFaceNetwork = Union[Any, torch.nn.Module]

__all__ = ["DetectorTask", "HuggingFaceDetector", "MetricsSpec"]


def _as_rgb_array(payload: Any) -> np.ndarray:
    """An image payload → an ``[H, W, 3]`` uint8 RGB array, the shape an image processor takes.

    Sources hand over decoded arrays in whatever mode the file had (cppe-5 mixes RGB, RGBA and
    grayscale rows — measured), and a CHW float tensor may arrive from a shared op chain. The
    processor owns resize/normalize; this owns only mode/layout coercion.
    """
    array = np.asarray(payload.detach().cpu() if isinstance(payload, torch.Tensor) else payload)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[2] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)  # CHW -> HWC
    if array.dtype != np.uint8:
        scaled = array * 255.0 if array.max() <= 1.0 else array
        array = np.clip(scaled, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        from PIL import Image as PILImage

        array = np.array(PILImage.fromarray(array).convert("RGB"))
    return array


def _record_target(record: Any, target_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """A record's detection target as ``(boxes xyxy absolute, labels)`` numpy arrays."""
    target = record[target_key]
    if isinstance(target, Regions):
        boxes, labels = target.boxes, target.labels
    else:
        data = item_data(target)
        boxes, labels = data["boxes"], data["labels"]
    to_np = lambda v: np.asarray(v.detach().cpu() if isinstance(v, torch.Tensor) else v)  # noqa: E731
    return to_np(boxes).reshape(-1, 4).astype(np.float64), to_np(labels).reshape(-1).astype(np.int64)


def _coco_annotations(index: int, boxes: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """One record's target in the COCO shape the image processor's ``annotations=`` takes."""
    annotations = []
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        w, h = float(x1 - x0), float(y1 - y0)
        annotations.append(
            {"bbox": [float(x0), float(y0), w, h], "category_id": int(label), "area": w * h, "iscrowd": 0}
        )
    return {"image_id": index, "annotations": annotations}


@configurable(task="detection", role="trainer", framework="huggingface")
class HuggingFaceDetector(TorchRunner, ProgressReporting):
    """Train, evaluate or predict an object detector with ``transformers.Trainer``.

    Args:
        model: The detection model — normally :class:`raidar.HuggingFaceDetectionModel`, wired
            ``!lazy:`` so ``initialize()`` can inject the resolved ``num_classes``. The wrapper's
            ``image_processor`` is what shapes every batch; a bare model without one is rejected
            with a clear error.
        train_set: Training source (``task="fit"``).
        test_set: Evaluation / prediction source.
        num_classes: Foreground class count K (0-indexed labels). Omitted, it is derived from the
            model or the train targets.
        task: Which entry point ``run()`` dispatches to.
        max_epochs / batch_size / num_workers / learning_rate / weight_decay: the shared knobs,
            translated into ``TrainingArguments``.
        score_threshold: Confidence floor for the decoded predictions the metrics and the sink
            see (the processor's ``post_process_object_detection`` threshold).
        report_to: Trainer's own tracking integrations. ``"none"`` on purpose.
        seed: Seed applied BEFORE the model is built and passed on to ``TrainingArguments``.
        output_dir: Where Trainer writes its state; defaults beside the sidecar.
        trackers: matrainer trackers — one or a list.
        callbacks: ``transformers.TrainerCallback`` instances.
        experiment_name / run_name: tracking identity.
        input_key / target_key: Record keys for the image and the detection ``Regions``.
        metrics: torchmetrics templates for the EVAL splits — the same vocabulary as the
            torch-family backends, because evaluation runs through OUR decode + torchmetrics
            pass (see the module docstring). ``val_metrics`` / ``test_metrics`` merge over it.
            No ``train_metrics``: the training pass computes the Hungarian loss and nothing else.
        class_names: Labels for the per-class metric report, in class-id order.
        plot_metrics: ALSO send each per-class metric to the trackers as a bar chart.
        metric_names: How recorded metrics are named. The Trainer writes ``train_*``/``eval_*``,
            which the default translates to the canonical ``train/*``/``val/*``; our own eval
            pass already writes ``<split>/<name>``.

    Post-construction slots (YAML / broadcasting): ``val_set``, ``training_arguments``,
    ``trainer``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        model: Mandatory[Lazy[HuggingFaceNetwork]],
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        num_classes: Optional[int] = None,
        task: DetectorTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 2,
        num_workers: int = 0,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        score_threshold: float = 0.25,
        report_to: str = "none",
        seed: int = 42,
        output_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[Any]]] = None,
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
        self.model: HuggingFaceNetwork = cast(HuggingFaceNetwork, model)
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: DetectorTask = task
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Stored rather than closed over, because `_training_arguments()` reads them at RUN time.
        self.num_workers = int(num_workers)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.score_threshold = float(score_threshold)
        self.report_to = report_to
        self.seed = int(seed)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # Sentinels, resolved in `initialize()` — the same COCO-mAP default as every backend.
        self.metrics: Any = metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False
        # Both transformers objects are DEFERRED SLOTS with STRING dotted targets: this module has
        # no top-level `import transformers`, and confluid resolves a dotted target at flow time.
        # `training_arguments` is EMPTY on purpose — every derived knob is applied in
        # `_training_arguments()` as a FLOOR, so a config overriding the slot ADDS to them
        # instead of silently dropping `remove_unused_columns=False` and the epoch count.
        self.training_arguments: Lazy["TrainingArguments"] = LazyClass("transformers.TrainingArguments")
        self.trainer: Lazy["Trainer"] = LazyClass("transformers.Trainer")
        self.callbacks: Optional[List[Lazy[Any]]] = callbacks
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self._computed_metrics: Dict[str, float] = {}

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override
        """Only ``fit`` optimizes — an evaluate/predict run is happy under inference mode."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        run_entrypoint(self, self.task)

    # ---- setup -------------------------------------------------------------- #
    def initialize(self) -> None:
        """Resolve ``num_classes``, then build the model and the metric collections."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()
        # Seed BEFORE the model is built, which is the whole point of the knob.
        # `transformers.Trainer` applies `TrainingArguments.seed` (42 by default) when the
        # TRAINING starts — by which time this wrapper has already constructed the network and
        # randomly initialized the detection head for `num_classes`, so that init was never
        # covered. Measured 2026-08-06: two runs of one config gave eval_loss 8.001 / 6.277 with
        # val/map_50 0.0685 / 0.1302, a 3.4x spread that made the backend uncomparable to itself.
        from transformers import set_seed

        set_seed(self.seed)
        self.model = flow(self.model, num_classes=self.num_classes)
        # Warm the image processor IN THE PARENT — the collate runs inside DataLoader workers,
        # and with `num_workers > 0` a lazily-built processor's first Hub fetch would happen in a
        # FORKED child, through `_scproxy` on macOS: a SIGSEGV with no Python traceback, surfacing
        # only as "DataLoader worker exited unexpectedly" (hit for real on the first cppe-5 run).
        # The same reasoning as `_prepare`'s `ensure_materialized`, applied to the processor.
        self._processor()
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
        if nc is None:
            raise RuntimeError("could not determine num_classes from the train targets.")
        self.num_classes = int(nc)

    def _initialize_eval(self) -> None:
        """Model-driven setup: the trained head's width is authoritative, not the eval split's."""
        nc = self._num_classes_without_data()
        if not nc:
            raise RuntimeError("could not determine num_classes — set it, or wire a model that exposes it.")
        self.num_classes = int(nc)
        self.test_set = self._prepare(self.test_set)
        self.val_set = self._prepare(self.val_set)

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (the fork guard)."""
        if source is None:
            return None
        return ensure_materialized(ensure_record_dataset(source))

    def _num_classes_without_data(self) -> Optional[int]:
        """The class count from the model or the config — everything short of walking the targets."""
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
        """0-indexed labels → K = max(label) + 1, walked once with progress."""
        from recordstream import iter_key

        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        logger.info("raidar: deriving num_classes from detection targets (pin num_classes to skip)")
        highest = -1
        for target in tqdm(iter_key(src, self.target_key), total=total, desc="deriving num_classes", unit="record"):
            labels = target.labels if isinstance(target, Regions) else target.get("labels")
            if labels is None:
                continue
            for value in np.asarray(labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels).reshape(-1):
                highest = max(highest, int(value))
        return highest + 1 if highest >= 0 else None

    def _resolve_task_defaults(self) -> None:
        """Fill the ``metrics`` sentinel — COCO mAP, identical to every other backend's choice."""
        if self.metrics is None:
            self.metrics = {"map": LazyClass[Metric](MeanAveragePrecision, iou_type="bbox")}

    def _build_split_metrics(self, spec: Any, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — merged by the fleet-shared rule."""
        return build_metric_collection(merge_metric_specs(self.metrics, spec)).clone(prefix=prefix)

    def _network(self) -> Any:
        """The live ``transformers`` model — the wrapper's, or a bare one wired directly."""
        return getattr(self.model, "model", self.model)

    def _processor(self) -> Any:
        """The checkpoint's image processor — the ONE component that shapes detection batches."""
        processor = getattr(self.model, "image_processor", None)
        if processor is None:
            raise TypeError(
                "HuggingFaceDetector needs the model's image processor — wire "
                "raidar.HuggingFaceDetectionModel (or any wrapper exposing `image_processor`)."
            )
        return processor

    # ---- data --------------------------------------------------------------- #
    def _to_hf_batch(self, records: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """A list of records → the mapping ``Trainer`` feeds the model, via the PROCESSOR.

        raidar hands over raw RGB arrays + COCO-style annotations; the processor does the
        resize / pad / normalize AND converts the boxes to the normalized cxcywh the DETR-family
        loss consumes. That is the whole reason no ToTensor / Normalize op appears in the
        huggingface scope block.
        """
        images, annotations = [], []
        for index, record in enumerate(records):
            images.append(_as_rgb_array(item_data(record[self.input_key])))
            boxes, labels = _record_target(record, self.target_key)
            annotations.append(_coco_annotations(index, boxes, labels))
        encoded = self._processor()(images=images, annotations=annotations, return_tensors="pt")
        return dict(encoded)

    # ---- the Trainer -------------------------------------------------------- #
    def _training_arguments(self) -> Any:
        """``TrainingArguments`` from the shared knobs — the flat config, in Trainer's words.

        A FLOOR, not an override: a key the config pinned on the ``training_arguments`` slot WINS,
        and only the ones it did not mention are injected (the workspace's standing
        ``TrainingArguments`` rule — replacing would silently drop ``remove_unused_columns=False``,
        which surfaces as a ``KeyError`` inside Trainer's own collator).
        """
        derived: Dict[str, Any] = {
            "output_dir": self._output_dir(),
            "num_train_epochs": self.max_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dataloader_num_workers": self.num_workers,
            # A generic map-style source is not a `datasets.Dataset`, so Trainer must not drop
            # "unused" columns by inspecting the model signature.
            "remove_unused_columns": False,
            "report_to": self.report_to,
            # The SAME seed `initialize()` already applied — one value, so the head's
            # initialization and the Trainer's data order cannot disagree about the run.
            "seed": self.seed,
            "run_name": self.run_name,
            "disable_tqdm": True,
            "save_strategy": "no",
            "eval_strategy": "epoch" if self.val_set is not None else "no",
        }
        pinned = getattr(self.training_arguments, "kwargs", None) or {}
        return flow(self.training_arguments, **{k: v for k, v in derived.items() if k not in pinned})

    def _build_trainer(self, train: Any = None, evaluate: Any = None) -> Any:
        """Flow the ``Trainer`` with everything only the RUN knows. No ``compute_metrics``:
        detection quality is scored by OUR decode + torchmetrics pass (module docstring)."""
        return flow(
            self.trainer,
            model=self._network(),
            args=self._training_arguments(),
            train_dataset=train,
            eval_dataset=evaluate,
            data_collator=self._to_hf_batch,
            callbacks=[flow(cb) for cb in (self.callbacks or [])] or None,
        )

    def _output_dir(self) -> str:
        """Where Trainer writes its state — beside the sidecar when one is configured."""
        if self.output_dir:
            return str(Path(self.output_dir).expanduser())
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent / self.run_name)
        return str(Path("./hf_runs") / self.run_name)

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"raidar huggingface training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("HuggingFaceDetector.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        trainer = self._build_trainer(train=self.train_set, evaluate=self.val_set)
        result = trainer.train()
        names = as_metric_names(self.metric_names)
        scored = names({k: float(v) for k, v in result.metrics.items() if _is_number(v)}, source="huggingface")
        if self.val_set is not None:
            scored.update(self._score(self.val_set, cast(MetricCollection, self.val_metrics)))
        self._computed_metrics = scored
        _log_trainer_history(trainer, trackers, names)
        for tracker in trackers:
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        """Score the VALIDATION split — the `evaluate` / `test` split is the only difference."""
        if self.val_set is None:
            raise ValueError("HuggingFaceDetector.evaluate() requires a 'val_set'.")
        self._computed_metrics = self._score(self.val_set, cast(MetricCollection, self.val_metrics))
        for tracker in self._live_trackers():
            tracker.log_metrics(self._computed_metrics)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        if self.test_set is None:
            raise ValueError("HuggingFaceDetector.test() requires a 'test_set'.")
        self._computed_metrics = self._score(self.test_set, cast(MetricCollection, self.test_metrics))
        for tracker in self._live_trackers():
            tracker.log_metrics(self._computed_metrics)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    @entrypoint("predict", role="predictor", primary=True)
    def predict(self) -> None:
        """Stream per-record decoded predictions to the sink, paired with each record's metadata."""
        if self.test_set is None:
            raise ValueError("HuggingFaceDetector.predict() requires a 'test_set'.")
        sink = flow(self.predictions_sink)
        for record, prediction in self._decoded_predictions(cast(Any, self.test_set)):
            if sink is not None:
                meta = {
                    key: item_data(value)
                    for key, value in record.items()
                    if key not in (self.input_key, self.target_key)
                }
                cast(PredictionsSink, sink).write(prediction, meta)
        if sink is not None and hasattr(sink, "close"):
            sink.close()
        logger.info("Prediction complete.")

    # ---- evaluation: OUR decode + the fleet's torchmetrics ------------------- #
    @torch.no_grad()
    def _decoded_predictions(self, dataset: Any) -> Iterable[Tuple[Any, Dict[str, torch.Tensor]]]:
        """Yield ``(record, prediction)`` pairs decoded back into ORIGINAL pixel space.

        The processor's ``post_process_object_detection`` inverts its own resize, which is what
        lets the predictions score against targets that never left pixel space.
        """
        network = self._network().eval()
        device = next(network.parameters()).device
        processor = self._processor()
        n = len(dataset)
        for start in range(0, n, self.batch_size):
            records = [dataset[i] for i in range(start, min(start + self.batch_size, n))]
            images = [_as_rgb_array(item_data(r[self.input_key])) for r in records]
            encoded = processor(images=images, return_tensors="pt").to(device)
            outputs = network(**encoded)
            sizes = torch.tensor([image.shape[:2] for image in images], device=device)
            decoded = processor.post_process_object_detection(
                outputs, threshold=self.score_threshold, target_sizes=sizes
            )
            for record, prediction in zip(records, decoded):
                yield record, {k: v.detach().cpu() for k, v in prediction.items()}

    def _score(self, dataset: Any, collection: MetricCollection) -> Dict[str, float]:
        """Evaluate a split through the SAME torchmetrics path as the torch-family backends."""
        for record, prediction in self._decoded_predictions(dataset):
            boxes, labels = _record_target(record, self.target_key)
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }
            collection.update([prediction], [target])
        computed = reportable_metrics(collection.compute())
        self._report_per_class(computed)
        values = scalar_metrics(computed)
        collection.reset()
        logger.info(f"Evaluation complete: {values}")
        return values

    def _report_per_class(self, values: Any) -> None:
        """Per-class VECTORS through the trackers' text (and figure) channel — the fleet rule."""
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

    # ---- tracking ------------------------------------------------------------ #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def trained_model(self) -> Any:
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _log_trainer_history(trainer: Any, trackers: Sequence[Tracker], names: MetricNames) -> None:
    """Forward EVERY logged step to the trackers, not just the final metrics.

    ``Trainer`` accumulates its own ``log_history`` and nothing else sees those rows, so without
    this a tracked run would show one point per RUN where every other backend shows a curve.
    """
    if not trackers:
        return
    for entry in getattr(trainer.state, "log_history", []) or []:
        values = names(
            {k: float(v) for k, v in entry.items() if _is_number(v) and k != "epoch"},
            source="huggingface",
        )
        if not values:
            continue
        step = int(entry.get("step", 0))
        for tracker in trackers:
            tracker.log_metrics(values, step=step)
