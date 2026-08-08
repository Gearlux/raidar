"""``HuggingFaceDetector`` — the detection run with ``transformers.Trainer`` driving it.

Since P6 (2026-08-08) this is ``(DetectionBase, matrainer.huggingface.HuggingFaceRunnable)``:
the machinery — the deferred ``TrainingArguments`` / ``Trainer`` slots, the run-time argument
FLOOR, the entry points with the plain-function tracking hookup, the ``log_history`` replay —
lives in :class:`matrainer.huggingface.trainer.HuggingFaceRunnable`; the ``num_classes``
resolution (model head → explicit → lazy kwarg → the announced ``Regions`` walk) in
:class:`~raidar.detection_base.DetectionBase`. What stays HERE is detection's huggingface
vocabulary, and each piece exists because detection is not classification:

1. ``_to_hf_batch`` — a list of RECORDS to the mapping ``Trainer`` feeds the model, through the
   checkpoint's OWN image processor: raidar hands over raw images + COCO-style annotations, and
   the processor does the resize / pad / normalize AND the box conversion to the normalized
   cxcywh the DETR-family loss consumes. Shaping belongs to the checkpoint, not to raidar.
2. **Evaluation is OUR pass, with the SAME torchmetrics the torch-family backends use.** A
   ``Trainer``'s ``compute_metrics`` hook receives per-batch stacked logits, but detection mAP
   needs decoded boxes in the ORIGINAL pixel space — the processor's
   ``post_process_object_detection`` does that decode, and the scores then flow through the same
   ``MeanAveragePrecision`` / ``DetectionConfusionMatrix`` templates as every other backend, so
   the ``metrics:`` slots keep ONE vocabulary across the fleet. The machinery's metric BUILDERS
   are therefore RESTORED to the family's torchmetrics ones (the machinery no-ops them for the
   name-list tasks), its ``_metrics_hook`` stays ``None`` (no Trainer hook — the Trainer still
   reports its own per-epoch ``eval_loss`` when a val split is wired), and both eval seams
   (``_fit_val_scores`` / ``_evaluate_split``) route to :meth:`_decoded_scores`.
3. **The model computes its own loss** (the DETR-family Hungarian matcher, from ``labels=``), so
   there is NO loss slot and NOTHING to register — ``FRAMEWORKS["huggingface"].registrations``
   is ``None``.
4. **The seed applies BEFORE the model is built** — ``TrainingArguments.seed`` alone comes too
   late to cover the randomly initialized detection head (measured: a 3.4x val/map_50 spread).

Labels are 0-indexed ``0..K-1`` with no background slot (wire
``CocoToTorchVisionDetection(label_offset=0)``); images stay RAW (no ToTensor) — the processor
normalizes.

Run it with ``raidar train cfg.yaml --framework huggingface``.
"""

from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from confluid import Lazy, LazyClass, Mandatory, configurable, flow
from loggair import get_logger
from matrainer.huggingface.trainer import HuggingFaceRunnable
from matrainer.metric_names import MetricNames
from matrainer.torch import scalar_metrics
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker
from matrainer.tracking.render import per_class_vectors, render_vector_as_figure, render_vector_as_table
from recordstream import PredictionsSink, RecordSource, Regions, item_data
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection_base import DetectionBase
from raidar.torch_detection import reportable_metrics

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
class HuggingFaceDetector(DetectionBase, HuggingFaceRunnable):
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
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            report_to=report_to,
            output_dir=output_dir,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.score_threshold = float(score_threshold)
        self.seed = int(seed)
        self.class_names: Optional[List[str]] = list(class_names) if class_names else None
        self.plot_metrics = plot_metrics

    # ---- setup: num_classes is the task base's; the seed and the processor warm are ours ---- #
    def initialize(self) -> None:
        """Seed, resolve ``num_classes`` + build the model (task base), then warm the processor.

        The seed goes FIRST, which is the whole point of the knob: ``transformers.Trainer``
        applies ``TrainingArguments.seed`` (42 by default) when the TRAINING starts — by which
        time the wrapper has already constructed the network and randomly initialized the
        detection head for ``num_classes``, so that init was never covered. Measured
        2026-08-06: two runs of one config gave eval_loss 8.001 / 6.277 with val/map_50
        0.0685 / 0.1302, a 3.4x spread that made the backend uncomparable to itself.
        """
        from transformers import set_seed

        set_seed(self.seed)
        super().initialize()
        # Warm the image processor IN THE PARENT — the collate runs inside DataLoader workers,
        # and with `num_workers > 0` a lazily-built processor's first Hub fetch would happen in a
        # FORKED child, through `_scproxy` on macOS: a SIGSEGV with no Python traceback, surfacing
        # only as "DataLoader worker exited unexpectedly" (hit for real on the first cppe-5 run).
        # The same reasoning as `_prepare`'s `ensure_materialized`, applied to the processor.
        self._processor()

    def _resolve_num_classes(self) -> Optional[int]:
        """The task base's ladder, made LOUD: a fit cannot proceed without a resolved count —
        the wrapper builds its head at that width, and failing here names the fix where the
        wrapper's own build error would not."""
        nc = super()._resolve_num_classes()
        if nc is None:
            raise RuntimeError(
                "could not determine num_classes — set it on the detector or the model, or wire a train_set."
            )
        return nc

    def _default_metrics(self) -> Any:
        """COCO mAP, identical to every other backend's choice — the FAMILY's torchmetrics hook."""
        return {"map": LazyClass[Metric](MeanAveragePrecision, iou_type="bbox")}

    # The machinery no-ops the metric builders for the name-list tasks; detection scores with
    # the fleet's torchmetrics, so the FAMILY builders are restored (the fastai-base precedent).
    def _build_fit_metrics(self, include_train: bool = True) -> None:
        TorchTrainerRunnable._build_fit_metrics(self, include_train=False)

    def _build_eval_metrics(self) -> None:
        TorchTrainerRunnable._build_eval_metrics(self)

    def _processor(self) -> Any:
        """The checkpoint's image processor — the ONE component that shapes detection batches."""
        processor = getattr(self.model, "image_processor", None)
        if processor is None:
            raise TypeError(
                "HuggingFaceDetector needs the model's image processor — wire "
                "raidar.HuggingFaceDetectionModel (or any wrapper exposing `image_processor`)."
            )
        return processor

    # ---- data ------------------------------------------------------------------------------- #
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

    # ---- the Trainer seams ------------------------------------------------------------------- #
    def _argument_floor(self) -> Dict[str, Any]:
        floor = super()._argument_floor()
        # The SAME seed `initialize()` already applied — one value, so the head's
        # initialization and the Trainer's data order cannot disagree about the run.
        floor["seed"] = self.seed
        return floor

    def _fit_val_scores(self, trainer: Any) -> Dict[str, float]:
        """The post-fit validation scores come from OUR decode + torchmetrics pass."""
        return self._decoded_scores(self.val_set, cast(MetricCollection, self.val_metrics))

    def _evaluate_split(self, source: Any, split: Literal["val", "test"]) -> Dict[str, float]:
        collection = self.val_metrics if split == "val" else self.test_metrics
        return self._decoded_scores(source, cast(MetricCollection, collection))

    # ---- the four entry points: predict is the one full override ----------------------------- #
    def predict(self) -> None:
        """Stream per-record decoded predictions to the sink, paired with each record's metadata."""
        if self.test_set is None:
            raise ValueError("HuggingFaceDetector.predict() requires a 'test_set'.")
        self.test_set = self._prepare(self.test_set)
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

    # ---- evaluation: OUR decode + the fleet's torchmetrics ----------------------------------- #
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

    def _decoded_scores(self, dataset: Any, collection: MetricCollection) -> Dict[str, float]:
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
