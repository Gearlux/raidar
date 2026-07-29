"""LightningDetector: a standard PyTorch-Lightning object-detection module (train + eval).

A self-contained ``pytorch_lightning.LightningModule`` for torchvision-native detection models —
its OWN detection step functions (``forward(images, targets) → loss dict`` in train,
``forward(images) → per-image predictions`` in eval), ``configure_optimizers``, and a ``run()``
that builds the loaders + ``pl.Trainer`` and calls ``fit`` / ``test`` / ``predict`` per the ``task``
knob. marainer is used ONLY for optional experiment tracking + the small ``ensure_record_dataset``
utility. Remove marainer and this class still trains.

Detection specifics it owns: the native torchvision model contract (loss dict summed to the
optimisation scalar; per-image ``{boxes, scores, labels}`` fed to ``MeanAveragePrecision``), and
``num_classes`` resolution (concrete model head → explicit arg → lazy kwarg → target walk).
"""

import inspect
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, Lazy, LazyClass, Mandatory, configurable, flow, get_hierarchy_from_instance, output
from confluid.registry import resolve_class
from loggair import get_logger
from marainer.lightning import TrackingCallback, activate_for_lightning, attach_lightning_progress, scalar_metrics
from marainer.run_names import generate_run_name
from marainer.tracking.base import Tracker
from pytorch_lightning.loggers import Logger
from recordstream import PredictionsSink, RecordSource, Regions, ensure_record_dataset
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from raidar.detection import DetectionBatchInput, detection_collate_fn

logger = get_logger(__name__)

#: Which entry point :meth:`LightningDetector.run` dispatches to.
DetectorTask = Literal["fit", "evaluate", "test", "predict"]


def _log_scalar_dict(module: L.LightningModule, metric: Any, prefix: str) -> None:
    """Compute *metric* and log every 0-dim floating Tensor value under *prefix* (MAP returns a dict)."""
    result = metric.compute()
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, Tensor) and v.ndim == 0 and v.is_floating_point():
                module.log(f"{prefix}{k}", v)
    elif isinstance(result, Tensor) and result.ndim == 0 and result.is_floating_point():
        module.log(f"{prefix}{type(metric).__name__}", result)


def _batch_detection(batch: Any) -> DetectionBatchInput:
    """Recover the :class:`~raidar.detection.DetectionBatchInput` (images + optional targets) from a batch.

    The detection collate (:func:`raidar.detection.detection_collate_fn`) returns a batched record
    dict whose ``"detection"`` key carries the :class:`~raidar.detection.DetectionBatchInput`
    VERBATIM (it is a custom container, not an item). A raw
    :class:`~raidar.detection.DetectionBatchInput` (fed directly, unwrapped) is passed through.
    """
    if isinstance(batch, dict):
        return cast(DetectionBatchInput, batch["detection"])
    return cast(DetectionBatchInput, batch)


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


@configurable(task="detection", role="trainer")
class LightningDetector(TorchRunner, ProgressReporting, L.LightningModule):
    """A standard Lightning object detector — train, evaluate, test, or predict via ``task``.

    Args:
        model: Torchvision-native detection model (loss dict in train, per-image predictions in eval).
            Wire ``!lazy:`` to have :meth:`initialize` flow it with the resolved ``num_classes``.
        train_set: Training dataset / Stream (for ``task="fit"``).
        test_set: Held-out eval dataset / Stream (for ``task="test"`` / ``"predict"``).
        num_classes: Output class count (torchvision-style incl. background). Omit to resolve from the
            model head or the train targets.
        task: Entry point :meth:`run` dispatches to.
        max_epochs / batch_size / num_workers / logger: ergonomic flat-config knobs.
        experiment_name / run_name: tracking identity.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the detection target ``Regions`` (default ``"target"``).

    Post-construction slots: ``val_set``, ``optimizer``, ``train/val/test_loader``, ``lightning``,
    ``val_metrics`` / ``test_metrics``, ``predictions_sink``, ``metrics_sidecar_path``, ``trackers``,
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
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        experiment_name: str = "raidar_detection",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.model: Lazy[nn.Module] = model
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.task: DetectorTask = task
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=_run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # Native detection models compute loss inside forward — no separate loss_fn callable.
        self.loss_fn = None
        self.optimizer: Lazy[Optimizer] = LazyClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        loader_kw = dict(
            collate_fn=partial(detection_collate_fn, input_key=self.input_key, target_key=self.target_key),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        self.train_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=True, **loader_kw)
        self.val_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(DataLoader, shuffle=False, **loader_kw)
        self.lightning: Lazy[L.Trainer] = LazyClass(L.Trainer, max_epochs=max_epochs, logger=logger)
        # Predictions are unavailable in training_step (model returns a loss dict); val/test get MAP.
        self.train_metrics: Optional[Lazy[Union[Metric, MetricCollection]]] = None
        self.val_metrics: Optional[Lazy[Union[Metric, MetricCollection]]] = LazyClass(
            MeanAveragePrecision, iou_type="bbox"
        )
        self.test_metrics: Optional[Lazy[Union[Metric, MetricCollection]]] = LazyClass(
            MeanAveragePrecision, iou_type="bbox"
        )
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        self.trackers: Optional[List[Tracker]] = None
        self.log_system_metrics: bool = True
        self._trackers: List[Tracker] = []

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

    def initialize(self) -> None:
        """Resolve ``num_classes`` and flow the model (fit: derive + inject; eval: read off the model)."""
        if self.task == "fit":
            self.train_set = ensure_record_dataset(flow(self.train_set)) if self.train_set is not None else None
            if self.val_set is not None:
                self.val_set = ensure_record_dataset(flow(self.val_set))
            if self.test_set is not None:
                self.test_set = ensure_record_dataset(flow(self.test_set))
            nc = self._resolve_num_classes()
            if nc is not None:
                self.num_classes = int(nc)
            if isinstance(self.model, Fluid):
                if self.num_classes is not None and self._model_accepts_num_classes():
                    self.model = flow(self.model, num_classes=self.num_classes)
                else:
                    self.model = flow(self.model)
        else:
            # Eval: the trained model's head is authoritative — flow first, read num_classes off it.
            self.model = flow(self.model)
            nc = _model_num_classes(cast(nn.Module, self.model))
            if nc:
                self.num_classes = int(nc)
            eval_slot = "val_set" if self.task == "evaluate" else "test_set"
            if getattr(self, eval_slot) is not None:
                setattr(self, eval_slot, ensure_record_dataset(flow(getattr(self, eval_slot))))

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

    # ---- detection step functions ------------------------------------------- #
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        batch_input = _batch_detection(batch)
        images: List[Tensor] = batch_input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch_input.targets
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
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        if not (len(loss_dict) == 1 and "loss" in loss_dict):
            for k, v in loss_dict.items():
                if isinstance(v, Tensor) and v.numel() == 1:
                    self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=bs)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        batch_input = _batch_detection(batch)
        images: List[Tensor] = batch_input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch_input.targets
        predictions = cast(nn.Module, self.model)(images)
        if self.val_metrics is not None and targets is not None:
            cast(Union[Metric, MetricCollection], self.val_metrics).update(predictions, targets)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch_input = _batch_detection(batch)
        images: List[Tensor] = batch_input.images
        targets: Optional[List[Dict[str, Tensor]]] = batch_input.targets
        predictions = cast(nn.Module, self.model)(images)
        if self.test_metrics is not None and targets is not None:
            cast(Union[Metric, MetricCollection], self.test_metrics).update(predictions, targets)
        if self.predictions_sink is not None:
            self._write_predictions(predictions, batch)
        return predictions

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        images: List[Tensor] = _batch_detection(batch).images
        predictions = cast(nn.Module, self.model)(images)
        if self.predictions_sink is not None:
            self._write_predictions(predictions, batch)
        return predictions

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is not None:
            _log_scalar_dict(self, self.val_metrics, prefix="val/")
            cast(Union[Metric, MetricCollection], self.val_metrics).reset()

    def on_test_epoch_end(self) -> None:
        # Log but DON'T reset — _finalize_metrics reads compute() for the sidecar after the loop.
        if self.test_metrics is not None:
            _log_scalar_dict(self, self.test_metrics, prefix="test/")

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points ---------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"raidar training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError("LightningDetector.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(flow(self.train_set))
        val_ds = ensure_record_dataset(flow(self.val_set)) if self.val_set is not None else None
        test_ds = ensure_record_dataset(flow(self.test_set)) if self.test_set is not None else None
        self.val_metrics = flow(self.val_metrics) if self.val_metrics is not None else None
        self.test_metrics = flow(self.test_metrics) if self.test_metrics is not None else None
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
        dataset = ensure_record_dataset(flow(self.test_set))
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
        dataset = ensure_record_dataset(flow(source))
        setattr(
            self, metrics_slot, flow(getattr(self, metrics_slot)) if getattr(self, metrics_slot) is not None else None
        )
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._finalize_metrics(getattr(self, metrics_slot))

    def _build_trainer(self) -> L.Trainer:
        trainer = cast(L.Trainer, flow(self.lightning))
        if self.trackers:
            self._trackers = list(self.trackers)
            cb = TrackingCallback(self._trackers, params=self._config_params())
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    def _config_params(self) -> Dict[str, Any]:
        """Flat ``{dotted-path: live value}`` of this runnable's configuration, for tracker params.

        Walks the FLOWED object graph (not the static signature), so it sees defaults the user
        never wrote into YAML plus post-construction keys. ``log_run_context`` keeps only the
        JSON-scalar values, so nested objects here are harmless. Best-effort by design: a
        tracking detail must never take down a training run.
        """
        try:
            return {path: value for path, (_, value, _) in get_hierarchy_from_instance(self).items()}
        except Exception:  # pragma: no cover - best effort
            return {}

    def _write_predictions(self, preds: Any, batch: Any) -> None:
        per_record_meta = None
        if isinstance(batch, dict):
            # The detection collate stashes the per-record metadata dicts under the "metadata" key.
            per_record_meta = batch.get("metadata")
        else:
            meta = getattr(batch, "metadata", None)
            if isinstance(meta, list):
                per_record_meta = meta
            elif isinstance(meta, dict):
                per_record_meta = meta.get("per_record")
        items = (
            preds["predictions"]
            if isinstance(preds, dict) and "predictions" in preds
            else (preds if isinstance(preds, list) else [preds])
        )
        for i in range(len(items)):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(items[i], m)

    def _finalize_metrics(self, metrics: Any) -> None:
        if metrics is None or self.metrics_sidecar_path is None:
            return
        from marainer.lightning import coerce_metric_value

        try:
            payload = coerce_metric_value(metrics.compute())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not compute final metrics for sidecar: {exc}")
            return
        wrapped = {"experiment_name": self.experiment_name, "run_name": self.run_name, "metrics": payload}
        out_path = Path(self.metrics_sidecar_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(wrapped, indent=2, sort_keys=True))
        logger.info(f"Wrote final metrics sidecar to {out_path}")

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
        if isinstance(metrics, Fluid) or metrics is None:
            raise RuntimeError("computed_metrics is only available after run().")
        return scalar_metrics(metrics.compute())


def _run_prefix(task: str) -> str:
    return {"fit": "train", "predict": "predict"}.get(task, "evaluate")


__all__ = ["DetectorTask", "LightningDetector", "_log_scalar_dict", "_model_num_classes"]
