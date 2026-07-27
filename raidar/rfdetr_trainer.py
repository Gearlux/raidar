"""Native RF-DETR trainer/evaluator — wraps RF-DETR's OWN PyTorch-Lightning stack.

Companion to :class:`raidar.detector.LightningDetector`. Where
that trainer drives the bare ``LWDETR`` module through marainer's generic loop,
this one hands off to RF-DETR's **own** training recipe — ``RFDETRModelModule`` +
``RFDETRDataModule`` + ``build_trainer`` (COCO-mAP eval via torchmetrics, EMA,
best-checkpoint saving, LR schedule). It is the RF-DETR analogue of
``raidar_ultralytics.UltralyticsTrainer``; because RF-DETR is Apache-2.0 it lives
in core raidar (the ultralytics sibling is split out only for AGPL reasons).

Two input modes (XOR), mirroring ``UltralyticsTrainer``:

* **Disk mode** — ``data: str | Path`` points at a COCO/YOLO/Roboflow dataset
  directory; we call RF-DETR's native ``model.train(dataset_dir=...)`` verbatim.
* **Stream mode** — ``train_set`` / ``val_set`` are :class:`recordstream.Stream` pipelines
  (+ ``num_classes`` + ``class_names``); we reimplement ``RFDETR.train()``'s tail
  with :class:`raidar.stream_rfdetr_dataset.StreamRFDETRDataModule`, so the Stream feeds
  RF-DETR's native loop with NO disk materialization.

Both trainer and evaluator are ``@configurable(task="detection", role=…)`` so
navigaitor surfaces them as alternate detection trainers/evaluators (detection is
a multi-trainer task) and `confluid`'s ``list_classes(task="detection", role=…)``
returns them beside the Lightning path.

**Label convention (footgun):** RF-DETR labels are **0-indexed ``0..K-1``** with
background reserved at index ``K`` — the OPPOSITE of torchvision's background-at-0.
Wire a 0-indexed target op (e.g. ``CocoToTorchVisionDetectionOp(label_offset=0)``),
NOT the ``+1`` shift the Faster R-CNN example uses.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

from confluid import configurable
from loggair import get_logger

logger = get_logger(__name__)

_VariantName = Literal["nano", "small", "medium", "large"]

#: Built when ``num_classes`` is omitted in disk mode — RF-DETR's ``train()``
#: re-aligns the head to the dataset's category count, so the initial value is
#: a placeholder that the native loop overwrites from ``dataset_dir``.
_DEFAULT_DISK_NUM_CLASSES = 90


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort scalar→float (handles 0-dim tensors); ``None`` if not scalar-numeric."""
    try:
        if hasattr(value, "item"):
            return float(value.item())  # 0-dim tensor / numpy scalar
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        # RuntimeError: multi-element tensor (e.g. `classes`) — not a scalar metric.
        return None


def _metrics_to_payload(metrics: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Coerce a metrics dict (e.g. PTL ``callback_metrics``) to JSON-safe floats, dropping non-scalars."""
    payload: Dict[str, float] = {}
    for key, value in (metrics or {}).items():
        coerced = _coerce_float(value)
        if coerced is not None:
            payload[str(key)] = coerced
    return payload


def _write_metrics_sidecar(metrics: Optional[Mapping[str, Any]], out_path: Optional[str]) -> None:
    """Dump a metrics dict to ``out_path`` as JSON (marainer's sidecar convention).

    Logger-agnostic stable schema downstream tooling (navigaitor's
    ``read_run_metrics``) can rely on regardless of which trainer produced the run.
    """
    if not out_path or not metrics:
        return
    payload = _metrics_to_payload(metrics)
    if not payload:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info(f"Wrote metrics sidecar: {out}")


def _sidecar_from_output_dir(output_dir: str, out_path: Optional[str]) -> None:
    """Disk-mode sidecar: distill the last row of RF-DETR's ``metrics.csv`` into ``out_path``.

    RF-DETR's CSVLogger writes a ``metrics.csv`` under ``output_dir``; we read the
    last non-empty row (best-effort — a missing file is logged, not fatal).
    """
    if not out_path:
        return
    candidates = sorted(Path(output_dir).rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        logger.info(f"metrics_output_path set but no metrics.csv found under {output_dir} — sidecar skipped.")
        return
    with candidates[0].open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if any(v not in ("", None) for v in row.values())]
    if not rows:
        return
    _write_metrics_sidecar(dict(rows[-1]), out_path)


def _enable_mlflow_system_metrics_if_tracking(enabled: bool) -> None:
    if enabled and os.environ.get("MLFLOW_TRACKING_URI"):
        from marainer.tracking.system import enable_mlflow_system_metrics

        enable_mlflow_system_metrics()


def _ordered_class_names(class_names: Optional[Dict[int, str]]) -> Optional[List[str]]:
    """``{0: "cat", 1: "dog"}`` -> ``["cat", "dog"]`` (index = class id)."""
    if not class_names:
        return None
    return [class_names[i] for i in sorted(class_names)]


def _trainer_device_kwargs(device: Optional[str]) -> Dict[str, Any]:
    """Reuse RF-DETR's own torch-device -> PTL ``{accelerator, devices}`` mapping."""
    from rfdetr.detr import RFDETR

    accelerator, devices = RFDETR._resolve_trainer_device_kwargs(device)
    kwargs: Dict[str, Any] = {}
    if accelerator is not None:
        kwargs["accelerator"] = accelerator
    if devices is not None:
        kwargs["devices"] = devices
    return kwargs


@configurable(task="detection", role="trainer")
class RFDETRTrainer:
    """Native RF-DETR trainer (disk + in-memory Stream), driving RF-DETR's own PTL loop.

    Args:
        data: Disk-mode dataset directory (COCO/YOLO/Roboflow). Mutually exclusive
            with ``train_set``/``val_set``.
        variant: ``"nano"`` (default) / ``"small"`` / ``"medium"`` / ``"large"``.
        pretrained: Start from RF-DETR's COCO-pretrained weights (ignored when
            ``checkpoint`` is given).
        resolution: Square input size; ``None`` uses the variant default.
        epochs: Training epochs.
        batch_size: Per-device batch size.
        lr: Base learning rate; ``None`` uses RF-DETR's default.
        device: Torch device string (``"cpu"``/``"cuda"``/``"cuda:1"``/``"mps"``);
            mapped to PTL accelerator/devices.
        output_dir: Where RF-DETR writes checkpoints / metrics.csv / logs.
        project: WandB/MLflow project name.
        run_name: WandB/MLflow run name (forwarded to RF-DETR's ``run`` knob;
            defaults to ``experiment_name``). NOTE: named ``run_name`` — not
            ``run`` — so it does not shadow the :meth:`run` method.
        experiment_name: Logical run label (also the default ``run_name``).
        num_workers: DataLoader workers.
        early_stopping: Enable RF-DETR's early stopping.
        eval_interval: Run validation every N epochs.
        run_test: Run the test split at the end of training.
        dataset_file: Disk-mode dataset layout (``"coco"``/``"yolo"``/``"roboflow"``).
        mlflow / tensorboard: Enable RF-DETR's native loggers.
        checkpoint: Resume / weight-init checkpoint (bypasses ``pretrained``).
        extra_train_kwargs: Escape hatch merged into the RF-DETR train config.
        metrics_output_path: Optional JSON sidecar for the final metrics.
        train_set / val_set / test_set: Stream-mode datasets (``test`` falls back to
            ``val``).
        num_classes: Stream-mode foreground class count K (labels are 0-indexed
            ``0..K-1``; head emits K+1 with background at K).
        class_names: Stream-mode ``{class_id: name}`` (required in Stream mode).
        aug_config: RF-DETR Albumentations augmentation override.
        log_system_metrics: Enable marainer MLflow system metrics when
            ``MLFLOW_TRACKING_URI`` is set.
    """

    def __init__(
        self,
        data: Any = None,
        variant: _VariantName = "nano",
        pretrained: bool = True,
        resolution: Optional[int] = None,
        epochs: int = 100,
        batch_size: int = 4,
        lr: Optional[float] = None,
        device: Optional[str] = None,
        output_dir: str = "output",
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        experiment_name: str = "rfdetr_run",
        num_workers: int = 2,
        early_stopping: bool = False,
        eval_interval: int = 1,
        run_test: bool = False,
        dataset_file: Literal["coco", "yolo", "roboflow"] = "roboflow",
        mlflow: bool = False,
        tensorboard: bool = False,
        checkpoint: Optional[str] = None,
        extra_train_kwargs: Optional[Dict[str, Any]] = None,
        metrics_output_path: Optional[str] = None,
        train_set: Any = None,
        val_set: Any = None,
        test_set: Any = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        aug_config: Optional[Dict[str, Any]] = None,
        log_system_metrics: bool = True,
    ) -> None:
        # Store config ONLY — no functional work (zero-arg / lazy-init mandate).
        self.data = data
        self.variant = variant
        self.pretrained = pretrained
        self.resolution = resolution
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = lr
        self.device = device
        self.output_dir = output_dir
        self.project = project
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.num_workers = int(num_workers)
        self.early_stopping = bool(early_stopping)
        self.eval_interval = int(eval_interval)
        self.run_test = bool(run_test)
        self.dataset_file = dataset_file
        self.mlflow = bool(mlflow)
        self.tensorboard = bool(tensorboard)
        self.checkpoint = checkpoint
        self.extra_train_kwargs = dict(extra_train_kwargs or {})
        self.metrics_output_path = metrics_output_path
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_names = dict(class_names) if class_names else None
        self.aug_config = dict(aug_config) if aug_config else None
        self.log_system_metrics = bool(log_system_metrics)

    def _validate_inputs(self) -> None:
        stream = self.train_set is not None or self.val_set is not None
        if self.data is not None and stream:
            raise ValueError(
                "RFDETRTrainer: pass EITHER `data` (disk dir) OR `train_set`+`val_set` (Stream) — not both."
            )
        if self.data is None and not stream:
            raise ValueError("RFDETRTrainer: must supply EITHER `data` OR `train_set`+`val_set`.")
        if stream:
            if self.train_set is None or self.val_set is None:
                raise ValueError("RFDETRTrainer Stream mode requires BOTH `train_set` and `val_set`.")
            if self.num_classes is None or not self.class_names:
                raise ValueError("RFDETRTrainer Stream mode requires `num_classes` and `class_names` (0-indexed).")

    def run(self) -> None:
        self._validate_inputs()
        _enable_mlflow_system_metrics_if_tracking(self.log_system_metrics)
        if self.data is not None:
            self._run_disk_mode()
        else:
            self._run_stream_mode()

    def _build_rf(self, num_classes: int) -> Any:
        from raidar.rfdetr import _build_rfdetr

        return _build_rfdetr(
            self.variant,
            num_classes=num_classes,
            pretrained_detector=self.pretrained and self.checkpoint is None,
            resolution=self.resolution,
        )

    def _run_disk_mode(self) -> None:
        """Hand off to RF-DETR's native ``.train()`` over an on-disk dataset directory."""
        rf = self._build_rf(self.num_classes if self.num_classes is not None else _DEFAULT_DISK_NUM_CLASSES)
        kwargs: Dict[str, Any] = dict(
            dataset_dir=str(self.data),
            dataset_file=self.dataset_file,
            epochs=self.epochs,
            batch_size=self.batch_size,
            output_dir=self.output_dir,
            device=self.device,
            project=self.project,
            run=self.run_name or self.experiment_name,
            num_workers=self.num_workers,
            early_stopping=self.early_stopping,
            eval_interval=self.eval_interval,
            run_test=self.run_test,
            mlflow=self.mlflow,
            tensorboard=self.tensorboard,
        )
        if self.lr is not None:
            kwargs["lr"] = self.lr
        if self.resolution is not None:
            kwargs["resolution"] = self.resolution
        if self.checkpoint is not None:
            kwargs["resume"] = self.checkpoint
        if self.aug_config:
            kwargs["aug_config"] = self.aug_config
        ordered = _ordered_class_names(self.class_names)
        if ordered:
            kwargs["class_names"] = ordered
        kwargs.update(self.extra_train_kwargs)
        rf.train(**kwargs)
        _sidecar_from_output_dir(self.output_dir, self.metrics_output_path)

    def _run_stream_mode(self) -> None:
        """Reimplement ``RFDETR.train()``'s tail (rfdetr/detr.py) with our StreamRFDETRDataModule.

        NO disk: our datamodule's ``setup()`` bypasses ``build_dataset``. ``dataset_dir="."``
        is a placeholder satisfying ``TrainConfig``'s required field/validator; it is never
        read (and ``train()``'s dataset-derived num_classes alignment never fires, because we
        pass ``num_classes`` explicitly and call the tail directly).
        """
        from rfdetr.training import RFDETRModelModule, build_trainer

        from raidar.stream_rfdetr_dataset import StreamRFDETRDataModule

        names = _ordered_class_names(self.class_names)
        assert self.num_classes is not None and names is not None  # _validate_inputs guarantees
        rf = self._build_rf(self.num_classes)
        model_config = rf.model_config

        cfg_kwargs: Dict[str, Any] = dict(
            dataset_dir=".",  # placeholder — bypassed by StreamRFDETRDataModule.setup()
            dataset_file=self.dataset_file,
            epochs=self.epochs,
            batch_size=self.batch_size,
            output_dir=self.output_dir,
            project=self.project,
            run=self.run_name or self.experiment_name,
            num_workers=self.num_workers,
            early_stopping=self.early_stopping,
            eval_interval=self.eval_interval,
            run_test=self.run_test,
            mlflow=self.mlflow,
            tensorboard=self.tensorboard,
            class_names=names,
        )
        if self.lr is not None:
            cfg_kwargs["lr"] = self.lr
        if self.checkpoint is not None:
            cfg_kwargs["resume"] = self.checkpoint
        if self.aug_config:
            cfg_kwargs["aug_config"] = self.aug_config
        cfg_kwargs.update(self.extra_train_kwargs)
        cfg = rf.get_train_config(**cfg_kwargs)

        module = RFDETRModelModule(model_config, cfg)
        datamodule = StreamRFDETRDataModule(
            model_config,
            cfg,
            train_stream=self.train_set,
            val_stream=self.val_set,
            test_stream=self.test_set,
            class_names=names,
        )
        trainer = build_trainer(cfg, model_config, **_trainer_device_kwargs(self.device))
        trainer.fit(module, datamodule, ckpt_path=cfg.resume or None)
        rf.model.model = module.model  # sync trained weights back for predict/export parity
        _write_metrics_sidecar(dict(trainer.callback_metrics), self.metrics_output_path)


@configurable(task="detection", role="evaluator")
class RFDETREvaluator:
    """Native RF-DETR evaluator — runs RF-DETR's validation/test loop over a checkpoint.

    RF-DETR has no standalone ``.evaluate()``; we drive its lower-level
    ``RFDETRModelModule`` + ``build_trainer`` and call ``trainer.validate``/``.test``
    (so the COCO-mAP ``COCOEvalCallback`` produces the metrics). The trained
    checkpoint is loaded the supported way — ``model_config.pretrain_weights =
    weights`` — so RF-DETR's weight loader handles head re-sizing.

    Args:
        weights: Path to the trained checkpoint (required at run time).
        data: Disk-mode dataset directory. Mutually exclusive with ``val_set``.
        variant: Model variant (must match the trained run).
        resolution: Square input size; ``None`` = variant default.
        batch_size: Eval batch size.
        device: Torch device string.
        output_dir: Working dir for RF-DETR's logger.
        dataset_file: Disk-mode dataset layout.
        experiment_name: Run label.
        metrics_output_path: Optional JSON metrics sidecar.
        val_set: Stream-mode eval dataset.
        num_classes: Foreground class count K (must match the trained run).
        class_names: ``{class_id: name}``.
        run_test: Run the test loop instead of validate.
        log_system_metrics: Enable marainer MLflow system metrics when tracking.
    """

    def __init__(
        self,
        weights: str = "",
        data: Any = None,
        variant: _VariantName = "nano",
        resolution: Optional[int] = None,
        batch_size: int = 4,
        device: Optional[str] = None,
        output_dir: str = "output",
        dataset_file: Literal["coco", "yolo", "roboflow"] = "roboflow",
        experiment_name: str = "rfdetr_eval",
        metrics_output_path: Optional[str] = None,
        val_set: Any = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        run_test: bool = False,
        log_system_metrics: bool = True,
    ) -> None:
        self.weights = weights
        self.data = data
        self.variant = variant
        self.resolution = resolution
        self.batch_size = int(batch_size)
        self.device = device
        self.output_dir = output_dir
        self.dataset_file = dataset_file
        self.experiment_name = experiment_name
        self.metrics_output_path = metrics_output_path
        self.val_set = val_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_names = dict(class_names) if class_names else None
        self.run_test = bool(run_test)
        self.log_system_metrics = bool(log_system_metrics)

    def _validate_inputs(self) -> None:
        if not self.weights:
            raise ValueError("RFDETREvaluator: `weights` (trained checkpoint path) is required.")
        stream = self.val_set is not None
        if self.data is not None and stream:
            raise ValueError("RFDETREvaluator: pass EITHER `data` (disk dir) OR `val_set` (Stream) — not both.")
        if self.data is None and not stream:
            raise ValueError("RFDETREvaluator: must supply EITHER `data` OR `val_set`.")
        if stream and (self.num_classes is None or not self.class_names):
            raise ValueError("RFDETREvaluator Stream mode requires `num_classes` and `class_names` (0-indexed).")

    def run(self) -> None:
        self._validate_inputs()
        _enable_mlflow_system_metrics_if_tracking(self.log_system_metrics)
        if self.data is not None:
            self._run_disk_mode()
        else:
            self._run_stream_mode()

    def _prepare(self, dataset_dir: str) -> Any:
        """Build the high-level RF-DETR + a checkpoint-loaded model_config + the eval TrainConfig."""
        from raidar.rfdetr import _build_rfdetr

        num_classes = self.num_classes if self.num_classes is not None else _DEFAULT_DISK_NUM_CLASSES
        rf = _build_rfdetr(self.variant, num_classes=num_classes, pretrained_detector=False, resolution=self.resolution)
        # The module rebuilds the model FROM model_config, so point it at the trained checkpoint
        # here (the supported _load_pretrain_weights path that handles head re-sizing).
        rf.model_config.pretrain_weights = self.weights
        names = _ordered_class_names(self.class_names)
        cfg = rf.get_train_config(
            dataset_dir=dataset_dir,
            dataset_file=self.dataset_file,
            batch_size=self.batch_size,
            output_dir=self.output_dir,
            run=self.experiment_name,
            run_test=self.run_test,
            **({"class_names": names} if names else {}),
        )
        return rf, cfg

    def _evaluate(self, module: Any, datamodule: Any, model_config: Any, cfg: Any) -> None:
        from rfdetr.training import build_trainer

        trainer = build_trainer(cfg, model_config, **_trainer_device_kwargs(self.device))
        results = trainer.test(module, datamodule) if self.run_test else trainer.validate(module, datamodule)
        metrics = results[0] if results else dict(trainer.callback_metrics)
        _write_metrics_sidecar(metrics, self.metrics_output_path)

    def _run_disk_mode(self) -> None:
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule

        rf, cfg = self._prepare(str(self.data))
        module = RFDETRModelModule(rf.model_config, cfg)
        datamodule = RFDETRDataModule(rf.model_config, cfg)
        self._evaluate(module, datamodule, rf.model_config, cfg)

    def _run_stream_mode(self) -> None:
        from rfdetr.training import RFDETRModelModule

        from raidar.stream_rfdetr_dataset import StreamRFDETRDataModule

        names = _ordered_class_names(self.class_names)
        rf, cfg = self._prepare(".")  # placeholder dir — StreamRFDETRDataModule bypasses build_dataset
        module = RFDETRModelModule(rf.model_config, cfg)
        datamodule = StreamRFDETRDataModule(
            rf.model_config,
            cfg,
            val_stream=self.val_set,
            test_stream=self.val_set if self.run_test else None,
            class_names=names,
        )
        self._evaluate(module, datamodule, rf.model_config, cfg)


__all__ = ["RFDETRTrainer", "RFDETREvaluator"]
