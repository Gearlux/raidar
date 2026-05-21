"""Native-Ultralytics training and evaluation, plug-compatible with marainer's Liquify CLI.

The default marainer ``train`` / ``evaluate`` commands drive PyTorch Lightning.
This module exposes parallel "Ultralytics-native" runners that drive
``ultralytics.engine.trainer.DetectionTrainer`` / ``DetectionValidator``
directly — keeping mosaic + albumentations + Ultralytics' own logger / TQDM
output, and accepting either a YOLO data.yaml on disk or a
:class:`dataflux.Flux` of detection samples.

Two ``@configurable`` classes form the user-facing surface:

* :class:`UltralyticsTrainer` — wraps ``ultralytics.YOLO.train``-style training.
  When ``data`` is a string path it forwards verbatim to ``YOLO.train(data=...)``;
  when ``data`` is a ``{"train": Flux, "val": Flux, "names": ..., "nc": ...}``
  dict it builds the model itself, swaps in a :class:`FluxDetectionTrainer`
  whose dataloaders draw from :class:`raidar.flux_yolo_dataset.FluxYOLODataset`,
  and runs the training loop.

* :class:`UltralyticsEvaluator` — same disk/Flux duality for validation. Disk
  mode uses ``YOLO.val``; Flux mode uses :class:`FluxDetectionValidator` over
  a :class:`FluxYOLODataset` with ``augment=False, rect=True``.

Both classes accept a ``metrics_output_path: Optional[str]`` — when set, the
final ``DetMetrics.results_dict`` is dumped as JSON, matching the
metrics-sidecar convention :class:`marainer.evaluator.LightningEvaluator` already
ships, so downstream tooling (e.g. Navigaitor's ``read_run_metrics``) doesn't
need to care which trainer produced the run.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from confluid import configurable
from logflow import get_logger
from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import callbacks
from ultralytics.utils.torch_utils import unwrap_model

from raidar.checkpoint import load_lightning_checkpoint
from raidar.flux_yolo_dataset import FluxYOLODataset
from raidar.ultralytics_common import build_yolo26

logger = get_logger(__name__)


_FluxBundle = Dict[str, Any]
"""ContextVar payload: ``{"train": Flux, "val": Flux, "data": data_dict,
"augment_override": Optional[bool]}``.

``augment_override`` lets the surrounding :class:`UltralyticsTrainer` /
:class:`UltralyticsEvaluator` force-on or force-off ultralytics' built-in
mosaic + albumentations pipeline for Flux mode. When ``None`` (default),
:meth:`FluxDetectionTrainer.build_dataset` keeps the legacy behaviour of
augmenting only in train mode.
"""

_FLUX_CTX: ContextVar[Optional[_FluxBundle]] = ContextVar("raidar_ultralytics_flux_ctx", default=None)


def _get_flux_bundle() -> _FluxBundle:
    bundle = _FLUX_CTX.get()
    if bundle is None:
        raise RuntimeError(
            "FluxDetectionTrainer / FluxDetectionValidator instantiated without an active flux context. "
            "Call UltralyticsTrainer.run() / UltralyticsEvaluator.run() — those set the ContextVar before "
            "instantiating the ultralytics machinery."
        )
    return bundle


class FluxDetectionTrainer(DetectionTrainer):
    """``DetectionTrainer`` whose dataloaders read from a DataFlux Flux."""

    def get_dataset(self) -> Dict[str, Any]:
        """Override BaseTrainer.get_dataset — return the pre-built data dict
        from the active flux ContextVar instead of resolving args.data on disk.
        """
        data: Dict[str, Any] = _get_flux_bundle()["data"]
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None) -> FluxYOLODataset:
        bundle = _get_flux_bundle()
        flux_key = "val" if mode == "val" else "train"
        flux = bundle[flux_key]
        gs = max(int(unwrap_model(self.model).stride.max()), 32)  # type: ignore[operator,arg-type]
        # Honour the surrounding trainer's augment override when set; fall back
        # to the legacy "augment only in train mode" default otherwise.
        override = bundle.get("augment_override")
        if override is None:
            augment = mode == "train"
        else:
            augment = bool(override)
        return FluxYOLODataset(
            flux=flux,
            data=self.data,
            imgsz=self.args.imgsz,
            augment=augment,
            hyp=self.args,
            batch_size=batch or self.args.batch,
            stride=gs,
            rect=(mode == "val"),
            prefix="",
            task=self.args.task,
        )

    def auto_batch(self, max_num_obj: int = 0, dataset_size: int = 0) -> int:  # noqa: ARG002
        # Ultralytics' default auto_batch calls build_dataset(self.data["train"], ...) —
        # safe under our overrides, but the parent path also assumes self.data["train"]
        # is a string path. Just disable autobatch and require an explicit batch.
        return int(self.args.batch)

    def plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:  # noqa: ARG002
        # Synthetic flux:// paths in batch["im_file"] would crash matplotlib.
        return

    def plot_training_labels(self) -> None:
        # Same reason — labels carry flux:// paths only.
        return

    def get_validator(self) -> "FluxDetectionValidator":
        """Build a Flux-aware validator so ``final_eval`` (which invokes
        ``self.validator(model=<.pt>)`` in standalone mode at the end of
        training) doesn't trip on ``check_det_dataset('<flux>')``.
        """
        from copy import copy as _copy

        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        validator = FluxDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=_copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.data = self.data  # pre-set so __call__ skips disk resolution
        return validator


class FluxDetectionValidator(DetectionValidator):
    """``DetectionValidator`` that accepts a pre-built dataloader + data dict
    and skips the standalone-mode disk-resolution path.

    When ``trainer is not None`` the parent's ``__call__`` already takes the
    trainer-mode path (no disk check); we only need to guard the standalone
    branch for ``model=...`` invocations.
    """

    def __call__(self, trainer: Any = None, model: Any = None) -> Dict[str, Any]:
        if trainer is None and self.dataloader is not None and self.data is not None:
            return self._call_standalone_flux(model)
        result: Dict[str, Any] = super().__call__(trainer=trainer, model=model)
        return result

    def _call_standalone_flux(self, model: Any) -> Dict[str, Any]:
        """Slim re-implementation of the standalone-mode validator loop with
        no on-disk dataset resolution.

        Trusts a pre-set ``self.data`` and ``self.dataloader`` instead of
        calling ``check_det_dataset(self.args.data)`` /
        ``self.get_dataloader(...)``. Accepts ``model`` as either a live
        ``nn.Module`` (Flux-mode evaluator) or a string/Path checkpoint
        (``DetectionTrainer.final_eval`` passes ``self.best``).
        """
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils import LOGGER, TQDM
        from ultralytics.utils.checks import check_imgsz
        from ultralytics.utils.ops import Profile
        from ultralytics.utils.torch_utils import select_device, smart_inference_mode

        @smart_inference_mode()
        def _run() -> Dict[str, Any]:
            self.training = False
            augment = self.args.augment

            if isinstance(model, (str, Path)):
                wrapped = AutoBackend(
                    model=str(model),
                    device=(select_device(self.args.device) if self.args.device else torch.device("cpu")),
                    dnn=False,
                    data=None,  # don't resolve self.args.data — we already pre-set self.data
                    fp16=self.args.half,
                    verbose=False,
                )
                inference_model: Any = wrapped
                self.device = wrapped.device
                stride = wrapped.stride
            else:
                self.device = select_device(self.args.device) if self.args.device else next(model.parameters()).device
                model.to(self.device)
                model.half() if self.args.half else model.float()
                model.eval()
                inference_model = model
                stride = getattr(model, "stride", torch.tensor([32]))

            self.stride = stride
            self.args.imgsz = check_imgsz(self.args.imgsz, stride=stride)

            dt = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
            self.init_metrics(unwrap_model(inference_model))  # type: ignore[arg-type]
            self.jdict: list[Any] = []  # type: ignore[assignment]
            for batch_i, batch in enumerate(bar):
                self.batch_i = batch_i  # type: ignore[assignment]
                with dt[0]:
                    batch = self.preprocess(batch)
                with dt[1]:
                    preds = inference_model(batch["img"], augment=augment)
                with dt[3]:
                    preds = self.postprocess(preds)
                self.update_metrics(preds, batch)

            self.gather_stats()
            stats: Dict[str, Any] = self.get_stats()
            self.speed = dict(
                zip(
                    self.speed.keys(),
                    (x.t / max(len(self.dataloader.dataset), 1) * 1e3 for x in dt),
                )
            )
            self.finalize_metrics()
            self.print_results()
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            return stats

        return _run()  # type: ignore[no-any-return]


def _build_data_dict(
    flux_data: Dict[str, Any],
    channels: int = 3,
) -> Dict[str, Any]:
    """Construct the ultralytics-shaped ``self.data`` dict from the user's
    Flux-mode ``data`` block.

    Args:
        flux_data: ``{"train": Flux, "val": Flux, "names": dict, "nc": int}``.
        channels: Image channel count (passed to YOLODataset as ``data["channels"]``).
    """
    if not isinstance(flux_data, dict):
        raise TypeError(f"Flux-mode `data` must be a dict; got {type(flux_data).__name__}.")
    if "train" not in flux_data and "val" not in flux_data:
        raise ValueError("Flux-mode `data` must include at least `train` or `val` keys.")
    names = flux_data.get("names")
    nc = flux_data.get("nc")
    if names is None or nc is None:
        raise ValueError("Flux-mode `data` must include `names: dict` and `nc: int`.")
    return {
        "path": ".",
        "train": ".",
        "val": ".",
        "test": ".",
        "names": names,
        "nc": int(nc),
        "channels": int(channels),
    }


def _write_metrics_sidecar(metrics: Any, out_path: Optional[str]) -> None:
    """Dump an ultralytics ``DetMetrics`` (or a dict-of-floats) to ``out_path``.

    Mirrors marainer's metrics-sidecar convention so Navigaitor / downstream
    tools see a stable schema regardless of which trainer produced the run.
    """
    if not out_path or metrics is None:
        return
    if hasattr(metrics, "results_dict"):
        payload: Dict[str, float] = {k: float(v) for k, v in metrics.results_dict.items()}
    elif isinstance(metrics, dict):
        payload = {k: float(v) for k, v in metrics.items()}
    else:
        logger.warning(
            f"metrics_output_path={out_path}: cannot serialize metrics of type "
            f"{type(metrics).__name__} — sidecar write skipped."
        )
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info(f"Wrote metrics sidecar: {out}")


@configurable
class UltralyticsTrainer:
    """Native-Ultralytics trainer (mosaic + albumentations + LOGGER) for
    marainer's CLI.

    Three accepted input shapes:

    * ``data: str | Path`` — a YOLO ``data.yaml`` on disk. Disk mode —
      forwards verbatim to ``YOLO.train(data=...)``.
    * ``data: dict`` ``{"train": Flux, "val": Flux, "names": dict, "nc": int}``
      — legacy Flux mode. Kept for back-compat.
    * ``train_set: Flux``, ``val_set: Flux``, ``num_classes: int``,
      ``class_names: dict[int, str]`` — ergonomic Flux mode, matches the
      sonair / deltaid / lightning-detection convention. Internally rebuilt
      into the legacy dict form.

    Augmentation control:

    * ``augment: Optional[bool]`` — when ``None`` (default), ultralytics'
      built-in mosaic + albumentations fire only during training (legacy
      behaviour). Set ``True`` to force-enable, or ``False`` to disable the
      built-in pipeline entirely so dataflux Flux ops upstream are the sole
      source of augmentation.
    * ``augmentation: Optional[Dict[str, Any]]`` — ultralytics augmentation
      hyperparams (``mosaic``, ``mixup``, ``hsv_h``, ``hsv_s``, ``hsv_v``,
      ``degrees``, ``translate``, ``scale``, ``shear``, ``perspective``,
      ``fliplr``, ``flipud``, ``copy_paste``, …). Key names match
      ultralytics' own config vocabulary; values are merged into the
      ultralytics overrides dict. Distinct from the raw ``extra_overrides``
      escape-hatch; reserved for the aug-scoped subset for readability.
      Setting ``augmentation`` while ``augment=False`` is a config smell and
      raises a ``ValueError``.
    """

    def __init__(
        self,
        data: Any = None,
        variant: Literal["n", "s", "m", "l", "x"] = "n",
        pretrained: bool = True,
        imgsz: int = 640,
        epochs: int = 100,
        batch: int = 16,
        device: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        experiment_name: str = "ultralytics_run",
        end2end: Optional[bool] = None,
        checkpoint: Optional[str] = None,
        extra_overrides: Optional[Dict[str, Any]] = None,
        metrics_output_path: Optional[str] = None,
        train_set: Any = None,
        val_set: Any = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        augment: Optional[bool] = None,
        augmentation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            end2end: Toggle YOLO26's NMS-free detection head. ``None`` (default)
                leaves whatever the model YAML sets — for ``yolo26{variant}.yaml``
                that's ``True``. Set ``False`` to disable the ``one2one`` branch;
                a known workaround for the MPS bool-indexing regression that
                crashes ``E2EDetectLoss`` mid-batch on Apple Silicon
                (``RuntimeError: shape mismatch`` inside ``tal.py``). Disabling
                end2end keeps the standard ``one2many`` head fully trainable —
                you trade NMS-free inference for a working MPS run.
            checkpoint: Optional path to a starting checkpoint. Accepts an
                Ultralytics ``.pt`` (e.g. a previous run's ``best.pt`` /
                ``last.pt``), a Lightning ``.ckpt``, or a plain state-dict.
                When set, the model is initialized from these weights instead
                of the variant's COCO-pretrained download — both ``pretrained``
                and the variant-based selection are bypassed for weight init
                (the variant still picks the architecture YAML in Flux mode).
                For full epoch/optimizer-state resume of an interrupted run,
                pair this (disk mode) with ``extra_overrides={"resume": True}``;
                Flux mode currently supports weight-only init.
        """
        # --- Input-form validation ----------------------------------------
        ergonomic_provided = train_set is not None or val_set is not None
        if data is not None and ergonomic_provided:
            raise ValueError(
                "UltralyticsTrainer: pass EITHER `data` (legacy str/dict) OR the ergonomic "
                "`train_set` + `val_set` (+ `num_classes` + `class_names`) form — not both."
            )
        if data is None and not ergonomic_provided:
            raise ValueError(
                "UltralyticsTrainer: must supply EITHER `data` OR `train_set` + `val_set` "
                "(+ `num_classes` + `class_names`)."
            )
        if ergonomic_provided:
            if train_set is None or val_set is None:
                raise ValueError("UltralyticsTrainer ergonomic form requires BOTH `train_set` and `val_set`.")
            if num_classes is None or class_names is None:
                raise ValueError(
                    "UltralyticsTrainer ergonomic form requires `num_classes` and `class_names: dict[int, str]`."
                )
            data = {
                "train": train_set,
                "val": val_set,
                "names": dict(class_names),
                "nc": int(num_classes),
            }
        # --- Augmentation knob validation ---------------------------------
        if augment is False and augmentation:
            raise ValueError(
                "UltralyticsTrainer: `augmentation=...` is meaningless when `augment=False`. "
                "Either set `augment=True`/`None` to enable ultralytics' aug pipeline, or drop "
                "the `augmentation` dict and supply augmentation via dataflux Flux ops upstream."
            )
        self.data = data
        self.variant = variant
        self.pretrained = pretrained
        self.imgsz = int(imgsz)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.device = device
        self.project = project
        self.name = name
        self.experiment_name = experiment_name
        self.end2end = end2end
        self.checkpoint = checkpoint
        self.extra_overrides = dict(extra_overrides or {})
        self.metrics_output_path = metrics_output_path
        self.augment = augment
        self.augmentation = dict(augmentation) if augmentation else {}

    def run(self) -> None:
        if isinstance(self.data, (str, Path)):
            self._run_disk_mode()
        elif isinstance(self.data, dict):
            self._run_flux_mode()
        else:
            raise TypeError(
                f"UltralyticsTrainer.data must be a str path or a Flux-mode dict; " f"got {type(self.data).__name__}."
            )

    def _run_disk_mode(self) -> None:
        """Hand off to ``ultralytics.YOLO.train`` verbatim."""
        if self.checkpoint is not None:
            model_spec = self.checkpoint
        else:
            model_spec = f"yolo26{self.variant}.pt" if self.pretrained else f"yolo26{self.variant}.yaml"
        yolo = YOLO(model_spec, task="detect")
        if self.end2end is not None:
            yolo.model.set_head_attr(end2end=self.end2end)  # type: ignore[union-attr]
        yolo.train(
            data=str(self.data),
            imgsz=self.imgsz,
            epochs=self.epochs,
            batch=self.batch,
            device=self.device,
            project=self.project,
            name=self.name or self.experiment_name,
            **self.extra_overrides,
        )
        _write_metrics_sidecar(getattr(yolo, "metrics", None), self.metrics_output_path)

    def _run_flux_mode(self) -> None:
        """Drive a :class:`FluxDetectionTrainer` directly, with our pre-built
        ``DetectionModel`` and the user's Fluxes wired in via the ContextVar.
        """
        flux_data = self.data
        if "train" not in flux_data:
            raise ValueError("UltralyticsTrainer.data (Flux mode) must include a `train` Flux.")
        if "val" not in flux_data:
            raise ValueError("UltralyticsTrainer.data (Flux mode) must include a `val` Flux.")

        data_dict = _build_data_dict(flux_data, channels=3)
        nc = data_dict["nc"]

        # When a user checkpoint is supplied its weights overwrite anything
        # `build_yolo26` would graft on, so skip the COCO download in that case.
        pretrained_detector = self.pretrained and self.checkpoint is None
        detection_model = build_yolo26(
            variant=self.variant,
            num_classes=nc,
            pretrained_detector=pretrained_detector,
            agnostic_nms=False,
            end2end=self.end2end,
        )
        if self.checkpoint is not None:
            load_lightning_checkpoint(detection_model, self.checkpoint)

        overrides: Dict[str, Any] = {
            "task": "detect",
            "mode": "train",
            "data": "<flux>",  # sentinel — get_dataset is overridden, this is never resolved
            "model": f"yolo26{self.variant}.yaml",
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "plots": False,  # synthetic flux:// paths break the plot helpers
        }
        if self.device is not None:
            overrides["device"] = self.device
        if self.project is not None:
            overrides["project"] = self.project
        if self.name is not None or self.experiment_name:
            overrides["name"] = self.name or self.experiment_name
        # Augmentation hyperparams (mosaic, mixup, hsv_h, ...) merge before
        # extra_overrides so users can still override any single key via the
        # raw escape-hatch dict.
        overrides.update(self.augmentation)
        overrides.update(self.extra_overrides)

        bundle: _FluxBundle = {
            "train": flux_data["train"],
            "val": flux_data["val"],
            "data": data_dict,
            "augment_override": self.augment,
        }
        token = _FLUX_CTX.set(bundle)
        try:
            trainer = FluxDetectionTrainer(overrides=overrides, _callbacks=callbacks.get_default_callbacks())
            # Inject our pre-built DetectionModel; setup_model() returns
            # immediately when self.model is already an nn.Module.
            trainer.model = detection_model  # type: ignore[assignment]
            trainer.train()
        finally:
            _FLUX_CTX.reset(token)

        _write_metrics_sidecar(
            (getattr(trainer.validator, "metrics", None) if getattr(trainer, "validator", None) else None),
            self.metrics_output_path,
        )


@configurable
class UltralyticsEvaluator:
    """Native-Ultralytics validator. Same disk/Flux duality as
    :class:`UltralyticsTrainer`, including the ergonomic
    ``val_set`` / ``num_classes`` / ``class_names`` form.
    """

    def __init__(
        self,
        weights: str,
        data: Any = None,
        variant: Literal["n", "s", "m", "l", "x"] = "n",
        imgsz: int = 640,
        batch: int = 16,
        device: Optional[str] = None,
        experiment_name: str = "ultralytics_eval",
        metrics_output_path: Optional[str] = None,
        val_set: Any = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        # Ergonomic form: build the legacy `data` dict from val_set + class_names + num_classes.
        ergonomic_provided = val_set is not None
        if data is not None and ergonomic_provided:
            raise ValueError(
                "UltralyticsEvaluator: pass EITHER `data` (legacy str/dict) OR the ergonomic "
                "`val_set` (+ `num_classes` + `class_names`) form — not both."
            )
        if data is None and not ergonomic_provided:
            raise ValueError(
                "UltralyticsEvaluator: must supply EITHER `data` OR `val_set` (+ `num_classes` + `class_names`)."
            )
        if ergonomic_provided:
            if num_classes is None or class_names is None:
                raise ValueError(
                    "UltralyticsEvaluator ergonomic form requires `num_classes` and `class_names: dict[int, str]`."
                )
            data = {
                "val": val_set,
                "names": dict(class_names),
                "nc": int(num_classes),
            }
        self.weights = weights
        self.data = data
        self.variant = variant
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.device = device
        self.experiment_name = experiment_name
        self.metrics_output_path = metrics_output_path

    def run(self) -> None:
        if isinstance(self.data, (str, Path)):
            self._run_disk_mode()
        elif isinstance(self.data, dict):
            self._run_flux_mode()
        else:
            raise TypeError(
                f"UltralyticsEvaluator.data must be a str path or a Flux-mode dict; " f"got {type(self.data).__name__}."
            )

    def _run_disk_mode(self) -> None:
        yolo = YOLO(self.weights, task="detect")
        yolo.val(
            data=str(self.data),
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            name=self.experiment_name,
        )
        _write_metrics_sidecar(getattr(yolo, "metrics", None), self.metrics_output_path)

    def _run_flux_mode(self) -> None:
        flux_data = self.data
        if "val" not in flux_data:
            raise ValueError("UltralyticsEvaluator.data (Flux mode) must include a `val` Flux.")

        data_dict = _build_data_dict(flux_data, channels=3)

        # Load the trained weights via YOLO so we get the proper DetectionModel
        # and its associated args namespace.
        yolo = YOLO(self.weights, task="detect")
        detection_model = yolo.model

        # Build a validator-flavored args namespace.
        args_ns = get_cfg(
            overrides={
                "task": "detect",
                "mode": "val",
                "data": "<flux>",
                "model": str(self.weights),
                "imgsz": self.imgsz,
                "batch": self.batch,
                "device": self.device or "",
                "plots": False,
                "name": self.experiment_name,
            }
        )
        save_dir = get_save_dir(args_ns)

        gs = max(int(unwrap_model(detection_model).stride.max()), 32)  # type: ignore[operator,arg-type]
        dataset = FluxYOLODataset(
            flux=flux_data["val"],
            data=data_dict,
            imgsz=self.imgsz,
            augment=False,
            hyp=args_ns,
            batch_size=self.batch,
            stride=gs,
            rect=True,
            prefix="",
            task="detect",
        )
        dataloader = build_dataloader(
            dataset,
            batch=self.batch,
            workers=0,
            shuffle=False,
            rank=-1,
            drop_last=False,
        )

        validator = FluxDetectionValidator(dataloader=dataloader, save_dir=save_dir, args=args_ns)
        validator.data = data_dict  # type: ignore[assignment]
        validator(model=detection_model)
        _write_metrics_sidecar(getattr(validator, "metrics", None), self.metrics_output_path)


__all__ = [
    "FluxDetectionTrainer",
    "FluxDetectionValidator",
    "UltralyticsEvaluator",
    "UltralyticsTrainer",
]
