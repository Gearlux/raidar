"""RFDETRDetector — the ``rfdetr`` framework: RF-DETR's OWN PyTorch-Lightning stack, merged.

One engine backend beside the others (`lightning` / `torch` / `huggingface` / …): train and eval
are ONE class with a ``task`` knob, dispatched through the ``@entrypoint`` markers. Because
RF-DETR is Apache-2.0 it lives in core raidar (the AGPL ultralytics counterpart stays in its
sibling package).

**Deliberately LIGHT — RF-DETR's machinery is reused, never restated.** The class owns only the
task dispatch, the dataset/tracking slots and a handful of ergonomic knobs; EVERY other RF-DETR
parameter travels through the ``train_config:`` pass-through mapping, which is merged OVER the
derived floor and handed verbatim to ``rf.get_train_config(**…)`` / ``rf.train(**…)`` — RF-DETR's
own pydantic ``TrainConfig`` validates it, so a knob its releases add is immediately configurable
with no raidar edit. A key the config pins WINS; the floor is a floor (the workspace's
``TrainingArguments`` precedent).

Two input modes (XOR):

* **Disk mode** — ``data: str | Path`` points at a COCO/YOLO/Roboflow dataset directory; RF-DETR's
  native ``.train()`` runs verbatim.
* **Stream mode** — ``train_set`` / ``val_set`` are :class:`recordstream.Stream` pipelines
  (+ ``num_classes`` + ``class_names``); ``RFDETR.train()``'s tail is reimplemented with
  :class:`raidar.stream_rfdetr_dataset.StreamRFDETRDataModule`, so the Stream feeds RF-DETR's
  native loop with NO disk materialization.

**Label convention (footgun):** RF-DETR labels are **0-indexed ``0..K-1``** with background
reserved at index ``K`` — the OPPOSITE of torchvision's background-at-0. Wire a 0-indexed target
op (``CocoToTorchVisionDetection(label_offset=0)``), NOT the ``+1`` shift the torchvision path uses.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from confluid import Partial, configurable, flow, output
from loggair import get_logger
from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, as_tracker_list
from matrainer.tracking.provenance import capture_log_file, log_run_context
from recordstream import RecordSource, ensure_materialized
from recordstream.runnable import ProgressReporting, TorchRunner, entrypoint, run_entrypoint

logger = get_logger(__name__)

#: Which entry point :meth:`RFDETRDetector.run` dispatches to. No ``predict``: RF-DETR's loop has
#: no prediction-sink path — prediction runs through the lightning backend + ``RFDETRModel``.
RFDETRTask = Literal["fit", "evaluate", "test"]

_VariantName = Literal["nano", "small", "medium", "large"]

#: Built when ``num_classes`` is omitted in disk mode — RF-DETR's ``train()`` re-aligns the head
#: to the dataset's category count, so the initial value is a placeholder the native loop
#: overwrites from ``dataset_dir``.
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


def _scalar_payload(metrics: Any) -> Dict[str, float]:
    """A metrics mapping reduced to its scalar-numeric entries, as plain floats."""
    payload: Dict[str, float] = {}
    for key, value in dict(metrics or {}).items():
        coerced = _coerce_float(value)
        if coerced is not None:
            payload[str(key)] = coerced
    return payload


def _last_csv_metrics(output_dir: str) -> Optional[Dict[str, Any]]:
    """Disk mode's final metrics: the last non-empty row of RF-DETR's ``metrics.csv``.

    RF-DETR's CSVLogger writes it under ``output_dir``; a missing file is logged, not fatal —
    the sidecar is then simply skipped.
    """
    candidates = sorted(Path(output_dir).rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        logger.info(f"No metrics.csv found under {output_dir} — metrics sidecar skipped.")
        return None
    with candidates[0].open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if any(v not in ("", None) for v in row.values())]
    return dict(rows[-1]) if rows else None


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


@configurable(task="detection", role="trainer", framework="rfdetr")
class RFDETRDetector(TorchRunner, ProgressReporting):
    """RF-DETR's native detection loop — train, evaluate, or test via ``task``.

    Args:
        data: Disk-mode dataset directory (COCO/YOLO/Roboflow). Mutually exclusive with
            ``train_set``/``val_set``.
        variant: ``"nano"`` (default) / ``"small"`` / ``"medium"`` / ``"large"``.
        pretrained: Start from RF-DETR's COCO-pretrained weights on a fit.
        weights: Trained checkpoint to EVALUATE (``task="evaluate"``/``"test"``) — loaded via
            RF-DETR's own ``model_config.pretrain_weights`` path, which handles head re-sizing.
        resolution: Square input size; ``None`` uses the variant default.
        task: Entry point :meth:`run` dispatches to (``fit`` / ``evaluate`` / ``test``).
        epochs: Training epochs.
        batch_size: Per-device batch size.
        device: Torch device string (``"cpu"``/``"cuda"``/``"cuda:1"``/``"mps"``); mapped to PTL
            accelerator/devices through RF-DETR's own resolver.
        output_dir: Where RF-DETR writes checkpoints / metrics.csv / logs.
        run_name: Run label forwarded to RF-DETR's ``run`` knob. NOTE: named ``run_name`` — not
            ``run`` — so it does not shadow the :meth:`run` method.
        experiment_name: Logical run label for the workspace tracking record.
        train_set / val_set / test_set: Stream-mode datasets (``test`` falls back to ``val``).
        num_classes: Stream-mode foreground class count K (labels are 0-indexed ``0..K-1``; the
            head emits K+1 with background at K).
        class_names: Stream-mode ``{class_id: name}`` (required in Stream mode).
        train_config: **The pass-through to EVERY other RF-DETR parameter** — ``lr``,
            ``lr_encoder``, ``weight_decay``, ``num_workers``, ``early_stopping`` and its knobs,
            ``eval_interval``, ``run_test``, ``dataset_file``, ``mlflow``/``tensorboard``/
            ``wandb``, ``resume``, ``aug_config``, gradient accumulation, EMA knobs, … — merged
            OVER the floor this class derives and handed verbatim to
            ``rf.get_train_config(**…)`` / ``rf.train(**…)``, where RF-DETR's own pydantic
            ``TrainConfig`` validates it. A key pinned here WINS over the floor.
        trackers: matrainer trackers — one or a list. RF-DETR's loop is upstream-owned, so they
            receive the run's PROVENANCE (git state, resolved config, hyperparameters, manifest,
            log file); RF-DETR's own logger knobs (via ``train_config``) keep the per-step curves.
        metric_names: How the recorded final metrics are named — see
            :class:`~matrainer.metric_names.MetricNames` (RF-DETR's loop IS PyTorch Lightning,
            so the translation declares ``source="lightning"``).
        metrics_sidecar_path: Optional JSON sidecar for the final metrics
            (written through ``matrainer.artifacts.write_metrics_sidecar``).
    """

    def __init__(
        self,
        data: Any = None,
        variant: _VariantName = "nano",
        pretrained: bool = True,
        weights: Optional[str] = None,
        resolution: Optional[int] = None,
        task: RFDETRTask = "fit",
        epochs: int = 100,
        batch_size: int = 4,
        device: Optional[str] = None,
        output_dir: str = "output",
        run_name: Optional[str] = None,
        experiment_name: str = "rfdetr_run",
        train_set: Optional[Partial[RecordSource]] = None,
        val_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        trackers: Optional[Union[Partial[Tracker], List[Partial[Tracker]]]] = None,
        metric_names: Optional[Partial[MetricNames]] = None,
        metrics_sidecar_path: Optional[str] = None,
    ) -> None:
        # Store config ONLY — no functional work (zero-arg / lazy-init mandate).
        self.data = data
        self.variant = variant
        self.pretrained = pretrained
        self.weights = weights
        self.resolution = resolution
        self.task: RFDETRTask = task
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.output_dir = output_dir
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.experiment_name = experiment_name
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_names = dict(class_names) if class_names else None
        self.train_config = dict(train_config) if train_config else {}
        # Stored VERBATIM (lazy-init mandate); `_live_trackers()` normalizes + flows at run time.
        self.trackers: Optional[Union[Partial[Tracker], List[Partial[Tracker]]]] = trackers
        self.metric_names: Optional[Partial[MetricNames]] = metric_names
        self.metrics_sidecar_path = metrics_sidecar_path
        self._computed_metrics: Optional[Dict[str, float]] = None

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test are inference-only."""
        return self.task == "fit"

    # ---- run dispatch ------------------------------------------------------- #
    def run(self) -> None:
        # The `@entrypoint` markers below ARE the dispatch table — a hand-written
        # {task: method} dict restates them and drifts (recordstream `docs/runnable.md`).
        run_entrypoint(self, self.task)

    # ---- input validation (lazy — at run time, per task) --------------------- #
    def _stream_wired(self) -> bool:
        return self.train_set is not None or self.val_set is not None

    def _validate_fit_inputs(self) -> None:
        stream = self._stream_wired()
        if self.data is not None and stream:
            raise ValueError(
                "RFDETRDetector: pass EITHER `data` (disk dir) OR `train_set`+`val_set` (Stream) — not both."
            )
        if self.data is None and not stream:
            raise ValueError("RFDETRDetector: must supply EITHER `data` OR `train_set`+`val_set`.")
        if stream:
            if self.train_set is None or self.val_set is None:
                raise ValueError("RFDETRDetector Stream mode requires BOTH `train_set` and `val_set`.")
            if self.num_classes is None or not self.class_names:
                raise ValueError("RFDETRDetector Stream mode requires `num_classes` and `class_names` (0-indexed).")

    def _validate_eval_inputs(self) -> None:
        if not self.weights:
            raise ValueError("RFDETRDetector: `weights` (trained checkpoint path) is required for evaluate/test.")
        stream = self.val_set is not None
        if self.data is not None and stream:
            raise ValueError("RFDETRDetector: pass EITHER `data` (disk dir) OR `val_set` (Stream) — not both.")
        if self.data is None and not stream:
            raise ValueError("RFDETRDetector: must supply EITHER `data` OR `val_set`.")
        if stream and (self.num_classes is None or not self.class_names):
            raise ValueError("RFDETRDetector Stream mode requires `num_classes` and `class_names` (0-indexed).")

    # ---- the RF-DETR config: derived floor + verbatim pass-through ----------- #
    def _config_kwargs(self, dataset_dir: str, run_test: bool = False) -> Dict[str, Any]:
        """The kwargs handed to ``rf.get_train_config(**…)`` / ``rf.train(**…)``.

        This class derives only the FLOOR — the handful of values it owns knobs for — and then
        merges ``train_config`` OVER it, verbatim. RF-DETR's own pydantic ``TrainConfig``
        validates the result, so every parameter it accepts (today's and any a release adds) is
        reachable from YAML without a raidar edit, and a pinned key always wins.
        """
        floor: Dict[str, Any] = dict(
            dataset_dir=dataset_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            output_dir=self.output_dir,
            run=self.run_name or self.experiment_name,
            run_test=run_test,
        )
        names = _ordered_class_names(self.class_names)
        if names:
            floor["class_names"] = names
        floor.update(self.train_config)
        return floor

    # ---- the entry points ---------------------------------------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    def fit(self) -> None:
        logger.info(f"rfdetr training: {self.experiment_name} ({self.run_name})")
        self._validate_fit_inputs()
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the hyperparameters
        # and the resolved config come from matrainer's ONE derivation. RF-DETR's own loop owns
        # the per-step metric CURVES (its logger knobs ride `train_config`); the trackers carry
        # the run RECORD — git state, config, manifest, log file — AND the final scores.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        if self.data is not None:
            self._fit_disk_mode()
        else:
            self._fit_stream_mode()
        self._log_final_metrics(trackers)
        write_metrics_sidecar(
            self.metrics_sidecar_path, self._computed_metrics or {}, self.experiment_name, self.run_name
        )
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every tracker.
        capture_log_file(trackers, manifest=manifest, status="success")

    @entrypoint("evaluate", role="evaluator")
    def evaluate(self) -> None:
        self._run_eval(run_test=False)

    @entrypoint("test", role="evaluator", primary=True)
    def test(self) -> None:
        self._run_eval(run_test=True)

    # ---- tracking helpers ---------------------------------------------------- #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot."""
        self.trackers = [flow(t) for t in as_tracker_list(self.trackers) or []]
        return cast(List[Tracker], self.trackers)

    def _record_metrics(self, metrics: Any) -> None:
        """Coerce + translate one run's final metrics and remember them for the sidecar/@output.

        The names come out of RF-DETR's PTL loop, so the translation declares
        ``source="lightning"`` — only the caller knows which engine spelled the keys.
        """
        names = as_metric_names(self.metric_names)
        self._computed_metrics = names(_scalar_payload(metrics), source="lightning")

    def _log_final_metrics(self, trackers: Sequence[Tracker]) -> None:
        """Put this run's final scores on the trackers, not only in the sidecar.

        RF-DETR owns the per-step CURVES — its own logger knobs ride ``train_config``, and this
        class deliberately does not duplicate them — but the run's final metrics are computed HERE
        (:meth:`_record_metrics`), so without this call a tracked run records its provenance and
        its log file while its ``mAP`` never reaches the tracker at all: the score is on disk, in
        the sidecar, and invisible in every UI and every cross-backend comparison.

        The sibling restoration project hit exactly this on its ``transformers.Trainer`` backend,
        where a backend sweep showed a completely BLANK row beside a log printing the score.
        """
        if not trackers or not self._computed_metrics:
            return
        for tracker in trackers:
            tracker.log_metrics(dict(self._computed_metrics))

    def _prepare_stream(self, source: Optional[Partial[RecordSource]]) -> Optional[Any]:
        """Flow a wired Stream slot LIVE and materialize it in THIS process.

        The CLI's manual materialization leaves a nested ``!class:`` dataset as a deferred marker
        (measured: RF-DETR's datamodule then dies with ``object of type 'Class' has no len()``),
        and ``ensure_materialized`` is the fork-safety guard — RF-DETR's loop runs DataLoader
        workers, and a lazy source first touched in a FORKED child builds through ``_scproxy`` on
        macOS, a SIGSEGV with no traceback.
        """
        if source is None:
            return None
        return ensure_materialized(flow(source))

    # ---- fit: the two input modes ------------------------------------------- #
    def _build_rf(self, num_classes: int, pretrained: bool) -> Any:
        from raidar.rfdetr import _build_rfdetr

        return _build_rfdetr(
            self.variant,
            num_classes=num_classes,
            pretrained_detector=pretrained,
            resolution=self.resolution,
        )

    def _seed_before_the_model_is_built(self) -> None:
        """Apply the configured seed HERE, because RF-DETR applies it too late to cover the model.

        ``RFDETRModelModule.on_fit_start`` calls ``seed_everything(train_config.seed + rank)`` —
        which is after BOTH fit paths have already run ``_build_rf``, i.e. after the network and
        its freshly-sized detection head exist. So the upstream call seeds the data order and the
        augmentation draws while the weights it trains were drawn from an unseeded RNG.

        Measured 2026-08-06: with ``train_config: {seed: 7}`` pinned and nothing else changed, two
        runs still gave mAP50:95 0.3595 / 0.3406 on mps AND 0.3777 / 0.3453 on cpu — so the device
        was not the cause (that hypothesis was tested and rejected); the model init was.

        ``seed_everything(..., workers=True)`` matches upstream's own call exactly, including the
        worker seeding. No configured seed = no call, so an unpinned run behaves as it always did.
        """
        seed = self.train_config.get("seed")
        if seed is None:
            return
        from pytorch_lightning import seed_everything

        seed_everything(int(seed), workers=True)

    def _fit_disk_mode(self) -> None:
        """Hand off to RF-DETR's native ``.train()`` over an on-disk dataset directory — verbatim."""
        self._seed_before_the_model_is_built()
        rf = self._build_rf(
            self.num_classes if self.num_classes is not None else _DEFAULT_DISK_NUM_CLASSES, self.pretrained
        )
        kwargs = self._config_kwargs(str(self.data))
        kwargs.setdefault("device", self.device)
        if self.resolution is not None:
            kwargs.setdefault("resolution", self.resolution)
        rf.train(**kwargs)
        final = _last_csv_metrics(self.output_dir)
        if final is not None:
            self._record_metrics(final)

    def _fit_stream_mode(self) -> None:
        """Reimplement ``RFDETR.train()``'s tail (rfdetr/detr.py) with our StreamRFDETRDataModule.

        NO disk: our datamodule's ``setup()`` bypasses ``build_dataset``. ``dataset_dir="."``
        is a placeholder satisfying ``TrainConfig``'s required field/validator; it is never
        read (and ``train()``'s dataset-derived num_classes alignment never fires, because we
        pass ``num_classes`` explicitly and call the tail directly).
        """
        from rfdetr.training import RFDETRModelModule, build_trainer

        from raidar.stream_rfdetr_dataset import StreamRFDETRDataModule

        self._seed_before_the_model_is_built()

        names = _ordered_class_names(self.class_names)
        assert self.num_classes is not None and names is not None  # _validate_fit_inputs guarantees
        self.train_set = self._prepare_stream(self.train_set)
        self.val_set = self._prepare_stream(self.val_set)
        self.test_set = self._prepare_stream(self.test_set)
        rf = self._build_rf(self.num_classes, self.pretrained)
        cfg = rf.get_train_config(**self._config_kwargs("."))

        module = RFDETRModelModule(rf.model_config, cfg)
        datamodule = StreamRFDETRDataModule(
            rf.model_config,
            cfg,
            train_stream=self.train_set,
            val_stream=self.val_set,
            test_stream=self.test_set,
            class_names=names,
        )
        trainer = build_trainer(cfg, rf.model_config, **_trainer_device_kwargs(self.device))
        trainer.fit(module, datamodule, ckpt_path=cfg.resume or None)
        rf.model.model = module.model  # sync trained weights back for predict/export parity
        self._record_metrics(dict(trainer.callback_metrics))

    # ---- evaluate / test ----------------------------------------------------- #
    def _run_eval(self, run_test: bool) -> None:
        """Run RF-DETR's validation/test loop over a trained checkpoint.

        RF-DETR has no standalone ``.evaluate()``; this drives its lower-level
        ``RFDETRModelModule`` + ``build_trainer`` and calls ``trainer.validate``/``.test`` (so the
        COCO-mAP ``COCOEvalCallback`` produces the metrics). The checkpoint is loaded the
        supported way — ``model_config.pretrain_weights = weights`` — so RF-DETR's weight loader
        handles head re-sizing.
        """
        logger.info(f"rfdetr evaluating ({'test' if run_test else 'validate'}): {self.experiment_name}")
        self._validate_eval_inputs()
        # Same wiring as `fit()`, and it was missing entirely: an evaluate run produced NO tracker
        # record at all — no provenance, no scores, and a manifest left `running` for matrainer's
        # atexit hook to relabel `interrupted` on a run that succeeded.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        if self.data is not None:
            self._eval_disk_mode(run_test)
        else:
            self._eval_stream_mode(run_test)
        self._log_final_metrics(trackers)
        write_metrics_sidecar(
            self.metrics_sidecar_path, self._computed_metrics or {}, self.experiment_name, self.run_name
        )
        capture_log_file(trackers, manifest=manifest, status="success")

    def _prepare_eval(self, dataset_dir: str, run_test: bool) -> Any:
        """Build the high-level RF-DETR + a checkpoint-loaded model_config + the eval TrainConfig."""
        num_classes = self.num_classes if self.num_classes is not None else _DEFAULT_DISK_NUM_CLASSES
        rf = self._build_rf(num_classes, pretrained=False)
        # The module rebuilds the model FROM model_config, so point it at the trained checkpoint
        # here (the supported _load_pretrain_weights path that handles head re-sizing).
        rf.model_config.pretrain_weights = self.weights
        cfg = rf.get_train_config(**self._config_kwargs(dataset_dir, run_test=run_test))
        return rf, cfg

    def _evaluate(self, module: Any, datamodule: Any, model_config: Any, cfg: Any, run_test: bool) -> None:
        from rfdetr.training import build_trainer

        trainer = build_trainer(cfg, model_config, **_trainer_device_kwargs(self.device))
        results = trainer.test(module, datamodule) if run_test else trainer.validate(module, datamodule)
        metrics = results[0] if results else dict(trainer.callback_metrics)
        self._record_metrics(metrics)

    def _eval_disk_mode(self, run_test: bool) -> None:
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule

        rf, cfg = self._prepare_eval(str(self.data), run_test)
        module = RFDETRModelModule(rf.model_config, cfg)
        datamodule = RFDETRDataModule(rf.model_config, cfg)
        self._evaluate(module, datamodule, rf.model_config, cfg, run_test)

    def _eval_stream_mode(self, run_test: bool) -> None:
        from rfdetr.training import RFDETRModelModule

        from raidar.stream_rfdetr_dataset import StreamRFDETRDataModule

        self.val_set = self._prepare_stream(self.val_set)
        names = _ordered_class_names(self.class_names)
        rf, cfg = self._prepare_eval(".", run_test)  # placeholder dir — StreamRFDETRDataModule bypasses build_dataset
        module = RFDETRModelModule(rf.model_config, cfg)
        datamodule = StreamRFDETRDataModule(
            rf.model_config,
            cfg,
            val_stream=self.val_set,
            test_stream=self.val_set if run_test else None,
            class_names=names,
        )
        self._evaluate(module, datamodule, rf.model_config, cfg, run_test)

    # ---- @outputs ------------------------------------------------------------ #
    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if self._computed_metrics is None:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)


__all__ = ["RFDETRDetector", "RFDETRTask"]
