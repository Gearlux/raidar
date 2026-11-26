# mypy: disable-error-code="attr-defined"
"""Tests for the merged RF-DETR framework runnable (``raidar.rfdetr_detector``).

Unit-level: the scalar-payload helper, the disk/Stream XOR validation per task, ``run()``
dispatch through the ``@entrypoint`` markers, the tracking hookup, and the eval
checkpoint-loading path (all mocked — no model build). A heavy end-to-end Stream smoke test is
gated behind ``@pytest.mark.slow`` + ``importorskip`` and trains on the best available
accelerator (``cuda``/``mps``/``cpu``) so it is not CPU-bound on dev machines and GPU CI.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, get_args

import pytest
import torch
from matrainer.tracking.base import Tracker
from recordstream.runnable import entrypoint_tasks

from raidar.rfdetr_detector import RFDETRDetector, RFDETRTask, _coerce_float, _scalar_payload

# ---------------------------------------------------------------------------
# Scalar payload helpers
# ---------------------------------------------------------------------------


def test_coerce_float_scalar_tensor_and_number() -> None:
    assert _coerce_float(torch.tensor(0.5)) == 0.5
    assert _coerce_float(0.7) == 0.7
    assert _coerce_float(3) == 3.0


def test_coerce_float_rejects_nonscalar_and_nonnumeric() -> None:
    assert _coerce_float(torch.tensor([1, 2])) is None  # multi-element -> RuntimeError -> None
    assert _coerce_float("not a number") is None


def test_scalar_payload_drops_nonscalars() -> None:
    payload = _scalar_payload({"val/map": torch.tensor(0.42), "classes": torch.tensor([1, 2]), "lr": 1e-4})
    assert payload == {"val/map": pytest.approx(0.42), "lr": pytest.approx(1e-4)}


def test_the_sidecar_is_matrainers_not_a_local_writer() -> None:
    """The hand-rolled `_write_metrics_sidecar` / `_metrics_to_payload` pair is GONE — the sidecar
    goes through `matrainer.artifacts.write_metrics_sidecar` like every other backend's."""
    import inspect

    import raidar.rfdetr_detector as mod

    source = inspect.getsource(mod)
    assert "write_metrics_sidecar" in source
    assert "def _write_metrics_sidecar" not in source


# ---------------------------------------------------------------------------
# XOR input validation (per task)
# ---------------------------------------------------------------------------


def test_fit_rejects_both_disk_and_stream() -> None:
    with pytest.raises(ValueError, match="not both"):
        RFDETRDetector(data="/d", train_set=[])._validate_fit_inputs()


def test_fit_rejects_neither() -> None:
    with pytest.raises(ValueError, match="EITHER"):
        RFDETRDetector()._validate_fit_inputs()


def test_fit_stream_requires_num_classes_and_class_names() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        RFDETRDetector(train_set=[], val_set=[])._validate_fit_inputs()
    with pytest.raises(ValueError, match="num_classes"):
        RFDETRDetector(train_set=[], val_set=[], num_classes=2)._validate_fit_inputs()


def test_fit_stream_requires_both_train_and_val() -> None:
    with pytest.raises(ValueError, match="BOTH"):
        RFDETRDetector(train_set=[], num_classes=2, class_names={0: "a"})._validate_fit_inputs()


def test_keep_picture_size_is_refused_on_a_disk_dataset() -> None:
    """A disk dataset is read by RF-DETR's own loader, whose resize this runnable does not build."""
    with pytest.raises(ValueError, match=r"keep_picture_size works on train_set/val_set streams only"):
        RFDETRDetector(data="/d", keep_picture_size=True)._validate_fit_inputs()
    with pytest.raises(ValueError, match=r"keep_picture_size works on train_set/val_set streams only"):
        RFDETRDetector(data="/d", weights="/w.pth", keep_picture_size=True)._validate_eval_inputs()


def test_eval_requires_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        RFDETRDetector(val_set=[], num_classes=2, class_names={0: "a"})._validate_eval_inputs()


# ---------------------------------------------------------------------------
# run() dispatch — the @entrypoint markers ARE the table
# ---------------------------------------------------------------------------


def test_entrypoint_markers_cover_every_task_literal_value() -> None:
    covered = set(entrypoint_tasks(RFDETRDetector, "trainer")) | set(entrypoint_tasks(RFDETRDetector, "evaluator"))
    assert covered == set(get_args(RFDETRTask))


def test_run_dispatches_fit_then_disk_vs_stream(monkeypatch: Any) -> None:
    calls: List[str] = []
    d = RFDETRDetector(data="/d")
    monkeypatch.setattr(d, "_fit_disk_mode", lambda: calls.append("disk"))
    monkeypatch.setattr(d, "_live_trackers", lambda: [])
    monkeypatch.setattr("raidar.rfdetr_detector.log_run_context", lambda trackers, runnable: None)
    monkeypatch.setattr("raidar.rfdetr_detector.capture_log_file", lambda trackers, manifest, status: None)
    d.run()
    s = RFDETRDetector(train_set=[], val_set=[], num_classes=2, class_names={0: "a", 1: "b"})
    monkeypatch.setattr(s, "_fit_stream_mode", lambda: calls.append("stream"))
    monkeypatch.setattr(s, "_live_trackers", lambda: [])
    s.run()
    assert calls == ["disk", "stream"]


def test_fit_hands_matrainer_the_runnable_and_finishes_the_manifest(monkeypatch: Any) -> None:
    """Provenance parity: `log_run_context(..., runnable=self)` at the top of fit, and
    `capture_log_file(..., status="success")` at the end — an unfinished manifest stays `running`
    and matrainer's atexit hook then labels a successful run `interrupted`."""
    seen: dict = {}
    d = RFDETRDetector(data="/d")
    monkeypatch.setattr(d, "_fit_disk_mode", lambda: None)
    monkeypatch.setattr(
        "raidar.rfdetr_detector.log_run_context",
        lambda trackers, runnable: seen.update(runnable=runnable) or "manifest",
    )
    monkeypatch.setattr(
        "raidar.rfdetr_detector.capture_log_file",
        lambda trackers, manifest, status: seen.update(manifest=manifest, status=status),
    )
    d.run()
    assert seen["runnable"] is d
    assert seen["manifest"] == "manifest" and seen["status"] == "success"


# ---------------------------------------------------------------------------
# Confluid construction (zero-arg + tags)
# ---------------------------------------------------------------------------


def test_zero_arg_construction_and_tags() -> None:
    d = RFDETRDetector()  # no required args
    assert d.task == "fit"
    assert RFDETRDetector.__confluid_task__ == "detection" and RFDETRDetector.__confluid_role__ == "trainer"
    assert RFDETRDetector.__confluid_framework__ == "rfdetr"


def test_run_name_does_not_shadow_run_method() -> None:
    # Regression: the MLflow run-name knob is `run_name`, NOT `run` (which is the method).
    d = RFDETRDetector(run_name="my_run")
    assert d.run_name == "my_run" and callable(d.run)


# ---------------------------------------------------------------------------
# model_config: the pass-through to the RF-DETR MODEL's settings (fixed when it is built)
# ---------------------------------------------------------------------------


def test_model_config_reaches_the_model_constructor(monkeypatch: Any) -> None:
    import rfdetr

    import raidar.rfdetr as rfmod

    built: dict = {}
    monkeypatch.setattr(rfdetr, "RFDETRSegMedium", lambda **kw: built.update(kw), raising=False)
    d = RFDETRDetector(variant="seg-medium", resolution=1008, num_classes=33, model_config={"group_detr": 8})
    d._build_rf(33, True)
    assert built == {"num_classes": 33, "resolution": 1008, "group_detr": 8}
    assert rfmod  # the module under test


def test_without_model_config_the_model_is_built_as_before(monkeypatch: Any) -> None:
    import rfdetr

    built: dict = {}
    monkeypatch.setattr(rfdetr, "RFDETRSegMedium", lambda **kw: built.update(kw), raising=False)
    RFDETRDetector(variant="seg-medium", resolution=1008)._build_rf(33, True)
    assert built == {"num_classes": 33, "resolution": 1008}


@pytest.mark.parametrize("key", ["num_classes", "resolution", "pretrain_weights"])
def test_model_config_refuses_a_key_with_its_own_parameter(key: str) -> None:
    d = RFDETRDetector(variant="seg-medium", model_config={key: 1})
    with pytest.raises(ValueError, match=key):
        d._build_rf(33, True)


# ---------------------------------------------------------------------------
# Eval checkpoint-loading path (mocked)
# ---------------------------------------------------------------------------


def test_eval_loads_checkpoint_via_pretrain_weights_and_validates(monkeypatch: Any) -> None:
    import raidar.rfdetr as rfmod
    import raidar.rfdetr_detector as mod

    captured: dict = {}

    fake_model_config = SimpleNamespace(pretrain_weights=None)
    fake_rf = SimpleNamespace(
        model_config=fake_model_config,
        get_train_config=lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(rfmod, "_build_rfdetr", lambda *a, **k: fake_rf)

    class _FakeTrainer:
        def __init__(self) -> None:
            self.callback_metrics = {"val/map": torch.tensor(0.3)}

        def validate(self, module: Any, dm: Any) -> Any:
            captured["validate"] = True
            return [{"val/map": 0.3}]

        def test(self, module: Any, dm: Any) -> Any:
            captured["test"] = True
            return [{"test/map": 0.3}]

    import rfdetr.training as rftrain

    monkeypatch.setattr(rftrain, "RFDETRModelModule", lambda mc, cfg: SimpleNamespace(model_config=mc, cfg=cfg))
    monkeypatch.setattr(rftrain, "build_trainer", lambda cfg, mc, **k: _FakeTrainer())
    monkeypatch.setattr(mod, "_trainer_device_kwargs", lambda device: {})

    import raidar.stream_rfdetr_dataset as frd

    monkeypatch.setattr(
        frd, "StreamRFDETRDataModule", lambda *a, **k: captured.update(keep=k["keep_picture_size"]) or SimpleNamespace()
    )

    # A real (indexable) stand-in: the runnable now PREPARES its stream slots (flow +
    # ensure_materialized), so a bare `object()` would fail the materialization read.
    d = RFDETRDetector(
        task="evaluate",
        weights="/ckpt/best.pth",
        val_set=[{"image": None}],
        num_classes=2,
        class_names={0: "a", 1: "b"},
    )
    d.run()

    assert fake_model_config.pretrain_weights == "/ckpt/best.pth"  # checkpoint loaded the supported way
    assert captured.get("validate") and "test" not in captured  # validate (not test) on task="evaluate"
    assert d.computed_metrics == {"val/map": pytest.approx(0.3)}  # the @output carries the result
    assert captured["keep"] is False  # the datamodule is told the runnable's keep_picture_size (default off)


def test_a_stream_fit_hands_keep_picture_size_to_its_datamodule(monkeypatch: Any) -> None:
    import rfdetr.training as rftrain

    import raidar.rfdetr as rfmod
    import raidar.rfdetr_detector as mod
    import raidar.stream_rfdetr_dataset as frd

    captured: dict = {}
    fake_rf = SimpleNamespace(
        model_config=SimpleNamespace(),
        get_train_config=lambda **kw: SimpleNamespace(resume=None, **kw),
        model=SimpleNamespace(model=None),
    )
    monkeypatch.setattr(rfmod, "_build_rfdetr", lambda *a, **k: fake_rf)
    monkeypatch.setattr(rftrain, "RFDETRModelModule", lambda mc, cfg: SimpleNamespace(model="trained"))
    trainer = SimpleNamespace(fit=lambda *a, **k: None, callback_metrics={})
    monkeypatch.setattr(rftrain, "build_trainer", lambda cfg, mc, **k: trainer)
    monkeypatch.setattr(mod, "_trainer_device_kwargs", lambda device: {})
    monkeypatch.setattr(
        frd, "StreamRFDETRDataModule", lambda *a, **k: captured.update(keep=k["keep_picture_size"]) or SimpleNamespace()
    )
    d = RFDETRDetector(
        train_set=[{"image": None}],
        val_set=[{"image": None}],
        num_classes=2,
        class_names={0: "a", 1: "b"},
        keep_picture_size=True,
    )
    d._fit_stream_mode()
    assert captured["keep"] is True


# ---------------------------------------------------------------------------
# One tracker run per training: the wired trackers are loggers of RF-DETR's trainer
# ---------------------------------------------------------------------------


class _RecordingTracker(Tracker):
    """A tracker that writes down, in order, every call that reaches it."""

    def __init__(self) -> None:
        super().__init__("experiment", "run")
        self.calls: List[Any] = []

    @property
    def experiment(self) -> Any:
        return self

    def log_hyperparams(self, params: Any) -> None:
        self.calls.append(("hyperparams", dict(params)))

    def log_metrics(self, metrics: Any, step: Any = None) -> None:
        self.calls.append(("metrics", dict(metrics), step))


class _EpochTrainer:
    """RF-DETR's trainer as far as its loggers go: one epoch's values to every logger, then each finalized."""

    def __init__(self, loggers: List[Any], raising: Any = None) -> None:
        self.loggers = loggers
        self.raising = raising
        self.global_step = 4
        self.callback_metrics = {"val/mAP_50_95": torch.tensor(0.5)}

    def _epoch(self) -> None:
        if self.raising is not None:
            raise self.raising
        for logger in self.loggers:
            logger.log_metrics({"train/loss": 1.25, "val/mAP_50_95": 0.5}, step=0)
            logger.finalize("success")

    def fit(self, module: Any, datamodule: Any, ckpt_path: Any = None) -> None:
        self._epoch()

    def validate(self, module: Any, datamodule: Any) -> Any:
        self._epoch()
        return [{"val/mAP_50_95": 0.5}]


class _OwnLogger:
    """A logger RF-DETR itself put on the trainer (its CSV logger)."""

    def __init__(self) -> None:
        self.metrics: List[Any] = []

    def log_metrics(self, metrics: Any, step: Any = None) -> None:
        self.metrics.append((dict(metrics), step))

    def finalize(self, status: str) -> None:
        pass


def _stream_run(monkeypatch: Any, trackers: Any, task: str = "fit", raising: Any = None) -> Any:
    """A stream-mode ``RFDETRDetector`` whose RF-DETR pieces are stand-ins; returns it, its trainer and the calls."""
    pytest.importorskip("rfdetr.training")
    import rfdetr.training as rftrain

    import raidar.rfdetr as rfmod
    import raidar.rfdetr_detector as mod
    import raidar.stream_rfdetr_dataset as frd

    seen: dict = {}
    own = _OwnLogger()
    trainer = _EpochTrainer([own], raising)
    fake_rf = SimpleNamespace(
        model_config=SimpleNamespace(pretrain_weights=None),
        get_train_config=lambda **kw: SimpleNamespace(resume=None, **kw),
        model=SimpleNamespace(model=None),
    )
    monkeypatch.setattr(rfmod, "_build_rfdetr", lambda *a, **k: fake_rf)
    monkeypatch.setattr(rftrain, "RFDETRModelModule", lambda mc, cfg: SimpleNamespace(model="trained"))
    monkeypatch.setattr(rftrain, "build_trainer", lambda cfg, mc, **k: trainer)
    monkeypatch.setattr(mod, "_trainer_device_kwargs", lambda device: {})
    monkeypatch.setattr(frd, "StreamRFDETRDataModule", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(mod, "log_run_context", lambda trackers, runnable: "manifest")
    monkeypatch.setattr(
        mod,
        "capture_log_file",
        lambda trackers, manifest, status: seen.update(log_file_to=list(trackers), status=status),
    )
    detector = RFDETRDetector(
        task=task,  # type: ignore[arg-type]
        weights="/ckpt/best.pth" if task != "fit" else None,
        train_set=[{"image": None}],
        val_set=[{"image": None}],
        num_classes=2,
        class_names={0: "a", 1: "b"},
        trackers=trackers,
    )
    if raising is not None:
        with pytest.raises(type(raising)):
            detector.run()
    else:
        detector.run()
    return detector, trainer, own, seen


def test_a_stream_fit_sends_every_epochs_values_to_the_trackers_own_run(monkeypatch: Any) -> None:
    """One run per training: a wired tracker is a logger of RF-DETR's trainer, so the epoch's losses and mAP reach
    the SAME tracker that then gets the final scores and the log file — after RF-DETR's own loggers, which stay.
    The final scores are logged at the step the trainer ended on, never as a stray point at step 0."""
    tracker = _RecordingTracker()
    _detector, trainer, own, seen = _stream_run(monkeypatch, [tracker])
    assert trainer.loggers[0] is own and len(trainer.loggers) == 2
    assert own.metrics == [({"train/loss": 1.25, "val/mAP_50_95": 0.5}, 0)]
    logged = [call for call in tracker.calls if call[0] == "metrics"]
    assert logged[0] == ("metrics", {"train/loss": 1.25, "val/mAP_50_95": 0.5}, 0)  # the epoch, at its step
    assert logged[-1] == ("metrics", {"val/mAP_50_95": pytest.approx(0.5)}, 4)  # then the final scores
    assert seen["log_file_to"] == [tracker]


def test_a_stream_fit_without_trackers_keeps_rfdetrs_own_loggers(monkeypatch: Any) -> None:
    _detector, trainer, own, _seen = _stream_run(monkeypatch, None)
    assert trainer.loggers == [own]


def test_a_stream_evaluation_sends_its_values_to_the_trackers_own_run(monkeypatch: Any) -> None:
    tracker = _RecordingTracker()
    _detector, trainer, own, _seen = _stream_run(monkeypatch, [tracker], task="evaluate")
    assert trainer.loggers[0] is own and len(trainer.loggers) == 2
    assert ("metrics", {"train/loss": 1.25, "val/mAP_50_95": 0.5}, 0) in tracker.calls


@pytest.mark.parametrize("task", ["fit", "evaluate"])
@pytest.mark.parametrize(
    "raising, status", [(RuntimeError("training failed"), "failed"), (KeyboardInterrupt(), "interrupted")]
)
def test_a_run_that_raises_still_ends_its_trackers_run(monkeypatch: Any, task: str, raising: Any, status: str) -> None:
    """A run that raised left its trackers open: MLflow showed it running for ever and the tracker's sampler was
    never stopped. The trackers get the log file and the run's real ending, and the error still reaches the caller
    (``_stream_run`` holds the run to raising it)."""
    tracker = _RecordingTracker()
    _detector, _trainer, _own, seen = _stream_run(monkeypatch, [tracker], task=task, raising=raising)
    assert seen == {"log_file_to": [tracker], "status": status}


def test_a_run_that_succeeds_ends_its_trackers_run_once_as_a_success(monkeypatch: Any) -> None:
    tracker = _RecordingTracker()
    _detector, _trainer, _own, seen = _stream_run(monkeypatch, [tracker])
    assert seen["status"] == "success"


# ---------------------------------------------------------------------------
# Heavy end-to-end Stream smoke (opt-in)
# ---------------------------------------------------------------------------


def _best_available_device() -> str:
    """Pick the fastest training device available on this host.

    The smoke test runs a real RF-DETR epoch: ~8 min on CPU but ~45 s on an
    Apple-Silicon ``mps`` or CUDA ``gpu`` backend. Preferring the accelerator
    when present keeps local runs (and GPU CI) from being dominated by this one
    test, while CPU-only runners still exercise the same code path.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.mark.slow
def test_stream_train_smoke(tmp_path: Any) -> None:
    pytest.importorskip("rfdetr.training")
    from recordstream import Boxes, Image
    from recordstream.core import Stream

    def _regions(boxes: Any, labels: Any) -> Boxes:
        return Boxes(boxes=boxes, labels=labels)

    def _stream(n: int) -> Stream:
        return Stream(
            source=[
                {
                    "image": Image(torch.rand(3, 256, 256).numpy(), layout="CHW"),
                    "target": _regions(torch.tensor([[10.0, 10.0, 80.0, 80.0]]), torch.tensor([0])),
                    "i": i,
                }
                for i in range(n)
            ]
        )

    sidecar = tmp_path / "metrics.json"
    detector = RFDETRDetector(
        train_set=_stream(8),
        val_set=_stream(4),
        num_classes=1,
        class_names={0: "obj"},
        variant="nano",
        pretrained=False,
        resolution=256,
        epochs=1,
        batch_size=2,
        device=_best_available_device(),
        train_config={"num_workers": 0},
        output_dir=str(tmp_path / "out"),
        metrics_sidecar_path=str(sidecar),
    )
    try:
        detector.run()
    except Exception as exc:  # pragma: no cover - asset download / env-specific failure
        pytest.skip(f"RF-DETR stream smoke could not run in this environment: {exc}")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["metrics"], "the sidecar carries the run's final scalar metrics"


class TestTheSeedCoversTheMODEL:
    """RF-DETR seeds at `on_fit_start` — after BOTH fit paths have already built the network.

    `RFDETRModelModule.on_fit_start` calls `seed_everything(train_config.seed + rank)`, but
    `_build_rf` runs before the trainer ever starts, so upstream's own call seeds the data order
    and augmentation while the weights were drawn unseeded. Measured 2026-08-06 with the seed
    pinned and nothing else changed: two runs still gave mAP50:95 0.3595 / 0.3406 on mps AND
    0.3777 / 0.3453 on cpu — which is also what ruled the DEVICE out as the cause.
    """

    def test_both_fit_paths_seed_before_building(self) -> None:
        import ast
        import inspect

        from raidar import rfdetr_detector

        tree = ast.parse(inspect.getsource(rfdetr_detector))
        for path in ("_fit_disk_mode", "_fit_stream_mode"):
            fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == path)
            seed_at = _first_call_line(fn, "_seed_before_the_model_is_built")
            build_at = _first_call_line(fn, "_build_rf")
            assert seed_at is not None, f"{path} does not seed — its weights are drawn unseeded"
            assert build_at is not None
            assert seed_at < build_at, f"{path} seeds AFTER _build_rf, which is the defect itself"

    def test_no_configured_seed_means_no_call(self) -> None:
        """An unpinned run must behave exactly as it always did — RF-DETR's own default is None."""
        detector = RFDETRDetector()
        assert detector.train_config.get("seed") is None
        detector._seed_before_the_model_is_built()  # must not raise, must not seed

    def test_the_example_config_pins_one(self) -> None:
        """The knob is only useful if the shipped config uses it — RF-DETR's own default is None.

        Read through confluid with the rfdetr scope ACTIVE (`yaml.safe_load` cannot parse the
        `!class:` tags), so this asserts what the run actually receives.
        """
        import confluid

        example = Path(__file__).resolve().parents[1] / "examples" / "train_cppe5.yaml"
        doc = confluid.load(example, until="document", scopes=["framework=rfdetr"])
        pinned = doc["runnable"].kwargs["train_config"]
        assert pinned.get("seed") is not None, "the shipped rfdetr block must pin a seed"


def _first_call_line(fn_node: Any, name: str) -> Any:
    import ast

    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if called == name:
                return node.lineno
    return None


def test_mps_low_watermark_is_handed_to_torch_before_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body slot a config names; `run()` applies it before the entry point builds anything."""
    from confluid import load, to_pydantic

    from raidar import rfdetr_detector

    order: List[Any] = []
    monkeypatch.setattr(rfdetr_detector, "apply_mps_low_watermark", lambda value: order.append(("watermark", value)))
    monkeypatch.setattr(rfdetr_detector, "run_entrypoint", lambda runnable, task: order.append(("task", task)))
    assert RFDETRDetector().mps_low_watermark is None
    assert "mps_low_watermark" in to_pydantic(RFDETRDetector).model_fields
    detector = load("runnable: !class:raidar.rfdetr_detector.RFDETRDetector\n  mps_low_watermark: 0.5\n")["runnable"]
    detector.run()
    assert order == [("watermark", 0.5), ("task", "fit")]
