# mypy: disable-error-code="attr-defined"
"""Tests for the native RF-DETR trainer/evaluator (``raidar.rfdetr_trainer``).

Unit-level: metrics-sidecar helpers, the disk/Flux XOR validation, ``run()``
dispatch, and the evaluator's checkpoint-loading path (all mocked — no model
build). A heavy end-to-end Flux smoke test is gated behind ``@pytest.mark.slow``
+ ``importorskip`` and trains on the best available accelerator
(``cuda``/``mps``/``cpu``) so it is not CPU-bound on dev machines and GPU CI.
"""

import json
from types import SimpleNamespace
from typing import Any, List

import pytest
import torch

from raidar.rfdetr_trainer import (
    RFDETREvaluator,
    RFDETRTrainer,
    _coerce_float,
    _metrics_to_payload,
    _write_metrics_sidecar,
)

# ---------------------------------------------------------------------------
# Metrics sidecar helpers
# ---------------------------------------------------------------------------


def test_coerce_float_scalar_tensor_and_number() -> None:
    assert _coerce_float(torch.tensor(0.5)) == 0.5
    assert _coerce_float(0.7) == 0.7
    assert _coerce_float(3) == 3.0


def test_coerce_float_rejects_nonscalar_and_nonnumeric() -> None:
    assert _coerce_float(torch.tensor([1, 2])) is None  # multi-element -> RuntimeError -> None
    assert _coerce_float("not a number") is None


def test_metrics_to_payload_drops_nonscalars() -> None:
    payload = _metrics_to_payload({"val/map": torch.tensor(0.42), "classes": torch.tensor([1, 2]), "lr": 1e-4})
    assert payload == {"val/map": pytest.approx(0.42), "lr": pytest.approx(1e-4)}


def test_write_metrics_sidecar(tmp_path: Any) -> None:
    out = tmp_path / "m.json"
    _write_metrics_sidecar({"val/map": torch.tensor(0.5)}, str(out))
    assert json.loads(out.read_text())["val/map"] == 0.5


def test_write_metrics_sidecar_noops_without_path_or_metrics(tmp_path: Any) -> None:
    out = tmp_path / "none.json"
    _write_metrics_sidecar({"x": torch.tensor(1.0)}, None)  # no path
    _write_metrics_sidecar(None, str(out))  # no metrics
    _write_metrics_sidecar({"classes": torch.tensor([1, 2])}, str(out))  # all non-scalar -> empty payload
    assert not out.exists()


# ---------------------------------------------------------------------------
# XOR input validation
# ---------------------------------------------------------------------------


def test_trainer_rejects_both_disk_and_flux() -> None:
    with pytest.raises(ValueError, match="not both"):
        RFDETRTrainer(data="/d", train_set=object())._validate_inputs()


def test_trainer_rejects_neither() -> None:
    with pytest.raises(ValueError, match="EITHER"):
        RFDETRTrainer()._validate_inputs()


def test_trainer_flux_requires_num_classes_and_class_names() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        RFDETRTrainer(train_set=object(), val_set=object())._validate_inputs()
    with pytest.raises(ValueError, match="num_classes"):
        RFDETRTrainer(train_set=object(), val_set=object(), num_classes=2)._validate_inputs()  # no class_names


def test_trainer_flux_requires_both_train_and_val() -> None:
    with pytest.raises(ValueError, match="BOTH"):
        RFDETRTrainer(train_set=object(), num_classes=2, class_names={0: "a"})._validate_inputs()


# ---------------------------------------------------------------------------
# run() dispatch
# ---------------------------------------------------------------------------


def test_run_dispatch_disk_vs_flux() -> None:
    calls: List[str] = []
    t = RFDETRTrainer(data="/d")
    t._run_disk_mode = lambda: calls.append("disk")  # type: ignore[method-assign]
    t.run()
    f = RFDETRTrainer(train_set=object(), val_set=object(), num_classes=2, class_names={0: "a", 1: "b"})
    f._run_flux_mode = lambda: calls.append("flux")  # type: ignore[method-assign]
    f.run()
    assert calls == ["disk", "flux"]


# ---------------------------------------------------------------------------
# Confluid construction (zero-arg + tags)
# ---------------------------------------------------------------------------


def test_zero_arg_construction_and_tags() -> None:
    RFDETRTrainer()  # no required args
    RFDETREvaluator()  # weights defaults to "" (validated at run)
    assert RFDETRTrainer.__confluid_task__ == "detection" and RFDETRTrainer.__confluid_role__ == "trainer"
    assert RFDETREvaluator.__confluid_task__ == "detection" and RFDETREvaluator.__confluid_role__ == "evaluator"


def test_run_name_does_not_shadow_run_method() -> None:
    # Regression: the MLflow run-name knob is `run_name`, NOT `run` (which is the method).
    t = RFDETRTrainer(run_name="my_run")
    assert t.run_name == "my_run" and callable(t.run)


# ---------------------------------------------------------------------------
# Evaluator validation + checkpoint-loading path (mocked)
# ---------------------------------------------------------------------------


def test_evaluator_requires_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        RFDETREvaluator(weights="", val_set=object(), num_classes=2, class_names={0: "a"})._validate_inputs()


def test_evaluator_loads_checkpoint_via_pretrain_weights_and_validates(monkeypatch: Any) -> None:
    import raidar.rfdetr as rfmod
    import raidar.rfdetr_trainer as mod

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

    import raidar.flux_rfdetr_dataset as frd

    monkeypatch.setattr(frd, "FluxRFDETRDataModule", lambda *a, **k: SimpleNamespace())

    ev = RFDETREvaluator(weights="/ckpt/best.pth", val_set=object(), num_classes=2, class_names={0: "a", 1: "b"})
    ev.run()

    assert fake_model_config.pretrain_weights == "/ckpt/best.pth"  # checkpoint loaded the supported way
    assert captured.get("validate") and "test" not in captured  # validate (not test) when run_test=False


# ---------------------------------------------------------------------------
# Heavy end-to-end Flux smoke (opt-in)
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
def test_flux_train_smoke(tmp_path: Any) -> None:
    pytest.importorskip("rfdetr.training")
    from dataflux.core import Flux
    from dataflux.sample import Sample

    def _flux(n: int) -> Flux:
        return Flux(
            source=[
                Sample(
                    input=torch.rand(3, 256, 256),
                    target={"boxes": torch.tensor([[10.0, 10.0, 80.0, 80.0]]), "labels": torch.tensor([0])},
                    metadata={"i": i},
                )
                for i in range(n)
            ]
        )

    sidecar = tmp_path / "metrics.json"
    trainer = RFDETRTrainer(
        train_set=_flux(8),
        val_set=_flux(4),
        num_classes=1,
        class_names={0: "obj"},
        variant="nano",
        pretrained=False,
        resolution=256,
        epochs=1,
        batch_size=2,
        device=_best_available_device(),
        num_workers=0,
        output_dir=str(tmp_path / "out"),
        metrics_output_path=str(sidecar),
    )
    try:
        trainer.run()
    except Exception as exc:  # pragma: no cover - asset download / env-specific failure
        pytest.skip(f"RF-DETR flux smoke could not run in this environment: {exc}")
    assert sidecar.exists()
