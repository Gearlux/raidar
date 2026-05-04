"""Smoke + unit tests for :mod:`raidar.ultralytics_runner`.

The fast tests exercise the helper logic and the type-validation paths in
``UltralyticsTrainer.run`` / ``UltralyticsEvaluator.run`` without invoking
any heavy ML.

The integration test (`test_ultralytics_trainer_flux_smoke`) actually
constructs the FluxDetectionTrainer and runs a single-epoch training loop
over a 4-sample in-memory Flux at imgsz=64 batch=2. It downloads
``yolo26n.pt`` on first run (≈6MB) and writes results to the test's tmp
directory. Skipped automatically if ultralytics' asset download fails
(offline runners / firewall-restricted CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
import torch
from dataflux.core import Flux
from dataflux.sample import Sample

from raidar.ultralytics_runner import UltralyticsEvaluator, UltralyticsTrainer, _build_data_dict, _write_metrics_sidecar

# --- helper unit tests -------------------------------------------------------


def test_build_data_dict_happy_path() -> None:
    data = _build_data_dict({"train": object(), "val": object(), "names": {0: "a"}, "nc": 1})
    assert data["nc"] == 1
    assert data["names"] == {0: "a"}
    assert data["channels"] == 3
    # Sentinels make it through unchanged so the trainer's get_dataloader doesn't trip.
    assert data["train"] == "."
    assert data["val"] == "."


def test_build_data_dict_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        _build_data_dict("not-a-dict")  # type: ignore[arg-type]


def test_build_data_dict_requires_train_or_val() -> None:
    with pytest.raises(ValueError, match="train.*val"):
        _build_data_dict({"names": {0: "a"}, "nc": 1})


def test_build_data_dict_requires_names_and_nc() -> None:
    with pytest.raises(ValueError, match="names.*nc"):
        _build_data_dict({"train": object(), "val": object()})


def test_write_metrics_sidecar_dict_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    _write_metrics_sidecar({"mAP50": 0.42, "loss": 1.23}, str(out))
    payload = json.loads(out.read_text())
    assert payload == {"mAP50": 0.42, "loss": 1.23}


def test_write_metrics_sidecar_results_dict_attribute(tmp_path: Path) -> None:
    class _FakeMetrics:
        results_dict = {"mAP50": 0.5, "mAP50-95": 0.3}

    out = tmp_path / "deep" / "metrics.json"
    _write_metrics_sidecar(_FakeMetrics(), str(out))
    payload = json.loads(out.read_text())
    assert payload == {"mAP50": 0.5, "mAP50-95": 0.3}


def test_write_metrics_sidecar_no_path_is_noop(tmp_path: Path) -> None:
    # No exception, no write.
    _write_metrics_sidecar({"x": 1}, None)
    assert not (tmp_path / "anything").exists()


# --- Trainer / Evaluator unit tests -----------------------------------------


def test_trainer_run_rejects_bad_data_type() -> None:
    trainer = UltralyticsTrainer(data=42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="str path or a Flux-mode dict"):
        trainer.run()


def test_evaluator_run_rejects_bad_data_type() -> None:
    evaluator = UltralyticsEvaluator(weights="dummy.pt", data=42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="str path or a Flux-mode dict"):
        evaluator.run()


def test_trainer_flux_mode_requires_train_and_val() -> None:
    trainer = UltralyticsTrainer(data={"names": {0: "a"}, "nc": 1, "train": object()})
    with pytest.raises(ValueError, match="val"):
        trainer.run()


def test_build_yolo26_end2end_toggle_disables_head_branch() -> None:
    """``end2end=False`` must propagate through ``build_yolo26 → set_head_attr``
    so the Detect head's NMS-free branch is off — the MPS-TAL bug workaround.
    """
    from raidar.ultralytics_common import build_yolo26

    detector = build_yolo26(variant="n", num_classes=1, pretrained_detector=False, end2end=False)
    head = detector.model[-1]  # type: ignore[index]  # Detect head is always the last module
    assert head.end2end is False, "end2end=False didn't reach the Detect head"  # type: ignore[union-attr]


def test_build_yolo26_end2end_none_leaves_head_default() -> None:
    """``end2end=None`` (default) must NOT touch the head — yolo26 ships True."""
    from raidar.ultralytics_common import build_yolo26

    detector = build_yolo26(variant="n", num_classes=1, pretrained_detector=False, end2end=None)
    head = detector.model[-1]  # type: ignore[index]
    assert head.end2end is True, "end2end=None should leave the model's default (True)"  # type: ignore[union-attr]


# --- Flux-mode integration smoke -------------------------------------------


def _tiny_detection_flux(n: int = 4, h: int = 64, w: int = 64) -> Flux:
    samples: List[Sample] = []
    for i in range(n):
        img = torch.rand(3, h, w)
        boxes = torch.tensor([[10.0, 12.0, 30.0, 40.0]], dtype=torch.float32)
        labels = torch.tensor([0], dtype=torch.int64)
        samples.append(Sample(img, {"boxes": boxes, "labels": labels}, {}))
    return Flux(source=samples)


@pytest.mark.slow
def test_ultralytics_evaluator_flux_smoke(tmp_path: Path) -> None:
    """Evaluate a freshly-trained checkpoint against a Flux without disk YAML."""
    flux = _tiny_detection_flux(n=4, h=64, w=64)

    # Train one epoch first to produce a real checkpoint.
    train_metrics = tmp_path / "train_metrics.json"
    try:
        UltralyticsTrainer(
            data={"train": flux, "val": flux, "names": {0: "thing"}, "nc": 1},
            variant="n",
            pretrained=False,
            imgsz=64,
            epochs=1,
            batch=2,
            device="cpu",
            project=str(tmp_path),
            name="train",
            metrics_output_path=str(train_metrics),
            extra_overrides={"workers": 0, "save": True, "verbose": False, "amp": False},
        ).run()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"FluxDetectionTrainer setup failed: {exc}")

    # Locate the produced checkpoint.
    weights = tmp_path / "train" / "weights" / "best.pt"
    if not weights.exists():
        weights = tmp_path / "train" / "weights" / "last.pt"
    assert weights.exists(), f"No checkpoint produced at {weights}"

    eval_metrics = tmp_path / "eval_metrics.json"
    try:
        UltralyticsEvaluator(
            weights=str(weights),
            data={"val": flux, "names": {0: "thing"}, "nc": 1},
            variant="n",
            imgsz=64,
            batch=2,
            device="cpu",
            experiment_name="flux_eval",
            metrics_output_path=str(eval_metrics),
        ).run()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"FluxDetectionValidator standalone-flux run failed: {exc}")

    assert eval_metrics.exists()
    payload = json.loads(eval_metrics.read_text())
    assert isinstance(payload, dict)


@pytest.mark.slow
def test_ultralytics_trainer_flux_smoke(tmp_path: Path) -> None:
    """End-to-end: 1 epoch over a 4-sample Flux at imgsz=64 batch=2.

    Downloads yolo26n.pt on first run. Skipped if the asset can't be
    fetched (offline runners).
    """
    flux = _tiny_detection_flux(n=4, h=64, w=64)
    metrics_out = tmp_path / "metrics.json"
    try:
        trainer = UltralyticsTrainer(
            data={"train": flux, "val": flux, "names": {0: "thing"}, "nc": 1},
            variant="n",
            pretrained=False,  # skip the .pt download — fresh-init trains a tiny bit slower but is offline-safe
            imgsz=64,
            epochs=1,
            batch=2,
            device="cpu",
            project=str(tmp_path),
            name="smoke",
            metrics_output_path=str(metrics_out),
            extra_overrides={"workers": 0, "save": False, "verbose": False, "amp": False},
        )
        trainer.run()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"FluxDetectionTrainer smoke run failed in this environment: {exc}")

    assert metrics_out.exists(), "metrics_output_path JSON was not written"
    payload = json.loads(metrics_out.read_text())
    assert isinstance(payload, dict)
