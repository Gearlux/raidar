"""Tests for LightningDetectionEvaluator."""

import json
from typing import Any, Dict, List

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from torch.utils.data import DataLoader

from raidar.detection import (
    DetectionBatchInput,
    DetectionMetricAdapter,
    DetectionOutput,
    DetectionStepOutput,
    detection_collate_fn,
)
from raidar.evaluator import LightningDetectionEvaluator

# ---------------------------------------------------------------------------
# Helpers (mirror test_detection_trainer.py for consistency)
# ---------------------------------------------------------------------------


class _TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        n = len(x.images)
        # Evaluator path always wants per-image predictions.
        preds: List[DetectionOutput] = []
        for _ in range(n):
            preds.append(
                DetectionOutput(
                    boxes=torch.zeros((0, 4), dtype=torch.float32),
                    scores=torch.zeros((0,), dtype=torch.float32),
                    labels=torch.zeros((0,), dtype=torch.int64),
                )
            )
        return {"losses": None, "predictions": preds}


def _make_flux(n: int = 4, size: int = 32) -> Flux:
    samples = [
        Sample(
            input=torch.rand(3, size, size),
            target={
                "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            metadata={"idx": i},
        )
        for i in range(n)
    ]
    return Flux(source=samples)


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_metrics_are_wired_on_construction() -> None:
    ev = LightningDetectionEvaluator(
        model=_TinyDetector(),
        dataset=_make_flux(4),
        num_classes=2,
    )
    assert ev.metrics is not None
    assert isinstance(ev.metrics, DetectionMetricAdapter)


def test_metrics_have_test_prefix() -> None:
    ev = LightningDetectionEvaluator(
        model=_TinyDetector(),
        dataset=_make_flux(4),
        num_classes=2,
    )
    keys = list(ev.metrics.keys())  # type: ignore[union-attr]
    assert all(k.startswith("test/") for k in keys), f"Expected test/ prefix, got: {keys}"


def test_default_loader_uses_detection_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    ev = LightningDetectionEvaluator(
        model=_TinyDetector(),
        dataset=_make_flux(4),
        num_classes=2,
    )
    assert isinstance(ev.loader, ConfluidClass)
    assert ev.loader.kwargs.get("collate_fn") is detection_collate_fn


# ---------------------------------------------------------------------------
# Full run with sidecar
# ---------------------------------------------------------------------------


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = LightningDetectionEvaluator(
        model=_TinyDetector(),
        dataset=_make_flux(4),
        num_classes=2,
        loader=Class(
            DataLoader,
            batch_size=2,
            collate_fn=detection_collate_fn,
        ),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
        experiment_name="det_test_exp",
        run_name="det_test_run",
        metrics_sidecar_path=str(sidecar),
    )
    ev.run()
    assert sidecar.exists(), "Sidecar must be written."
    payload = json.loads(sidecar.read_text())
    assert payload["experiment_name"] == "det_test_exp"
    assert "metrics" in payload
    metric_values = payload["metrics"]
    # MAP keys map / map_50 / mar_* should all be in the sidecar.
    assert any(k.endswith("map") for k in metric_values)
    for v in metric_values.values():
        assert isinstance(v, (int, float))


# ---------------------------------------------------------------------------
# Predictions-sink smoke
# ---------------------------------------------------------------------------


def test_evaluator_writes_to_predictions_sink(tmp_path: Any) -> None:
    sink = _RecordingSink()
    ev = LightningDetectionEvaluator(
        model=_TinyDetector(),
        dataset=_make_flux(4),
        num_classes=2,
        loader=Class(
            DataLoader,
            batch_size=2,
            collate_fn=detection_collate_fn,
        ),
        predictions_sink=sink,
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    ev.run()
    assert sink.closed, ".close() must be invoked at end of run."
    assert len(sink.writes) >= 1
