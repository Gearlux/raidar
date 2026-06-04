"""Tests for the native :class:`raidar.evaluator.LightningDetectionEvaluator`.

The evaluator overrides ``test_step`` to drive a torchvision-native detector
(``model(images) → List[Dict]``) and a raw ``MeanAveragePrecision`` directly —
no ``DetectionStepOutput`` / ``DetectionMetricAdapter`` layer.
"""

import json
from typing import Any, Dict, List

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from confluid.fluid import Class as ConfluidClass
from dataflux.core import Flux
from dataflux.sample import Sample
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import DetectionOutput, detection_collate_fn
from raidar.evaluator import LightningDetectionEvaluator


def _make_evaluator(**overrides: Any) -> LightningDetectionEvaluator:
    """Build a LightningDetectionEvaluator, applying overrides as post-construction attributes.

    Mirrors the minimal-ctor pattern: only 'model', 'dataset', 'num_classes',
    'experiment_name', 'run_name' are constructor params; deferred slots
    (``loader`` / ``lightning`` / ``metrics`` / ``predictions_sink`` /
    ``metrics_sidecar_path`` / ``class_names``) are set after construction.
    """
    _ctor_keys = ("model", "dataset", "num_classes", "experiment_name", "run_name")
    ctor = {k: overrides.pop(k) for k in _ctor_keys if k in overrides}
    ev = LightningDetectionEvaluator(**ctor)
    for _k, _v in overrides.items():
        setattr(ev, _k, _v)
    return ev


# ---------------------------------------------------------------------------
# Helpers (native detector: model(images) → List[DetectionOutput])
# ---------------------------------------------------------------------------


class _NativeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: List[Tensor]) -> List[DetectionOutput]:
        preds: List[DetectionOutput] = []
        for _ in images:
            preds.append(
                DetectionOutput(
                    boxes=torch.zeros((0, 4), dtype=torch.float32),
                    scores=torch.zeros((0,), dtype=torch.float32),
                    labels=torch.zeros((0,), dtype=torch.int64),
                )
            )
        return preds


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


def test_metrics_default_to_lazy_map() -> None:
    ev = _make_evaluator(model=_NativeDetector(), dataset=_make_flux(4), num_classes=2)
    assert isinstance(ev.metrics, ConfluidClass)
    assert ev.metrics.target is MeanAveragePrecision
    assert ev.metrics.kwargs.get("iou_type") == "bbox"


def test_default_loader_uses_detection_collate_fn() -> None:
    ev = _make_evaluator(model=_NativeDetector(), dataset=_make_flux(4), num_classes=2)
    assert isinstance(ev.loader, ConfluidClass)
    assert ev.loader.kwargs.get("collate_fn") is detection_collate_fn


# ---------------------------------------------------------------------------
# Full run with sidecar
# ---------------------------------------------------------------------------


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = _make_evaluator(
        model=_NativeDetector(),
        dataset=_make_flux(4),
        num_classes=2,
        loader=Class(DataLoader, batch_size=2, collate_fn=detection_collate_fn),
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
    # Raw MeanAveragePrecision surfaces the `map` scalar (among map_50 / mar_* etc.).
    assert any(str(k).endswith("map") for k in metric_values)
    assert isinstance(metric_values["map"], (int, float))


# ---------------------------------------------------------------------------
# Predictions-sink smoke
# ---------------------------------------------------------------------------


def test_evaluator_writes_to_predictions_sink(tmp_path: Any) -> None:
    sink = _RecordingSink()
    ev = _make_evaluator(
        model=_NativeDetector(),
        dataset=_make_flux(4),
        num_classes=2,
        loader=Class(DataLoader, batch_size=2, collate_fn=detection_collate_fn),
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
    assert len(sink.writes) == 4
    for pred, meta in sink.writes:
        assert set(pred.keys()) == {"boxes", "scores", "labels"}
        assert isinstance(meta, dict)
