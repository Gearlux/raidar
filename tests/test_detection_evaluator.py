"""Tests for :class:`raidar.detector.LightningDetector` in the evaluation path (task='test').

The detector's overridden ``test_step`` drives a torchvision-native detector
(``model(images) → List[Dict]``) and a raw ``MeanAveragePrecision`` directly — no adapter layer.
"""

import json
from typing import Any, Dict, List

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, LazyClass
from confluid.fluid import Class as ConfluidClass
from sampleflux.core import Flux
from sampleflux.sample import Sample
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import DetectionOutput, detection_collate_fn
from raidar.detector import LightningDetector


def _evaluator(**overrides: Any) -> LightningDetector:
    """Build a test-path LightningDetector (maps ``dataset=`` → ``test_set``, sets task='test')."""
    ctor: Dict[str, Any] = {"task": "test"}
    for key in ("model", "experiment_name", "run_name"):
        if key in overrides:
            ctor[key] = overrides.pop(key)
    if "dataset" in overrides:
        ctor["test_set"] = overrides.pop("dataset")
    ev = LightningDetector(**ctor)
    for _k, _v in overrides.items():
        setattr(ev, _k, _v)
    return ev


class _NativeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: List[Tensor]) -> List[DetectionOutput]:
        return [
            DetectionOutput(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64),
            )
            for _ in images
        ]


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


def test_metrics_default_to_lazy_map() -> None:
    ev = _evaluator(model=_NativeDetector(), dataset=_make_flux(4))
    assert isinstance(ev.test_metrics, ConfluidClass)
    assert ev.test_metrics.target is MeanAveragePrecision
    assert ev.test_metrics.kwargs.get("iou_type") == "bbox"


def test_default_loader_uses_detection_collate_fn() -> None:
    ev = _evaluator(model=_NativeDetector(), dataset=_make_flux(4))
    assert isinstance(ev.test_loader, ConfluidClass)
    assert ev.test_loader.kwargs.get("collate_fn") is detection_collate_fn


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = _evaluator(
        model=_NativeDetector(),
        dataset=_make_flux(4),
        test_loader=Class(DataLoader, batch_size=2, collate_fn=detection_collate_fn),
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
    metric_values = payload["metrics"]
    assert any(str(k).endswith("map") for k in metric_values)
    assert isinstance(metric_values["map"], (int, float))


def test_evaluator_writes_to_predictions_sink(tmp_path: Any) -> None:
    sink = _RecordingSink()
    ev = _evaluator(
        model=_NativeDetector(),
        dataset=_make_flux(4),
        test_loader=Class(DataLoader, batch_size=2, collate_fn=detection_collate_fn),
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


class _NumClassesDetector(nn.Module):
    """Native detector exposing ``num_classes`` — stands in for a trained model at eval time."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: List[Tensor]) -> List[DetectionOutput]:
        return _NativeDetector.forward(self, images)  # type: ignore[arg-type]


def test_initialize_reads_num_classes_off_trained_model() -> None:
    # On the eval path, initialize() flows the model first, then reads num_classes off the LIVE model.
    ev = _evaluator(model=LazyClass(_NumClassesDetector, num_classes=5), dataset=_make_flux(4))
    assert ev.num_classes is None  # not resolved until initialize()
    ev.initialize()
    assert isinstance(ev.model, _NumClassesDetector)
    assert ev.num_classes == 5


def test_initialize_reads_num_classes_from_torchvision_head() -> None:
    model = nn.Module()
    model.roi_heads = nn.Module()  # type: ignore[assignment]
    model.roi_heads.box_predictor = nn.Module()  # type: ignore[attr-defined]
    model.roi_heads.box_predictor.cls_score = nn.Linear(10, 7)  # type: ignore[attr-defined]
    ev = _evaluator(model=model, dataset=_make_flux(4))
    ev.initialize()
    assert ev.num_classes == 7


def test_initialize_best_effort_when_model_lacks_num_classes() -> None:
    # The MAP metric is class-agnostic — a detector that exposes no count leaves num_classes None.
    ev = _evaluator(model=_NativeDetector(), dataset=_make_flux(4))
    ev.initialize()
    assert ev.num_classes is None


def test_initialize_normalizes_dataset_to_flux() -> None:
    ev = _evaluator(model=_NativeDetector(), dataset=_make_flux(4))
    ev.initialize()
    assert isinstance(ev.test_set, Flux)


def test_run_calls_initialize(tmp_path: Any) -> None:
    ev = _evaluator(
        model=LazyClass(_NumClassesDetector, num_classes=6),
        dataset=_make_flux(4),
        test_loader=Class(DataLoader, batch_size=2, collate_fn=detection_collate_fn),
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
    assert isinstance(ev.model, _NumClassesDetector)
    assert ev.num_classes == 6
