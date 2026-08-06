"""Tests for :class:`raidar.lightning_detector.LightningDetector` in the evaluation path (task='test').

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
from recordstream import Image, Regions, collate_list
from recordstream.core import Stream
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from raidar.lightning_detector import LightningDetector
from raidar.torch_detection import DetectionOutput


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


def _regions(boxes: Any, labels: Any) -> Regions:
    """A detection-target Regions carrying tensor boxes/labels (as the typed detection op produces)."""
    return Regions(boxes=boxes, labels=labels)


def _make_stream(n: int = 4, size: int = 32) -> Stream:
    records = [
        {
            "image": Image(torch.rand(3, size, size).numpy(), layout="CHW"),
            "target": _regions(
                torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int64),
            ),
            "idx": i,
        }
        for i in range(n)
    ]
    return Stream(source=records)


class _RecordingSink:
    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def test_metrics_default_is_a_sentinel_resolved_at_initialize() -> None:
    """`None` means "not chosen" — the COCO-mAP default arrives in `initialize()`, prefixed for
    the split, so an explicitly wired metric is never silently replaced."""
    ev = _evaluator(model=_NativeDetector(), dataset=_make_stream(4))
    assert ev.metrics is None and ev.test_metrics is None
    ev.initialize()
    assert isinstance(ev.test_metrics, MetricCollection)
    assert any(str(k).startswith("test/") for k in ev.test_metrics.keys())
    assert any(isinstance(m, MeanAveragePrecision) for m in ev.test_metrics.values())


def test_default_loader_DECLARES_the_list_collate() -> None:

    ev = _evaluator(model=_NativeDetector(), dataset=_make_stream(4))
    assert isinstance(ev.test_loader, ConfluidClass)
    collate = ev.test_loader.kwargs.get("collate_fn")
    # DECLARED, not inherited: the native forward takes List[Tensor], so the loaders ask
    # recordstream for the unstacked batch rather than relying on ToTensor's plain tensors
    # happening to skip stacking.
    assert collate is collate_list


def test_evaluator_run_writes_metrics_sidecar(tmp_path: Any) -> None:
    sidecar = tmp_path / "eval_metrics.json"
    ev = _evaluator(
        model=_NativeDetector(),
        dataset=_make_stream(4),
        test_loader=Class(DataLoader, batch_size=2, collate_fn=collate_list),
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
    # The split's collection is prefixed, so the sidecar records `test/map` etc. — the same
    # canonical `<split>/<name>` every backend writes.
    assert any(str(k).endswith("map") for k in metric_values)
    assert isinstance(metric_values["test/map"], (int, float))
    # MAP's `classes` bookkeeping entry (observed ids, not a metric) is filtered out.
    assert "test/classes" not in metric_values


def test_evaluator_writes_to_predictions_sink(tmp_path: Any) -> None:
    sink = _RecordingSink()
    ev = _evaluator(
        model=_NativeDetector(),
        dataset=_make_stream(4),
        test_loader=Class(DataLoader, batch_size=2, collate_fn=collate_list),
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
    ev = _evaluator(model=LazyClass(_NumClassesDetector, num_classes=5), dataset=_make_stream(4))
    assert ev.num_classes is None  # not resolved until initialize()
    ev.initialize()
    assert isinstance(ev.model, _NumClassesDetector)
    assert ev.num_classes == 5


def test_initialize_reads_num_classes_from_torchvision_head() -> None:
    model = nn.Module()
    model.roi_heads = nn.Module()  # type: ignore[assignment]
    model.roi_heads.box_predictor = nn.Module()  # type: ignore[attr-defined]
    model.roi_heads.box_predictor.cls_score = nn.Linear(10, 7)  # type: ignore[attr-defined]
    ev = _evaluator(model=model, dataset=_make_stream(4))
    ev.initialize()
    assert ev.num_classes == 7


def test_initialize_best_effort_when_model_lacks_num_classes() -> None:
    # The MAP metric is class-agnostic — a detector that exposes no count leaves num_classes None.
    ev = _evaluator(model=_NativeDetector(), dataset=_make_stream(4))
    ev.initialize()
    assert ev.num_classes is None


def test_initialize_normalizes_dataset_to_stream() -> None:
    ev = _evaluator(model=_NativeDetector(), dataset=_make_stream(4))
    ev.initialize()
    assert isinstance(ev.test_set, Stream)


def test_run_calls_initialize(tmp_path: Any) -> None:
    ev = _evaluator(
        model=LazyClass(_NumClassesDetector, num_classes=6),
        dataset=_make_stream(4),
        test_loader=Class(DataLoader, batch_size=2, collate_fn=collate_list),
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
