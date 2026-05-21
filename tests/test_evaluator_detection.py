"""Exercise the Marainer Evaluator with a raidar detection model end-to-end."""

from typing import Any, Dict, List

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from marainer.evaluator import LightningEvaluator
from raidar.detection import DetectionMetricAdapter, detection_collate_fn
from raidar.faster_rcnn import FasterRCNNModel


class _RecordingSink:
    """Minimal sink that records everything handed to write()."""

    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def test_evaluator_runs_on_detection_task(tmp_path: Any) -> None:
    model = FasterRCNNModel(
        num_classes=2,
        variant="v2",
        pretrained_backbone=False,
        trainable_backbone_layers=1,
        min_size=64,
        max_size=128,
    )
    samples = [
        Sample(
            input=torch.rand(3, 64, 64),
            target={
                "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            metadata={"idx": i},
        )
        for i in range(4)
    ]
    flux = Flux(source=samples)

    sink = _RecordingSink()
    evaluator = LightningEvaluator(
        model=model,
        dataset=flux,
        loader=Class(DataLoader, batch_size=2, collate_fn=detection_collate_fn),
        metrics=DetectionMetricAdapter(MeanAveragePrecision(iou_type="bbox")),
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
    evaluator.run()

    # Expect 4 per-sample writes (one per image across 2 batches of 2).
    assert len(sink.writes) == 4
    assert sink.closed is True
    # Each recorded prediction is a typed DetectionOutput dict.
    for pred, meta in sink.writes:
        assert set(pred.keys()) == {"boxes", "scores", "labels"}
        assert isinstance(meta, dict)
        assert "idx" in meta
