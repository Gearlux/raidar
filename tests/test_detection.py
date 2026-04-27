"""Tests for the generic detection adapters in :mod:`raidar.detection`.

Detector-specific tests (Faster R-CNN model + the BN-frozen dual-forward
helper) live in :mod:`tests.test_faster_rcnn`.
"""

from typing import Dict, List

import torch
from dataflux.sample import Sample
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import (
    DetectionBatchInput,
    DetectionMetricAdapter,
    DetectionStepOutput,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)

# --- helpers ---------------------------------------------------------------


def _make_images(n: int = 2, size: int = 64) -> List[Tensor]:
    return [torch.rand(3, size, size) for _ in range(n)]


def _make_targets(n: int = 2) -> List[Dict[str, Tensor]]:
    targets = []
    for _ in range(n):
        targets.append(
            {
                "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        )
    return targets


def _synthetic_detection_samples(n: int, size: int = 64) -> List[Sample]:
    samples: List[Sample] = []
    for _ in range(n):
        samples.append(
            Sample(
                input=torch.rand(3, size, size),
                target={
                    "boxes": torch.tensor(
                        [[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32
                    ),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
                metadata={},
            )
        )
    return samples


# --- DetectionBatchInput ---------------------------------------------------


def test_detection_batch_input_shape() -> None:
    imgs = _make_images(3)
    tgts = _make_targets(3)
    bi = DetectionBatchInput(images=imgs, targets=tgts)
    assert bi.shape == (3,)
    assert bi.shape[0] == 3
    assert len(bi) == 3


def test_detection_batch_input_targets_optional() -> None:
    bi = DetectionBatchInput(images=_make_images(2))
    assert bi.targets is None
    assert bi.shape[0] == 2


# --- TorchvisionDetectionLoss ----------------------------------------------


def test_torchvision_detection_loss_sums_dict_with_weights() -> None:
    loss_fn = TorchvisionDetectionLoss(weights={"loss_a": 2.0})
    step: DetectionStepOutput = DetectionStepOutput(
        losses={"loss_a": torch.tensor(3.0), "loss_b": torch.tensor(1.0)},
        predictions=None,
    )
    out = loss_fn(step, target=None)
    assert torch.allclose(out, torch.tensor(7.0))  # 2.0*3 + 1.0*1


def test_torchvision_detection_loss_zero_when_losses_is_none() -> None:
    loss_fn = TorchvisionDetectionLoss()
    step: DetectionStepOutput = DetectionStepOutput(
        losses=None,
        predictions=[
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        ],
    )
    out = loss_fn(step, target=None)
    assert out.dim() == 0
    assert float(out) == 0.0
    assert not out.requires_grad


# --- DetectionMetricAdapter ------------------------------------------------


def test_metric_adapter_no_op_when_predictions_none() -> None:
    adapter = DetectionMetricAdapter(MeanAveragePrecision(iou_type="bbox"))
    step: DetectionStepOutput = DetectionStepOutput(
        losses={"loss_a": torch.tensor(1.0)},
        predictions=None,
    )
    # Should not raise and should not update internal state.
    adapter.update(step, target=None)
    # Forward path also no-ops.
    out = adapter(step, target=None)
    assert out == {}


def test_metric_adapter_forwards_to_inner_when_predictions_present() -> None:
    adapter = DetectionMetricAdapter(MeanAveragePrecision(iou_type="bbox"))
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    step: DetectionStepOutput = DetectionStepOutput(losses=None, predictions=preds)
    adapter.update(step, targets)
    computed = adapter.compute()
    assert isinstance(computed, dict)
    # MeanAveragePrecision emits scalar `map` among other keys.
    assert (
        "map" in computed
        or "MeanAveragePrecision_map" in computed
        or any("map" in str(k) for k in computed.keys())
    )


# --- detection_collate_fn --------------------------------------------------


def test_detection_collate_preserves_lists_and_sets_batch_input() -> None:
    samples = _synthetic_detection_samples(3, size=32)
    batch = detection_collate_fn(samples)
    assert isinstance(batch, Sample)
    assert isinstance(batch.input, DetectionBatchInput)
    assert batch.input.shape[0] == 3
    assert len(batch.input.images) == 3
    assert len(batch.input.targets) == 3
    # batch.target is the same list of dicts (for metrics/loss).
    assert batch.target is batch.input.targets
    # Per-sample metadata preserved.
    assert isinstance(batch.metadata, dict) and "per_sample" in batch.metadata
