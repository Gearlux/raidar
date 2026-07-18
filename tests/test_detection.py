"""Tests for the native detection plumbing in :mod:`raidar.detection`.

The marainer-generic adapter layer (``DetectionStepOutput`` /
``TorchvisionDetectionLoss`` / ``DetectionMetricAdapter``) was removed —
detection runs exclusively through the specialised trainer/evaluator. What
remains here is the batch container + collate the native path uses; the
``DetectionConfusionMatrix`` metric is covered in
:mod:`tests.test_detection_confusion_matrix`.
"""

from typing import Dict, List

import torch
from sampleflux.sample import Sample
from torch import Tensor

from raidar.detection import DetectionBatchInput, detection_collate_fn

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
                    "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
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


# --- detection_collate_fn --------------------------------------------------


def test_detection_collate_preserves_lists_and_sets_batch_input() -> None:
    samples = _synthetic_detection_samples(3, size=32)
    batch = detection_collate_fn(samples)
    assert isinstance(batch, Sample)
    assert isinstance(batch.input, DetectionBatchInput)
    assert batch.input.shape[0] == 3
    assert len(batch.input.images) == 3
    assert batch.input.targets is not None
    assert len(batch.input.targets) == 3
    # batch.target is the same list of dicts (for metrics).
    assert batch.target is batch.input.targets
    # Per-sample metadata preserved.
    assert isinstance(batch.metadata, dict) and "per_sample" in batch.metadata
