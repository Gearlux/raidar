"""Tests for the native detection plumbing in :mod:`raidar.detection`.

The matrainer-generic adapter layer (``DetectionStepOutput`` /
``TorchvisionDetectionLoss`` / ``DetectionMetricAdapter``) was removed —
detection runs exclusively through the specialised trainer/evaluator. What
remains here is the batch container + collate the native path uses; the
``DetectionConfusionMatrix`` metric is covered in
:mod:`tests.test_detection_confusion_matrix`.
"""

from typing import Any, Dict, List

import torch
from recordstream import Image, Record, Regions
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


def _regions(boxes: Any, labels: Any) -> Regions:
    """A detection-target Regions carrying tensor boxes/labels (as the typed detection op produces)."""
    return Regions(boxes=boxes, labels=labels)


def _synthetic_detection_records(n: int, size: int = 64) -> List[Record]:
    """Per-record records shaped like a detection source: Image under "image" + Regions under "target"."""
    records: List[Record] = []
    for _ in range(n):
        records.append(
            {
                "image": Image(torch.rand(3, size, size).numpy(), layout="CHW"),
                "target": _regions(
                    torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                    torch.tensor([1], dtype=torch.int64),
                ),
            }
        )
    return records


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
    records = _synthetic_detection_records(3, size=32)
    batch = detection_collate_fn(records)
    assert isinstance(batch, dict)
    # The DetectionBatchInput rides VERBATIM under the batched record's "detection" key.
    bi = batch["detection"]
    assert isinstance(bi, DetectionBatchInput)
    assert bi.shape[0] == 3
    assert len(bi.images) == 3
    assert bi.targets is not None
    assert len(bi.targets) == 3
    # The "targets" field is the SAME list of dicts (for metrics).
    assert batch["targets"] is bi.targets
    # Each target was unwrapped from the Regions item into the torchvision {boxes, labels} dict.
    assert set(bi.targets[0].keys()) == {"boxes", "labels"}
    # Per-record metadata preserved as a list.
    assert isinstance(batch["metadata"], list) and len(batch["metadata"]) == 3
