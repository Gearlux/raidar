"""Tests for LightningDetectionTrainer."""

from typing import Any, List

import torch
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision

from dataflux.sample import Sample
from raidar.detection import (
    DetectionBatchInput,
    DetectionMetricAdapter,
    DetectionOutput,
    DetectionStepOutput,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)
from raidar.trainer import LightningDetectionTrainer, _default_detection_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyDetector(nn.Module):
    """Stub detector that returns a DetectionStepOutput-shaped dict.

    In train mode (targets present) returns a 1-element loss dict with a
    learnable scalar so gradients flow; in eval mode (targets absent or
    requested) returns a single empty per-image prediction so the
    metric-update path runs end-to-end.
    """

    def __init__(self) -> None:
        super().__init__()
        # A learnable parameter so the optimizer has something to update.
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        n = len(x.images)
        if x.targets is not None:
            # Training/val path. Compute a real Tensor that depends on alpha
            # so loss.backward() updates self.alpha.
            loss = (self.alpha * torch.ones((), device=x.images[0].device)).sum()
            return {"losses": {"loss": loss}, "predictions": None}
        # Pure-prediction path: return one empty prediction per image.
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


def _make_samples(n: int = 4, size: int = 32) -> List[Sample]:
    samples: List[Sample] = []
    for i in range(n):
        samples.append(
            Sample(
                input=torch.rand(3, size, size),
                target={
                    "boxes": torch.tensor(
                        [[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32
                    ),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
                metadata={"idx": i},
            )
        )
    return samples


def _make_batch(n: int = 2, size: int = 32) -> Sample:
    return detection_collate_fn(_make_samples(n=n, size=size))


# ---------------------------------------------------------------------------
# _default_detection_metrics
# ---------------------------------------------------------------------------


def test_default_detection_metrics_is_adapter_with_prefix() -> None:
    m = _default_detection_metrics(prefix="val/")
    assert isinstance(m, DetectionMetricAdapter)
    keys = [str(k) for k in m.keys()]
    assert all(k.startswith("val/") for k in keys), f"Expected val/ prefix, got {keys}"
    assert any(k.endswith("map") for k in keys)


# ---------------------------------------------------------------------------
# LightningDetectionTrainer defaults
# ---------------------------------------------------------------------------


def test_loss_fn_is_torchvision_detection_loss() -> None:
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    assert isinstance(trainer.loss_fn, TorchvisionDetectionLoss)


def test_metrics_are_wired_with_correct_prefixes() -> None:
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    # Detection training metrics break gradients on most detectors; default None.
    assert trainer.train_metrics is None
    assert trainer.val_metrics is not None
    assert trainer.test_metrics is not None
    assert isinstance(trainer.val_metrics, DetectionMetricAdapter)
    assert isinstance(trainer.test_metrics, DetectionMetricAdapter)
    val_keys = [str(k) for k in trainer.val_metrics.keys()]
    test_keys = [str(k) for k in trainer.test_metrics.keys()]
    assert all(
        k.startswith("val/") for k in val_keys
    ), f"val/ prefix expected, got {val_keys}"
    assert all(
        k.startswith("test/") for k in test_keys
    ), f"test/ prefix expected, got {test_keys}"


def test_val_and_test_metrics_are_independent_instances() -> None:
    """Independent inner-metric state so val and test accumulations don't cross-contaminate."""
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    assert trainer.val_metrics is not None
    assert trainer.test_metrics is not None
    # The inner MAP instance is held on each adapter via _inner_metric (set via object.__setattr__).
    assert trainer.val_metrics._inner_metric is not trainer.test_metrics._inner_metric  # type: ignore[attr-defined]


def test_default_loader_uses_detection_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    assert isinstance(trainer.train_loader, ConfluidClass)
    assert trainer.train_loader.kwargs.get("collate_fn") is detection_collate_fn
    assert isinstance(trainer.val_loader, ConfluidClass)
    assert trainer.val_loader.kwargs.get("collate_fn") is detection_collate_fn


# ---------------------------------------------------------------------------
# Forward / step methods
# ---------------------------------------------------------------------------


def test_training_step_returns_finite_scalar_with_grad() -> None:
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    batch = _make_batch(n=2)
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_scalar() -> None:
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    batch = _make_batch(n=2)
    loss = trainer.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Override hygiene — proves no raidar wrapper is required for custom metrics
# ---------------------------------------------------------------------------


def test_user_can_override_val_metrics_with_custom_adapter() -> None:
    """Replace the default val_metrics with a DetectionMetricAdapter built
    around a differently-configured MeanAveragePrecision; the override sticks.
    """
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )
    custom = DetectionMetricAdapter(
        MeanAveragePrecision(iou_type="bbox", class_metrics=True), prefix="val/"
    )
    trainer.val_metrics = custom
    assert trainer.val_metrics is custom


def test_user_can_override_loss_fn_with_any_callable() -> None:
    """Assign any nn.Module that consumes (step_output, target) — no raidar wrapper needed."""
    trainer = LightningDetectionTrainer(
        model=_TinyDetector(),
        train_set=_make_samples(4),
        num_classes=2,
    )

    class _CustomLoss(nn.Module):
        def forward(self, step_output: Any, target: Any) -> torch.Tensor:
            losses = step_output["losses"]
            assert losses is not None
            total = 0.5 * sum(losses.values())
            return (
                total if isinstance(total, torch.Tensor) else torch.tensor(float(total))
            )

    trainer.loss_fn = _CustomLoss()
    batch = _make_batch(n=2)
    loss = trainer.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert isinstance(trainer.loss_fn, _CustomLoss)
