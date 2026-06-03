"""Tests for the specialised :class:`raidar.lightning_trainer.LightningDetectionTrainer`.

These tests cover the **native** path — overridden ``training_step`` /
``validation_step`` / ``test_step`` consume torchvision-native models
(``forward(images, targets) → loss dict`` / ``forward(images) → predictions
list``) and torchmetrics detection metrics directly, with no
``TorchvisionDetectionLoss`` / ``DetectionMetricAdapter`` layer involved.
"""

from typing import Any, Dict, List, Optional, Union

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from dataflux.core import Flux
from dataflux.sample import Sample
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import DetectionConfusionMatrixNative, detection_collate_fn
from raidar.lightning_trainer import LightningDetectionTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NativeDetector(nn.Module):
    """Stub detector that conforms to the torchvision-native contract.

    * Train mode: ``forward(images, targets) → {"loss_a": .., "loss_b": ..}``
      where each loss depends on a learnable parameter (so gradients flow).
    * Eval mode: ``forward(images) → List[Dict[str, Tensor]]`` with one empty
      prediction per image (enough for ``MeanAveragePrecision.update``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            device = images[0].device
            return {
                "loss_a": (self.alpha * torch.ones((), device=device)).sum(),
                "loss_b": (0.5 * self.alpha * torch.ones((), device=device)).sum(),
            }
        out: List[Dict[str, Tensor]] = []
        for _ in images:
            out.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )
        return out


def _make_samples(n: int = 4, size: int = 32) -> List[Sample]:
    samples: List[Sample] = []
    for i in range(n):
        samples.append(
            Sample(
                input=torch.rand(3, size, size),
                target={
                    "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
                metadata={"idx": i},
            )
        )
    return samples


def _make_batch(n: int = 2, size: int = 32) -> Sample:
    return detection_collate_fn(_make_samples(n=n, size=size))


def _make_flux(n: int = 4, size: int = 32) -> Flux:
    # A Flux (torch Dataset) — the trainer's train_set/val_set/test_set are typed
    # Dataset[Any] | Fluid; a bare list[Sample] is normalized to a Flux only at run
    # time (ensure_sample_dataset), so wire datasets as a Flux up front.
    return Flux(source=_make_samples(n=n, size=size))


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_loss_fn_is_none() -> None:
    """Specialised trainer drops the loss_fn parameter entirely — native
    detection models compute loss inside forward."""
    trainer = LightningDetectionTrainer(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    assert trainer.loss_fn is None


def test_train_metrics_default_none() -> None:
    trainer = LightningDetectionTrainer(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    assert trainer.train_metrics is None


def test_val_test_metrics_default_to_lazy_map() -> None:
    """Default ``val_metrics`` / ``test_metrics`` are ``Class(MeanAveragePrecision, ...)``
    so they're independent and materialise through the base trainer's ``flow()`` step."""
    from confluid.fluid import Class as ConfluidClass

    trainer = LightningDetectionTrainer(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    assert isinstance(trainer.val_metrics, ConfluidClass)
    assert isinstance(trainer.test_metrics, ConfluidClass)
    assert trainer.val_metrics is not trainer.test_metrics
    assert trainer.val_metrics.target is MeanAveragePrecision
    assert trainer.val_metrics.kwargs.get("iou_type") == "bbox"


def test_user_supplied_instance_metric_passes_through() -> None:
    """An already-instantiated metric is held verbatim — no Lazy wrapping needed."""
    custom = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    trainer = LightningDetectionTrainer(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.val_metrics = custom
    assert trainer.val_metrics is custom


def test_metric_collection_with_native_confusion_matrix_construction() -> None:
    """A MetricCollection of native detection metrics (MAP + confusion matrix)
    constructs cleanly — proving the specialised trainer accepts compositions
    without adapter wrapping."""
    coll = MetricCollection(
        {
            "map": MeanAveragePrecision(iou_type="bbox"),
            "confusion": DetectionConfusionMatrixNative(iou_threshold=0.5),
        }
    )
    trainer = LightningDetectionTrainer(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.val_metrics = coll
    assert trainer.val_metrics is coll


def test_default_loader_uses_detection_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    trainer = LightningDetectionTrainer(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    assert isinstance(trainer.train_loader, ConfluidClass)
    assert trainer.train_loader.kwargs.get("collate_fn") is detection_collate_fn
    assert isinstance(trainer.val_loader, ConfluidClass)
    assert trainer.val_loader.kwargs.get("collate_fn") is detection_collate_fn


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def test_training_step_sums_native_loss_dict_with_grad() -> None:
    trainer = LightningDetectionTrainer(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    trainer.train()
    batch = _make_batch(n=2)
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_rejects_missing_targets() -> None:
    from raidar.detection import DetectionBatchInput

    trainer = LightningDetectionTrainer(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    trainer.train()
    batch = Sample(
        input=DetectionBatchInput(images=[torch.rand(3, 32, 32)], targets=None),
        target=[],
        metadata={"per_sample": [{}]},
    )
    with pytest.raises(ValueError, match="requires targets"):
        trainer.training_step(batch, 0)


def test_training_step_rejects_non_dict_model_output() -> None:
    """If the model returns a list (eval-shape) in train mode, fail loudly."""

    class _BrokenDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(1.0))

        def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
        ) -> Any:
            return [{"boxes": torch.zeros((0, 4))} for _ in images]

    trainer = LightningDetectionTrainer(
        model=_BrokenDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    trainer.train()
    batch = _make_batch(n=2)
    with pytest.raises(TypeError, match="must return Dict"):
        trainer.training_step(batch, 0)


def test_validation_step_updates_metric_with_predictions() -> None:
    trainer = LightningDetectionTrainer(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.val_set = _make_flux(2)
    trainer.val_metrics = MeanAveragePrecision(iou_type="bbox")
    trainer.eval()
    batch = _make_batch(n=2)
    # validation_step doesn't return a loss in the native path; just executes.
    trainer.validation_step(batch, 0)
    # After update, the metric holds state — compute() returns a dict with map keys.
    assert isinstance(trainer.val_metrics, MeanAveragePrecision)


def test_test_step_uses_test_metric() -> None:
    trainer = LightningDetectionTrainer(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.test_set = _make_flux(2)
    trainer.test_metrics = MeanAveragePrecision(iou_type="bbox")
    trainer.eval()
    batch = _make_batch(n=2)
    trainer.test_step(batch, 0)
    assert isinstance(trainer.test_metrics, MeanAveragePrecision)


# ---------------------------------------------------------------------------
# End-to-end Lightning fit() against an in-memory Flux — no raidar adapters
# ---------------------------------------------------------------------------


def test_end_to_end_native_trainer_fit(tmp_path: Any) -> None:
    """Specialised trainer runs through fit() with no raidar loss/metric adapter:
    pass raw native detector + raw torchmetrics MAP, observe train/loss and
    val/map.
    """
    train_flux = _make_flux(4)
    val_flux = _make_flux(2)

    trainer = LightningDetectionTrainer(model=_NativeDetector(), train_set=train_flux, num_classes=2)
    trainer.val_set = val_flux

    lightning_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
    )
    # The base trainer's run() flow()s val_metrics from the Class wrapper. Mirror
    # that here so the metric materialises before fit().
    from confluid import flow

    trainer.val_metrics = flow(trainer.val_metrics)
    trainer.test_metrics = flow(trainer.test_metrics)
    trainer.train_loader = torch.utils.data.DataLoader(train_flux, batch_size=2, collate_fn=detection_collate_fn)
    trainer.val_loader = torch.utils.data.DataLoader(val_flux, batch_size=2, collate_fn=detection_collate_fn)

    lightning_trainer.fit(
        model=trainer,
        train_dataloaders=trainer.train_loader,
        val_dataloaders=trainer.val_loader,
    )
    metrics = lightning_trainer.callback_metrics
    assert any(k.startswith("train/loss") for k in metrics), f"expected train/loss in {list(metrics)}"
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss)
    # MAP scalars should appear under val/ prefix.
    assert any(k.startswith("val/map") for k in metrics), f"expected val/map* in {list(metrics)}"
