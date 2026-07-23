"""Tests for the specialised :class:`raidar.lightning_trainer.LightningDetector`.

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
from confluid import LazyClass
from sampleflux.core import Flux
from sampleflux.sample import Sample
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import DetectionConfusionMatrix, detection_collate_fn
from raidar.detector import LightningDetector, _model_num_classes

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
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    assert trainer.loss_fn is None


def test_train_metrics_default_none() -> None:
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    assert trainer.train_metrics is None


def test_val_test_metrics_default_to_lazy_map() -> None:
    """Default ``val_metrics`` / ``test_metrics`` are ``Class(MeanAveragePrecision, ...)``
    so they're independent and materialise through the base trainer's ``flow()`` step."""
    from confluid.fluid import Class as ConfluidClass

    trainer = LightningDetector(
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
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.val_metrics = custom
    assert trainer.val_metrics is custom


def test_metric_collection_with_native_confusion_matrix_construction() -> None:
    """A MetricCollection of native detection metrics (MAP + confusion matrix)
    constructs cleanly — proving the specialised trainer accepts compositions
    without adapter wrapping."""
    coll = MetricCollection(
        {
            "map": MeanAveragePrecision(iou_type="bbox"),
            "confusion": DetectionConfusionMatrix(iou_threshold=0.5),
        }
    )
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.val_metrics = coll
    assert trainer.val_metrics is coll


def test_default_loader_uses_detection_collate_fn() -> None:
    from confluid.fluid import Class as ConfluidClass

    trainer = LightningDetector(
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
    trainer = LightningDetector(
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

    trainer = LightningDetector(
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

    trainer = LightningDetector(
        model=_BrokenDetector(),
        train_set=_make_flux(4),
        num_classes=2,
    )
    trainer.train()
    batch = _make_batch(n=2)
    with pytest.raises(TypeError, match="must return Dict"):
        trainer.training_step(batch, 0)


def test_validation_step_updates_metric_with_predictions() -> None:
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
    trainer.val_set = _make_flux(2)
    trainer.val_metrics = MeanAveragePrecision(iou_type="bbox")
    trainer.eval()
    batch = _make_batch(n=2)
    # validation_step doesn't return a loss in the native path; just executes.
    trainer.validation_step(batch, 0)
    # After update, the metric holds state — compute() returns a dict with map keys.
    assert isinstance(trainer.val_metrics, MeanAveragePrecision)


def test_test_step_uses_test_metric() -> None:
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_flux(4), num_classes=2)
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

    trainer = LightningDetector(model=_NativeDetector(), train_set=train_flux, num_classes=2)
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


# ---------------------------------------------------------------------------
# initialize(): num_classes resolution + lazy-model flow (sonair-style)
# ---------------------------------------------------------------------------


class _NumClassesDetector(nn.Module):
    """Stub whose ctor records ``num_classes`` — stands in for a ``!lazy:`` detector
    builder (e.g. torchvision's ``fasterrcnn_resnet50_fpn``), so ``initialize()``'s
    flow-injection of the resolved count is testable without a heavy download."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = nn.Parameter(torch.tensor(1.0))


def _labelled_flux(label_sets: List[List[int]], size: int = 16) -> Flux:
    """A detection Flux whose per-image targets carry the given label lists."""
    samples = [
        Sample(
            input=torch.rand(3, size, size),
            target={
                "boxes": torch.zeros((len(lbls), 4), dtype=torch.float32),
                "labels": torch.tensor(lbls, dtype=torch.int64),
            },
            metadata={},
        )
        for lbls in label_sets
    ]
    return Flux(source=samples)


def test_initialize_derives_num_classes_from_targets_and_flows_lazy_model() -> None:
    # Labels up to 2 → derived num_classes = max + 1 = 3; injected into the lazy model.
    trainer = LightningDetector(model=LazyClass(_NumClassesDetector), train_set=_labelled_flux([[1, 2], [1]]))
    trainer.initialize()
    assert trainer.num_classes == 3
    assert isinstance(trainer.model, _NumClassesDetector)
    assert trainer.model.num_classes == 3


def test_initialize_explicit_num_classes_injected_into_lazy_model() -> None:
    trainer = LightningDetector(model=LazyClass(_NumClassesDetector), train_set=_labelled_flux([[1]]), num_classes=7)
    trainer.initialize()
    assert isinstance(trainer.model, _NumClassesDetector)
    assert trainer.num_classes == 7 and trainer.model.num_classes == 7


def test_initialize_lazy_model_pinned_num_classes_used() -> None:
    # num_classes pinned on the lazy model's kwargs is used as-is.
    trainer = LightningDetector(model=LazyClass(_NumClassesDetector, num_classes=9), train_set=_labelled_flux([[1]]))
    trainer.initialize()
    assert isinstance(trainer.model, _NumClassesDetector)
    assert trainer.num_classes == 9 and trainer.model.num_classes == 9


def test_initialize_concrete_model_num_classes_is_authoritative() -> None:
    # A concrete model's head wins over a (mismatched) explicit trainer arg.
    model = _NumClassesDetector(num_classes=4)
    trainer = LightningDetector(model=model, train_set=_labelled_flux([[1]]), num_classes=99)
    trainer.initialize()
    assert trainer.num_classes == 4
    assert trainer.model is model


def test_initialize_model_without_num_classes_param_flowed_plainly() -> None:
    # A lazy model whose builder takes no num_classes is flowed plainly (no kwarg error).
    trainer = LightningDetector(model=LazyClass(_NativeDetector), train_set=_labelled_flux([[1]]), num_classes=3)
    trainer.initialize()
    assert isinstance(trainer.model, _NativeDetector)
    assert trainer.num_classes == 3


def test_trained_model_output_raises_before_run_returns_after_flow() -> None:
    trainer = LightningDetector(model=LazyClass(_NumClassesDetector), train_set=_labelled_flux([[1]]))
    with pytest.raises(RuntimeError, match="only available after run"):
        _ = trainer.trained_model
    trainer.initialize()  # flows the model into a live instance
    assert isinstance(trainer.trained_model, _NumClassesDetector)


def test_model_num_classes_reads_attr_and_torchvision_head() -> None:
    # Explicit num_classes attribute (raidar/ultralytics wrappers).
    assert _model_num_classes(_NumClassesDetector(num_classes=6)) == 6
    # torchvision two-stage box-predictor head shape (Faster R-CNN).
    m = nn.Module()
    m.roi_heads = nn.Module()
    m.roi_heads.box_predictor = nn.Module()
    m.roi_heads.box_predictor.cls_score = nn.Linear(10, 5)
    assert _model_num_classes(m) == 5
    # Neither present → None.
    assert _model_num_classes(nn.Linear(3, 3)) is None
