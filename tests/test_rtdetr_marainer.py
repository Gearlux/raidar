"""Tests for :class:`raidar.rtdetr_marainer.RTDETRMarainerModel`.

The marainer-generic wrapper returns :class:`DetectionStepOutput` from a
:class:`DetectionBatchInput` so RT-DETR plugs into the unchanged base
:class:`marainer.trainer.LightningTrainer` flow.
"""

from typing import Any, Dict, List

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from marainer.trainer import LightningTrainer
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import DetectionBatchInput, DetectionMetricAdapter, TorchvisionDetectionLoss, detection_collate_fn
from raidar.rtdetr_marainer import RTDETRMarainerModel, _compute_rtdetr_loss_without_bn_drift

IMGSZ = 128


def _make_images(n: int = 2, size: int = IMGSZ) -> List[Tensor]:
    return [torch.rand(3, size, size) for _ in range(n)]


def _make_targets(n: int = 2) -> List[Dict[str, Tensor]]:
    return [
        {
            "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        }
        for _ in range(n)
    ]


def _synthetic_detection_samples(n: int, size: int = IMGSZ) -> List[Sample]:
    return [
        Sample(
            input=torch.rand(3, size, size),
            target={
                "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            },
            metadata={},
        )
        for _ in range(n)
    ]


class _RTDETRLikeWithBN(nn.Module):
    """Mimics RTDETRDetectionModel's API around a real BN layer."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            stacked = x["img"]
            y = self.head(self.bn(stacked))
            total = y.abs().mean()
            items = torch.tensor([float(y.mean()), float(y.abs().mean()), float(y.var())])
            return total, items
        y = self.head(self.bn(x))
        b = x.shape[0]
        preds = torch.zeros((b, 300, 6))
        return preds, {}


# --- BN-frozen helper ------------------------------------------------------


def test_rtdetr_bn_running_stats_preserved_by_loss_recompute() -> None:
    mod = _RTDETRLikeWithBN()
    mod.eval()
    assert mod.bn.running_mean is not None
    assert mod.bn.running_var is not None
    rm_before = mod.bn.running_mean.clone()
    rv_before = mod.bn.running_var.clone()
    for _ in range(3):
        batch = {"img": torch.stack(_make_images(2))}
        _compute_rtdetr_loss_without_bn_drift(mod, batch)
    assert torch.equal(mod.bn.running_mean, rm_before)
    assert torch.equal(mod.bn.running_var, rv_before)
    assert not mod.training


def test_rtdetr_compute_losses_returns_no_grad() -> None:
    mod = _RTDETRLikeWithBN()
    mod.eval()
    batch = {"img": torch.stack(_make_images(2))}
    total, items = _compute_rtdetr_loss_without_bn_drift(mod, batch)
    assert total.shape == ()
    assert items.shape == (3,)
    assert not total.requires_grad


# --- RTDETRMarainerModel forward ------------------------------------------


def _build_small_marainer_rtdetr() -> RTDETRMarainerModel:
    return RTDETRMarainerModel(
        num_classes=2,
        variant="l",
        pretrained_detector=False,
        imgsz=IMGSZ,
        score_threshold=0.0,
    )


@pytest.fixture(scope="module")
def small_marainer_rtdetr() -> RTDETRMarainerModel:
    return _build_small_marainer_rtdetr()


def test_train_mode_returns_losses_only(small_marainer_rtdetr: RTDETRMarainerModel) -> None:
    small_marainer_rtdetr.train()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_marainer_rtdetr(batch)
    assert step["predictions"] is None
    assert step["losses"] is not None
    assert step["losses"]["loss"].requires_grad


def test_train_mode_requires_targets(small_marainer_rtdetr: RTDETRMarainerModel) -> None:
    small_marainer_rtdetr.train()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    with pytest.raises(ValueError, match="requires targets"):
        small_marainer_rtdetr(batch)


def test_eval_with_targets_returns_both(small_marainer_rtdetr: RTDETRMarainerModel) -> None:
    small_marainer_rtdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_marainer_rtdetr(batch)
    assert step["losses"] is not None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_eval_without_targets_returns_predictions_only(
    small_marainer_rtdetr: RTDETRMarainerModel,
) -> None:
    small_marainer_rtdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=None)
    step = small_marainer_rtdetr(batch)
    assert step["losses"] is None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_compute_val_loss_false_skips_loss(small_marainer_rtdetr: RTDETRMarainerModel) -> None:
    small_marainer_rtdetr.compute_val_loss = False
    small_marainer_rtdetr.eval()
    try:
        batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
        step = small_marainer_rtdetr(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_marainer_rtdetr.compute_val_loss = True


# --- End-to-end with base marainer LightningTrainer + adapters ------------


def test_end_to_end_base_trainer_runs_with_marainer_rtdetr(tmp_path: Any) -> None:
    model = _build_small_marainer_rtdetr()
    train_flux = Flux(source=_synthetic_detection_samples(4))
    val_flux = Flux(source=_synthetic_detection_samples(2))

    trainer = LightningTrainer(
        model=model,
        train_set=train_flux,
        val_set=val_flux,
        optimizer=Class(torch.optim.SGD, lr=1e-3),
    )
    trainer.loss_fn = TorchvisionDetectionLoss()  # type: ignore[assignment]
    trainer.val_metrics = DetectionMetricAdapter(MeanAveragePrecision(iou_type="bbox"))
    trainer.train_loader = DataLoader(train_flux, batch_size=2, collate_fn=detection_collate_fn)
    trainer.val_loader = DataLoader(val_flux, batch_size=2, collate_fn=detection_collate_fn)

    lightning_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
    )
    lightning_trainer.fit(
        model=trainer,
        train_dataloaders=trainer.train_loader,
        val_dataloaders=trainer.val_loader,
    )
    metrics = lightning_trainer.callback_metrics
    assert any(k.startswith("train/loss") for k in metrics)
    assert any(k.startswith("val/loss") for k in metrics)
