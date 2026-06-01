"""Tests for :class:`raidar.rfdetr_marainer.RFDETRMarainerModel`.

The marainer-generic wrapper returns :class:`DetectionStepOutput` from a
:class:`DetectionBatchInput` so RF-DETR plugs into the unchanged base
:class:`marainer.trainer.LightningTrainer` flow.
"""

from typing import Any, Dict, List

import pytest
import pytorch_lightning as L
import torch
from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from marainer.trainer import LightningTrainer
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import DetectionBatchInput, DetectionMetricAdapter, TorchvisionDetectionLoss, detection_collate_fn
from raidar.rfdetr_marainer import RFDETRMarainerModel

RESOLUTION = 128


def _make_images(n: int = 2, size: int = RESOLUTION) -> List[Tensor]:
    return [torch.rand(3, size, size) for _ in range(n)]


def _make_targets(n: int = 2) -> List[Dict[str, Tensor]]:
    return [
        {
            "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        }
        for _ in range(n)
    ]


def _synthetic_detection_samples(n: int, size: int = RESOLUTION) -> List[Sample]:
    return [
        Sample(
            input=torch.rand(3, size, size),
            target={
                "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            },
            metadata={},
        )
        for _ in range(n)
    ]


def _build_small_marainer_rfdetr() -> RFDETRMarainerModel:
    return RFDETRMarainerModel(
        num_classes=2,
        variant="nano",
        pretrained_detector=False,
        resolution=RESOLUTION,
        score_threshold=0.0,
    )


@pytest.fixture(scope="module")
def small_marainer_rfdetr() -> RFDETRMarainerModel:
    return _build_small_marainer_rfdetr()


def test_train_mode_returns_losses_only(small_marainer_rfdetr: RFDETRMarainerModel) -> None:
    small_marainer_rfdetr.train()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_marainer_rfdetr(batch)
    assert step["predictions"] is None
    assert step["losses"] is not None
    assert step["losses"]["loss"].requires_grad


def test_train_mode_requires_targets(small_marainer_rfdetr: RFDETRMarainerModel) -> None:
    small_marainer_rfdetr.train()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    with pytest.raises(ValueError, match="requires targets"):
        small_marainer_rfdetr(batch)


def test_eval_with_targets_returns_both(small_marainer_rfdetr: RFDETRMarainerModel) -> None:
    small_marainer_rfdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_marainer_rfdetr(batch)
    assert step["losses"] is not None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_eval_without_targets_returns_predictions_only(
    small_marainer_rfdetr: RFDETRMarainerModel,
) -> None:
    small_marainer_rfdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=None)
    step = small_marainer_rfdetr(batch)
    assert step["losses"] is None
    assert step["predictions"] is not None


def test_compute_val_loss_false_skips_loss(small_marainer_rfdetr: RFDETRMarainerModel) -> None:
    small_marainer_rfdetr.compute_val_loss = False
    small_marainer_rfdetr.eval()
    try:
        batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
        step = small_marainer_rfdetr(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_marainer_rfdetr.compute_val_loss = True


def test_end_to_end_base_trainer_runs_with_marainer_rfdetr(tmp_path: Any) -> None:
    model = _build_small_marainer_rfdetr()
    train_flux = Flux(source=_synthetic_detection_samples(4))
    val_flux = Flux(source=_synthetic_detection_samples(2))

    trainer = LightningTrainer(model=model, train_set=train_flux)
    trainer.val_set = val_flux
    trainer.optimizer = Class(torch.optim.SGD, lr=1e-4)
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
