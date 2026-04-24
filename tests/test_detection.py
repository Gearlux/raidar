"""Tests for the detection adapters wired into the base Marainer Trainer."""

from typing import Any, Dict, List, Optional

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from marainer.trainer import Trainer
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from raidar.detection import (
    DetectionBatchInput,
    DetectionMetricAdapter,
    DetectionStepOutput,
    FasterRCNNLoss,
    FasterRCNNModel,
    _compute_losses_without_bn_drift,
    detection_collate_fn,
)

# --- fixtures & helpers ----------------------------------------------------


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


class _DetectorWithBN(nn.Module):
    """Mimics the torchvision detector API around a real BN layer.

    In train mode with targets, returns a loss dict (scalar losses that pull
    from the BN'd activations, so any running-stat drift would be observable).
    In eval mode, returns the list-of-dicts predictions shape torchvision uses.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 1, 3, padding=1)

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Any:
        stacked = torch.stack(images)
        y = self.head(self.bn(stacked))
        if self.training:
            assert targets is not None
            return {
                "loss_a": y.mean(),
                "loss_b": y.abs().mean(),
            }
        return [
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
            for _ in images
        ]


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


# --- _compute_losses_without_bn_drift --------------------------------------


def test_bn_running_stats_preserved_by_eval_loss_computation() -> None:
    mod = _DetectorWithBN()
    mod.eval()

    # Snapshot BN running stats.
    rm_before = mod.bn.running_mean.clone()
    rv_before = mod.bn.running_var.clone()

    for _ in range(3):
        _compute_losses_without_bn_drift(mod, _make_images(2), _make_targets(2))

    assert torch.equal(mod.bn.running_mean, rm_before)
    assert torch.equal(mod.bn.running_var, rv_before)
    assert not mod.training  # restored


def test_compute_losses_returns_dict_in_eval_mode() -> None:
    mod = _DetectorWithBN()
    mod.eval()
    losses = _compute_losses_without_bn_drift(mod, _make_images(2), _make_targets(2))
    assert isinstance(losses, dict)
    assert set(losses.keys()) == {"loss_a", "loss_b"}
    for v in losses.values():
        assert isinstance(v, Tensor)
        assert v.dim() == 0
        assert not v.requires_grad  # computed under torch.no_grad


# --- FasterRCNNLoss --------------------------------------------------------


def test_faster_rcnn_loss_sums_dict_with_weights() -> None:
    loss_fn = FasterRCNNLoss(weights={"loss_a": 2.0})
    step: DetectionStepOutput = DetectionStepOutput(
        losses={"loss_a": torch.tensor(3.0), "loss_b": torch.tensor(1.0)},
        predictions=None,
    )
    out = loss_fn(step, target=None)
    assert torch.allclose(out, torch.tensor(7.0))  # 2.0*3 + 1.0*1


def test_faster_rcnn_loss_zero_when_losses_is_none() -> None:
    loss_fn = FasterRCNNLoss()
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


# --- FasterRCNNModel (real torchvision, small inputs) ----------------------


def _build_small_faster_rcnn() -> FasterRCNNModel:
    # No pretrained weights so the test runs offline; v2 for speed consistency.
    return FasterRCNNModel(
        num_classes=2,
        variant="v2",
        pretrained_backbone=False,
        trainable_backbone_layers=1,
        min_size=64,
        max_size=128,
    )


@pytest.fixture(scope="module")
def small_faster_rcnn() -> FasterRCNNModel:
    return _build_small_faster_rcnn()


def test_faster_rcnn_train_mode_returns_losses_only(
    small_faster_rcnn: FasterRCNNModel,
) -> None:
    small_faster_rcnn.train()
    batch = DetectionBatchInput(
        images=_make_images(2, size=64), targets=_make_targets(2)
    )
    step = small_faster_rcnn(batch)
    assert step["predictions"] is None
    assert step["losses"] is not None
    # Torchvision's four standard loss keys are present (plus possibly more in v2).
    for key in [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]:
        assert key in step["losses"], f"expected torchvision loss key {key!r}"
        assert step["losses"][key].dim() == 0


def test_faster_rcnn_eval_with_targets_returns_both(
    small_faster_rcnn: FasterRCNNModel,
) -> None:
    small_faster_rcnn.eval()
    batch = DetectionBatchInput(
        images=_make_images(2, size=64), targets=_make_targets(2)
    )
    step = small_faster_rcnn(batch)
    assert step["losses"] is not None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2
    for p in step["predictions"]:
        assert set(p.keys()) == {"boxes", "scores", "labels"}
        assert p["boxes"].dtype == torch.float32
        assert p["scores"].dtype == torch.float32
        assert p["labels"].dtype == torch.int64


def test_faster_rcnn_eval_without_targets_returns_predictions_only(
    small_faster_rcnn: FasterRCNNModel,
) -> None:
    small_faster_rcnn.eval()
    batch = DetectionBatchInput(images=_make_images(2, size=64), targets=None)
    step = small_faster_rcnn(batch)
    assert step["losses"] is None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_faster_rcnn_compute_val_loss_false_skips_dual_forward(
    small_faster_rcnn: FasterRCNNModel,
) -> None:
    small_faster_rcnn.compute_val_loss = False
    small_faster_rcnn.eval()
    try:
        batch = DetectionBatchInput(
            images=_make_images(2, size=64), targets=_make_targets(2)
        )
        step = small_faster_rcnn(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_faster_rcnn.compute_val_loss = True


# --- End-to-end: adapter pattern inside the unchanged base Trainer ---------


def test_end_to_end_base_trainer_runs_with_detection_adapters(tmp_path: Any) -> None:
    # Fresh model — this test trains, so keep it isolated from fixtures.
    model = _build_small_faster_rcnn()
    train_samples = _synthetic_detection_samples(4, size=64)
    val_samples = _synthetic_detection_samples(2, size=64)
    train_flux = Flux(source=train_samples)
    val_flux = Flux(source=val_samples)

    trainer = Trainer(
        model=model,
        train_set=train_flux,
        val_set=val_flux,
        optimizer=Class(torch.optim.SGD, lr=1e-3),
    )
    trainer.loss_fn = FasterRCNNLoss(
        weights={"loss_classifier": 1.0, "loss_box_reg": 1.0}
    )
    trainer.val_metrics = DetectionMetricAdapter(MeanAveragePrecision(iou_type="bbox"))
    trainer.train_loader = DataLoader(
        train_flux, batch_size=2, collate_fn=detection_collate_fn
    )
    trainer.val_loader = DataLoader(
        val_flux, batch_size=2, collate_fn=detection_collate_fn
    )

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
    # train/loss logged by the base Trainer step.
    assert any(
        k.startswith("train/loss") for k in metrics
    ), f"expected train/loss in {list(metrics)}"
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss), f"train/loss not finite: {train_loss}"
    # val/loss is now a REAL scalar (from the BN-frozen dual-forward), not a zero placeholder.
    assert any(
        k.startswith("val/loss") for k in metrics
    ), f"expected val/loss in {list(metrics)}"
    val_loss = next(v for k, v in metrics.items() if k.startswith("val/loss"))
    assert torch.isfinite(val_loss)
