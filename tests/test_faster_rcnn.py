"""Tests for the Faster R-CNN model wrapper and its BN-frozen dual-forward helper.

Generic detection adapter tests (DetectionBatchInput, TorchvisionDetectionLoss,
DetectionMetricAdapter, detection_collate_fn) live in
:mod:`tests.test_detection`.
"""

from typing import Any, Dict, List, Optional, cast

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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from raidar.detection import DetectionBatchInput, DetectionMetricAdapter, TorchvisionDetectionLoss, detection_collate_fn
from raidar.faster_rcnn import FasterRCNNModel, _compute_losses_without_bn_drift

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
                    "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
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


# --- _compute_losses_without_bn_drift --------------------------------------


def test_bn_running_stats_preserved_by_eval_loss_computation() -> None:
    mod = _DetectorWithBN()
    mod.eval()

    # BatchNorm2d's running_mean / running_var are typed Optional because they
    # can be None when track_running_stats=False; with the default they're
    # always real tensors.
    assert mod.bn.running_mean is not None
    assert mod.bn.running_var is not None
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
    batch = DetectionBatchInput(images=_make_images(2, size=64), targets=_make_targets(2))
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
    batch = DetectionBatchInput(images=_make_images(2, size=64), targets=_make_targets(2))
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
        batch = DetectionBatchInput(images=_make_images(2, size=64), targets=_make_targets(2))
        step = small_faster_rcnn(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_faster_rcnn.compute_val_loss = True


def test_faster_rcnn_pretrained_detector_swaps_predictor_for_num_classes() -> None:
    """`pretrained_detector=True` keeps backbone/RPN/ROI weights but swaps the head.

    Building the full pretrained model loads a 91-class COCO predictor; the
    constructor must replace it with a fresh ``FastRCNNPredictor`` whose
    ``cls_score`` has the configured ``num_classes`` outputs (and whose weights
    are random — distinct from the COCO weights even bit-pattern-wise).

    This test forces the pretrained-weights download path; skip cleanly when
    the runtime can't reach the torchvision weights mirror so CI without
    network access still passes the rest of the suite.
    """
    try:
        model = FasterRCNNModel(
            num_classes=3,  # 1 background + 2 foreground
            variant="v2",
            pretrained_detector=True,
            trainable_backbone_layers=1,
            min_size=64,
            max_size=128,
        )
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"Skipping: pretrained weights unavailable ({exc})")

    predictor = cast(FastRCNNPredictor, model.detector.roi_heads.box_predictor)  # type: ignore[union-attr]
    assert isinstance(predictor, FastRCNNPredictor)
    # cls_score is `[in_features → num_classes]` for FastRCNNPredictor.
    assert predictor.cls_score.out_features == 3
    # bbox_pred is `[in_features → num_classes * 4]`.
    assert predictor.bbox_pred.out_features == 12

    # The model must still run a forward pass — pretrained body + fresh head.
    model.eval()
    batch = DetectionBatchInput(images=_make_images(1, size=64), targets=None)
    step = model(batch)
    assert step["predictions"] is not None and len(step["predictions"]) == 1


def test_faster_rcnn_pretrained_detector_default_off() -> None:
    """Default ``pretrained_detector=False`` runs offline (no weights download)."""
    # Just instantiating succeeds — this is the path used by the small fixture.
    model = FasterRCNNModel(
        num_classes=2,
        variant="v2",
        pretrained_backbone=False,
        pretrained_detector=False,
        trainable_backbone_layers=1,
        min_size=64,
        max_size=128,
    )
    predictor = cast(FastRCNNPredictor, model.detector.roi_heads.box_predictor)  # type: ignore[union-attr]
    assert predictor.cls_score.out_features == 2


def test_faster_rcnn_loads_from_checkpoint_path(tmp_path: Any) -> None:
    """``checkpoint_path`` round-trips a Lightning-style ``state_dict`` into the model."""
    src = _build_small_faster_rcnn()
    # Mutate one parameter so we can detect whether the load reached the model.
    src_predictor = cast(FastRCNNPredictor, src.detector.roi_heads.box_predictor)  # type: ignore[union-attr]
    sentinel = src_predictor.cls_score.weight.data
    sentinel.fill_(0.42)

    # Lightning wraps the model under ``self.model = model``, so the saved
    # state_dict's keys are prefixed with ``model.``. Mimic that here.
    wrapped = {f"model.{k}": v for k, v in src.state_dict().items()}
    ckpt = tmp_path / "fake_lightning.ckpt"
    torch.save({"state_dict": wrapped, "epoch": 1}, ckpt)

    loaded = FasterRCNNModel(
        num_classes=2,
        variant="v2",
        pretrained_backbone=False,
        pretrained_detector=False,
        trainable_backbone_layers=1,
        min_size=64,
        max_size=128,
        checkpoint_path=str(ckpt),
    )
    loaded_predictor = cast(FastRCNNPredictor, loaded.detector.roi_heads.box_predictor)  # type: ignore[union-attr]
    restored = loaded_predictor.cls_score.weight.data
    assert torch.equal(restored, sentinel)


# --- End-to-end: adapter pattern inside the unchanged base Trainer ---------


def test_end_to_end_base_trainer_runs_with_detection_adapters(tmp_path: Any) -> None:
    # Fresh model — this test trains, so keep it isolated from fixtures.
    model = _build_small_faster_rcnn()
    train_samples = _synthetic_detection_samples(4, size=64)
    val_samples = _synthetic_detection_samples(2, size=64)
    train_flux = Flux(source=train_samples)
    val_flux = Flux(source=val_samples)

    trainer = LightningTrainer(model=model, train_set=train_flux)
    trainer.val_set = val_flux
    trainer.optimizer = Class(torch.optim.SGD, lr=1e-3)
    trainer.loss_fn = TorchvisionDetectionLoss(  # type: ignore[assignment]
        weights={"loss_classifier": 1.0, "loss_box_reg": 1.0}
    )
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
    # train/loss logged by the base Trainer step.
    assert any(k.startswith("train/loss") for k in metrics), f"expected train/loss in {list(metrics)}"
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss), f"train/loss not finite: {train_loss}"
    # val/loss is now a REAL scalar (from the BN-frozen dual-forward), not a zero placeholder.
    assert any(k.startswith("val/loss") for k in metrics), f"expected val/loss in {list(metrics)}"
    val_loss = next(v for k, v in metrics.items() if k.startswith("val/loss"))
    assert torch.isfinite(val_loss)
