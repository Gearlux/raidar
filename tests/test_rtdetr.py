"""Tests for the RT-DETR model wrapper and its BN-frozen dual-forward helper.

Generic detection adapter tests live in :mod:`tests.test_detection`; the
parallel YOLO26 tests live in :mod:`tests.test_yolo26`.
"""

from typing import Any, Dict, List, cast

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
from ultralytics.nn.modules.head import RTDETRDecoder

from raidar.detection import DetectionBatchInput, DetectionMetricAdapter, TorchvisionDetectionLoss, detection_collate_fn
from raidar.rtdetr import (
    RTDETRModel,
    _build_rtdetr_batch_dict,
    _compute_rtdetr_loss_without_bn_drift,
    _decode_rtdetr_predictions,
)

# --- fixtures & helpers ----------------------------------------------------

# Smallest practical RT-DETR input. Must be a multiple of stride 32 AND yield
# enough multi-scale feature tokens (h/8)² + (h/16)² + (h/32)² to satisfy the
# decoder's top-k=300 query selection. At imgsz=128 the total is 16²+8²+4²=336.
IMGSZ = 128


def _make_images(n: int = 2, size: int = IMGSZ) -> List[Tensor]:
    return [torch.rand(3, size, size) for _ in range(n)]


def _make_targets(n: int = 2) -> List[Dict[str, Tensor]]:
    targets = []
    for _ in range(n):
        targets.append(
            {
                "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            }
        )
    return targets


def _synthetic_detection_samples(n: int, size: int = IMGSZ) -> List[Sample]:
    samples: List[Sample] = []
    for _ in range(n):
        samples.append(
            Sample(
                input=torch.rand(3, size, size),
                target={
                    "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                },
                metadata={},
            )
        )
    return samples


class _RTDETRLikeWithBN(nn.Module):
    """Mimics RTDETRDetectionModel's API around a real BN layer.

    Forward signature matches the RT-DETR contract: with a batch dict
    (training / loss path), returns ``(scalar_total, items[3])`` — the first
    element is **already a 0-dim scalar**, unlike YOLO26's ``[3]`` per-component
    tensor. With a tensor (eval path), returns ``(predictions, intermediate)``
    where predictions has shape ``[B, 300, 4 + nc]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            stacked = x["img"]
            y = self.head(self.bn(stacked))
            total = y.abs().mean()  # 0-dim scalar (matches RT-DETR loss contract)
            items = torch.tensor([float(y.mean()), float(y.abs().mean()), float(y.var())])
            return total, items
        y = self.head(self.bn(x))
        # (B, 300, 4 + nc=2) synthetic — caller will not inspect contents,
        # only needs the no-grad BN stat assertion path.
        b = x.shape[0]
        preds = torch.zeros((b, 300, 6))
        return preds, {}


# --- _compute_rtdetr_loss_without_bn_drift ---------------------------------


def test_rtdetr_bn_running_stats_preserved_by_loss_recompute() -> None:
    mod = _RTDETRLikeWithBN()
    mod.eval()

    # BatchNorm2d's running_mean / running_var are typed Optional because they
    # can be None when track_running_stats=False; with the default they're
    # always real tensors.
    assert mod.bn.running_mean is not None
    assert mod.bn.running_var is not None
    rm_before = mod.bn.running_mean.clone()
    rv_before = mod.bn.running_var.clone()

    for _ in range(3):
        batch = {"img": torch.stack(_make_images(2))}
        _compute_rtdetr_loss_without_bn_drift(mod, batch)

    assert torch.equal(mod.bn.running_mean, rm_before)
    assert torch.equal(mod.bn.running_var, rv_before)
    assert not mod.training  # restored


def test_rtdetr_compute_losses_returns_no_grad() -> None:
    mod = _RTDETRLikeWithBN()
    mod.eval()
    batch = {"img": torch.stack(_make_images(2))}
    total, items = _compute_rtdetr_loss_without_bn_drift(mod, batch)
    assert total.shape == ()  # 0-dim scalar (RT-DETR contract)
    assert items.shape == (3,)
    assert not total.requires_grad  # computed under torch.no_grad


# --- _build_rtdetr_batch_dict ----------------------------------------------


def test_build_rtdetr_batch_dict_translates_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ), torch.rand(3, IMGSZ, IMGSZ)]
    targets = [
        {
            "boxes": torch.tensor([[32.0, 32.0, 96.0, 96.0]]),  # cxcywh = (0.5, 0.5, 0.5, 0.5)
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 64.0, 128.0]]),  # cxcywh = (0.25, 0.5, 0.5, 1.0)
            "labels": torch.tensor([0], dtype=torch.int64),
        },
    ]
    out = _build_rtdetr_batch_dict(images, targets, IMGSZ)
    assert out["img"].shape == (2, 3, IMGSZ, IMGSZ)
    assert out["cls"].shape == (2, 1)
    assert out["bboxes"].shape == (2, 4)
    assert out["batch_idx"].tolist() == [0.0, 1.0]
    # First box: cxcywh = (0.5, 0.5, 0.5, 0.5)
    assert torch.allclose(out["bboxes"][0], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    # Second box: (0.25, 0.5, 0.5, 1.0)
    assert torch.allclose(out["bboxes"][1], torch.tensor([0.25, 0.5, 0.5, 1.0]))
    # cls becomes float
    assert out["cls"].dtype == torch.float32
    assert out["cls"].flatten().tolist() == [1.0, 0.0]


def test_build_rtdetr_batch_dict_handles_empty_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ)]
    targets: List[Dict[str, Tensor]] = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    out = _build_rtdetr_batch_dict(images, targets, IMGSZ)
    assert out["img"].shape == (1, 3, IMGSZ, IMGSZ)
    assert out["cls"].shape == (0, 1)
    assert out["bboxes"].shape == (0, 4)
    assert out["batch_idx"].shape == (0,)


def test_build_rtdetr_batch_dict_rejects_wrong_size_image() -> None:
    images = [torch.rand(3, 32, 32)]  # not IMGSZ
    targets = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    with pytest.raises(ValueError, match="requires all images to be"):
        _build_rtdetr_batch_dict(images, targets, IMGSZ)


# --- _decode_rtdetr_predictions --------------------------------------------


def test_decode_rtdetr_predictions_filters_by_score_and_lifts_to_pixels() -> None:
    """RT-DETR's decoder returns ``[B, 300, 4 + nc]`` in normalized cxcywh +
    per-class sigmoid space. The decode helper must split, argmax over class,
    threshold, convert cxcywh→xyxy, and multiply by ``imgsz`` to lift into
    pixel coordinates.
    """
    nc = 2
    raw = torch.zeros((2, 300, 4 + nc))
    # Batch 0, query 0: cxcywh=(0.5, 0.5, 0.5, 0.5) → xyxy_norm=(0.25, 0.25, 0.75, 0.75)
    raw[0, 0, :4] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    raw[0, 0, 4:] = torch.tensor([0.9, 0.05])  # max=0.9, cls=0
    # Batch 0, query 1: low score — must be dropped.
    raw[0, 1, :4] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    raw[0, 1, 4:] = torch.tensor([0.05, 0.03])  # max=0.05 < 0.1 threshold
    # Batch 1, query 0: cxcywh=(0.25, 0.5, 0.5, 1.0) → xyxy_norm=(0.0, 0.0, 0.5, 1.0)
    raw[1, 0, :4] = torch.tensor([0.25, 0.5, 0.5, 1.0])
    raw[1, 0, 4:] = torch.tensor([0.0, 0.5])  # max=0.5, cls=1

    preds = _decode_rtdetr_predictions(raw, imgsz=IMGSZ, score_threshold=0.1)

    assert len(preds) == 2

    # Batch 0: one survivor at pixel-space [0.25, 0.25, 0.75, 0.75] * IMGSZ, score 0.9, cls 0
    assert preds[0]["boxes"].shape == (1, 4)
    assert preds[0]["boxes"].dtype == torch.float32
    assert preds[0]["scores"].dtype == torch.float32
    assert preds[0]["labels"].dtype == torch.int64
    expected0 = torch.tensor([0.25, 0.25, 0.75, 0.75]) * float(IMGSZ)
    assert torch.allclose(preds[0]["boxes"][0], expected0)
    assert preds[0]["scores"].tolist() == pytest.approx([0.9])
    assert preds[0]["labels"].tolist() == [0]

    # Batch 1: one survivor at pixel-space [0.0, 0.0, 0.5, 1.0] * IMGSZ, score 0.5, cls 1
    assert preds[1]["boxes"].shape == (1, 4)
    expected1 = torch.tensor([0.0, 0.0, 0.5, 1.0]) * float(IMGSZ)
    assert torch.allclose(preds[1]["boxes"][0], expected1)
    assert preds[1]["scores"].tolist() == pytest.approx([0.5])
    assert preds[1]["labels"].tolist() == [1]


# --- RTDETRModel construction ----------------------------------------------


def test_rtdetr_rejects_imgsz_not_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        RTDETRModel(num_classes=2, imgsz=63, pretrained_detector=False)


def test_rtdetr_pretrained_resnet_variants_rejected() -> None:
    """Resnet variants ship architecture-only — no public pretrained .pt.
    Asking for ``pretrained_detector=True`` must fail upfront with a clear
    message rather than blowing up later inside ``attempt_download_asset``.
    """
    with pytest.raises(ValueError, match="resnet"):
        RTDETRModel(num_classes=2, variant="resnet50", pretrained_detector=True, imgsz=IMGSZ)


def _build_small_rtdetr() -> RTDETRModel:
    # Offline build (no weights download) at the smallest valid imgsz.
    return RTDETRModel(
        num_classes=2,
        variant="l",
        pretrained_detector=False,
        imgsz=IMGSZ,
        score_threshold=0.0,  # don't filter — tests want the raw output count
    )


@pytest.fixture(scope="module")
def small_rtdetr() -> RTDETRModel:
    return _build_small_rtdetr()


def test_rtdetr_train_mode_returns_loss_only(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.train()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_rtdetr(batch)
    assert step["predictions"] is None
    assert step["losses"] is not None
    assert "loss" in step["losses"]
    loss = step["losses"]["loss"]
    assert loss.dim() == 0
    assert loss.requires_grad


def test_rtdetr_train_mode_requires_targets(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.train()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    with pytest.raises(ValueError, match="requires targets"):
        small_rtdetr(batch)


def test_rtdetr_eval_with_targets_returns_both(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_rtdetr(batch)
    assert step["losses"] is not None
    assert "loss" in step["losses"]
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2
    for p in step["predictions"]:
        assert set(p.keys()) == {"boxes", "scores", "labels"}
        assert p["boxes"].dtype == torch.float32
        assert p["scores"].dtype == torch.float32
        assert p["labels"].dtype == torch.int64


def test_rtdetr_eval_without_targets_returns_predictions_only(
    small_rtdetr: RTDETRModel,
) -> None:
    small_rtdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=None)
    step = small_rtdetr(batch)
    assert step["losses"] is None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_rtdetr_compute_val_loss_false_skips_loss(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.compute_val_loss = False
    small_rtdetr.eval()
    try:
        batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
        step = small_rtdetr(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_rtdetr.compute_val_loss = True


def test_rtdetr_pretrained_detector_swaps_head_for_num_classes() -> None:
    """`pretrained_detector=True` keeps backbone/encoder/bbox-head weights but
    fits the classifier to ``num_classes``.

    Building RT-DETR with ``num_classes != 80`` triggers our weight-load path
    that copies all shape-compatible parameters from the COCO .pt and skips
    the decoder's classification heads (which are nc-specific). The final
    ``RTDETRDecoder`` head must have ``nc == num_classes`` and the canonical
    ``num_queries == 300``.

    Skip cleanly when the weights mirror is unreachable so CI without
    network access still passes the rest of the suite.
    """
    try:
        model = RTDETRModel(
            num_classes=3,
            variant="l",
            pretrained_detector=True,
            imgsz=IMGSZ,
        )
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"Skipping: pretrained weights unavailable ({exc})")

    head = cast(RTDETRDecoder, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert isinstance(head, RTDETRDecoder)
    assert head.nc == 3
    assert head.num_queries == 300

    # Forward must still run — pretrained body + fresh classifier head.
    model.eval()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    step = model(batch)
    assert step["predictions"] is not None and len(step["predictions"]) == 1


def test_rtdetr_pretrained_detector_default_off() -> None:
    """Default ``pretrained_detector=False`` runs offline (no weights download)."""
    model = RTDETRModel(
        num_classes=2,
        variant="l",
        pretrained_detector=False,
        imgsz=IMGSZ,
    )
    head = cast(RTDETRDecoder, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.nc == 2
    assert head.num_queries == 300


# --- End-to-end: adapter pattern inside the unchanged base Trainer ---------


def test_end_to_end_base_trainer_runs_with_rtdetr(tmp_path: Any) -> None:
    """The whole point of raidar's adapter pattern: RTDETRModel drops into
    the base Marainer Trainer with the same TorchvisionDetectionLoss and
    DetectionMetricAdapter the YOLO26 / Faster R-CNN flow uses."""
    model = _build_small_rtdetr()
    train_samples = _synthetic_detection_samples(4)
    val_samples = _synthetic_detection_samples(2)
    train_flux = Flux(source=train_samples)
    val_flux = Flux(source=val_samples)

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
    assert any(k.startswith("train/loss") for k in metrics), f"expected train/loss in {list(metrics)}"
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss), f"train/loss not finite: {train_loss}"
    assert any(k.startswith("val/loss") for k in metrics), f"expected val/loss in {list(metrics)}"
    val_loss = next(v for k, v in metrics.items() if k.startswith("val/loss"))
    assert torch.isfinite(val_loss)
