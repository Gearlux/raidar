"""Tests for the YOLO26 model wrapper and its BN-frozen dual-forward helper.

Generic detection adapter tests live in :mod:`tests.test_detection`; the
parallel Faster R-CNN tests live in :mod:`tests.test_faster_rcnn`.
"""

from typing import Any, Dict, List

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
    TorchvisionDetectionLoss,
    detection_collate_fn,
)
from raidar.yolo26 import (
    YOLO26Model,
    _build_yolo_batch_dict,
    _compute_yolo_loss_without_bn_drift,
    _decode_yolo_predictions,
)

# --- fixtures & helpers ----------------------------------------------------

# Smallest valid YOLO input (must be a multiple of stride=32).
IMGSZ = 64


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
                    "boxes": torch.tensor(
                        [[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32
                    ),
                    "labels": torch.tensor([0], dtype=torch.int64),
                },
                metadata={},
            )
        )
    return samples


class _YOLOLikeWithBN(nn.Module):
    """Mimics Ultralytics' DetectionModel API around a real BN layer.

    Forward signature matches the YOLO contract: with a batch dict (training
    / loss path), returns ``(per_component_tensor[3], detached_items[3])``;
    with a tensor (eval path), returns ``(predictions, intermediate_dict)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            stacked = x["img"]
            y = self.head(self.bn(stacked))
            grad_loss = torch.stack([y.mean(), y.abs().mean(), y.var()])
            detached = grad_loss.detach()
            return grad_loss, detached
        y = self.head(self.bn(x))
        # (B, N=4, 6) synthetic: 4 anchors per image, [x1, y1, x2, y2, score, cls]
        b = x.shape[0]
        preds = torch.zeros((b, 4, 6))
        return preds, {}


# --- _compute_yolo_loss_without_bn_drift -----------------------------------


def test_yolo_bn_running_stats_preserved_by_loss_recompute() -> None:
    mod = _YOLOLikeWithBN()
    mod.eval()

    rm_before = mod.bn.running_mean.clone()
    rv_before = mod.bn.running_var.clone()

    for _ in range(3):
        batch = {"img": torch.stack(_make_images(2))}
        _compute_yolo_loss_without_bn_drift(mod, batch)

    assert torch.equal(mod.bn.running_mean, rm_before)
    assert torch.equal(mod.bn.running_var, rv_before)
    assert not mod.training  # restored


def test_yolo_compute_losses_returns_no_grad() -> None:
    mod = _YOLOLikeWithBN()
    mod.eval()
    batch = {"img": torch.stack(_make_images(2))}
    grad_loss, items = _compute_yolo_loss_without_bn_drift(mod, batch)
    assert grad_loss.shape == (3,)
    assert items.shape == (3,)
    assert not grad_loss.requires_grad  # computed under torch.no_grad


# --- _build_yolo_batch_dict ------------------------------------------------


def test_build_yolo_batch_dict_translates_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ), torch.rand(3, IMGSZ, IMGSZ)]
    targets = [
        {
            "boxes": torch.tensor(
                [[16.0, 16.0, 48.0, 48.0]]
            ),  # cxcywh = (0.5, 0.5, 0.5, 0.5)
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor(
                [[0.0, 0.0, 32.0, 64.0]]
            ),  # cxcywh = (0.25, 0.5, 0.5, 1.0)
            "labels": torch.tensor([0], dtype=torch.int64),
        },
    ]
    out = _build_yolo_batch_dict(images, targets, IMGSZ)
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


def test_build_yolo_batch_dict_handles_empty_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ)]
    targets: List[Dict[str, Tensor]] = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    out = _build_yolo_batch_dict(images, targets, IMGSZ)
    assert out["img"].shape == (1, 3, IMGSZ, IMGSZ)
    assert out["cls"].shape == (0, 1)
    assert out["bboxes"].shape == (0, 4)
    assert out["batch_idx"].shape == (0,)


def test_build_yolo_batch_dict_rejects_wrong_size_image() -> None:
    images = [torch.rand(3, 32, 32)]  # not IMGSZ
    targets = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    with pytest.raises(ValueError, match="requires all images to be"):
        _build_yolo_batch_dict(images, targets, IMGSZ)


# --- _decode_yolo_predictions ----------------------------------------------


def test_decode_yolo_predictions_filters_by_score() -> None:
    raw = torch.tensor(
        [
            [
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],  # keep (high score)
                [5.0, 5.0, 15.0, 15.0, 0.05, 1.0],  # drop (low score)
            ],
            [
                [1.0, 1.0, 5.0, 5.0, 0.5, 1.0],  # keep
                [2.0, 2.0, 8.0, 8.0, 0.5, 0.0],  # keep
            ],
        ]
    )
    preds = _decode_yolo_predictions(raw, score_threshold=0.1)
    assert len(preds) == 2
    assert preds[0]["boxes"].shape == (1, 4)
    assert preds[0]["scores"].tolist() == pytest.approx([0.9])
    assert preds[0]["labels"].tolist() == [0]
    assert preds[1]["boxes"].shape == (2, 4)
    assert preds[1]["labels"].tolist() == [1, 0]


# --- YOLO26Model construction ----------------------------------------------


def test_yolo26_rejects_imgsz_not_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        YOLO26Model(num_classes=2, imgsz=63, pretrained_detector=False)


def _build_small_yolo26() -> YOLO26Model:
    # Offline build (no weights download) at the smallest valid imgsz.
    return YOLO26Model(
        num_classes=2,
        variant="n",
        pretrained_detector=False,
        imgsz=IMGSZ,
        score_threshold=0.0,  # don't filter — tests want the raw output count
    )


@pytest.fixture(scope="module")
def small_yolo26() -> YOLO26Model:
    return _build_small_yolo26()


def test_yolo26_train_mode_returns_loss_only(small_yolo26: YOLO26Model) -> None:
    small_yolo26.train()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_yolo26(batch)
    assert step["predictions"] is None
    assert step["losses"] is not None
    assert "loss" in step["losses"]
    loss = step["losses"]["loss"]
    assert loss.dim() == 0
    assert loss.requires_grad


def test_yolo26_train_mode_requires_targets(small_yolo26: YOLO26Model) -> None:
    small_yolo26.train()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    with pytest.raises(ValueError, match="requires targets"):
        small_yolo26(batch)


def test_yolo26_eval_with_targets_returns_both(small_yolo26: YOLO26Model) -> None:
    small_yolo26.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_yolo26(batch)
    assert step["losses"] is not None
    assert "loss" in step["losses"]
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2
    for p in step["predictions"]:
        assert set(p.keys()) == {"boxes", "scores", "labels"}
        assert p["boxes"].dtype == torch.float32
        assert p["scores"].dtype == torch.float32
        assert p["labels"].dtype == torch.int64


def test_yolo26_eval_without_targets_returns_predictions_only(
    small_yolo26: YOLO26Model,
) -> None:
    small_yolo26.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=None)
    step = small_yolo26(batch)
    assert step["losses"] is None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_yolo26_compute_val_loss_false_skips_loss(small_yolo26: YOLO26Model) -> None:
    small_yolo26.compute_val_loss = False
    small_yolo26.eval()
    try:
        batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
        step = small_yolo26(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_yolo26.compute_val_loss = True


def test_yolo26_pretrained_detector_swaps_head_for_num_classes() -> None:
    """`pretrained_detector=True` keeps backbone/neck/regression weights but
    fits the classifier to ``num_classes``.

    Building YOLO26 with ``num_classes != 80`` triggers our weight-load path
    that copies all shape-compatible parameters from the COCO .pt and skips
    the head's classification convolutions (which are nc-specific). The
    final ``Detect`` head must have ``nc == num_classes``.

    Skip cleanly when the weights mirror is unreachable so CI without
    network access still passes the rest of the suite.
    """
    try:
        model = YOLO26Model(
            num_classes=3,
            variant="n",
            pretrained_detector=True,
            imgsz=IMGSZ,
        )
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"Skipping: pretrained weights unavailable ({exc})")

    head = model.detector.model[-1]  # type: ignore[index, union-attr]
    assert head.nc == 3
    # End-to-end head is the YOLO26 default and the configuration we exercise.
    assert head.end2end is True

    # Forward must still run — pretrained body + fresh classifier head.
    model.eval()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    step = model(batch)
    assert step["predictions"] is not None and len(step["predictions"]) == 1


def test_yolo26_pretrained_detector_default_off() -> None:
    """Default ``pretrained_detector=False`` runs offline (no weights download)."""
    model = YOLO26Model(
        num_classes=2,
        variant="n",
        pretrained_detector=False,
        imgsz=IMGSZ,
    )
    head = model.detector.model[-1]  # type: ignore[index, union-attr]
    assert head.nc == 2


# --- End-to-end: adapter pattern inside the unchanged base Trainer ---------


def test_end_to_end_base_trainer_runs_with_yolo26(tmp_path: Any) -> None:
    """The whole point of raidar's adapter pattern: YOLO26Model drops into
    the base Marainer Trainer with the same TorchvisionDetectionLoss and
    DetectionMetricAdapter the Faster R-CNN flow uses."""
    model = _build_small_yolo26()
    train_samples = _synthetic_detection_samples(4)
    val_samples = _synthetic_detection_samples(2)
    train_flux = Flux(source=train_samples)
    val_flux = Flux(source=val_samples)

    trainer = Trainer(
        model=model,
        train_set=train_flux,
        val_set=val_flux,
        optimizer=Class(torch.optim.SGD, lr=1e-3),
    )
    trainer.loss_fn = TorchvisionDetectionLoss()
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
    assert any(
        k.startswith("train/loss") for k in metrics
    ), f"expected train/loss in {list(metrics)}"
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss), f"train/loss not finite: {train_loss}"
    assert any(
        k.startswith("val/loss") for k in metrics
    ), f"expected val/loss in {list(metrics)}"
    val_loss = next(v for k, v in metrics.items() if k.startswith("val/loss"))
    assert torch.isfinite(val_loss)
