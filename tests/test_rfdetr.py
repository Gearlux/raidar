"""Tests for the RF-DETR model wrapper.

Generic detection adapter tests live in :mod:`tests.test_detection`; the
parallel YOLO26 / RT-DETR tests live in :mod:`tests.test_yolo26` and
:mod:`tests.test_rtdetr`.
"""

from typing import Any, Dict, List

import pytest
import pytorch_lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from marainer.trainer import LightningTrainer
from raidar.detection import (
    DetectionBatchInput,
    DetectionMetricAdapter,
    TorchvisionDetectionLoss,
    detection_collate_fn,
)
from raidar.rfdetr import RFDETRModel, _build_rfdetr_targets, _decode_rfdetr_predictions

# --- fixtures & helpers ----------------------------------------------------

# Smallest practical RF-DETR input. Nano hardcodes patch_size=16 and
# num_windows=2, so resolution must be a multiple of 32. At resolution=128
# each attention window sees 4x4 patches, which keeps the transformer
# non-degenerate while staying CPU-fast. Offline-ness is provided by
# `pretrained_detector=False` below — Nano's patch_size=16 disagrees with
# DINOv2's native 14 at every resolution, so the backbone bootstraps from
# random regardless.
RESOLUTION = 128


def _make_images(n: int = 2, size: int = RESOLUTION) -> List[Tensor]:
    return [torch.rand(3, size, size) for _ in range(n)]


def _make_targets(n: int = 2) -> List[Dict[str, Tensor]]:
    targets = []
    for _ in range(n):
        targets.append(
            {
                "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            }
        )
    return targets


def _synthetic_detection_samples(n: int, size: int = RESOLUTION) -> List[Sample]:
    samples: List[Sample] = []
    for _ in range(n):
        samples.append(
            Sample(
                input=torch.rand(3, size, size),
                target={
                    "boxes": torch.tensor(
                        [[8.0, 8.0, 24.0, 24.0]], dtype=torch.float32
                    ),
                    "labels": torch.tensor([0], dtype=torch.int64),
                },
                metadata={},
            )
        )
    return samples


# --- _build_rfdetr_targets -------------------------------------------------


def test_build_rfdetr_targets_translates_xyxy_to_cxcywh_normalized() -> None:
    """RF-DETR's criterion expects per-image targets with cxcywh-normalized
    boxes; raidar carries them in xyxy-pixel format. The helper converts on
    a per-image basis and preserves int64 labels."""
    targets = [
        {
            "boxes": torch.tensor(
                [[32.0, 32.0, 96.0, 96.0]]
            ),  # cxcywh = (0.5, 0.5, 0.5, 0.5)
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor(
                [[0.0, 0.0, 64.0, 128.0]]
            ),  # cxcywh = (0.25, 0.5, 0.5, 1.0)
            "labels": torch.tensor([0], dtype=torch.int64),
        },
    ]
    out = _build_rfdetr_targets(
        targets, resolution=RESOLUTION, device=torch.device("cpu")
    )
    assert len(out) == 2
    assert torch.allclose(out[0]["boxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(out[1]["boxes"], torch.tensor([[0.25, 0.5, 0.5, 1.0]]))
    assert out[0]["boxes"].dtype == torch.float32
    assert out[0]["labels"].dtype == torch.int64
    assert out[0]["labels"].tolist() == [1]


def test_build_rfdetr_targets_handles_empty_targets() -> None:
    targets: List[Dict[str, Tensor]] = [
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
    ]
    out = _build_rfdetr_targets(
        targets, resolution=RESOLUTION, device=torch.device("cpu")
    )
    assert out[0]["boxes"].shape == (0, 4)
    assert out[0]["labels"].shape == (0,)


# --- _decode_rfdetr_predictions --------------------------------------------


def test_decode_rfdetr_predictions_filters_background_and_threshold() -> None:
    """``PostProcess`` output is a list of per-image dicts with raw label
    indices in ``[0, num_classes_internal-1]``. Index ``num_classes`` is the
    no-object/background slot. The decode helper drops those and applies
    ``score_threshold``."""
    num_classes = 2
    # Per-image dicts as PostProcess returns (already in pixel space).
    results = [
        {
            "scores": torch.tensor([0.9, 0.8, 0.05, 0.7]),
            "labels": torch.tensor([0, 1, 0, num_classes]),  # last is background
            "boxes": torch.tensor(
                [
                    [10.0, 10.0, 30.0, 30.0],  # keep
                    [5.0, 5.0, 15.0, 15.0],  # keep
                    [0.0, 0.0, 4.0, 4.0],  # drop (low score)
                    [20.0, 20.0, 40.0, 40.0],  # drop (background label)
                ]
            ),
        },
        {
            "scores": torch.tensor([0.6, 0.4]),
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]]),
        },
    ]
    preds = _decode_rfdetr_predictions(
        results, num_classes=num_classes, score_threshold=0.5
    )

    assert len(preds) == 2
    # Image 0: two survivors (background and low-score dropped).
    assert preds[0]["boxes"].shape == (2, 4)
    assert preds[0]["scores"].dtype == torch.float32
    assert preds[0]["labels"].dtype == torch.int64
    assert preds[0]["labels"].tolist() == [0, 1]
    assert preds[0]["scores"].tolist() == pytest.approx([0.9, 0.8])
    # Image 1: only the 0.6 survives.
    assert preds[1]["boxes"].shape == (1, 4)
    assert preds[1]["labels"].tolist() == [1]
    assert preds[1]["scores"].tolist() == pytest.approx([0.6])


# --- RFDETRModel construction ----------------------------------------------


def test_rfdetr_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        RFDETRModel(
            num_classes=2,
            variant="xxlarge",  # type: ignore[arg-type]
            pretrained_detector=False,
            resolution=RESOLUTION,
        )


def _build_small_rfdetr() -> RFDETRModel:
    # Smallest offline build: nano variant at resolution=64.
    return RFDETRModel(
        num_classes=2,
        variant="nano",
        pretrained_detector=False,
        resolution=RESOLUTION,
        score_threshold=0.0,  # don't filter — tests want to inspect raw output
    )


@pytest.fixture(scope="module")
def small_rfdetr() -> RFDETRModel:
    return _build_small_rfdetr()


def test_rfdetr_train_mode_returns_loss_only(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.train()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_rfdetr(batch)
    assert step["predictions"] is None
    assert step["losses"] is not None
    assert "loss" in step["losses"]
    loss = step["losses"]["loss"]
    assert loss.dim() == 0
    assert loss.requires_grad


def test_rfdetr_train_mode_requires_targets(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.train()
    batch = DetectionBatchInput(images=_make_images(1), targets=None)
    with pytest.raises(ValueError, match="requires targets"):
        small_rfdetr(batch)


def test_rfdetr_eval_with_targets_returns_both(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
    step = small_rfdetr(batch)
    assert step["losses"] is not None
    assert "loss" in step["losses"]
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2
    for p in step["predictions"]:
        assert set(p.keys()) == {"boxes", "scores", "labels"}
        assert p["boxes"].dtype == torch.float32
        assert p["scores"].dtype == torch.float32
        assert p["labels"].dtype == torch.int64


def test_rfdetr_eval_without_targets_returns_predictions_only(
    small_rfdetr: RFDETRModel,
) -> None:
    small_rfdetr.eval()
    batch = DetectionBatchInput(images=_make_images(2), targets=None)
    step = small_rfdetr(batch)
    assert step["losses"] is None
    assert step["predictions"] is not None
    assert len(step["predictions"]) == 2


def test_rfdetr_compute_val_loss_false_skips_loss(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.compute_val_loss = False
    small_rfdetr.eval()
    try:
        batch = DetectionBatchInput(images=_make_images(2), targets=_make_targets(2))
        step = small_rfdetr(batch)
        assert step["losses"] is None
        assert step["predictions"] is not None
    finally:
        small_rfdetr.compute_val_loss = True


def test_rfdetr_rejects_wrong_size_image(small_rfdetr: RFDETRModel) -> None:
    """The wrapper requires images already at ``resolution × resolution``."""
    small_rfdetr.eval()
    batch = DetectionBatchInput(images=[torch.rand(3, 32, 32)], targets=None)
    with pytest.raises(ValueError, match="requires all images to be"):
        small_rfdetr(batch)


def test_rfdetr_class_embed_matches_num_classes(small_rfdetr: RFDETRModel) -> None:
    """RF-DETR uses ``num_classes + 1`` output logits internally — the +1 is
    the no-object/background slot. The decode helper drops the background
    class so callers see exactly ``num_classes`` real labels."""
    assert (
        small_rfdetr.detector.class_embed.out_features == small_rfdetr.num_classes + 1
    )
    assert small_rfdetr.detector.num_queries == 300


def test_rfdetr_pretrained_detector_swaps_head_for_num_classes() -> None:
    """``pretrained_detector=True`` downloads RF-DETR's COCO checkpoint and
    re-initializes the detection head for ``num_classes``. Skip cleanly when
    the weights mirror is unreachable so CI without network access still
    passes the rest of the suite. Use the variant default resolution so the
    DINOv2 backbone weights apply (their patch_size matches at default)."""
    try:
        model = RFDETRModel(
            num_classes=3,
            variant="nano",
            pretrained_detector=True,
        )
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"Skipping: pretrained weights unavailable ({exc})")

    # +1 for the no-object slot.
    assert model.detector.class_embed.out_features == 4
    assert model.detector.num_queries == 300

    # Forward must still run end-to-end at the variant's default resolution.
    model.eval()
    imgs = [torch.rand(3, model.resolution, model.resolution)]
    batch = DetectionBatchInput(images=imgs, targets=None)
    step = model(batch)
    assert step["predictions"] is not None and len(step["predictions"]) == 1


def test_rfdetr_pretrained_detector_default_off() -> None:
    """``pretrained_detector=False`` runs offline (no checkpoint download)."""
    model = RFDETRModel(
        num_classes=2, variant="nano", pretrained_detector=False, resolution=RESOLUTION
    )
    assert model.detector.class_embed.out_features == 3  # num_classes + 1


# --- End-to-end: adapter pattern inside the unchanged base Trainer ---------


def test_end_to_end_base_trainer_runs_with_rfdetr(tmp_path: Any) -> None:
    """The whole point of raidar's adapter pattern: RFDETRModel drops into
    the base Marainer Trainer with the same TorchvisionDetectionLoss and
    DetectionMetricAdapter the YOLO26 / RT-DETR / Faster R-CNN flows use."""
    model = _build_small_rfdetr()
    train_samples = _synthetic_detection_samples(4)
    val_samples = _synthetic_detection_samples(2)
    train_flux = Flux(source=train_samples)
    val_flux = Flux(source=val_samples)

    trainer = LightningTrainer(
        model=model,
        train_set=train_flux,
        val_set=val_flux,
        optimizer=Class(torch.optim.SGD, lr=1e-4),
    )
    trainer.loss_fn = TorchvisionDetectionLoss()  # type: ignore[assignment]
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
