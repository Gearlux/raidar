"""Tests for :class:`raidar.yolo26.YOLO26Model` (torchvision-native).

The native model returns a loss dict in train mode and a per-image
predictions list in eval mode — no ``DetectionStepOutput`` here. The
marainer-generic ``DetectionStepOutput`` wrapper is exercised in
:mod:`tests.test_yolo26_marainer`.
"""

from typing import Any, Dict, List, cast

import pytest
import torch
from torch import Tensor
from ultralytics.nn.modules.head import Detect

from raidar.yolo26 import YOLO26Model, _build_yolo_batch_dict, _decode_yolo_predictions

# Smallest valid YOLO input (must be a multiple of stride=32).
IMGSZ = 64


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


# --- _build_yolo_batch_dict ------------------------------------------------


def test_build_yolo_batch_dict_translates_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ), torch.rand(3, IMGSZ, IMGSZ)]
    targets = [
        {
            "boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 32.0, 64.0]]),
            "labels": torch.tensor([0], dtype=torch.int64),
        },
    ]
    out = _build_yolo_batch_dict(images, targets, IMGSZ)
    assert out["img"].shape == (2, 3, IMGSZ, IMGSZ)
    assert out["cls"].shape == (2, 1)
    assert out["bboxes"].shape == (2, 4)
    assert out["batch_idx"].tolist() == [0.0, 1.0]
    assert torch.allclose(out["bboxes"][0], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.allclose(out["bboxes"][1], torch.tensor([0.25, 0.5, 0.5, 1.0]))
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
    images = [torch.rand(3, 32, 32)]
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
                [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                [5.0, 5.0, 15.0, 15.0, 0.05, 1.0],
            ],
            [
                [1.0, 1.0, 5.0, 5.0, 0.5, 1.0],
                [2.0, 2.0, 8.0, 8.0, 0.5, 0.0],
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


def test_decode_yolo_predictions_handles_end2end_false_raw_layout() -> None:
    """``Detect.forward(end2end=False)`` emits ``(B, 4+nc, anchors)`` — NMS
    unifies that with the end2end=True ``(B, max_det, 6)`` layout."""
    nc, num_anchors = 2, 64
    raw = torch.zeros(1, 4 + nc, num_anchors)
    raw[0, 0, 0] = 10.0
    raw[0, 1, 0] = 10.0
    raw[0, 2, 0] = 8.0
    raw[0, 3, 0] = 8.0
    raw[0, 4 + 1, 0] = 0.9
    preds = _decode_yolo_predictions(raw, score_threshold=0.5)
    assert len(preds) == 1
    assert preds[0]["boxes"].shape[0] >= 1
    assert int(preds[0]["labels"][0].item()) == 1


# --- YOLO26Model construction ----------------------------------------------


def test_yolo26_rejects_imgsz_not_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        YOLO26Model(num_classes=2, imgsz=63, pretrained_detector=False)


def _build_small_yolo26() -> YOLO26Model:
    return YOLO26Model(
        num_classes=2,
        variant="n",
        pretrained_detector=False,
        imgsz=IMGSZ,
        score_threshold=0.0,
    )


@pytest.fixture(scope="module")
def small_yolo26() -> YOLO26Model:
    return _build_small_yolo26()


# --- Native forward contract ----------------------------------------------


def test_train_mode_returns_loss_dict(small_yolo26: YOLO26Model) -> None:
    small_yolo26.train()
    out = small_yolo26(_make_images(2), _make_targets(2))
    assert isinstance(out, dict)
    assert "loss" in out
    assert out["loss"].dim() == 0
    assert out["loss"].requires_grad


def test_train_mode_requires_targets(small_yolo26: YOLO26Model) -> None:
    small_yolo26.train()
    with pytest.raises(ValueError, match="requires targets"):
        small_yolo26(_make_images(1))


def test_eval_mode_returns_predictions_list(small_yolo26: YOLO26Model) -> None:
    small_yolo26.eval()
    preds = small_yolo26(_make_images(2))
    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert set(p.keys()) == {"boxes", "scores", "labels"}
        assert p["boxes"].dtype == torch.float32
        assert p["scores"].dtype == torch.float32
        assert p["labels"].dtype == torch.int64


def test_eval_mode_ignores_targets(small_yolo26: YOLO26Model) -> None:
    """Eval mode returns predictions regardless of whether targets are passed."""
    small_yolo26.eval()
    preds = small_yolo26(_make_images(2), _make_targets(2))
    assert isinstance(preds, list)
    assert len(preds) == 2


def test_yolo26_pretrained_detector_swaps_head_for_num_classes() -> None:
    """``pretrained_detector=True`` keeps backbone/neck/regression weights but
    fits the classifier to ``num_classes``. Skip if weights mirror unreachable."""
    try:
        model = YOLO26Model(num_classes=3, variant="n", pretrained_detector=True, imgsz=IMGSZ)
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"Skipping: pretrained weights unavailable ({exc})")

    head = cast(Detect, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.nc == 3
    assert head.end2end is True

    model.eval()
    preds = model(_make_images(1))
    assert isinstance(preds, list) and len(preds) == 1


def test_yolo26_pretrained_detector_default_off() -> None:
    model = YOLO26Model(num_classes=2, variant="n", pretrained_detector=False, imgsz=IMGSZ)
    head = cast(Detect, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.nc == 2


def test_yolo26_agnostic_nms_propagates_to_head() -> None:
    model = YOLO26Model(
        num_classes=2,
        variant="n",
        pretrained_detector=False,
        imgsz=IMGSZ,
        agnostic_nms=True,
    )
    head = cast(Detect, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.agnostic_nms is True
    model.eval()
    preds = model(_make_images(1))
    assert isinstance(preds, list) and len(preds) == 1


def test_yolo26_end2end_default_leaves_yaml_setting() -> None:
    model = YOLO26Model(num_classes=2, variant="n", pretrained_detector=False, imgsz=IMGSZ)
    head = cast(Detect, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.end2end is True


def test_yolo26_end2end_false_propagates_to_head() -> None:
    model = YOLO26Model(
        num_classes=2,
        variant="n",
        pretrained_detector=False,
        imgsz=IMGSZ,
        end2end=False,
    )
    head = cast(Detect, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.end2end is False
    model.eval()
    preds = model(_make_images(2))
    assert isinstance(preds, list) and len(preds) == 2


# --- Specialised trainer end-to-end ---------------------------------------


def test_native_yolo26_drops_into_specialised_trainer(tmp_path: Any) -> None:
    """The whole point of the specialised path: native YOLO26 + raw
    torchmetrics MAP, no raidar adapter layer."""
    import pytorch_lightning as L
    from dataflux.core import Flux
    from dataflux.sample import Sample
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision

    from raidar.detection import detection_collate_fn
    from raidar.lightning_trainer import LightningDetectionTrainer

    samples = [
        Sample(
            input=torch.rand(3, IMGSZ, IMGSZ),
            target={
                "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
            },
            metadata={},
        )
        for _ in range(4)
    ]
    train_flux = Flux(source=samples)
    val_flux = Flux(source=samples[:2])

    model = _build_small_yolo26()
    trainer = LightningDetectionTrainer(
        model=model,
        train_set=train_flux,
        val_set=val_flux,
        num_classes=2,
        val_metrics=MeanAveragePrecision(iou_type="bbox"),
        test_metrics=MeanAveragePrecision(iou_type="bbox"),
    )
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
