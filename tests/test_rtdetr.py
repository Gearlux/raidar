"""Tests for :class:`raidar.rtdetr.RTDETRModel` (torchvision-native).

The native model returns a loss dict in train mode and a per-image
predictions list in eval mode. The marainer-generic wrapper (which returns
``DetectionStepOutput``) is exercised in :mod:`tests.test_rtdetr_marainer`.
"""

from typing import Dict, List, cast

import pytest
import torch
from torch import Tensor
from ultralytics.nn.modules.head import RTDETRDecoder

from raidar.rtdetr import RTDETRModel, _build_rtdetr_batch_dict, _decode_rtdetr_predictions

# Smallest practical RT-DETR input. Must yield enough multi-scale tokens to
# satisfy the decoder's top-k=300 query selection: (h/8)²+(h/16)²+(h/32)².
# At imgsz=128: 16²+8²+4²=336.
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


# --- _build_rtdetr_batch_dict ----------------------------------------------


def test_build_rtdetr_batch_dict_translates_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ), torch.rand(3, IMGSZ, IMGSZ)]
    targets = [
        {
            "boxes": torch.tensor([[32.0, 32.0, 96.0, 96.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 64.0, 128.0]]),
            "labels": torch.tensor([0], dtype=torch.int64),
        },
    ]
    out = _build_rtdetr_batch_dict(images, targets, IMGSZ)
    assert out["img"].shape == (2, 3, IMGSZ, IMGSZ)
    assert out["cls"].shape == (2, 1)
    assert out["bboxes"].shape == (2, 4)
    assert torch.allclose(out["bboxes"][0], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.allclose(out["bboxes"][1], torch.tensor([0.25, 0.5, 0.5, 1.0]))


def test_build_rtdetr_batch_dict_handles_empty_targets() -> None:
    images = [torch.rand(3, IMGSZ, IMGSZ)]
    targets: List[Dict[str, Tensor]] = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    out = _build_rtdetr_batch_dict(images, targets, IMGSZ)
    assert out["cls"].shape == (0, 1)
    assert out["bboxes"].shape == (0, 4)


def test_build_rtdetr_batch_dict_rejects_wrong_size_image() -> None:
    images = [torch.rand(3, 32, 32)]
    targets = [
        {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    with pytest.raises(ValueError, match="requires all images to be"):
        _build_rtdetr_batch_dict(images, targets, IMGSZ)


# --- _decode_rtdetr_predictions --------------------------------------------


def test_decode_rtdetr_predictions_filters_and_lifts_to_pixels() -> None:
    nc = 2
    raw = torch.zeros((2, 300, 4 + nc))
    raw[0, 0, :4] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    raw[0, 0, 4:] = torch.tensor([0.9, 0.05])
    raw[0, 1, :4] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    raw[0, 1, 4:] = torch.tensor([0.05, 0.03])
    raw[1, 0, :4] = torch.tensor([0.25, 0.5, 0.5, 1.0])
    raw[1, 0, 4:] = torch.tensor([0.0, 0.5])

    preds = _decode_rtdetr_predictions(raw, imgsz=IMGSZ, score_threshold=0.1)
    assert len(preds) == 2
    expected0 = torch.tensor([0.25, 0.25, 0.75, 0.75]) * float(IMGSZ)
    assert torch.allclose(preds[0]["boxes"][0], expected0)
    assert preds[0]["scores"].tolist() == pytest.approx([0.9])
    assert preds[0]["labels"].tolist() == [0]
    expected1 = torch.tensor([0.0, 0.0, 0.5, 1.0]) * float(IMGSZ)
    assert torch.allclose(preds[1]["boxes"][0], expected1)
    assert preds[1]["labels"].tolist() == [1]


# --- RTDETRModel construction ----------------------------------------------


def test_rtdetr_rejects_imgsz_not_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        RTDETRModel(num_classes=2, imgsz=63, pretrained_detector=False)


def test_rtdetr_pretrained_resnet_variants_rejected() -> None:
    with pytest.raises(ValueError, match="resnet"):
        RTDETRModel(num_classes=2, variant="resnet50", pretrained_detector=True, imgsz=IMGSZ)


def _build_small_rtdetr() -> RTDETRModel:
    return RTDETRModel(
        num_classes=2,
        variant="l",
        pretrained_detector=False,
        imgsz=IMGSZ,
        score_threshold=0.0,
    )


@pytest.fixture(scope="module")
def small_rtdetr() -> RTDETRModel:
    return _build_small_rtdetr()


# --- Native forward contract ----------------------------------------------


def test_train_mode_returns_loss_dict(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.train()
    out = small_rtdetr(_make_images(2), _make_targets(2))
    assert isinstance(out, dict)
    assert "loss" in out
    assert out["loss"].dim() == 0
    assert out["loss"].requires_grad


def test_train_mode_requires_targets(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.train()
    with pytest.raises(ValueError, match="requires targets"):
        small_rtdetr(_make_images(1))


def test_eval_mode_returns_predictions_list(small_rtdetr: RTDETRModel) -> None:
    small_rtdetr.eval()
    preds = small_rtdetr(_make_images(2))
    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert set(p.keys()) == {"boxes", "scores", "labels"}
        assert p["boxes"].dtype == torch.float32
        assert p["scores"].dtype == torch.float32
        assert p["labels"].dtype == torch.int64


def test_rtdetr_pretrained_detector_default_off() -> None:
    model = RTDETRModel(num_classes=2, variant="l", pretrained_detector=False, imgsz=IMGSZ)
    head = cast(RTDETRDecoder, model.detector.model[-1])  # type: ignore[index, union-attr]
    assert head.nc == 2
    assert head.num_queries == 300
