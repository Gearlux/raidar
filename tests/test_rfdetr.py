"""Tests for :class:`raidar.rfdetr.RFDETRModel` (torchvision-native).

The native model returns a loss dict in train mode and a per-image
predictions list in eval mode — the only detection path (the matrainer-generic
adapter wrappers were removed).
"""

from typing import Dict, List

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from raidar.rfdetr import RFDETRModel, _build_rfdetr_targets, _decode_rfdetr_predictions

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


# --- _build_rfdetr_targets -------------------------------------------------


def test_build_rfdetr_targets_translates_xyxy_to_cxcywh_normalized() -> None:
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
    out = _build_rfdetr_targets(targets, resolution=RESOLUTION, device=torch.device("cpu"))
    assert len(out) == 2
    assert torch.allclose(out[0]["boxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(out[1]["boxes"], torch.tensor([[0.25, 0.5, 0.5, 1.0]]))


def test_build_rfdetr_targets_handles_empty_targets() -> None:
    targets: List[Dict[str, Tensor]] = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    out = _build_rfdetr_targets(targets, resolution=RESOLUTION, device=torch.device("cpu"))
    assert out[0]["boxes"].shape == (0, 4)


# --- _decode_rfdetr_predictions --------------------------------------------


def test_decode_rfdetr_predictions_filters_background_and_threshold() -> None:
    num_classes = 2
    results = [
        {
            "scores": torch.tensor([0.9, 0.8, 0.05, 0.7]),
            "labels": torch.tensor([0, 1, 0, num_classes]),
            "boxes": torch.tensor(
                [
                    [10.0, 10.0, 30.0, 30.0],
                    [5.0, 5.0, 15.0, 15.0],
                    [0.0, 0.0, 4.0, 4.0],
                    [20.0, 20.0, 40.0, 40.0],
                ]
            ),
        },
        {
            "scores": torch.tensor([0.6, 0.4]),
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]]),
        },
    ]
    preds = _decode_rfdetr_predictions(results, num_classes=num_classes, score_threshold=0.5)
    assert len(preds) == 2
    assert preds[0]["labels"].tolist() == [0, 1]
    assert preds[1]["labels"].tolist() == [1]


# --- RFDETRModel construction ----------------------------------------------


def test_rfdetr_rejects_unknown_variant() -> None:
    # Confluid's ``@configurable`` now validates kwargs against the
    # auto-generated pydantic schema before the body runs. Since
    # ``variant`` is annotated as ``Literal[...]``, pydantic surfaces the
    # mismatch first; the body's hand-rolled ``ValueError`` would still
    # fire if validation were skipped.
    with pytest.raises((ValueError, ValidationError)):
        RFDETRModel(
            num_classes=2,
            variant="xxlarge",  # type: ignore[arg-type]
            pretrained_detector=False,
            resolution=RESOLUTION,
        )


def _build_small_rfdetr() -> RFDETRModel:
    return RFDETRModel(
        num_classes=2,
        variant="nano",
        pretrained_detector=False,
        resolution=RESOLUTION,
        score_threshold=0.0,
    )


@pytest.fixture(scope="module")
def small_rfdetr() -> RFDETRModel:
    return _build_small_rfdetr()


# --- Native forward contract ----------------------------------------------


def test_train_mode_returns_loss_dict(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.train()
    out = small_rfdetr(_make_images(2), _make_targets(2))
    assert isinstance(out, dict)
    assert "loss" in out
    assert out["loss"].requires_grad


def test_train_mode_requires_targets(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.train()
    with pytest.raises(ValueError, match="requires targets"):
        small_rfdetr(_make_images(1))


def test_eval_mode_returns_predictions_list(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.eval()
    preds = small_rfdetr(_make_images(2))
    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert set(p.keys()) == {"boxes", "scores", "labels"}


def test_rfdetr_rejects_wrong_size_image(small_rfdetr: RFDETRModel) -> None:
    small_rfdetr.eval()
    with pytest.raises(ValueError, match="requires all images to be"):
        small_rfdetr([torch.rand(3, 32, 32)])


def test_rfdetr_class_embed_matches_num_classes(small_rfdetr: RFDETRModel) -> None:
    assert small_rfdetr.detector.class_embed.out_features == small_rfdetr.num_classes + 1
    assert small_rfdetr.detector.num_queries == 300


def test_rfdetr_pretrained_detector_default_off() -> None:
    model = RFDETRModel(num_classes=2, variant="nano", pretrained_detector=False, resolution=RESOLUTION)
    assert model.detector.class_embed.out_features == 3
