"""Tests for :class:`raidar.rfdetr.RFDETRModel` (torchvision-native).

The native model returns a loss dict in train mode and a per-image
predictions list in eval mode — the only detection path (the matrainer-generic
adapter wrappers were removed).
"""

from typing import Any, Dict, List, cast

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from raidar.rfdetr import RFDETRModel, _build_rfdetr_targets, _checkpoint_num_classes, _decode_rfdetr_predictions

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


class TestNumClassesIsBakedIntoTheCheckpoint:
    """`checkpoint_path` alone must be enough (user ruling 2026-08-31: "only use checkpoint_path").

    The head size travels in the checkpoint's own `class_embed` shape ([K+1, hidden] — the
    no-object slot rides at index K), and the pretrained COCO detector's count is a known
    constant — so `num_classes` stays unset unless the user is building a FRESH random head.
    """

    def test_an_rfdetr_style_checkpoint_derives_its_foreground_count(self, tmp_path: Any) -> None:
        path = tmp_path / "checkpoint_best_total.pth"
        torch.save({"model": {"class_embed.weight": torch.zeros(6, 256)}}, path)
        assert _checkpoint_num_classes(str(path)) == 5

    def test_a_lightning_prefixed_checkpoint_derives_too(self, tmp_path: Any) -> None:
        path = tmp_path / "epoch=3.ckpt"
        torch.save({"state_dict": {"model.detector.class_embed.weight": torch.zeros(6, 256)}}, path)
        assert _checkpoint_num_classes(str(path)) == 5

    def test_a_headless_or_missing_file_derives_nothing(self, tmp_path: Any) -> None:
        headless = tmp_path / "other.pt"
        torch.save({"model": {"backbone.weight": torch.zeros(2, 2)}}, headless)
        assert _checkpoint_num_classes(str(headless)) is None
        assert _checkpoint_num_classes(str(tmp_path / "absent.pth")) is None

    def test_resolution_order_explicit_then_checkpoint_then_pretrained(self, tmp_path: Any) -> None:
        path = tmp_path / "trained.pth"
        torch.save({"model": {"class_embed.weight": torch.zeros(6, 256)}}, path)
        explicit = RFDETRModel(num_classes=2, checkpoint_path=str(path))
        assert explicit._resolve_num_classes() == 2, "an explicit value always wins"
        derived = RFDETRModel(checkpoint_path=str(path))
        assert derived._resolve_num_classes() == 5, "unset -> read off the checkpoint"
        pretrained = RFDETRModel()
        assert pretrained._resolve_num_classes() == 90, "unset, no checkpoint -> the COCO head's own count"

    def test_no_head_to_read_from_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            RFDETRModel(pretrained_detector=False)._resolve_num_classes()


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
    # `.backbone`, not `.detector`: the model builds lazily, so reading the raw slot would only
    # work when an earlier test in this module happened to have run a forward pass first.
    # `cast`: `class_embed` / `num_queries` are rfdetr's own attributes, not `nn.Module`'s.
    detector = cast(Any, small_rfdetr.backbone)
    assert small_rfdetr.num_classes is not None, "the fixture pins an explicit count"
    assert detector.class_embed.out_features == small_rfdetr.num_classes + 1
    assert detector.num_queries == 300


def test_rfdetr_pretrained_detector_default_off() -> None:
    model = RFDETRModel(num_classes=2, variant="nano", pretrained_detector=False, resolution=RESOLUTION)
    assert cast(Any, model.backbone).class_embed.out_features == 3  # the property triggers the build


# ---------------------------------------------------------------------------
# lazy construction (Partial-Init / Zero-Arg mandate)
# ---------------------------------------------------------------------------


def test_construction_materializes_nothing_and_does_not_import_the_library() -> None:
    """Building the wrapper must be free — no parameters, and no ``rfdetr`` import.

    Before this was made lazy the constructor allocated 30.1M parameters and read the
    pretrained checkpoint from ``~/.roboflow`` (measured: 2.1 s), so merely asking this class
    for its schema paid for a whole network. The IMPORT is part of the cost the constructor
    must not pay either — it is the only reason the normalization buffers are registered in
    ``solidify()`` rather than ``__init__``, which is easy to "tidy" back without noticing the
    half-second it puts on every first construction.
    """
    import subprocess
    import sys

    # A subprocess, because `rfdetr` may already be in sys.modules from an earlier test.
    source = (
        "import sys;"
        "from raidar.rfdetr import RFDETRModel;"
        "m = RFDETRModel(num_classes=2, variant='nano', pretrained_detector=False);"
        "print(sum(p.numel() for p in m.parameters()), 'rfdetr.detr' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True)

    assert result.stdout.strip().endswith("0 False"), result.stdout


def test_solidify_builds_once_and_keeps_the_normalization_buffers_in_the_state_dict() -> None:
    """The buffers moved into ``solidify()``; a built model's checkpoint must be unchanged."""
    model = RFDETRModel(num_classes=2, variant="nano", pretrained_detector=False, resolution=RESOLUTION)

    built = model.solidify()

    assert model.solidify() is built, "the build must be cached, not repeated"
    assert model.backbone is built, "the property and solidify() share one cache"
    assert sum(p.numel() for p in model.parameters()) > 0
    state = model.state_dict()
    assert "_mean" in state and "_std" in state
    assert model.resolution == RESOLUTION
