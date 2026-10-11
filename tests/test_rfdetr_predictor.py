"""Tests for :class:`raidar.rfdetr_predictor.RFDETRPredictor` — prediction from one of RF-DETR's own checkpoints.

The fixture is a REAL checkpoint in the shape RF-DETR's training writes (`model`, `args`, `model_name`,
`model_config`): a seg-nano at 96 px with TWO query groups — a group count the variant's default (13) does not
have, which is exactly what made a trained seg model unloadable without restating its training settings.
"""

import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

pytest.importorskip("rfdetr")

from raidar.rfdetr import RFDETRPredictor  # noqa: E402 - after the importorskip

CLASS_NAMES = ["tone", "chirp", "burst"]


@pytest.fixture(scope="module")
def seg_checkpoint(tmp_path_factory: Any) -> str:
    """A seg-nano checkpoint trained with `group_detr: 2` at 96 px, 3 classes — random weights, never downloaded."""
    import rfdetr

    rf = rfdetr.RFDETRSegNano(num_classes=3, pretrain_weights=None, resolution=96, group_detr=2, device="cpu")
    path = tmp_path_factory.mktemp("seg") / "checkpoint_best_total.pth"
    network = rf.model.model
    assert network is not None, "built, not optimized for inference"
    torch.save(
        {
            "model": network.state_dict(),
            "args": {"class_names": CLASS_NAMES},
            "model_name": "RFDETRSegNano",
            "model_config": rf.model_config.model_dump(),
        },
        path,
    )
    return str(path)


@pytest.fixture(scope="module")
def predictor(seg_checkpoint: str) -> RFDETRPredictor:
    model = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0)
    return model.eval().to("cpu")


class TestConstruction:
    def test_construction_reads_no_file_and_imports_no_library(self, tmp_path: Any) -> None:
        """The lazy-init mandate: a path that does not exist is not looked at until the first prediction."""
        model = RFDETRPredictor(checkpoint_path=str(tmp_path / "absent.pth"))
        assert model._rf is None

    def test_a_missing_checkpoint_is_named_when_the_model_is_built(self, tmp_path: Any) -> None:
        missing = tmp_path / "absent.pth"
        with pytest.raises(FileNotFoundError, match="absent.pth"):
            RFDETRPredictor(checkpoint_path=str(missing)).solidify()

    def test_no_checkpoint_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_path"):
            RFDETRPredictor().solidify()

    def test_it_is_offered_where_detection_models_are(self) -> None:
        from confluid.registry import get_registry

        assert getattr(RFDETRPredictor, "__confluid_category__") == "detection_model"
        assert "RFDETRPredictor" in get_registry().list_classes(category="detection_model")


class TestTheCheckpointDescribesTheModel:
    def test_variant_groups_resolution_and_classes_come_from_the_file(self, predictor: RFDETRPredictor) -> None:
        rf = predictor.solidify()
        assert type(rf).__name__ == "RFDETRSegNano"
        assert rf.model_config.group_detr == 2, "the TRAINED group count, not the variant's default 13"
        assert rf.model_config.resolution == 96
        assert predictor.class_names == CLASS_NAMES

    def test_solidify_builds_once(self, predictor: RFDETRPredictor) -> None:
        assert predictor.solidify() is predictor.solidify()

    def test_model_refs_name_the_checkpoint(self, predictor: RFDETRPredictor, seg_checkpoint: str) -> None:
        refs = predictor.model_refs
        assert len(refs) == 1 and refs[0].uri.endswith(os.path.basename(seg_checkpoint))


class TestPrediction:
    def _predict(self, predictor: RFDETRPredictor, images: List[Any]) -> List[Dict[str, torch.Tensor]]:
        with torch.no_grad():
            return predictor(images)

    def test_boxes_and_masks_are_in_the_pictures_own_pixels(self, predictor: RFDETRPredictor) -> None:
        """Any picture size: RF-DETR resizes to its 96 px and hands everything back at 100 x 140."""
        (out,) = self._predict(predictor, [torch.rand(3, 100, 140)])
        assert set(out) == {"boxes", "scores", "labels", "masks"}
        n = len(out["scores"])
        assert n > 0, "threshold 0 keeps every query"
        assert out["boxes"].shape == (n, 4)
        assert out["masks"].shape == (n, 100, 140) and out["masks"].dtype == torch.bool
        assert float(out["boxes"][:, [0, 2]].max()) <= 140 + 1e-3
        assert float(out["boxes"][:, [1, 3]].max()) <= 100 + 1e-3

    def test_no_label_outside_the_vocabulary_survives(self, predictor: RFDETRPredictor) -> None:
        """RF-DETR's head scores a no-object slot too (measured: label 3 of a 3-class head); it is no class."""
        (out,) = self._predict(predictor, [torch.rand(3, 96, 96)])
        assert out["labels"].dtype == torch.int64
        assert set(out["labels"].tolist()) <= set(range(len(CLASS_NAMES)))

    def test_a_grey_picture_is_read_as_its_three_equal_channels(self, predictor: RFDETRPredictor) -> None:
        """Training read a one-channel picture as RGB (`convert("RGB")`): a 2-D or 1-channel one predicts the same."""
        grey = torch.rand(96, 96)
        (two_d,) = self._predict(predictor, [grey])
        (one,) = self._predict(predictor, [grey[None]])
        (rgb,) = self._predict(predictor, [grey.expand(3, 96, 96).clone()])
        for out in (two_d, one):
            assert torch.allclose(out["scores"], rgb["scores"])
            assert torch.allclose(out["boxes"], rgb["boxes"])

    def test_a_uint8_picture_predicts_as_its_float_twin(self, predictor: RFDETRPredictor) -> None:
        pixels = np.random.default_rng(0).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        (as_uint8,) = self._predict(predictor, [pixels])
        (as_float,) = self._predict(predictor, [torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0])
        assert torch.allclose(as_uint8["scores"], as_float["scores"], atol=1e-5)

    def test_several_pictures_give_one_answer_each(self, predictor: RFDETRPredictor) -> None:
        outs = self._predict(predictor, [torch.rand(3, 96, 96), torch.rand(3, 64, 80)])
        assert len(outs) == 2
        assert outs[1]["masks"].shape[1:] == (64, 80)

    def test_the_threshold_drops_low_scores(self, seg_checkpoint: str) -> None:
        strict = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.99).eval().to("cpu")
        (out,) = self._predict(strict, [torch.rand(3, 96, 96)])
        assert all(float(s) > 0.99 for s in out["scores"])

    def test_a_detection_checkpoint_returns_no_masks(self, tmp_path: Any) -> None:
        import rfdetr

        rf = rfdetr.RFDETRNano(num_classes=2, pretrain_weights=None, resolution=96, device="cpu")
        path = tmp_path / "checkpoint_best_total.pth"
        network = rf.model.model
        assert network is not None, "built, not optimized for inference"
        torch.save(
            {
                "model": network.state_dict(),
                "args": {"class_names": ["a", "b"]},
                "model_name": "RFDETRNano",
                "model_config": rf.model_config.model_dump(),
            },
            path,
        )
        model = RFDETRPredictor(checkpoint_path=str(path), score_threshold=0.0).eval().to("cpu")
        (out,) = self._predict(model, [torch.rand(3, 96, 96)])
        assert set(out) == {"boxes", "scores", "labels"}
