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

from raidar.rfdetr import RFDETRPredictor, _any_class_logits  # noqa: E402 - after the importorskip

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


def _noisy_or(prob: torch.Tensor) -> torch.Tensor:
    """The chance that at least one of independent class scores is right: 1 - prod(1 - p) over the last axis."""
    return 1 - torch.prod(1 - prob, dim=-1)


class TestTheAnyClassRewrite:
    """`_any_class_logits` on hand-made score tables — what RF-DETR's own top-k then ranks."""

    def test_each_query_keeps_its_best_class_carrying_the_any_class_score(self) -> None:
        """A query of the 2457 MHz link on `files#31` (rfdetr_a, measured 2026-10-12): LTE uplink 0.274, 802.11a/g
        0.127, BR/EDR 0.126, DVB-RCS2 0.092 and the other 28 classes together 0.373 — no class over the 0.3 cut,
        any class 0.685. The 28 are spread evenly: 1 - (1 - p)^28 = 0.373."""
        rest = 1 - (1 - 0.373) ** (1 / 28)
        prob = torch.tensor([[[0.274, 0.127, 0.126, 0.092] + [rest] * 28]])
        counted = torch.ones(32, dtype=torch.bool)
        out = _any_class_logits(torch.logit(prob), counted).sigmoid()
        assert float(out[0, 0, 0]) == pytest.approx(0.685, abs=1e-3)
        assert torch.all(out[0, 0, 1:] == 0), "every other class of the query scores nothing"

    def test_a_query_of_empty_air_stays_low(self) -> None:
        """`files#31` query 148: its highest classes 0.014, 0.008, 0.004, 0.003, the other 28 together 0.030."""
        rest = 1 - (1 - 0.030) ** (1 / 28)
        prob = torch.tensor([[[0.014, 0.008, 0.004, 0.003] + [rest] * 28]])
        out = _any_class_logits(torch.logit(prob), torch.ones(32, dtype=torch.bool)).sigmoid()
        assert float(out.max()) == pytest.approx(0.058, abs=1e-3)

    def test_an_uncounted_slot_neither_scores_nor_names(self) -> None:
        """The head's no-object slot (and any class left out) adds nothing and is never the name, however high."""
        prob = torch.tensor([[[0.2, 0.1, 0.9]]])
        out = _any_class_logits(torch.logit(prob), torch.tensor([True, True, False])).sigmoid()
        assert int(out[0, 0].argmax()) == 0
        assert float(out[0, 0, 0]) == pytest.approx(1 - 0.8 * 0.9, abs=1e-6)
        assert float(out[0, 0, 2]) == 0.0

    def test_a_certain_class_stays_certain(self) -> None:
        """A score of 1.0 (logit +inf) gives any-class 1.0, not NaN."""
        logits = torch.tensor([[[float("inf"), 0.0]]])
        out = _any_class_logits(logits, torch.ones(2, dtype=torch.bool)).sigmoid()
        assert float(out[0, 0, 0]) == pytest.approx(1.0) and not torch.isnan(out).any()


class TestAnyClassScoring:
    """`score_mode: any_class` on the real seg checkpoint: one detection per query, scored by any class."""

    @staticmethod
    def _predict_with_logits(model: RFDETRPredictor, image: torch.Tensor) -> tuple:
        """The predictor's answer and the network's raw class scores ``[Q, C]`` that produced it."""
        seen: List[torch.Tensor] = []

        def grab(_module: Any, _inputs: Any, output: Any) -> None:
            seen.append((output["pred_logits"] if isinstance(output, dict) else output[1]).detach()[0])

        handle = model.solidify().model.model.register_forward_hook(grab)
        try:
            with torch.no_grad():
                (out,) = model([image])
        finally:
            handle.remove()
        return out, seen[0]

    def test_class_mode_is_the_default_and_unchanged(self, seg_checkpoint: str) -> None:
        image = torch.rand(3, 96, 96, generator=torch.Generator().manual_seed(1))
        default = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0).eval().to("cpu")
        explicit = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0, score_mode="class")
        explicit = explicit.eval().to("cpu")
        assert default.score_mode == "class"
        with torch.no_grad():
            (a,) = default([image])
            (b,) = explicit([image])
        for key in ("boxes", "scores", "labels", "masks"):
            assert torch.equal(a[key], b[key])

    def test_no_query_answers_twice_and_each_score_is_its_any_class_chance(self, seg_checkpoint: str) -> None:
        image = torch.rand(3, 96, 96, generator=torch.Generator().manual_seed(2))
        model = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0, score_mode="any_class")
        out, logits = self._predict_with_logits(model.eval().to("cpu"), image)
        prob = logits.sigmoid()[:, : len(CLASS_NAMES)]  # the no-object slot is no class
        expected = _noisy_or(prob)
        order = torch.argsort(expected, descending=True)
        n = len(out["scores"])
        assert n == logits.shape[0], "every query once (threshold 0), never one query under two classes"
        assert torch.allclose(out["scores"], expected[order][:n], atol=1e-5)
        assert torch.equal(out["labels"], prob.argmax(-1)[order][:n])

    def test_class_mode_answers_one_query_under_several_classes(self, seg_checkpoint: str) -> None:
        """The contrast that makes the test above mean something: per class, one query (its box AND its mask)
        comes back under two labels."""
        model = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0).eval().to("cpu")
        with torch.no_grad():
            (out,) = model([torch.rand(3, 96, 96, generator=torch.Generator().manual_seed(2))])
        rows = list(zip(out["boxes"].tolist(), out["masks"], out["labels"].tolist()))
        twice = any(
            a[0] == b[0] and torch.equal(a[1], b[1]) and a[2] != b[2] for i, a in enumerate(rows) for b in rows[i + 1 :]
        )
        assert twice

    def test_any_class_keeps_every_query_class_mode_keeps(self, seg_checkpoint: str) -> None:
        """Any class is at least the best class, so at one cut it keeps a superset of the queries."""
        image = torch.rand(3, 96, 96, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            (every,) = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0).eval().to("cpu")([image])
        cut = float(torch.sort(every["scores"], descending=True).values[10])
        with torch.no_grad():
            (per_class,) = (
                RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=cut).eval().to("cpu")([image])
            )
            (any_class,) = (
                RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=cut, score_mode="any_class")
                .eval()
                .to("cpu")([image])
            )
        assert len(per_class["scores"]) > 0
        kept = {tuple(box) for box in any_class["boxes"].tolist()}
        assert {tuple(box) for box in per_class["boxes"].tolist()} <= kept

    def test_a_query_keeps_its_mask_beside_its_box(self, seg_checkpoint: str) -> None:
        image = torch.rand(3, 100, 140, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            (per_class,) = (
                RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0).eval().to("cpu")([image])
            )
            (any_class,) = (
                RFDETRPredictor(checkpoint_path=seg_checkpoint, score_threshold=0.0, score_mode="any_class")
                .eval()
                .to("cpu")([image])
            )
        assert any_class["masks"].shape == (len(any_class["scores"]), 100, 140)

        # A query is known by its box — but an untrained model clamps many queries to the same edge box, so only
        # a box that ONE query gave (one box, one mask in either answer) identifies its query.
        def by_box(out: Dict[str, torch.Tensor]) -> Dict[tuple, List[torch.Tensor]]:
            found: Dict[tuple, List[torch.Tensor]] = {}
            for box, mask in zip(out["boxes"].tolist(), out["masks"]):
                found.setdefault(tuple(box), []).append(mask)
            return found

        ours, theirs = by_box(any_class), by_box(per_class)
        unique = [box for box, masks in ours.items() if len(masks) == 1 and box in theirs]
        unique = [box for box in unique if all(torch.equal(m, theirs[box][0]) for m in theirs[box])]
        assert len(unique) > 10
        for box in unique:
            assert torch.equal(ours[box][0], theirs[box][0])

    def test_an_excluded_class_neither_scores_nor_names(self, seg_checkpoint: str) -> None:
        image = torch.rand(3, 96, 96, generator=torch.Generator().manual_seed(5))
        model = RFDETRPredictor(
            checkpoint_path=seg_checkpoint, score_threshold=0.0, score_mode="any_class", any_class_excludes=["chirp"]
        )
        out, logits = self._predict_with_logits(model.eval().to("cpu"), image)
        prob = logits.sigmoid()[:, [0, 2]]  # tone and burst
        expected = torch.sort(_noisy_or(prob), descending=True).values
        assert 1 not in out["labels"].tolist()
        assert torch.allclose(out["scores"], expected[: len(out["scores"])], atol=1e-5)

    def test_an_exclusion_the_checkpoint_does_not_know_is_named(self, seg_checkpoint: str) -> None:
        model = RFDETRPredictor(checkpoint_path=seg_checkpoint, score_mode="any_class", any_class_excludes=["nope"])
        with pytest.raises(ValueError, match=r"'nope'.*tone, chirp, burst"):
            model.solidify()

    def test_an_unknown_score_mode_is_refused(self) -> None:
        """confluid's validation refuses it first (pydantic's error is a ValueError) ..."""
        with pytest.raises(ValueError, match=r"score_mode[\s\S]*'class' or 'any_class'"):
            RFDETRPredictor(score_mode="per_query")  # type: ignore[arg-type]

    def test_an_unknown_score_mode_is_refused_with_validation_off(self) -> None:
        """... and the constructor itself when that validation is off (`CONFLUID_VALIDATE_INIT=off`)."""
        from confluid.validation import override_init_mode

        with (
            override_init_mode("off"),
            pytest.raises(ValueError, match=r"score_mode 'per_query'.*'class', 'any_class'"),
        ):
            RFDETRPredictor(score_mode="per_query")  # type: ignore[arg-type]
