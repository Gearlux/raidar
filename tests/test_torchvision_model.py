"""TorchVisionDetectionModel — the torchvision zoo behind the raidar wrapper contract.

The decode tests run against a STUB detector (pre-seeding ``.detector`` short-circuits
``solidify()``'s build), so the score cut and the label-indexing knob are pinned on exact
rows without building a real backbone. One real build (the smallest variant, no weights,
no downloads) pins the builder path and the checkpoint round-trip.
"""

from typing import Any, Dict, List, Optional, cast

import pytest
import torch
from torch import Tensor, nn

from raidar.torchvision_model import (
    TORCHVISION_VARIANTS,
    TorchVisionDetectionModel,
    TorchVisionVariant,
    _checkpoint_num_classes,
    _resolve_weights,
)

# The smallest, fastest builder — used wherever a REAL model must be built.
_SMALL: TorchVisionVariant = "fasterrcnn_mobilenet_v3_large_320_fpn"


class _StubDetector(nn.Module):
    """A detector double honouring torchvision's forward contract."""

    def __init__(self, results: List[Dict[str, Tensor]]) -> None:
        super().__init__()
        self.results = results
        self.seen: List[Any] = []

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None) -> Any:
        self.seen.append((type(images).__name__, len(images), targets is not None))
        if targets is not None:
            return {"loss_classifier": torch.tensor(1.0), "loss_box_reg": torch.tensor(2.0)}
        return self.results


def _stubbed(results: List[Dict[str, Tensor]], **kwargs: Any) -> TorchVisionDetectionModel:
    model = TorchVisionDetectionModel(**kwargs)
    model.detector = _StubDetector(results)
    return model.eval()


def _one_image_results() -> List[Dict[str, Tensor]]:
    """Five detections mirroring the measured fasterrcnn output shape (labels 1-indexed)."""
    return [
        {
            "boxes": torch.tensor(
                [
                    [312.0, 131.0, 1748.0, 1008.0],
                    [980.0, 132.0, 1714.0, 948.0],
                    [314.0, 154.0, 1070.0, 947.0],
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                ]
            ),
            "scores": torch.tensor([0.95, 0.85, 0.57, 0.44, 0.27]),
            "labels": torch.tensor([1, 1, 1, 2, 3]),
        }
    ]


class TestConstructionIsLazy:
    def test_zero_arg_builds_and_materializes_nothing(self) -> None:
        """No builder call, no weight download at construction — the build is solidify()'s cost.

        (An import-based laziness probe is NOT possible here: torchvision imports its whole model
        zoo eagerly at package import, and bare ``torchvision`` arrives transitively with
        ``matrainer.torch`` — same as ``raidar.rfdetr``. The measurable claim is the unbuilt slot.)
        """
        assert TorchVisionDetectionModel().detector is None

    def test_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="variant"):
            TorchVisionDetectionModel(variant=cast(Any, "yolo_v5"))

    def test_unknown_weights_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            TorchVisionDetectionModel(weights=cast(Any, "IMAGENET1K_V1"))

    def test_unknown_label_indexing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="label_indexing"):
            TorchVisionDetectionModel(label_indexing=cast(Any, "offset_1"))


class TestEvalDecoding:
    def test_the_score_cut_drops_low_rows(self) -> None:
        model = _stubbed(_one_image_results(), score_threshold=0.5)
        (out,) = model([torch.rand(3, 8, 8)])
        assert len(out["scores"]) == 3
        assert [round(float(s), 2) for s in out["scores"]] == [0.95, 0.85, 0.57]

    def test_zero_based_shifts_the_labels_down(self) -> None:
        model = _stubbed(_one_image_results(), score_threshold=0.0, label_indexing="zero_based")
        (out,) = model([torch.rand(3, 8, 8)])
        assert out["labels"].tolist() == [0, 0, 0, 1, 2]

    def test_one_based_emits_verbatim(self) -> None:
        model = _stubbed(_one_image_results(), score_threshold=0.0, label_indexing="one_based")
        (out,) = model([torch.rand(3, 8, 8)])
        assert out["labels"].tolist() == [1, 1, 1, 2, 3]

    def test_background_rows_are_dropped_in_both_modes(self) -> None:
        results = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 6.0, 6.0]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 4]),  # 0 = torchvision's background slot
            }
        ]
        for indexing, expected in (("zero_based", [3]), ("one_based", [4])):
            model = _stubbed(results, score_threshold=0.0, label_indexing=cast(Any, indexing))
            (out,) = model([torch.rand(3, 8, 8)])
            assert out["labels"].tolist() == expected, indexing

    def test_an_empty_result_stays_empty(self) -> None:
        results = [
            {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}
        ]
        model = _stubbed(results)
        (out,) = model([torch.rand(3, 8, 8)])
        assert out["boxes"].shape == (0, 4) and len(out["labels"]) == 0

    def test_train_mode_passes_the_loss_dict_through(self) -> None:
        model = _stubbed(_one_image_results()).train()
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 5.0, 5.0]]), "labels": torch.tensor([1])}]
        out = model([torch.rand(3, 8, 8)], targets)
        assert set(out) == {"loss_classifier", "loss_box_reg"}

    def test_train_mode_requires_targets(self) -> None:
        model = _stubbed(_one_image_results()).train()
        with pytest.raises(ValueError, match="targets"):
            model([torch.rand(3, 8, 8)])


class TestWeightsResolution:
    def test_every_variant_resolves_default_and_coco_v1(self) -> None:
        for variant in TORCHVISION_VARIANTS:
            for name in ("DEFAULT", "COCO_V1"):
                member = _resolve_weights(variant, name)
                assert member is not None, f"{variant}/{name}"
                assert member.url.startswith("https://"), f"{variant}/{name}"

    def test_none_resolves_to_no_weights(self) -> None:
        assert _resolve_weights(_SMALL, "none") is None

    def test_a_name_the_variant_lacks_warns_and_falls_back_to_random(self) -> None:
        """Future torchvision skew: a valid member missing for one variant WARNS, never raises."""
        from loggair import get_logger

        logger = get_logger("raidar.torchvision_model")
        collected: List[str] = []
        sink_id = logger.add(lambda message: collected.append(str(message)), level="WARNING")
        try:
            member = _resolve_weights(_SMALL, cast(Any, "COCO_V99"))
            logger.complete()
        finally:
            logger.remove(sink_id)
        assert member is None
        assert any("COCO_V99" in line and _SMALL in line for line in collected)


class TestSolidifyAndCheckpoint:
    def test_solidify_builds_once_with_random_init(self) -> None:
        model = TorchVisionDetectionModel(variant=_SMALL, weights="none", num_classes=3)
        built = model.solidify()
        assert isinstance(built, nn.Module)
        assert model.solidify() is built, "the build must be cached, not repeated"

    def test_a_saved_state_dict_round_trips_through_checkpoint_path(self, tmp_path: Any) -> None:
        trained = TorchVisionDetectionModel(variant=_SMALL, weights="none", num_classes=3)
        trained.solidify()
        path = tmp_path / "trained.pt"
        torch.save(trained.state_dict(), path)

        reloaded = TorchVisionDetectionModel(variant=_SMALL, weights="none", num_classes=3, checkpoint_path=str(path))
        reloaded.solidify()
        name, wanted = next(iter(trained.state_dict().items()))
        assert torch.equal(reloaded.state_dict()[name], wanted)


class TestNumClassesIsBakedIntoTheCheckpoint:
    """A trained checkpoint knows its own head size — config must not have to restate it."""

    @pytest.mark.parametrize("variant", TORCHVISION_VARIANTS)
    def test_every_variant_derives_its_head_size_from_a_checkpoint(
        self, variant: TorchVisionVariant, tmp_path: Any
    ) -> None:
        model = TorchVisionDetectionModel(variant=variant, weights="none", num_classes=3)
        model.solidify()
        path = tmp_path / "w.pt"
        torch.save(model.state_dict(), path)
        assert _checkpoint_num_classes(variant, str(path)) == 3, variant

    def test_a_checkpoint_alone_sizes_the_head(self, tmp_path: Any) -> None:
        """checkpoint_path with num_classes left None: the head comes out sized to the checkpoint."""
        trained = TorchVisionDetectionModel(variant=_SMALL, weights="none", num_classes=3)
        trained.solidify()
        path = tmp_path / "trained.pt"
        torch.save(trained.state_dict(), path)

        reloaded = TorchVisionDetectionModel(variant=_SMALL, weights="none", checkpoint_path=str(path))
        built = cast(Any, reloaded.solidify())
        assert built.roi_heads.box_predictor.cls_score.out_features == 3
        name, wanted = next(iter(trained.state_dict().items()))
        assert torch.equal(reloaded.state_dict()[name], wanted)

    def test_a_lightning_prefixed_checkpoint_still_derives(self, tmp_path: Any) -> None:
        """Suffix matching survives Lightning's ``model.`` and the wrapper's ``detector.`` prefixes."""
        model = TorchVisionDetectionModel(variant=_SMALL, weights="none", num_classes=4)
        model.solidify()
        prefixed = {f"model.{k}": v for k, v in model.state_dict().items()}
        path = tmp_path / "lightning.ckpt"
        torch.save({"state_dict": prefixed, "epoch": 1}, path)
        assert _checkpoint_num_classes(_SMALL, str(path)) == 4

    def test_an_explicit_num_classes_wins_and_a_disagreement_is_loud(self, tmp_path: Any) -> None:
        trained = TorchVisionDetectionModel(variant=_SMALL, weights="none", num_classes=3)
        trained.solidify()
        path = tmp_path / "trained.pt"
        torch.save(trained.state_dict(), path)
        with pytest.raises(RuntimeError, match="size mismatch"):
            TorchVisionDetectionModel(
                variant=_SMALL, weights="none", num_classes=5, checkpoint_path=str(path)
            ).solidify()

    def test_a_checkpoint_without_the_head_warns_and_derives_nothing(self, tmp_path: Any) -> None:
        from loggair import get_logger

        path = tmp_path / "headless.pt"
        torch.save({"some.other.weight": torch.zeros(2, 2)}, path)
        logger = get_logger("raidar.torchvision_model")
        collected: List[str] = []
        sink_id = logger.add(lambda message: collected.append(str(message)), level="WARNING")
        try:
            derived = _checkpoint_num_classes(_SMALL, str(path))
            logger.complete()
        finally:
            logger.remove(sink_id)
        assert derived is None
        assert any(_SMALL in line for line in collected)

    def test_a_missing_file_derives_nothing_silently(self, tmp_path: Any) -> None:
        """The documented FileNotFoundError stays the loader's — the peek must not preempt it."""
        assert _checkpoint_num_classes(_SMALL, str(tmp_path / "absent.pt")) is None


class TestModelPredictIntegration:
    def test_detection_records_flow_through_a_list_batch(self) -> None:
        """The annotaide-facing contract: ModelPredict hands a LIST, gets per-record Boxes."""
        from recordstream.ops.predict import ModelPredict

        per_image = {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([2]),
        }
        model = _stubbed([per_image, per_image])
        op = ModelPredict(model=model, kind="detection", key="model_input", output="target", device="cpu")
        records = [{"model_input": torch.rand(3, 8, 8)} for _ in range(2)]
        out = op.predict_batch(records)
        stub = cast(_StubDetector, model.detector)
        assert stub.seen[0][0] == "list" and stub.seen[0][1] == 2
        for record in out:
            assert record["target"].labels == [1]  # zero_based: torchvision's 2 -> 1


class TestModelRefs:
    def test_pretrained_weights_are_named(self) -> None:
        model = TorchVisionDetectionModel(variant=_SMALL, weights="DEFAULT")
        (ref,) = model.model_refs
        assert ref.uri == f"torchvision://{_SMALL}?weights=DEFAULT"
        assert ref.url and ref.url.startswith("https://")

    def test_a_checkpoint_is_named_and_random_init_is_not(self, tmp_path: Any) -> None:
        path = tmp_path / "w.pt"
        path.touch()
        model = TorchVisionDetectionModel(variant=_SMALL, weights="none", checkpoint_path=str(path))
        (ref,) = model.model_refs
        assert ref.uri.startswith("file://") and ref.url is None
        assert TorchVisionDetectionModel(weights="none").model_refs == []
