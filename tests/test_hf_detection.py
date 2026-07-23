"""Tests for :mod:`raidar.hf_detection` (D-FINE / RT-DETRv2 transformers wrappers).

The native wrappers return a single-key loss dict in train mode and a per-image
``DetectionOutput`` list in eval mode — the same contract torchvision detectors
and :class:`raidar.rfdetr.RFDETRModel` speak, so they run through the generic
:class:`raidar.lightning_trainer.LightningDetector` with no adapter.

Pure-helper / map / guard tests run with no network. The real-model forward and
end-to-end tests build the SMALLEST checkpoints with ``pretrained=False`` (random
weights from config — no weight download); the train-loop smoke is ``slow``.
"""

from typing import Any, Dict, List, get_args

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from raidar.hf_detection import (
    _DFINE_CHECKPOINTS,
    _RTDETRV2_CHECKPOINTS,
    DFINEModel,
    HFDetectionModel,
    RTDETRv2Model,
    _build_hf_detection_targets,
    _decode_hf_predictions,
    _DfineVariant,
    _RTDETRv2Variant,
)

# ---------------------------------------------------------------------------
# _build_hf_detection_targets — xyxy-pixel → cxcywh-normalised {class_labels, boxes}
# ---------------------------------------------------------------------------


def test_build_targets_translates_xyxy_to_cxcywh_normalized() -> None:
    targets = [{"boxes": torch.tensor([[10.0, 20.0, 110.0, 220.0]]), "labels": torch.tensor([2])}]
    # sizes are (H, W) = (200, 100): cx=(10+110)/2/100=0.6, cy=(20+220)/2/200=0.6, w=100/100=1, h=200/200=1
    out = _build_hf_detection_targets(targets, [(200, 100)], torch.device("cpu"))
    assert len(out) == 1
    assert torch.allclose(out[0]["boxes"], torch.tensor([[0.6, 0.6, 1.0, 1.0]]))
    assert out[0]["class_labels"].tolist() == [2]
    assert out[0]["class_labels"].dtype == torch.long


def test_build_targets_normalisation_is_scale_invariant_per_image() -> None:
    # Same relative box, two image sizes → identical normalised coords.
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 320.0, 320.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[0.0, 0.0, 64.0, 64.0]]), "labels": torch.tensor([0])},
    ]
    out = _build_hf_detection_targets(targets, [(640, 640), (128, 128)], torch.device("cpu"))
    assert torch.allclose(out[0]["boxes"], out[1]["boxes"])
    assert torch.allclose(out[0]["boxes"], torch.tensor([[0.25, 0.25, 0.5, 0.5]]))


def test_build_targets_handles_empty() -> None:
    targets: List[Dict[str, Tensor]] = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]
    out = _build_hf_detection_targets(targets, [(640, 640)], torch.device("cpu"))
    assert out[0]["boxes"].shape == (0, 4)
    assert out[0]["class_labels"].shape == (0,)


# ---------------------------------------------------------------------------
# _decode_hf_predictions — processor output → DetectionOutput
# ---------------------------------------------------------------------------


def test_decode_predictions_maps_to_detection_output() -> None:
    results = [
        {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            "scores": torch.tensor([0.9, 0.6]),
            "labels": torch.tensor([1, 0]),
        },
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)},
    ]
    preds = _decode_hf_predictions(results)
    assert len(preds) == 2
    assert set(preds[0]) == {"boxes", "scores", "labels"}
    assert preds[0]["boxes"].shape == (2, 4)
    assert preds[0]["labels"].dtype == torch.long
    assert preds[0]["scores"].dtype == torch.float32
    assert preds[1]["boxes"].shape == (0, 4)


# ---------------------------------------------------------------------------
# variant → checkpoint maps (closed Literal is the single source of truth)
# ---------------------------------------------------------------------------


def test_dfine_checkpoint_map_covers_every_variant() -> None:
    assert set(_DFINE_CHECKPOINTS) == set(get_args(_DfineVariant))
    assert all(v.startswith("ustc-community/dfine-") for v in _DFINE_CHECKPOINTS.values())


def test_rtdetrv2_checkpoint_map_covers_every_variant() -> None:
    assert set(_RTDETRV2_CHECKPOINTS) == set(get_args(_RTDETRv2Variant))
    assert all(v.startswith("PekingU/rtdetr_v2_") for v in _RTDETRV2_CHECKPOINTS.values())


# ---------------------------------------------------------------------------
# Construction guards (no network)
# ---------------------------------------------------------------------------


def test_image_size_must_be_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        DFINEModel(image_size=100)


def test_dfine_rejects_unknown_variant() -> None:
    # Confluid validates the Literal kwarg via pydantic before the body; the body's
    # hand-rolled ValueError is the fallback if validation were skipped.
    with pytest.raises((ValueError, ValidationError)):
        DFINEModel(variant="enormous")  # type: ignore[arg-type]


def test_rtdetrv2_rejects_unknown_variant() -> None:
    with pytest.raises((ValueError, ValidationError)):
        RTDETRv2Model(variant="r999")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Real-model forward contract (smallest checkpoints, pretrained=False = no download)
# ---------------------------------------------------------------------------

_NUM_CLASSES = 3
_IMG = 256  # multiple of 32; smaller than 640 to keep the forward light


@pytest.fixture(scope="module")
def models() -> Dict[str, HFDetectionModel]:
    pytest.importorskip("transformers")
    return {
        "dfine": DFINEModel(num_classes=_NUM_CLASSES, variant="nano", pretrained=False, image_size=_IMG),
        "rtdetrv2": RTDETRv2Model(num_classes=_NUM_CLASSES, variant="r18vd", pretrained=False, image_size=_IMG),
    }


def _images() -> List[Tensor]:
    # Variable-size [0,1] images — the wrapper resizes internally.
    return [torch.rand(3, 200, 256), torch.rand(3, 256, 200)]


def _targets() -> List[Dict[str, Tensor]]:
    return [
        {"boxes": torch.tensor([[10.0, 20.0, 110.0, 180.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)},  # empty image
    ]


@pytest.mark.parametrize("key", ["dfine", "rtdetrv2"])
def test_train_mode_returns_loss_dict(models: Dict[str, HFDetectionModel], key: str) -> None:
    model = models[key]
    model.train()
    out = model(_images(), _targets())
    assert isinstance(out, dict) and set(out) == {"loss"}
    assert out["loss"].requires_grad and torch.isfinite(out["loss"])


@pytest.mark.parametrize("key", ["dfine", "rtdetrv2"])
def test_train_mode_requires_targets(models: Dict[str, HFDetectionModel], key: str) -> None:
    model = models[key]
    model.train()
    with pytest.raises(ValueError, match="requires targets"):
        model(_images())


@pytest.mark.parametrize("key", ["dfine", "rtdetrv2"])
def test_eval_mode_returns_predictions_in_input_pixel_space(models: Dict[str, HFDetectionModel], key: str) -> None:
    model = models[key]
    model.eval()
    imgs = _images()
    with torch.no_grad():
        preds = model(imgs)
    assert isinstance(preds, list) and len(preds) == len(imgs)
    for img, p in zip(imgs, preds):
        assert set(p) == {"boxes", "scores", "labels"}
        assert p["boxes"].ndim == 2 and p["boxes"].shape[-1] == 4
        if len(p["scores"]):
            # post_process applied the score threshold; labels are 0-indexed < K.
            assert float(p["scores"].min()) >= model.score_threshold - 1e-6
            assert int(p["labels"].max()) < model.num_classes


@pytest.mark.parametrize("key,cls", [("dfine", DFINEModel), ("rtdetrv2", RTDETRv2Model)])
def test_num_classes_attr_drives_head(models: Dict[str, HFDetectionModel], key: str, cls: Any) -> None:
    model = models[key]
    # The wrapper exposes num_classes (read by the trainer/evaluator) and re-sized the head.
    assert model.num_classes == _NUM_CLASSES
    assert model.model.config.num_labels == _NUM_CLASSES


# ---------------------------------------------------------------------------
# End-to-end through the generic LightningDetector (slow — real backbone)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("cls,variant", [(DFINEModel, "nano"), (RTDETRv2Model, "r18vd")])
def test_end_to_end_lightning_fit(cls: Any, variant: str, tmp_path: Any) -> None:
    pytest.importorskip("transformers")
    import pytorch_lightning as L
    from confluid import flow
    from sampleflux.core import Flux
    from sampleflux.sample import Sample

    from raidar.detection import detection_collate_fn
    from raidar.detector import LightningDetector

    def _flux(n: int) -> Flux:
        samples = [
            Sample(
                input=torch.rand(3, 128, 128),
                target={
                    "boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                },
                metadata={"idx": i},
            )
            for i in range(n)
        ]
        return Flux(source=samples)

    model = cls(num_classes=2, variant=variant, pretrained=False, image_size=_IMG)
    trainer = LightningDetector(model=model, train_set=_flux(4), num_classes=2)
    trainer.val_set = _flux(2)
    trainer.val_metrics = flow(trainer.val_metrics)
    trainer.train_loader = torch.utils.data.DataLoader(_flux(4), batch_size=2, collate_fn=detection_collate_fn)
    trainer.val_loader = torch.utils.data.DataLoader(_flux(2), batch_size=2, collate_fn=detection_collate_fn)

    lightning_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
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
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss)
