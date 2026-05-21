"""Tests for the ergonomic Flux interface + augmentation knobs on
:class:`raidar.ultralytics_trainer.UltralyticsTrainer` / ``UltralyticsEvaluator``.
"""

from typing import Any, Dict, List, cast

import pytest
import torch
from dataflux.core import Flux
from dataflux.sample import Sample

from raidar.flux_yolo_dataset import FluxYOLODataset
from raidar.ultralytics_trainer import _FLUX_CTX, FluxDetectionTrainer, UltralyticsEvaluator, UltralyticsTrainer


def _make_flux(n: int = 4, size: int = 64) -> Flux:
    samples: List[Sample] = []
    for i in range(n):
        samples.append(
            Sample(
                input=torch.rand(3, size, size),
                target={
                    "boxes": torch.tensor([[4.0, 4.0, 28.0, 28.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
                metadata={"idx": i},
            )
        )
    return Flux(source=samples)


class _StubTrainer:
    """Minimal stand-in carrying the attrs ``FluxDetectionTrainer.build_dataset``
    reads from ``self`` (``self.data``, ``self.args``, ``self.model``).

    Lets the augmentation-plumbing tests probe ``build_dataset`` without
    spinning up a full ultralytics training pipeline. Uses ultralytics'
    own ``get_cfg`` so ``self.args`` carries every hyperparam ultralytics
    expects (``mask_ratio``, ``overlap_mask``, ``mosaic``, …) rather than a
    hand-rolled namespace that misses keys.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        from ultralytics.cfg import get_cfg

        self.data = data
        self.args = get_cfg(
            overrides={
                "task": "detect",
                "mode": "train",
                "imgsz": 64,
                "batch": 2,
                "data": "<stub>",
            }
        )
        self.model = type("M", (), {"stride": torch.tensor([32])})()


def _make_bundle(trainer: UltralyticsTrainer) -> Dict[str, Any]:
    return {
        "train": trainer.data["train"],
        "val": trainer.data["val"],
        "data": {
            "path": ".",
            "train": ".",
            "val": ".",
            "test": ".",
            "names": trainer.data["names"],
            "nc": trainer.data["nc"],
            "channels": 3,
        },
        "augment_override": trainer.augment,
    }


# ---------------------------------------------------------------------------
# UltralyticsTrainer ergonomic form
# ---------------------------------------------------------------------------


def test_ergonomic_form_constructs_legacy_data_dict() -> None:
    train = _make_flux(2)
    val = _make_flux(2)
    trainer = UltralyticsTrainer(
        train_set=train,
        val_set=val,
        num_classes=2,
        class_names={0: "background", 1: "drone"},
        variant="n",
        epochs=1,
        batch=2,
    )
    assert isinstance(trainer.data, dict)
    assert trainer.data["train"] is train
    assert trainer.data["val"] is val
    assert trainer.data["nc"] == 2
    assert trainer.data["names"] == {0: "background", 1: "drone"}


def test_ergonomic_form_rejects_both_data_and_train_set() -> None:
    with pytest.raises(ValueError, match="EITHER"):
        UltralyticsTrainer(
            data="dummy.yaml",
            train_set=_make_flux(2),
            val_set=_make_flux(2),
            num_classes=2,
            class_names={0: "bg", 1: "fg"},
        )


def test_ergonomic_form_rejects_missing_class_names() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        UltralyticsTrainer(
            train_set=_make_flux(2),
            val_set=_make_flux(2),
            # missing num_classes + class_names
        )


def test_legacy_data_dict_form_still_works() -> None:
    trainer = UltralyticsTrainer(
        data={
            "train": _make_flux(2),
            "val": _make_flux(2),
            "names": {0: "bg", 1: "fg"},
            "nc": 2,
        },
        variant="n",
    )
    assert isinstance(trainer.data, dict)
    assert trainer.data["nc"] == 2


def test_neither_data_nor_train_set_raises() -> None:
    with pytest.raises(ValueError, match="EITHER `data` OR `train_set`"):
        UltralyticsTrainer(variant="n")


# ---------------------------------------------------------------------------
# Augmentation knobs
# ---------------------------------------------------------------------------


def test_augment_false_disables_flux_yolo_dataset_augmentation() -> None:
    """``augment=False`` propagates through the FLUX context so
    ``FluxDetectionTrainer.build_dataset()`` builds the dataset with
    ``augment=False`` regardless of train/val mode.
    """
    trainer = UltralyticsTrainer(
        train_set=_make_flux(2),
        val_set=_make_flux(2),
        num_classes=2,
        class_names={0: "bg", 1: "fg"},
        augment=False,
    )
    bundle = _make_bundle(trainer)
    token = _FLUX_CTX.set(bundle)
    try:
        stub = cast(FluxDetectionTrainer, _StubTrainer(data=bundle["data"]))
        ds = FluxDetectionTrainer.build_dataset(stub, img_path=".", mode="train", batch=2)
        assert isinstance(ds, FluxYOLODataset)
        assert ds.augment is False
    finally:
        _FLUX_CTX.reset(token)


def test_augment_true_explicit_keeps_augmentation_in_train_mode() -> None:
    trainer = UltralyticsTrainer(
        train_set=_make_flux(2),
        val_set=_make_flux(2),
        num_classes=2,
        class_names={0: "bg", 1: "fg"},
        augment=True,
    )
    bundle = _make_bundle(trainer)
    token = _FLUX_CTX.set(bundle)
    try:
        stub = cast(FluxDetectionTrainer, _StubTrainer(data=bundle["data"]))
        ds_train = FluxDetectionTrainer.build_dataset(stub, img_path=".", mode="train", batch=2)
        # Even in val mode, augment_override=True forces augment=True.
        ds_val = FluxDetectionTrainer.build_dataset(stub, img_path=".", mode="val", batch=2)
        assert ds_train.augment is True
        assert ds_val.augment is True
    finally:
        _FLUX_CTX.reset(token)


def test_augment_none_default_keeps_legacy_per_mode_behavior() -> None:
    trainer = UltralyticsTrainer(
        train_set=_make_flux(2),
        val_set=_make_flux(2),
        num_classes=2,
        class_names={0: "bg", 1: "fg"},
        # augment defaults to None.
    )
    bundle = _make_bundle(trainer)
    token = _FLUX_CTX.set(bundle)
    try:
        stub = cast(FluxDetectionTrainer, _StubTrainer(data=bundle["data"]))
        ds_train = FluxDetectionTrainer.build_dataset(stub, img_path=".", mode="train", batch=2)
        ds_val = FluxDetectionTrainer.build_dataset(stub, img_path=".", mode="val", batch=2)
        assert ds_train.augment is True
        assert ds_val.augment is False
    finally:
        _FLUX_CTX.reset(token)


def test_augmentation_dict_stored_for_overrides_merge() -> None:
    trainer = UltralyticsTrainer(
        train_set=_make_flux(2),
        val_set=_make_flux(2),
        num_classes=2,
        class_names={0: "bg", 1: "fg"},
        augmentation={"mosaic": 0.5, "mixup": 0.3, "fliplr": 0.1},
    )
    assert trainer.augmentation == {"mosaic": 0.5, "mixup": 0.3, "fliplr": 0.1}


def test_augmentation_with_augment_false_raises() -> None:
    with pytest.raises(ValueError, match="meaningless"):
        UltralyticsTrainer(
            train_set=_make_flux(2),
            val_set=_make_flux(2),
            num_classes=2,
            class_names={0: "bg", 1: "fg"},
            augment=False,
            augmentation={"mosaic": 0.5},
        )


# ---------------------------------------------------------------------------
# UltralyticsEvaluator ergonomic form
# ---------------------------------------------------------------------------


def test_evaluator_ergonomic_form() -> None:
    ev = UltralyticsEvaluator(
        weights="dummy.pt",
        val_set=_make_flux(2),
        num_classes=2,
        class_names={0: "bg", 1: "fg"},
    )
    assert isinstance(ev.data, dict)
    assert ev.data["nc"] == 2
    assert ev.data["names"] == {0: "bg", 1: "fg"}


def test_evaluator_ergonomic_rejects_both_data_and_val_set() -> None:
    with pytest.raises(ValueError, match="EITHER"):
        UltralyticsEvaluator(
            weights="dummy.pt",
            data="dummy.yaml",
            val_set=_make_flux(2),
            num_classes=2,
            class_names={0: "bg", 1: "fg"},
        )


def test_evaluator_neither_data_nor_val_set_raises() -> None:
    with pytest.raises(ValueError, match="EITHER `data` OR `val_set`"):
        UltralyticsEvaluator(weights="dummy.pt")
