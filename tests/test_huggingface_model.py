# mypy: disable-error-code="attr-defined"
"""The ONE Hub zoo wrapper — light, generic, checkpoint-as-a-string.

There are deliberately no per-checkpoint preset classes (the old ``DFINEModel``/``RTDETRv2Model``
are DELETED): any Hub object-detection checkpoint is a ``model_name``. These tests are
build-free — the wrapper's whole contract below the build line is lazy storage + validation +
``model_refs``.
"""

import pytest

from raidar.huggingface_model import HuggingFaceDetectionModel


def test_zero_arg_construction_stores_only() -> None:
    model = HuggingFaceDetectionModel()
    assert model.model_name == "" and model.num_classes == 0
    assert model._model is None and model._processor is None


def test_solidify_requires_a_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        HuggingFaceDetectionModel(num_classes=5).solidify()


def test_solidify_requires_a_positive_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        HuggingFaceDetectionModel(model_name="ustc-community/dfine-small-coco").solidify()


def test_discovery_tags() -> None:
    assert HuggingFaceDetectionModel.__confluid_task__ == "detection"
    assert HuggingFaceDetectionModel.__confluid_role__ == "model"
    assert HuggingFaceDetectionModel.__confluid_framework__ == "huggingface"


def test_model_refs_names_the_checkpoint_without_building() -> None:
    """STORED CONFIG ONLY: the property is asked of a wrapper that may never run."""
    model = HuggingFaceDetectionModel(model_name="ustc-community/dfine-small-coco", num_classes=5)
    refs = model.model_refs
    assert len(refs) == 1 and refs[0].role == "pretrained"
    assert "dfine-small-coco" in refs[0].uri
    assert model._model is None, "model_refs must not build the model"


def test_model_refs_is_empty_for_random_weights_or_no_name() -> None:
    """Random weights referenced no artifact — inventing a ref would name something nobody loaded."""
    assert HuggingFaceDetectionModel(model_name="x/y", num_classes=2, pretrained=False).model_refs == []
    assert HuggingFaceDetectionModel().model_refs == []
