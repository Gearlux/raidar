"""Each detector names the weight artifacts it loads (``matrainer.uri``).

These wrappers build their detector in ``__init__`` — constructing one downloads COCO-pretrained
weights — so the property is exercised against a stand-in carrying the attributes it reads. That
is a limitation of these classes' construction, not of the property: it reads stored configuration
only, which is exactly what makes the stand-in a faithful test.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from matrainer.uri import ModelRef

from raidar.hf_detection import HFDetectionModel
from raidar.rfdetr import RFDETR_URI_PREFIX, RFDETRModel


def refs_of(cls: Any, **attributes: Any) -> List[ModelRef]:
    """Read ``cls.model_refs`` against a stand-in holding the attributes it consults."""
    return list(cls.model_refs.fget(SimpleNamespace(**attributes)))


def test_a_hub_detector_names_its_repository() -> None:
    refs = refs_of(HFDetectionModel, model_id="ustc-community/dfine-small-coco", pretrained=True, checkpoint_path=None)
    assert [r.uri for r in refs] == ["hf://models/ustc-community/dfine-small-coco"]
    assert refs[0].url == "https://huggingface.co/ustc-community/dfine-small-coco"
    assert refs[0].role == "pretrained"


def test_a_hub_detector_built_from_config_reports_nothing() -> None:
    assert refs_of(HFDetectionModel, model_id="org/net", pretrained=False, checkpoint_path=None) == []


def test_a_checkpoint_is_a_second_entry_with_its_own_role(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    refs = refs_of(HFDetectionModel, model_id="org/net", pretrained=True, checkpoint_path=str(checkpoint))
    assert [r.role for r in refs] == ["pretrained", "checkpoint"]
    assert refs[1].uri == checkpoint.resolve().as_uri()


def test_the_subclasses_inherit_the_property() -> None:
    """They differ only in which Hub id they default to, so there is one implementation."""
    from raidar.hf_detection import DFINEModel, RTDETRv2Model

    for cls in (DFINEModel, RTDETRv2Model):
        assert cls.model_refs is HFDetectionModel.model_refs


def test_an_rfdetr_variant_names_the_zoo_entry() -> None:
    """The library fetches a variant's checkpoint from its own release assets — no page to link."""
    refs = refs_of(RFDETRModel, variant="nano", pretrained_detector=True, checkpoint_path=None)
    assert [r.uri for r in refs] == [f"{RFDETR_URI_PREFIX}nano"]
    assert refs[0].url is None


def test_an_rfdetr_trained_from_scratch_reports_only_its_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    refs = refs_of(RFDETRModel, variant="small", pretrained_detector=False, checkpoint_path=str(checkpoint))
    assert [r.role for r in refs] == ["checkpoint"]


def test_a_detector_with_neither_reports_nothing() -> None:
    assert refs_of(RFDETRModel, variant="nano", pretrained_detector=False, checkpoint_path=None) == []
