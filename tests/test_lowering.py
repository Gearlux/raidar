"""The RF-DETR predictor's lowering is the library's own inference path, fixed-size: measured against the library's
pieces (its resize, its normalization, its post-processing) on a stand-in network, and against the predictor itself
on a real checkpoint when one is named."""

import os
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from recordstream import Image
from torch import nn
from torpaido.lowering import model_lowering_for

from raidar.lowering import lower_rfdetr_predictor
from raidar.rfdetr import RFDETRPredictor, _AnyClassPostProcess

rfdetr_postprocess = pytest.importorskip("rfdetr.models.postprocess")

QUERIES, CLASSES, RESOLUTION = 6, 4, 32  # the head's slots: three classes and the no-object slot


class _Network(nn.Module):
    """A detector head whose logits and boxes depend on the picture's mean colour — so the resize and the
    normalization in front of it matter to the test."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.head = nn.Linear(3, QUERIES * CLASSES + QUERIES * 4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.head(x.mean(dim=(2, 3)))
        logits = out[:, : QUERIES * CLASSES].view(-1, QUERIES, CLASSES) * 4.0
        boxes = torch.sigmoid(out[:, QUERIES * CLASSES :].view(-1, QUERIES, 4))
        return {"pred_logits": logits, "pred_boxes": boxes}


def _fake_rf(num_select: int = 5) -> Any:
    model = SimpleNamespace(
        model=_Network().eval(),
        postprocess=rfdetr_postprocess.PostProcess(num_select=num_select),
        resolution=RESOLUTION,
        device=torch.device("cpu"),
        args=SimpleNamespace(num_classes=CLASSES - 1),
    )
    return SimpleNamespace(
        model=model, class_names=["a", "b", "c"], means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]
    )


class _Context:
    """What a lowering asks the program for, stood in: modules and buffers kept by name."""

    def __init__(self) -> None:
        self.held: Dict[str, Any] = {}

    def module(self, name: str, module: nn.Module) -> nn.Module:
        held: nn.Module = self.held.setdefault(name, module)
        return held

    def buffer(self, name: str, factory: Any) -> torch.Tensor:
        if name not in self.held:
            self.held[name] = factory()
        tensor: torch.Tensor = self.held[name]
        return tensor


def _pictures(seed: int = 0, count: int = 2, height: int = 37, width: int = 53) -> torch.Tensor:
    return torch.rand(count, 3, height, width, generator=torch.Generator().manual_seed(seed))


def _library_reference(
    rf: Any, pictures: torch.Tensor, threshold: float, any_class: Any = None
) -> List[Dict[str, Any]]:
    """What ``RFDETR.predict`` computes on tensors: its resize, its normalization, its post-processing, its cut."""
    resized = torch.stack([TF.resize(p, [RESOLUTION, RESOLUTION], antialias=False) for p in pictures])
    normalized = TF.normalize(resized, rf.means, rf.stds)
    with torch.no_grad():
        outputs = rf.model.model(normalized)
    target_sizes = torch.tensor([[pictures.shape[-2], pictures.shape[-1]]] * pictures.shape[0])
    postprocess = any_class if any_class is not None else rf.model.postprocess
    results = postprocess(outputs, target_sizes=target_sizes)
    rows = []
    for result in results:
        keep = (result["scores"] > threshold) & (result["labels"] < len(rf.class_names))
        rows.append({key: result[key][keep] for key in ("boxes", "scores", "labels")})
    return rows


def _kept(found: Dict[str, Any], index: int) -> Dict[str, torch.Tensor]:
    keep = found["keep"][index]
    return {key: found[key][index][keep] for key in ("boxes", "scores", "labels")}


def test_the_lowering_is_registered_for_the_predictor() -> None:
    assert model_lowering_for(RFDETRPredictor(checkpoint_path="x")) is lower_rfdetr_predictor


def test_the_kept_rows_are_the_library_s_detections() -> None:
    predictor = RFDETRPredictor(checkpoint_path="unused.pth", score_threshold=0.3)
    predictor._rf = _fake_rf()
    pictures = _pictures()

    found = lower_rfdetr_predictor(predictor, pictures, _Context())  # type: ignore[arg-type]
    expected = _library_reference(predictor._rf, pictures, 0.3)

    assert found["class_count"] == 3 and found["boxes"].shape == (2, 5, 4)
    for index in range(2):
        kept = _kept(found, index)
        assert kept["labels"].tolist() == expected[index]["labels"].tolist()
        assert torch.allclose(kept["scores"], expected[index]["scores"], atol=1e-6)
        assert torch.allclose(kept["boxes"], expected[index]["boxes"], atol=1e-4)
    assert kept["boxes"].numel() > 0, "the stand-in should keep at least one row for the test to mean anything"


def test_the_any_class_mode_rewrites_the_logits_as_the_predictor_does() -> None:
    predictor = RFDETRPredictor(checkpoint_path="unused.pth", score_threshold=0.2, score_mode="any_class")
    rf = _fake_rf()
    predictor._rf = rf
    rf.model.postprocess = _AnyClassPostProcess(
        rf.model.postprocess, predictor._counted_slots(rf)
    )  # what solidify does
    pictures = _pictures(seed=1)

    found = lower_rfdetr_predictor(predictor, pictures, _Context())  # type: ignore[arg-type]
    expected = _library_reference(rf, pictures, 0.2, any_class=rf.model.postprocess)

    for index in range(2):
        kept = _kept(found, index)
        assert kept["labels"].tolist() == expected[index]["labels"].tolist()
        assert torch.allclose(kept["scores"], expected[index]["scores"], atol=1e-6)
        assert torch.allclose(kept["boxes"], expected[index]["boxes"], atol=1e-4)


def test_the_no_object_slot_is_never_kept() -> None:
    predictor = RFDETRPredictor(checkpoint_path="unused.pth", score_threshold=0.0)
    predictor._rf = _fake_rf(num_select=QUERIES * CLASSES)

    found = lower_rfdetr_predictor(predictor, _pictures(seed=2), _Context())  # type: ignore[arg-type]

    assert (found["labels"][found["keep"]] < 3).all()
    assert (found["labels"] == 3).any(), "the stand-in must select the no-object slot somewhere for the test to bite"


@pytest.mark.slow
def test_a_real_checkpoint_gives_the_predictor_s_own_boxes() -> None:
    path = os.environ.get("RAIDAR_RFDETR_CHECKPOINT")
    if not path or not os.path.exists(path):
        pytest.skip("set RAIDAR_RFDETR_CHECKPOINT to an RF-DETR checkpoint to run the lowering against the library")
    predictor = RFDETRPredictor(checkpoint_path=path, score_threshold=0.3)
    picture = (np.random.default_rng(0).random((320, 320, 3)) * 255).astype(np.uint8)
    expected = predictor([Image(picture)])[0]

    pictures = torch.from_numpy(picture).permute(2, 0, 1).float()[None] / 255.0
    found = lower_rfdetr_predictor(predictor, pictures, _Context())  # type: ignore[arg-type]
    kept = _kept(found, 0)

    assert kept["labels"].tolist() == expected["labels"].tolist()
    assert torch.allclose(kept["boxes"], expected["boxes"], atol=1e-2)
