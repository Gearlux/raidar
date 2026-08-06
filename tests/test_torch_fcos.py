"""``TorchFcos`` — the spec's torch implementation, checked against the native contract.

The spec-vs-implementation parameter agreement lives in ``test_fcos_spec.py`` (the fleet gate);
this file owns what is torch's alone: the lazy build, the native forward contract both ways, the
loss being differentiable, and a real overfit proving the loss actually learns.
"""

from typing import Dict, List

import pytest
import torch

from raidar.fcos.torch_fcos import TorchFcos


def _images(n: int = 2, size: int = 64) -> List[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.rand(3, size, size, generator=generator) for _ in range(n)]


def _targets(n: int = 2) -> List[Dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        for _ in range(n)
    ]


def _model(**overrides: object) -> TorchFcos:
    kwargs: Dict[str, object] = {"num_classes": 3, "input_shape": (64, 64, 3)}
    kwargs.update(overrides)
    return TorchFcos(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Lazy construction
# --------------------------------------------------------------------------- #
def test_zero_arg_construction_builds_nothing() -> None:
    model = TorchFcos()
    assert model.net is None


def test_solidify_without_num_classes_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        TorchFcos().solidify()


def test_the_build_is_cached_and_seeded() -> None:
    model = _model()
    assert model.solidify() is model.solidify()
    twin = _model()
    for p1, p2 in zip(model.backbone.parameters(), twin.backbone.parameters()):
        assert torch.equal(p1, p2)


# --------------------------------------------------------------------------- #
# The native contract
# --------------------------------------------------------------------------- #
def test_training_forward_returns_the_loss_dict() -> None:
    model = _model()
    model.train()
    losses = model(_images(), _targets())
    assert set(losses.keys()) == {"loss_cls", "loss_box", "loss_ctr"}
    total = sum(losses.values())
    assert total.requires_grad
    total.backward()  # differentiable end to end


def test_training_without_targets_raises() -> None:
    model = _model()
    model.train()
    with pytest.raises(ValueError, match="targets"):
        model(_images())


def test_eval_forward_returns_per_image_predictions() -> None:
    model = _model()
    model.eval()
    with torch.no_grad():
        predictions = model(_images(3))
    assert isinstance(predictions, list) and len(predictions) == 3
    for prediction in predictions:
        assert set(prediction.keys()) == {"boxes", "scores", "labels"}
        assert prediction["boxes"].shape[-1] == 4


def test_a_wrong_input_size_raises_naming_the_resize_rule() -> None:
    model = _model()
    model.train()
    with pytest.raises(ValueError, match="[Rr]esize"):
        model([torch.rand(3, 96, 96)], _targets(1))


def test_an_empty_target_image_still_trains() -> None:
    """A no-object image contributes classification loss only — the fg-less branch."""
    model = _model()
    model.train()
    empty = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    losses = model(_images(1), empty)
    assert float(losses["loss_box"].detach()) == 0.0 and float(losses["loss_ctr"].detach()) == 0.0
    assert float(losses["loss_cls"].detach()) > 0.0
    sum(losses.values()).backward()


# --------------------------------------------------------------------------- #
# The loss learns — a real overfit on one image
# --------------------------------------------------------------------------- #
def test_overfitting_one_image_recovers_its_box() -> None:
    torch.manual_seed(0)
    model = _model(score_threshold=0.3)
    images, targets = _images(1), _targets(1)
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=5e-3)

    model.train()
    first = float(sum(model(images, targets).values()))
    for _ in range(150):
        optimizer.zero_grad()
        losses = model(images, targets)
        sum(losses.values()).backward()
        optimizer.step()
    last = float(sum(model(images, targets).values()))
    assert last < first * 0.5, f"loss did not descend: {first:.3f} -> {last:.3f}"

    model.eval()
    with torch.no_grad():
        prediction = model(images)[0]
    assert len(prediction["boxes"]), "an overfit model must detect its training box"
    assert prediction["labels"][0] == 1
    # The top box should land near the target (IoU is checked loosely — 150 Adam steps).
    box = prediction["boxes"][0]
    expected = targets[0]["boxes"][0]
    lt = torch.maximum(box[:2], expected[:2])
    rb = torch.minimum(box[2:], expected[2:])
    intersection = float((rb - lt).clamp(min=0).prod())
    union = float(box[2:].sub(box[:2]).prod() + expected[2:].sub(expected[:2]).prod()) - intersection
    assert intersection / union > 0.5, f"IoU {intersection / union:.2f} too low: {box} vs {expected}"
