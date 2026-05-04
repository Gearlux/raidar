"""Tests for :func:`raidar.checkpoint.load_lightning_checkpoint`."""

from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn

from raidar.checkpoint import load_lightning_checkpoint


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)


def _make_lightning_ckpt(path: Path, weights: torch.Tensor) -> None:
    """Save a synthetic Lightning ``.ckpt`` (top-level ``state_dict``, ``model.`` prefix)."""
    torch.save({"state_dict": {"model.linear.weight": weights}, "epoch": 1}, path)


def _make_plain_state_dict(path: Path, weights: torch.Tensor) -> None:
    """Save a plain ``torch.save(model.state_dict(), ...)`` with no Lightning wrapper."""
    torch.save({"linear.weight": weights}, path)


def test_strips_lightning_model_prefix(tmp_path: Path) -> None:
    weights = torch.full((2, 4), 7.0)
    ckpt = tmp_path / "lightning.ckpt"
    _make_lightning_ckpt(ckpt, weights)

    inner = _Inner()
    inner.linear.weight.data.zero_()
    load_lightning_checkpoint(inner, str(ckpt))
    assert torch.equal(inner.linear.weight.data, weights)


def test_accepts_plain_state_dict(tmp_path: Path) -> None:
    """No top-level ``state_dict`` key — the loaded object IS the state dict."""
    weights = torch.full((2, 4), 3.0)
    ckpt = tmp_path / "plain.pt"
    _make_plain_state_dict(ckpt, weights)

    inner = _Inner()
    load_lightning_checkpoint(inner, str(ckpt))
    assert torch.equal(inner.linear.weight.data, weights)


def test_strict_true_raises_on_mismatch(tmp_path: Path) -> None:
    ckpt = tmp_path / "extra.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.zeros(2, 4),
                "model.does_not_exist": torch.zeros(1),
            }
        },
        ckpt,
    )
    with pytest.raises(RuntimeError, match="(?i)unexpected"):
        load_lightning_checkpoint(_Inner(), str(ckpt), strict=True)


def test_strict_false_tolerates_extra_keys(tmp_path: Path) -> None:
    ckpt = tmp_path / "extra.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.full((2, 4), 5.0),
                "model.does_not_exist": torch.zeros(1),
            }
        },
        ckpt,
    )
    inner = _Inner()
    load_lightning_checkpoint(inner, str(ckpt), strict=False)
    assert torch.equal(inner.linear.weight.data, torch.full((2, 4), 5.0))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lightning_checkpoint(_Inner(), str(tmp_path / "nope.ckpt"))


# --- Ultralytics-format checkpoints -----------------------------------------


class _UltralyticsLikeModel(nn.Module):
    """Stand-in for ultralytics' ``DetectionModel``: an ``nn.Sequential`` of
    layers exposed as ``self.model``, so the saved state_dict has keys like
    ``model.0.weight`` — the namespace shape ultralytics' ``best.pt``
    actually carries."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 2, bias=False))


class _Wrapper(nn.Module):
    """Stand-in for raidar's :class:`~raidar.yolo26.YOLO26Model`: the real
    detector lives under ``self.detector``, so the wrapper's state_dict
    keys are prefixed ``detector.<...>``."""

    def __init__(self) -> None:
        super().__init__()
        self.detector = _UltralyticsLikeModel()


def _layer(m: _UltralyticsLikeModel) -> nn.Linear:
    """Type-narrowing accessor — ``nn.Sequential.__getitem__`` is typed
    ``Module | Tensor``, but we know index 0 is the ``Linear`` we put there."""
    return cast(nn.Linear, m.model[0])


def _make_ultralytics_ckpt(path: Path, weights: torch.Tensor) -> None:
    """Save a synthetic Ultralytics-shape checkpoint: ``raw["model"]`` is a
    *live* nn.Module instance (not a flat state-dict)."""
    src = _UltralyticsLikeModel()
    with torch.no_grad():
        _layer(src).weight.copy_(weights)
    torch.save({"model": src, "epoch": 1, "optimizer": None}, path)


def test_loads_ultralytics_format_into_matching_module(tmp_path: Path) -> None:
    """Direct load: target module's state_dict shape matches the source's."""
    weights = torch.full((2, 4), 9.0)
    ckpt = tmp_path / "best.pt"
    _make_ultralytics_ckpt(ckpt, weights)

    target = _UltralyticsLikeModel()
    _layer(target).weight.data.zero_()
    load_lightning_checkpoint(target, str(ckpt))
    assert torch.equal(_layer(target).weight.data, weights)


def test_loads_ultralytics_format_into_wrapper_with_named_child(tmp_path: Path) -> None:
    """Auto-retarget: source keys (``model.0.weight``) don't match the
    wrapper's keys (``detector.model.0.weight``) but DO match its
    ``detector`` child — so the helper redirects the load there.
    Mirrors how ``YOLO26Model(checkpoint_path=<best.pt>)`` should work."""
    weights = torch.full((2, 4), 4.0)
    ckpt = tmp_path / "best.pt"
    _make_ultralytics_ckpt(ckpt, weights)

    wrapper = _Wrapper()
    _layer(wrapper.detector).weight.data.zero_()
    load_lightning_checkpoint(wrapper, str(ckpt))
    assert torch.equal(_layer(wrapper.detector).weight.data, weights)


def test_ultralytics_format_does_not_strip_model_prefix(tmp_path: Path) -> None:
    """Regression: ``model.0.weight`` is the real namespace level
    (DetectionModel.model is an nn.Sequential), not Lightning's wrapper
    prefix. Stripping it would leave a key ``"0.weight"`` that no module
    expects, and a strict load would explode. ``strict=True`` here is the
    canary."""
    weights = torch.full((2, 4), 2.0)
    ckpt = tmp_path / "best.pt"
    _make_ultralytics_ckpt(ckpt, weights)

    target = _UltralyticsLikeModel()
    load_lightning_checkpoint(target, str(ckpt), strict=True)
    assert torch.equal(_layer(target).weight.data, weights)
