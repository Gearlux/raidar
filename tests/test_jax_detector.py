# mypy: disable-error-code="attr-defined"
"""The JAX backend — the jitted spec loss, the dense-target bridge, and a real two-epoch fit."""

import ast
from pathlib import Path
from typing import Any, Dict, List, get_args

import numpy as np
import pytest
import torch
from recordstream import Image, Record, Regions
from recordstream.core import Stream
from recordstream.runnable import entrypoint_tasks

from raidar.fcos.targets import FcosTargets
from raidar.jax_detector import DetectorTask, JaxDetector

SOURCE = Path(__file__).resolve().parent.parent / "raidar" / "jax_detector.py"


def _make_stream(n: int = 4, size: int = 64) -> Stream:
    records: List[Record] = [
        {
            "image": Image(np.random.default_rng(i).random((3, size, size)).astype("float32"), layout="CHW"),
            "target": Regions(
                boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]], dtype=torch.float32),
                labels=torch.tensor([1], dtype=torch.int64),
            ),
            "idx": i,
        }
        for i in range(n)
    ]
    return Stream(source=records, ops=[FcosTargets(num_classes=3, input_shape=(size, size, 3))])


def _detector(**overrides: Any) -> JaxDetector:
    from raidar.fcos.flax_fcos import FlaxFcos

    kwargs: Dict[str, Any] = {
        "model": FlaxFcos(num_classes=3, input_shape=(64, 64, 3)),
        "num_classes": 3,
        "max_epochs": 2,
        "batch_size": 2,
    }
    kwargs.update(overrides)
    return JaxDetector(**kwargs)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_tags_and_entrypoints() -> None:
    assert JaxDetector.__confluid_task__ == "detection"
    assert JaxDetector.__confluid_role__ == "trainer"
    assert JaxDetector.__confluid_framework__ == "jax"
    covered = (
        set(entrypoint_tasks(JaxDetector, "trainer"))
        | set(entrypoint_tasks(JaxDetector, "evaluator"))
        | set(entrypoint_tasks(JaxDetector, "predictor"))
    )
    assert covered == set(get_args(DetectorTask))


def test_the_steps_are_jitted_once_and_reused() -> None:
    """A fresh ``nnx.jit`` per epoch retraces the whole graph every time — `fit` must call
    `_make_steps` OUTSIDE its epoch loop (read from the source, as the deltaid pin does)."""
    source = SOURCE.read_text()
    fit = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "fit")
    calls_in_loops = [
        ast.unparse(call)
        for loop in ast.walk(fit)
        if isinstance(loop, (ast.For, ast.While))
        for call in ast.walk(loop)
        if isinstance(call, ast.Call) and "make_steps" in ast.unparse(call)
    ]
    assert not calls_in_loops, f"_make_steps must be called once, outside the epoch loop: {calls_in_loops}"
    assert "_make_steps" in source


def test_an_unknown_optimizer_name_raises() -> None:
    from raidar.jax_detector import _as_optimizer

    with pytest.raises(ValueError, match="optax"):
        _as_optimizer("definitely_not_an_optimizer", 1e-3)


# --------------------------------------------------------------------------- #
# End to end — a real jitted fit + the numpy scoring path
# --------------------------------------------------------------------------- #
def test_fit_end_to_end_scores_validation_with_the_spec_ap(tmp_path: Any) -> None:
    import json

    sidecar = tmp_path / "metrics.json"
    detector = _detector(train_set=_make_stream(6), task="fit")
    detector.val_set = _make_stream(2)
    detector.metrics_sidecar_path = str(sidecar)
    detector.run()

    metrics = detector.computed_metrics
    assert "train/loss" in metrics and "val/loss" in metrics, metrics
    assert "val/map_50" in metrics and "val/precision" in metrics and "val/recall" in metrics
    assert json.loads(sidecar.read_text())["metrics"]
    history = detector.history
    assert len(history["train/loss"]) == 2


def test_the_loss_descends_over_epochs() -> None:
    detector = _detector(train_set=_make_stream(6), task="fit", max_epochs=4)
    detector.run()
    losses = detector.history["train/loss"]
    assert losses[-1] < losses[0], f"loss did not descend: {losses}"


def test_test_scores_through_the_same_numpy_path() -> None:
    detector = _detector(task="test")
    detector.test_set = _make_stream(4)
    detector.run()
    assert "test/map_50" in detector.computed_metrics


def test_predict_writes_per_record_with_metadata() -> None:
    class _Sink:
        def __init__(self) -> None:
            self.writes: List[Any] = []
            self.closed = False

        def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
            self.writes.append((prediction, metadata))

        def close(self) -> None:
            self.closed = True

    sink = _Sink()
    detector = _detector(task="predict")
    detector.test_set = _make_stream(3)
    detector.predictions_sink = sink
    detector.run()
    assert len(sink.writes) == 3 and sink.closed
    for prediction, meta in sink.writes:
        assert set(prediction.keys()) == {"boxes", "scores", "labels"}
        assert "idx" in meta
