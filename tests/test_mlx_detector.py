# mypy: disable-error-code="attr-defined"
"""The MLX backend — the mx spec loss, the dense-target bridge, and a real two-epoch fit.

Apple silicon only: the whole file is skipped where ``mlx`` is not installed, which the extra's
platform marker makes the normal state everywhere else.
"""

import importlib.util
from typing import Any, Dict, List, get_args

import numpy as np
import pytest
import torch
from recordstream import Image, Record, Regions
from recordstream.core import Stream
from recordstream.runnable import entrypoint_tasks

if importlib.util.find_spec("mlx") is None:  # pragma: no cover - non-Apple hardware
    pytest.skip("mlx is Apple-silicon only", allow_module_level=True)

from raidar.fcos.targets import FcosTargets  # noqa: E402
from raidar.mlx_detector import DetectorTask, MlxDetector, _as_optimizer  # noqa: E402


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


def _detector(**overrides: Any) -> MlxDetector:
    from raidar.fcos.mlx_fcos import MlxFcos

    kwargs: Dict[str, Any] = {
        "model": MlxFcos(num_classes=3, input_shape=(64, 64, 3)),
        "num_classes": 3,
        "max_epochs": 2,
        "batch_size": 2,
    }
    kwargs.update(overrides)
    return MlxDetector(**kwargs)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_tags_and_entrypoints() -> None:
    assert MlxDetector.__confluid_task__ == "detection"
    assert MlxDetector.__confluid_role__ == "trainer"
    assert MlxDetector.__confluid_framework__ == "mlx"
    covered = (
        set(entrypoint_tasks(MlxDetector, "trainer"))
        | set(entrypoint_tasks(MlxDetector, "evaluator"))
        | set(entrypoint_tasks(MlxDetector, "predictor"))
    )
    assert covered == set(get_args(DetectorTask))


def test_the_built_network_lives_in_a_public_attribute() -> None:
    """MLX's ``valid_parameter_filter`` skips underscore-prefixed attributes — as ``_model`` the
    whole network is invisible to ``parameters()`` and the optimizer updates NOTHING, forever,
    without raising (the segmentation project's measured trap)."""
    from raidar.fcos.mlx_fcos import MlxFcos

    model = MlxFcos(num_classes=3, input_shape=(64, 64, 3))
    model.solidify()
    assert model.net is not None
    assert "net" in model.parameters(), "the built network must appear in the parameter tree"


def test_an_unknown_optimizer_name_raises() -> None:
    with pytest.raises(ValueError, match="mlx"):
        _as_optimizer("definitely_not_an_optimizer", 1e-3)


# --------------------------------------------------------------------------- #
# End to end — a real fit + the numpy scoring path
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
