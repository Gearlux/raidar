# mypy: disable-error-code="attr-defined"
"""The bare-torch control loop — same config surface as the Lightning backend, no framework.

The end-to-end tests run a REAL two-epoch fit over a synthetic in-memory stream on CPU, because
the loop IS the module's content: forward, backward, step, validate, report.
"""

import ast
import inspect
import json
from typing import Any, Dict, List, Optional, Union

import pytest
import torch
import torch.nn as nn
from recordstream import Image, Record, Regions, collate_list
from recordstream.core import Stream
from recordstream.runnable import entrypoint_tasks
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from raidar.torch_detector import TorchDetector


class _NativeDetector(nn.Module):
    """Stub conforming to the torchvision-native contract (loss dict / prediction list)."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            device = images[0].device
            return {"loss": (self.alpha * torch.ones((), device=device)) ** 2}
        return [
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
            for _ in images
        ]


def _make_stream(n: int = 4, size: int = 32) -> Stream:
    records: List[Record] = [
        {
            "image": Image(torch.rand(3, size, size).numpy(), layout="CHW"),
            "target": Regions(
                boxes=torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                labels=torch.tensor([1], dtype=torch.int64),
            ),
            "idx": i,
        }
        for i in range(n)
    ]
    return Stream(source=records)


# --------------------------------------------------------------------------- #
# The backend needs NO training framework
# --------------------------------------------------------------------------- #
def test_it_needs_no_training_framework() -> None:
    """AST-scan the module's imports: `pytorch_lightning` must not appear, directly or through
    `matrainer.lightning` (whose package __init__ runs require("pytorch_lightning"))."""
    import raidar.torch_detector as mod

    tree = ast.parse(inspect.getsource(mod))
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [name for name in imported if "pytorch_lightning" in name or name.startswith("matrainer.lightning")]
    assert not offenders, f"the framework-free backend imports a framework: {offenders}"


# --------------------------------------------------------------------------- #
# Construction / defaults — the Lightning backend's surface, minus the Trainer
# --------------------------------------------------------------------------- #
def test_zero_arg_signature_tags_and_task_default() -> None:
    params = inspect.signature(TorchDetector.__init__).parameters
    assert "logger" not in params and "lightning" not in params
    assert "trackers" in params and "metric_names" in params
    assert TorchDetector.__confluid_task__ == "detection"
    assert TorchDetector.__confluid_role__ == "trainer"
    assert TorchDetector.__confluid_framework__ == "torch"


def test_entrypoint_markers_cover_every_task() -> None:
    covered = (
        set(entrypoint_tasks(TorchDetector, "trainer"))
        | set(entrypoint_tasks(TorchDetector, "evaluator"))
        | set(entrypoint_tasks(TorchDetector, "predictor"))
    )
    assert covered == {"fit", "evaluate", "test", "predict"}


def test_no_loss_and_no_train_metrics_slot() -> None:
    """The native contract computes loss inside forward, and a train split has no predictions."""
    detector = TorchDetector(model=_NativeDetector())
    assert detector.loss_fn is None
    assert "train_metrics" not in inspect.signature(TorchDetector.__init__).parameters


def test_metric_slots_default_to_sentinels_resolved_at_initialize() -> None:
    detector = TorchDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2, device="cpu")
    assert detector.metrics is None and detector.val_metrics is None
    detector.initialize()
    assert isinstance(detector.val_metrics, MetricCollection)
    assert any(isinstance(m, MeanAveragePrecision) for m in detector.val_metrics.values())
    assert any(str(k).startswith("val/") for k in detector.val_metrics.keys())


def test_an_explicit_device_is_taken_at_its_word() -> None:
    detector = TorchDetector(model=_NativeDetector(), train_set=_make_stream(2), num_classes=2, device="cpu")
    detector.initialize()
    assert detector._device.type == "cpu"


# --------------------------------------------------------------------------- #
# End to end — the loop trains, reports, checkpoints
# --------------------------------------------------------------------------- #
def test_fit_end_to_end_trains_reports_and_checkpoints(tmp_path: Any) -> None:
    sidecar = tmp_path / "metrics.json"
    detector = TorchDetector(
        model=_NativeDetector(),
        train_set=_make_stream(6),
        num_classes=2,
        task="fit",
        max_epochs=2,
        batch_size=2,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    detector.val_set = _make_stream(2)
    detector.metrics_sidecar_path = str(sidecar)
    detector.run()

    history = detector.history
    assert len(history["train/loss"]) == 2, history
    assert history["train/loss"][1] < history["train/loss"][0], "the loss must DESCEND — the loop must train"
    assert any(k.startswith("val/map") for k in detector.computed_metrics), detector.computed_metrics
    assert (tmp_path / "ckpt" / "model.pt").exists()
    payload = json.loads(sidecar.read_text())
    assert payload["metrics"], "the sidecar carries the run's final scalars"
    assert "val/classes" not in payload["metrics"], "MAP's bookkeeping key is filtered"


def test_evaluate_scores_the_val_split(tmp_path: Any) -> None:
    detector = TorchDetector(model=_NativeDetector(), num_classes=2, task="evaluate", device="cpu")
    detector.val_set = _make_stream(4)
    detector.run()
    assert any(k.startswith("val/map") for k in detector.computed_metrics)


def test_predict_writes_per_record_to_the_sink(tmp_path: Any) -> None:
    class _Sink:
        def __init__(self) -> None:
            self.writes: List[Any] = []
            self.closed = False

        def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
            self.writes.append((prediction, metadata))

        def close(self) -> None:
            self.closed = True

    sink = _Sink()
    detector = TorchDetector(model=_NativeDetector(), num_classes=2, task="predict", device="cpu")
    detector.test_set = _make_stream(4)
    detector.predictions_sink = sink
    detector.run()
    assert len(sink.writes) == 4 and sink.closed
    for prediction, meta in sink.writes:
        assert set(prediction.keys()) == {"boxes", "scores", "labels"}
        assert isinstance(meta, dict)


def test_default_loader_DECLARES_the_list_collate() -> None:

    from confluid.fluid import Class as ConfluidClass

    detector = TorchDetector(model=_NativeDetector())
    assert isinstance(detector.train_loader, ConfluidClass)
    collate = detector.train_loader.kwargs.get("collate_fn")
    # DECLARED, not inherited: the native forward takes List[Tensor], so the loaders ask
    # recordstream for the unstacked batch rather than relying on ToTensor's plain tensors
    # happening to skip stacking.
    assert collate is collate_list


def test_training_rejects_a_missing_target() -> None:

    detector = TorchDetector(model=_NativeDetector(), num_classes=2, device="cpu")
    detector._device = torch.device("cpu")
    with pytest.raises(ValueError, match="targets"):
        detector._train_epoch(
            [collate_list([{"image": torch.rand(3, 8, 8)}])],  # type: ignore[arg-type]
            torch.optim.SGD(detector.model.parameters(), lr=0.1),  # type: ignore[union-attr]
        )
