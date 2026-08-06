# mypy: disable-error-code="attr-defined"
"""The fastai backend — the no-op-loss bridge over the loss-inside-forward contract.

The end-to-end test runs a REAL two-epoch ``fit_one_cycle`` over a synthetic stream on CPU: the
bridge (collate → adapter → pass-through loss → metrics callback) is the module's content, and
only a live Learner exercises it.
"""

from typing import Any, Dict, List, Optional, Union, get_args

import torch
import torch.nn as nn
from recordstream import Image, Record, Regions
from recordstream.core import Stream
from recordstream.runnable import entrypoint_tasks
from torch import Tensor

from raidar.fastai_detector import (
    DetectorTask,
    FastaiDetector,
    NativeDetectionAdapter,
    fastai_detection_collate,
    passthrough_loss,
)


class _NativeDetector(nn.Module):
    """Stub conforming to the torchvision-native contract, with a learnable loss."""

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
# The bridge pieces
# --------------------------------------------------------------------------- #
def test_the_collate_packs_targets_into_xb() -> None:
    """fastai calls `model(*xb)` — the train-mode forward needs the targets IN that call."""
    records = list(_make_stream(2))
    (images, targets), dummy = fastai_detection_collate(records)
    assert len(images) == 2 and len(targets) == 2
    assert dummy.shape == (2,)


def test_the_adapter_returns_the_summed_loss_in_training() -> None:
    adapter = NativeDetectionAdapter(_NativeDetector())
    adapter.train()
    (payload, _) = fastai_detection_collate(list(_make_stream(2)))
    loss = adapter(payload)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0 and loss.requires_grad


def test_the_adapter_returns_predictions_in_eval() -> None:
    adapter = NativeDetectionAdapter(_NativeDetector())
    adapter.eval()
    (payload, _) = fastai_detection_collate(list(_make_stream(2)))
    predictions = adapter(payload)
    assert isinstance(predictions, list) and len(predictions) == 2


def test_the_passthrough_loss_passes_the_scalar_and_zeroes_the_rest() -> None:
    scalar = torch.tensor(3.5)
    assert passthrough_loss(scalar, None) is scalar
    # Eval: the "prediction" is a list, for which no loss exists under this contract.
    zero = passthrough_loss([{"boxes": torch.zeros((0, 4))}], None)
    assert float(zero) == 0.0


# --------------------------------------------------------------------------- #
# Construction / tags / dispatch
# --------------------------------------------------------------------------- #
def test_tags_and_entrypoints() -> None:
    assert FastaiDetector.__confluid_task__ == "detection"
    assert FastaiDetector.__confluid_role__ == "trainer"
    assert FastaiDetector.__confluid_framework__ == "torch"  # the axis names what it is WIRED TO
    covered = (
        set(entrypoint_tasks(FastaiDetector, "trainer"))
        | set(entrypoint_tasks(FastaiDetector, "evaluator"))
        | set(entrypoint_tasks(FastaiDetector, "predictor"))
    )
    assert covered == set(get_args(DetectorTask))


def test_the_schedule_import_is_load_bearing() -> None:
    """`fit_one_cycle` is a @patch installed as `fastai.callback.schedule`'s import side effect."""
    import inspect

    import raidar.fastai_detector as mod

    assert "from fastai.callback import schedule" in inspect.getsource(mod)


# --------------------------------------------------------------------------- #
# End to end — a real Learner over the bridge
# --------------------------------------------------------------------------- #
def test_fit_end_to_end_through_a_real_learner(tmp_path: Any) -> None:
    import json

    sidecar = tmp_path / "metrics.json"
    detector = FastaiDetector(
        model=_NativeDetector(),
        train_set=_make_stream(6),
        num_classes=2,
        task="fit",
        max_epochs=2,
        batch_size=2,
        learning_rate=0.1,
        device="cpu",
    )
    detector.val_set = _make_stream(2)
    detector.metrics_sidecar_path = str(sidecar)
    detector.run()

    metrics = detector.computed_metrics
    assert "train/loss" in metrics, metrics
    assert any(k.startswith("val/map") for k in metrics), "the torchmetrics pass must score validation"
    assert "val/loss" not in metrics, "the no-op contract records NO meaningful valid_loss — it must be dropped"
    payload = json.loads(sidecar.read_text())
    assert payload["metrics"]


def test_test_scores_through_the_same_collection(tmp_path: Any) -> None:
    detector = FastaiDetector(model=_NativeDetector(), num_classes=2, task="test", device="cpu")
    detector.test_set = _make_stream(4)
    detector.run()
    assert any(k.startswith("test/map") for k in detector.computed_metrics)


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
    detector = FastaiDetector(model=_NativeDetector(), num_classes=2, task="predict", device="cpu")
    detector.test_set = _make_stream(3)
    detector.predictions_sink = sink
    detector.run()
    assert len(sink.writes) == 3 and sink.closed
    for prediction, meta in sink.writes:
        assert set(prediction.keys()) == {"boxes", "scores", "labels"}
        assert "idx" in meta, "predict must keep the record metadata the fastai collate drops"
