# mypy: disable-error-code="attr-defined"
"""The Keras backend — dense-target batching, the spec loss hookup, and a real two-epoch fit.

The end-to-end test runs a REAL ``model.fit`` over a synthetic stream: the bridge (FcosTargets →
collate → ``_to_arrays`` → the three spec losses → numpy AP scoring) is the module's content, and
only a live fit exercises it.
"""

from typing import Any, Dict, List, get_args

import numpy as np
import pytest
import torch
from recordstream import Image, Record, Regions
from recordstream.core import Stream
from recordstream.runnable import entrypoint_tasks

from raidar.fcos.targets import FcosTargets
from raidar.keras_detector import DetectorTask, KerasDetector


def _record(size: int = 64) -> Record:
    return {
        "image": Image(np.random.default_rng(0).random((3, size, size)).astype("float32"), layout="CHW"),
        "target": Regions(
            boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]], dtype=torch.float32),
            labels=torch.tensor([1], dtype=torch.int64),
        ),
        "idx": 7,
    }


def _make_stream(n: int = 4, size: int = 64) -> Stream:
    records: List[Record] = []
    for i in range(n):
        record = _record(size)
        record["idx"] = i
        records.append(record)
    return Stream(source=records, ops=[FcosTargets(num_classes=3, input_shape=(size, size, 3))])


# --------------------------------------------------------------------------- #
# FcosTargets — the dense-target op
# --------------------------------------------------------------------------- #
class TestFcosTargets:
    def test_it_writes_the_three_dense_maps(self) -> None:
        out = FcosTargets(num_classes=3, input_shape=(64, 64, 3))(_record())
        assert out is not None
        assert out["fcos_cls"].shape == (8, 8, 3)
        assert out["fcos_box"].shape == (8, 8, 4)
        assert out["fcos_ctr"].shape == (8, 8)

    def test_it_DESTROYS_NOTHING(self) -> None:
        """The op ADDS the dense maps; the Regions stays for the eval pass to score against.

        It briefly replaced the target with a plain `{boxes, labels}` dict, believing a
        variable-N region column could not survive collation. Measured false (2026-08-05): a
        `Regions`' boxes/labels are declared item ATTRS, which the collate lists per record
        rather than stacking — so the degrade bought nothing and cost `canvas`/`extras`.
        """
        record = _record()
        record["target"] = Regions(
            boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
            labels=torch.tensor([1]),
            canvas=(64, 64),
        )
        out = FcosTargets(num_classes=3, input_shape=(64, 64, 3))(record)
        assert out is not None
        assert isinstance(out["target"], Regions), "the target item must survive the op"
        assert out["target"].canvas == (64, 64), "frame metadata the old degrade threw away"

    def test_torch_boxes_are_read_as_readily_as_lists(self) -> None:
        """np.asarray reads a CPU tensor as readily as a list — the assignment needs numpy."""
        record = _record()
        assert isinstance(record["target"].boxes, torch.Tensor)
        out = FcosTargets(num_classes=3, input_shape=(64, 64, 3))(record)
        assert out is not None and out["fcos_cls"].sum() > 0, "the boxes reached the assignment"

    def test_a_frame_mismatch_is_caught_at_the_first_record(self) -> None:
        """The knob-level error `canvas` made possible — see `_check_frame`.

        Not a correctness fix: a real mismatch also dies downstream in the model
        ("expected shape=(None, 128, 128, 3), found shape=(2, 256, 256, 3)"). This one fires
        earlier and names the CONFIG lines instead of the tensors.
        """
        from recordstream.ops.target import ResizeDetection

        record = {
            "image": Image(np.zeros((300, 300, 3), dtype="uint8")),
            "target": Regions(boxes=torch.tensor([[30.0, 30.0, 250.0, 250.0]]), labels=torch.tensor([1])),
        }
        resized = ResizeDetection(width=256, height=256)(record)
        assert resized is not None
        with pytest.raises(ValueError, match=r"\(256, 256\) \(H, W\) frame but this op assigns against \(128, 128\)"):
            FcosTargets(num_classes=3, input_shape=(128, 128, 3))(dict(resized))
        # The matching frame passes, and a target with NO canvas skips THIS check — since
        # 2026-08-06 that means only a record with no image to read the frame from.
        assert FcosTargets(num_classes=3, input_shape=(256, 256, 3))(dict(resized)) is not None
        no_canvas = dict(record)
        no_canvas["target"] = Regions(boxes=torch.tensor([[3.0, 3.0, 40.0, 40.0]]), labels=torch.tensor([1]))
        assert FcosTargets(num_classes=3, input_shape=(64, 64, 3))(no_canvas) is not None

    def test_a_canvas_less_record_is_still_caught_by_the_box_EXTENT(self) -> None:
        """The canvas check is not the only line of defence, and this is the case that matters.

        An image-only resize moves the pixels and leaves the boxes in the source frame — AND
        leaves the canvas agreeing with `input_shape`, so the knob check passes while the image
        matches too (passing the model's own size check). The spec's `assign_targets` guard
        catches it from the box coordinates alone, with no canvas needed. This test asserted the
        opposite until 2026-08-05: it pinned the silent pass as correct.
        """
        record = {
            "image": Image(np.zeros((64, 64, 3), dtype="uint8")),
            "target": Regions(boxes=torch.tensor([[30.0, 30.0, 250.0, 250.0]]), labels=torch.tensor([1])),
        }
        with pytest.raises(ValueError, match="not in this frame"):
            FcosTargets(num_classes=3, input_shape=(64, 64, 3))(record)

    def test_zero_arg_construction_validates_lazily(self) -> None:
        op = FcosTargets()
        with pytest.raises(ValueError, match="num_classes"):
            op(_record())

    def test_a_missing_regions_target_raises_naming_the_ordering(self) -> None:
        record = _record()
        record["target"] = {"bbox": []}
        with pytest.raises(ValueError, match="detection-target op"):
            FcosTargets(num_classes=3, input_shape=(64, 64, 3))(record)

    def test_the_dense_maps_stack_while_the_variable_n_target_does_not(self) -> None:
        """The whole point, and BOTH halves of it: the maps batch regularly (that is what the
        op is for) while the variable-N ground truth rides along, read back by `batch_regions`
        — no ragged tensors, no padding, nothing destroyed."""
        from recordstream import batch_regions, collate_records

        two_boxes = _record()
        two_boxes["target"] = Regions(
            boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0], [2.0, 2.0, 20.0, 20.0]]),
            labels=torch.tensor([1, 2]),
        )
        op = FcosTargets(num_classes=3, input_shape=(64, 64, 3))
        # An op's contract is `Record -> Optional[Record]` (None = drop); assert the encoding
        # happened rather than feeding a maybe-None into the collate.
        encoded = [op(_record()), op(two_boxes)]
        assert all(record is not None for record in encoded)
        batch = collate_records([record for record in encoded if record is not None])
        assert np.asarray(batch["fcos_cls"]).shape == (2, 8, 8, 3)
        targets = batch_regions(batch, "target")
        assert [t["boxes"].shape[0] for t in targets] == [1, 2]


# --------------------------------------------------------------------------- #
# Construction / tags / dispatch
# --------------------------------------------------------------------------- #
def test_tags_and_entrypoints() -> None:
    assert KerasDetector.__confluid_task__ == "detection"
    assert KerasDetector.__confluid_role__ == "trainer"
    assert KerasDetector.__confluid_framework__ == "keras"
    covered = (
        set(entrypoint_tasks(KerasDetector, "trainer"))
        | set(entrypoint_tasks(KerasDetector, "evaluator"))
        | set(entrypoint_tasks(KerasDetector, "predictor"))
    )
    assert covered == set(get_args(DetectorTask))


# --------------------------------------------------------------------------- #
# End to end — a real fit + the numpy scoring path
# --------------------------------------------------------------------------- #
def _detector(**overrides: Any) -> KerasDetector:
    from raidar.fcos.keras_fcos import KerasFcos

    kwargs: Dict[str, Any] = {
        "model": KerasFcos(num_classes=3, input_shape=(64, 64, 3)),
        "num_classes": 3,
        "max_epochs": 2,
        "batch_size": 2,
    }
    kwargs.update(overrides)
    return KerasDetector(**kwargs)


def test_fit_end_to_end_scores_validation_with_the_spec_ap(tmp_path: Any) -> None:
    import json

    sidecar = tmp_path / "metrics.json"
    detector = _detector(train_set=_make_stream(6), task="fit")
    detector.val_set = _make_stream(2)
    detector.metrics_sidecar_path = str(sidecar)
    detector.run()

    metrics = detector.computed_metrics
    assert "train/loss" in metrics, metrics
    assert "val/map_50" in metrics and "val/precision" in metrics and "val/recall" in metrics
    assert json.loads(sidecar.read_text())["metrics"]
    assert "loss" in detector.history  # Keras's own vocabulary stays on the history


def test_test_scores_through_the_same_numpy_path(tmp_path: Any) -> None:
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
        assert "idx" in meta, "predict must keep the record metadata beside each prediction"
