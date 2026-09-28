# mypy: disable-error-code="attr-defined"
"""The huggingface backend — Trainer-driven, processor-shaped, torchmetrics-scored.

Unit-level and download-free: the batch/annotation shaping helpers, the TrainingArguments floor,
the num_classes derivation and the dispatch markers. The end-to-end fine-tune is the example
config's job (verified against a real Hub checkpoint).
"""

import ast
import inspect
from typing import Any, Optional, cast, get_args

import numpy as np
import pytest
import torch
from recordstream import Boxes
from recordstream.runnable import entrypoint_tasks

from raidar.huggingface_detector import (
    DetectorTask,
    HuggingFaceDetector,
    _as_rgb_array,
    _coco_annotations,
    _record_target,
)


class _StubModel:
    """A live model stand-in with the two attributes the detector consults."""

    num_classes = 5
    image_processor = object()


# --------------------------------------------------------------------------- #
# Construction / tags / dispatch
# --------------------------------------------------------------------------- #
def test_tags_and_entrypoints() -> None:
    assert HuggingFaceDetector.__confluid_task__ == "detection"
    assert HuggingFaceDetector.__confluid_role__ == "trainer"
    assert HuggingFaceDetector.__confluid_framework__ == "huggingface"
    covered = (
        set(entrypoint_tasks(HuggingFaceDetector, "trainer"))
        | set(entrypoint_tasks(HuggingFaceDetector, "evaluator"))
        | set(entrypoint_tasks(HuggingFaceDetector, "predictor"))
    )
    assert covered == set(get_args(DetectorTask))


def test_no_loss_slot_and_no_train_metrics() -> None:
    """The model computes its own Hungarian loss; the training pass computes nothing else."""
    params = inspect.signature(HuggingFaceDetector.__init__).parameters
    assert "loss_fn" not in params and "train_metrics" not in params
    assert "trackers" in params and "metric_names" in params


def test_transformers_stays_a_deferred_dotted_target() -> None:
    """No module-level `import transformers` — the slots are STRING dotted paths confluid
    resolves at flow time, so `import raidar.huggingface_detector` stays light."""
    import raidar.huggingface_detector as mod

    source = inspect.getsource(mod)
    assert "\nimport transformers" not in source and "\nfrom transformers import" not in source
    detector = HuggingFaceDetector(model=_StubModel())
    assert cast(Any, detector.training_arguments).target == "transformers.TrainingArguments"
    assert cast(Any, detector.trainer).target == "transformers.Trainer"


# --------------------------------------------------------------------------- #
# The TrainingArguments floor
# --------------------------------------------------------------------------- #
def test_the_floor_is_derived_and_a_pinned_key_wins(monkeypatch: Any, tmp_path: Any) -> None:
    detector = HuggingFaceDetector(model=_StubModel(), max_epochs=3, batch_size=4, output_dir=str(tmp_path))
    captured: dict = {}

    def _fake_flow(slot: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return slot

    monkeypatch.setattr("matrainer.huggingface.trainer.flow", _fake_flow)
    cast(Any, detector.training_arguments).kwargs.update({"num_train_epochs": 99})  # the config pinned a key
    detector._training_arguments()
    assert "num_train_epochs" not in captured, "a pinned key must WIN — the floor may not restate it"
    assert captured["per_device_train_batch_size"] == 4
    assert captured["remove_unused_columns"] is False
    assert captured["report_to"] == "none"


# --------------------------------------------------------------------------- #
# Batch shaping — raw RGB + COCO annotations for the processor
# --------------------------------------------------------------------------- #
def test_as_rgb_array_coerces_every_source_mode() -> None:
    rgba = (np.arange(4 * 5 * 4).reshape(4, 5, 4) % 256).astype(np.uint8)
    gray = (np.arange(4 * 5).reshape(4, 5) % 256).astype(np.uint8)
    chw_float = torch.rand(3, 4, 5)
    for payload in (rgba, gray, chw_float):
        out = _as_rgb_array(payload)
        assert out.shape == (4, 5, 3) and out.dtype == np.uint8


def test_coco_annotations_carry_xywh_and_area() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 60.0]])
    labels = np.array([2])
    ann = _coco_annotations(7, boxes, labels)
    assert ann["image_id"] == 7
    entry = ann["annotations"][0]
    assert entry["bbox"] == [10.0, 20.0, 20.0, 40.0]  # xyxy -> xywh
    assert entry["category_id"] == 2 and entry["area"] == pytest.approx(800.0)


def test_record_target_unwraps_regions() -> None:
    record = {"target": Boxes(boxes=torch.tensor([[1.0, 2.0, 3.0, 4.0]]), labels=torch.tensor([1]))}
    boxes, labels = _record_target(record, "target")
    assert boxes.shape == (1, 4) and labels.tolist() == [1]


def test_a_model_without_a_processor_is_rejected() -> None:
    detector = HuggingFaceDetector(model=torch.nn.Identity())
    with pytest.raises(TypeError, match="image_processor"):
        detector._processor()


# --------------------------------------------------------------------------- #
# num_classes resolution
# --------------------------------------------------------------------------- #
def test_num_classes_is_derived_from_0_indexed_targets() -> None:
    records = [
        {"target": Boxes(boxes=torch.zeros((1, 4)), labels=torch.tensor([3]))},
        {"target": Boxes(boxes=torch.zeros((1, 4)), labels=torch.tensor([1]))},
    ]
    detector = HuggingFaceDetector(model=torch.nn.Identity())
    assert detector._num_classes_from_targets(records) == 4  # max label 3 -> K = 4


def test_a_live_models_head_is_authoritative() -> None:
    detector = HuggingFaceDetector(model=_StubModel(), num_classes=99)
    assert detector._resolve_num_classes() == 5


class TestTheRunIsSeeded:
    """The seed must be applied BEFORE the model is built, which is the whole point of the knob.

    `transformers.Trainer` applies `TrainingArguments.seed` (42 by default) when TRAINING starts —
    by which time this backend has already constructed the network and randomly initialized the
    detection head for `num_classes`, so that init was never covered. Measured 2026-08-06: two
    runs of one config gave eval_loss 8.001 / 6.277 with val/map_50 0.0685 / 0.1302, a 3.4x
    spread; with the seed applied first, two runs land IDENTICAL (train/loss 40.41484375,
    val/map_50 0.046988025307655334).

    Pinned by ORDER rather than behaviour: asserting on real weights would download a Hub
    checkpoint, and the defect was never that the call is missing — it is where it sits.
    """

    def test_initialize_seeds_BEFORE_it_builds_the_model(self) -> None:
        import inspect

        from raidar import huggingface_detector

        tree = ast.parse(inspect.getsource(huggingface_detector))
        initialize = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "initialize"
        )
        lines = ast.dump(initialize)
        seed_at = _index_of_call(initialize, "set_seed")
        # The model flow lives in the task base since P6 — `super().initialize()` is where the
        # build happens, so the pinned order is: seed strictly before that call.
        build_at = _index_of_call(initialize, "initialize")
        assert seed_at is not None, f"initialize() must seed before building the model: {lines[:80]}"
        assert build_at is not None
        assert seed_at < build_at, "the seed is applied AFTER the model is built — the head init is uncovered"

    def test_the_same_seed_reaches_training_arguments(self, monkeypatch: Any) -> None:
        """ONE value, so the head's init and the Trainer's data order cannot disagree."""
        detector = HuggingFaceDetector(model=_StubModel(), seed=1234)
        assert detector.seed == 1234
        captured: dict = {}

        def _fake_flow(slot: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return slot

        monkeypatch.setattr("matrainer.huggingface.trainer.flow", _fake_flow)
        detector._training_arguments()
        assert captured["seed"] == 1234

    def test_the_default_matches_the_transformers_default(self) -> None:
        """42 — so adding the knob changed the Trainer's own behaviour for nobody."""
        from transformers import TrainingArguments

        assert HuggingFaceDetector(model=_StubModel()).seed == TrainingArguments.__dataclass_fields__["seed"].default


def _index_of_call(fn_node: ast.FunctionDef, name: str) -> Optional[int]:
    """Line offset of the first call to `name` inside a function body, or None."""
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            func = node.func
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            if called == name:
                return node.lineno
    return None
