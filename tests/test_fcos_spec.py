"""The shared FCOS spec — the derivation, the numpy reference, and each implementation's claim.

The parameter-count test is the one holding the fleet together: every implementation is checked
against the spec's DERIVED count (the layer order is prose, so the number is the contract). A
new engine implementation joins ``IMPLEMENTATIONS`` here and nothing else.
"""

import importlib.util
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

from raidar.fcos import spec

# Engine-implementation builders, extended one per phase. Each returns an object exposing
# `parameter_count` after being configured with num_classes=5 and default widths.
IMPLEMENTATIONS: Dict[str, Callable[[], Any]] = {}


def _torch_fcos() -> Any:
    from raidar.fcos.torch_fcos import TorchFcos

    return TorchFcos(num_classes=5)


def _keras_fcos() -> Any:
    from raidar.fcos.keras_fcos import KerasFcos

    return KerasFcos(num_classes=5)


def _tensorflow_fcos() -> Any:
    from raidar.fcos.tensorflow_fcos import TensorFlowFcos

    return TensorFlowFcos(num_classes=5)


def _flax_fcos() -> Any:
    from raidar.fcos.flax_fcos import FlaxFcos

    return FlaxFcos(num_classes=5)


def _mlx_fcos() -> Any:
    from raidar.fcos.mlx_fcos import MlxFcos

    return MlxFcos(num_classes=5)


IMPLEMENTATIONS["torch"] = _torch_fcos
IMPLEMENTATIONS["keras"] = _keras_fcos
IMPLEMENTATIONS["tensorflow"] = _tensorflow_fcos
IMPLEMENTATIONS["jax"] = _flax_fcos
if importlib.util.find_spec("mlx") is not None:  # Apple silicon only — the extra's own marker
    IMPLEMENTATIONS["mlx"] = _mlx_fcos


# --------------------------------------------------------------------------- #
# The derivation
# --------------------------------------------------------------------------- #
def test_the_derived_count_matches_the_documented_number() -> None:
    assert spec.trainable_parameter_count(5) == 110_234


def test_stride_and_grid_follow_from_the_widths() -> None:
    assert spec.stride() == 8
    assert spec.grid_size((256, 256, 3)) == (32, 32)
    assert spec.grid_size((128, 256, 3), widths=(8, 16)) == (32, 64)


def test_an_indivisible_input_raises_naming_the_rule() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        spec.grid_size((250, 256, 3))


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_implementation_reports_the_derived_count(name: str) -> None:
    assert IMPLEMENTATIONS[name]().parameter_count == spec.trainable_parameter_count(5)


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_the_agreement_holds_under_custom_widths(name: str) -> None:
    """The count must be DERIVED, not memorized — custom widths catch a hardcoded number."""
    builder = IMPLEMENTATIONS[name]
    model = builder()
    model.widths = (8, 16)
    model.head_width = 24
    for cache in ("net", "_model"):  # force a rebuild under the new widths, whichever cache name
        if hasattr(model, cache):
            setattr(model, cache, None)
    assert model.parameter_count == spec.trainable_parameter_count(5, widths=(8, 16), head_width=24)


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_implementation_is_SEEDED(name: str) -> None:
    """An unseeded engine is not comparable to itself, so a refactor cannot be re-run to check it.

    Keras was the one implementation without the knob, and it showed: three runs of an identical
    config gave 2.27 / 2.18 / 2.12 while its three siblings landed bit-identical, so the one
    engine that could not confirm a refactor was silently the one you would ask. Derived over
    IMPLEMENTATIONS rather than listed, so the next engine cannot join without it.
    """
    model = IMPLEMENTATIONS[name]()
    assert hasattr(model, "seed"), f"{name} has no seed knob — its runs cannot be compared"
    assert isinstance(model.seed, int)


def test_the_keras_seed_actually_reproduces_the_weights() -> None:
    """The knob is only worth having if it BINDS — assert on the built weights, not the API."""
    from raidar.fcos.keras_fcos import KerasFcos

    def first_kernel(seed: int) -> Any:
        model = KerasFcos(num_classes=5, input_shape=(64, 64, 3), seed=seed)
        return np.asarray(model.solidify().get_weights()[0])

    a, b, other = first_kernel(0), first_kernel(0), first_kernel(1)
    assert np.array_equal(a, b), "the same seed must rebuild the same weights"
    assert not np.array_equal(a, other), "a different seed must not — else it is seeding nothing"


# --------------------------------------------------------------------------- #
# Assignment — the FCOS center-sampling reference
# --------------------------------------------------------------------------- #
def test_assignment_marks_the_cells_under_a_box_center() -> None:
    # A box centered on (64, 64) with the default 8-px stride: cell (7, 7) owns point (60, 60),
    # cell (8, 8) owns (68, 68) — both inside the box AND within 1.5 strides of its center.
    boxes = np.array([[32.0, 32.0, 96.0, 96.0]])
    out = spec.assign_targets(boxes, np.array([2]), num_classes=5)
    assert out["foreground"][7, 7] and out["foreground"][8, 8]
    # Center sampling: a cell inside the box but far from its center stays NEGATIVE.
    assert not out["foreground"][4, 4]
    assert out["cls"][7, 7].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    # The (l, t, r, b) distances from cell (7, 7)'s point (60, 60).
    assert out["box"][7, 7].tolist() == [28.0, 28.0, 36.0, 36.0]
    assert 0.0 < out["centerness"][7, 7] <= 1.0


def test_a_cell_claimed_twice_takes_the_smaller_box() -> None:
    boxes = np.array([[0.0, 0.0, 256.0, 256.0], [56.0, 56.0, 72.0, 72.0]])
    out = spec.assign_targets(boxes, np.array([0, 3]), num_classes=5)
    assert out["cls"][7, 7, 3] == 1.0 and out["cls"][7, 7, 0] == 0.0


def test_an_empty_image_assigns_all_negative() -> None:
    out = spec.assign_targets(np.zeros((0, 4)), np.zeros((0,)), num_classes=5)
    assert not out["foreground"].any()
    assert out["cls"].sum() == 0.0


def test_a_perfectly_centered_cell_has_centerness_one() -> None:
    # Box centered exactly on cell (7, 7)'s point (60, 60).
    out = spec.assign_targets(np.array([[40.0, 40.0, 80.0, 80.0]]), np.array([1]), num_classes=5)
    assert out["centerness"][7, 7] == pytest.approx(1.0)


class TestBoxesMustBeInTheFrame:
    """The spec's ONE silent failure, guarded in the one place both callers share.

    Boxes left in a stale coordinate system (an image-only resize moves pixels, not boxes)
    produce NO error of their own: every box falls outside the grid, nothing is positive, and
    training runs against pure background — measured as `loss_box`/`loss_ctr` pinned at `-0.0`
    while `loss_cls` climbs. Both `FcosTargets` and `TorchFcos` reach assignment through
    `assign_targets`, so guarding it there covers every engine.
    """

    def test_stale_frame_boxes_raise_naming_the_cause(self) -> None:
        with pytest.raises(ValueError, match="not in this frame"):
            spec.assign_targets(
                np.array([[50.0, 50.0, 426.0, 426.0]]), np.array([1]), input_shape=(128, 128, 3), num_classes=3
            )

    def test_the_message_names_the_op_that_moves_BOTH(self) -> None:
        with pytest.raises(ValueError, match="ResizeDetection"):
            spec.assign_targets(
                np.array([[0.0, 0.0, 999.0, 999.0]]), np.array([0]), input_shape=(128, 128, 3), num_classes=3
            )

    def test_a_box_just_past_the_edge_is_TOLERATED(self) -> None:
        """Annotations are not always clipped to the image — a pixel or two over is not a bug."""
        out = spec.assign_targets(
            np.array([[10.0, 10.0, 129.0, 129.0]]), np.array([1]), input_shape=(128, 128, 3), num_classes=3
        )
        assert out["foreground"].any()

    def test_an_empty_target_is_not_a_frame_error(self) -> None:
        out = spec.assign_targets(np.zeros((0, 4)), np.zeros((0,)), input_shape=(128, 128, 3), num_classes=3)
        assert not out["foreground"].any()

    def test_the_guard_reaches_the_torch_model_too(self) -> None:
        """`TorchFcos` validates the IMAGE size itself but never saw the boxes' frame."""
        import torch

        from raidar.fcos.torch_fcos import TorchFcos

        model = TorchFcos(num_classes=3, input_shape=(64, 64, 3))
        model.train()
        targets = [{"boxes": torch.tensor([[10.0, 10.0, 300.0, 300.0]]), "labels": torch.tensor([1])}]
        with pytest.raises(ValueError, match="not in this frame"):
            model([torch.rand(3, 64, 64)], targets)


def test_assignment_without_num_classes_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        spec.assign_targets(np.zeros((0, 4)), np.zeros((0,)))


# --------------------------------------------------------------------------- #
# Decode — sigmoid/exp transforms, thresholds, NMS
# --------------------------------------------------------------------------- #
def _raw_maps(rows: int = 32, cols: int = 32, num_classes: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_logits = np.full((rows, cols, num_classes), -20.0, dtype=np.float32)
    box_raw = np.zeros((rows, cols, 4), dtype=np.float32)
    ctr_logit = np.full((rows, cols), -20.0, dtype=np.float32)
    return cls_logits, box_raw, ctr_logit


def test_decode_recovers_a_planted_box() -> None:
    cls_logits, box_raw, ctr_logit = _raw_maps()
    # Plant class 2 at cell (7, 7) = point (60, 60), distances exp(raw) * 8 = 28.4 px each way.
    cls_logits[7, 7, 2] = 20.0
    ctr_logit[7, 7] = 20.0
    box_raw[7, 7] = np.log(28.0 / 8.0)
    out = spec.decode_predictions(cls_logits, box_raw, ctr_logit)
    assert len(out["boxes"]) == 1
    assert out["labels"].tolist() == [2]
    assert out["scores"][0] == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(out["boxes"][0], [32.0, 32.0, 88.0, 88.0], atol=1e-3)


def test_decode_scores_gate_on_centerness_too() -> None:
    """score = sqrt(cls * ctr): a confident class on a zero-centerness cell must not survive."""
    cls_logits, box_raw, ctr_logit = _raw_maps()
    cls_logits[7, 7, 2] = 20.0  # ctr stays -20 -> score ~ sqrt(1 * 0) ~ 0
    out = spec.decode_predictions(cls_logits, box_raw, ctr_logit, score_threshold=0.05)
    assert len(out["boxes"]) == 0


def test_decode_nms_merges_overlapping_cells_and_keeps_distinct_classes() -> None:
    cls_logits, box_raw, ctr_logit = _raw_maps()
    for cell in (7, 8):  # two adjacent cells predicting the SAME box for class 1
        cls_logits[cell, cell, 1] = 20.0
        ctr_logit[cell, cell] = 20.0
        box_raw[cell, cell] = np.log(30.0 / 8.0)
    cls_logits[20, 20, 4] = 20.0  # a distinct class elsewhere survives independently
    ctr_logit[20, 20] = 20.0
    box_raw[20, 20] = np.log(20.0 / 8.0)
    out = spec.decode_predictions(cls_logits, box_raw, ctr_logit, iou_threshold=0.5)
    assert sorted(out["labels"].tolist()) == [1, 4]


def test_decode_clips_to_the_image() -> None:
    cls_logits, box_raw, ctr_logit = _raw_maps()
    cls_logits[0, 0, 0] = 20.0
    ctr_logit[0, 0] = 20.0
    box_raw[0, 0] = np.log(100.0 / 8.0)  # extends far past the top-left corner
    out = spec.decode_predictions(cls_logits, box_raw, ctr_logit)
    assert out["boxes"][0][0] == 0.0 and out["boxes"][0][1] == 0.0


def test_decode_caps_max_detections() -> None:
    cls_logits, box_raw, ctr_logit = _raw_maps()
    for i in range(0, 32, 4):
        cls_logits[i, i, 0] = 20.0
        ctr_logit[i, i] = 20.0
        box_raw[i, i] = np.log(10.0 / 8.0)
    out = spec.decode_predictions(cls_logits, box_raw, ctr_logit, iou_threshold=0.99, max_detections=3)
    assert len(out["boxes"]) == 3


# --------------------------------------------------------------------------- #
# AP50 — the numpy scoring the non-torch engines report
# --------------------------------------------------------------------------- #
def _prediction(boxes: List[List[float]], scores: List[float], labels: List[int]) -> Dict[str, np.ndarray]:
    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(scores, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def _target(boxes: List[List[float]], labels: List[int]) -> Dict[str, np.ndarray]:
    return {"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4), "labels": np.array(labels, dtype=np.int64)}


def test_perfect_predictions_score_one() -> None:
    box = [10.0, 10.0, 50.0, 50.0]
    out = spec.average_precision_50([_prediction([box], [0.9], [1])], [_target([box], [1])])
    assert out == {"map_50": 1.0, "precision": 1.0, "recall": 1.0}


def test_a_wrong_class_match_is_a_miss() -> None:
    box = [10.0, 10.0, 50.0, 50.0]
    out = spec.average_precision_50([_prediction([box], [0.9], [2])], [_target([box], [1])])
    assert out["map_50"] == 0.0 and out["recall"] == 0.0


def test_a_duplicate_detection_counts_once() -> None:
    """Greedy matching: the second (lower-scored) hit on an already-matched box is a FP."""
    box = [10.0, 10.0, 50.0, 50.0]
    out = spec.average_precision_50([_prediction([box, box], [0.9, 0.8], [1, 1])], [_target([box], [1])])
    assert out["recall"] == 1.0
    assert out["precision"] == 0.5
    assert out["map_50"] == 1.0  # the TP ranks first, so AP is unhurt


def test_ap_ranks_by_score_across_images() -> None:
    """A confident FP above a TP drags AP below 1 — the ranking is split-wide, not per-image."""
    box = [10.0, 10.0, 50.0, 50.0]
    far = [100.0, 100.0, 140.0, 140.0]
    out = spec.average_precision_50(
        [_prediction([far], [0.95], [1]), _prediction([box], [0.6], [1])],
        [_target([box], [1]), _target([box], [1])],
    )
    # Recall tops out at 0.5 (one of two boxes found) with precision 0.5 there: AP = 0.25.
    assert out["map_50"] == pytest.approx(0.25)


def test_map_averages_over_ground_truth_classes_only() -> None:
    """A class predicted but absent from ALL targets must not dilute (or inflate) the mean."""
    box = [10.0, 10.0, 50.0, 50.0]
    out = spec.average_precision_50(
        [_prediction([box, [60.0, 60.0, 90.0, 90.0]], [0.9, 0.9], [1, 3])], [_target([box], [1])]
    )
    assert out["map_50"] == 1.0  # class 3 contributes no AP term; it does cost precision
    assert out["precision"] == 0.5


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="predictions for"):
        spec.average_precision_50([], [_target([[0.0, 0.0, 1.0, 1.0]], [0])])
