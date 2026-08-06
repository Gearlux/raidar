"""Tests for :class:`raidar.torch_detection.DetectionConfusionMatrix`.

Synthetic-box scenarios drive every TP / FP / FN branch of the greedy
class-aware matcher. ``DetectionConfusionMatrix`` is a native
``torchmetrics.Metric`` (``update(preds, targets)`` over per-image lists) that
drops straight into a ``MetricCollection`` alongside ``MeanAveragePrecision``.
"""

from typing import Dict, List

import pytest
import torch
from torch import Tensor

from raidar.torch_detection import DetectionConfusionMatrix, DetectionOutput


def _pred(boxes: List[List[float]], scores: List[float], labels: List[int]) -> DetectionOutput:
    return DetectionOutput(
        boxes=(torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))),
        scores=(torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,))),
        labels=(torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)),
    )


def _gt(boxes: List[List[float]], labels: List[int]) -> Dict[str, Tensor]:
    return {
        "boxes": (torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))),
        "labels": (torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)),
    }


# --- matching branches ------------------------------------------------------


def test_confusion_perfect_overlap_counts_tp() -> None:
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    preds = [_pred([[0.0, 0.0, 10.0, 10.0]], [0.9], [1])]
    targets = [_gt([[0.0, 0.0, 10.0, 10.0]], [1])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 1 and int(out["fp"]) == 0 and int(out["fn"]) == 0
    assert float(out["precision"]) == 1.0
    assert float(out["recall"]) == 1.0
    assert float(out["f1"]) == 1.0


def test_confusion_low_iou_is_fp_plus_fn() -> None:
    """An IoU below the threshold leaves the prediction as FP and the GT as FN."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    # Boxes barely touch — IoU = 0
    preds = [_pred([[0.0, 0.0, 5.0, 5.0]], [0.9], [1])]
    targets = [_gt([[20.0, 20.0, 25.0, 25.0]], [1])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 1 and int(out["fn"]) == 1


def test_confusion_class_mismatch_is_fp_plus_fn() -> None:
    """Same-IoU but different class: prediction is FP, GT is FN."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    preds = [_pred([[0.0, 0.0, 10.0, 10.0]], [0.9], [2])]
    targets = [_gt([[0.0, 0.0, 10.0, 10.0]], [1])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 1 and int(out["fn"]) == 1


def test_confusion_score_below_threshold_drops_pred() -> None:
    """Predictions filtered by score: the GT becomes an FN with no FP."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.5)
    preds = [_pred([[0.0, 0.0, 10.0, 10.0]], [0.2], [1])]
    targets = [_gt([[0.0, 0.0, 10.0, 10.0]], [1])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 0 and int(out["fn"]) == 1


def test_confusion_no_predictions_all_fn() -> None:
    cm = DetectionConfusionMatrix()
    preds = [_pred([], [], [])]
    targets = [_gt([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], [1, 1])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 0 and int(out["fn"]) == 2
    assert float(out["precision"]) == 0.0
    assert float(out["recall"]) == 0.0
    assert float(out["f1"]) == 0.0


def test_confusion_no_ground_truth_all_fp() -> None:
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    preds = [_pred([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], [0.9, 0.8], [1, 1])]
    targets = [_gt([], [])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 2 and int(out["fn"]) == 0


def test_confusion_greedy_consumes_high_score_first() -> None:
    """Two predictions overlap one GT: only the highest-score wins TP, the other becomes FP."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    preds = [
        _pred(
            [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.0, 9.0]],
            [0.5, 0.9],  # second pred has higher score → consumed first
            [1, 1],
        )
    ]
    targets = [_gt([[0.0, 0.0, 10.0, 10.0]], [1])]
    cm.update(preds, targets)
    out = cm.compute()
    assert int(out["tp"]) == 1 and int(out["fp"]) == 1 and int(out["fn"]) == 0


def test_confusion_class_aware_matching_with_two_gts() -> None:
    """Predictions can only match same-class GTs, even when IoU favors a wrong-class one."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    # Two GTs at the same location but different classes; pred is class 2.
    preds = [_pred([[0.0, 0.0, 10.0, 10.0]], [0.9], [2])]
    targets = [
        _gt(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
            [1, 2],
        )
    ]
    cm.update(preds, targets)
    out = cm.compute()
    # No same-class GT overlapping → pred is FP. GT[1] (class 2 at far box) is FN.
    # GT[0] (class 1) is also FN.
    assert int(out["tp"]) == 0 and int(out["fp"]) == 1 and int(out["fn"]) == 2


def test_confusion_partial_iou_at_threshold_boundary_is_tp() -> None:
    """IoU == iou_threshold counts as TP (>= comparison)."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    # Construct exact 0.5 IoU: 10x10 vs 10x10 shifted by `s` in x;
    # need (10-s)*10 / (200 - (10-s)*10) = 0.5 → 10-s = 6.667, s = 3.333.
    s = 3.333333333333333
    cm.update(
        [_pred([[0.0, 0.0, 10.0, 10.0]], [0.9], [1])],
        [_gt([[s, 0.0, s + 10.0, 10.0]], [1])],
    )
    out = cm.compute()
    assert int(out["tp"]) + int(out["fp"]) == 1
    # It should be TP at exactly-0.5 to within float precision.
    assert int(out["tp"]) == 1


def test_confusion_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        DetectionConfusionMatrix(iou_threshold=0.0)
    with pytest.raises(ValueError, match="iou_threshold"):
        DetectionConfusionMatrix(iou_threshold=1.5)
    with pytest.raises(ValueError, match="score_threshold"):
        DetectionConfusionMatrix(score_threshold=-0.1)
    with pytest.raises(ValueError, match="score_threshold"):
        DetectionConfusionMatrix(score_threshold=1.5)


def test_confusion_rejects_mismatched_lengths() -> None:
    cm = DetectionConfusionMatrix()
    with pytest.raises(ValueError, match="must match per-image"):
        cm.update(
            [_pred([[0, 0, 1, 1]], [0.9], [1])],
            [_gt([[0, 0, 1, 1]], [1]), _gt([], [])],
        )


def test_confusion_update_silent_on_none_inputs() -> None:
    """update(None, anything) and update(anything, None) are no-ops, not errors."""
    cm = DetectionConfusionMatrix()
    cm.update(None, None)  # type: ignore[arg-type]
    cm.update([_pred([], [], [])], None)  # type: ignore[arg-type]
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 0 and int(out["fn"]) == 0


def test_confusion_target_without_boxes_treated_as_empty() -> None:
    """A target dict missing 'boxes'/'labels' should be treated as empty GT, not crash."""
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    cm.update(
        [_pred([[0, 0, 10, 10]], [0.9], [1])],
        [{}],  # type: ignore[list-item]
    )
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 1 and int(out["fn"]) == 0


def test_confusion_handles_inputs_on_non_cpu_device_via_implicit_move() -> None:
    """Inputs on a non-CPU device (MPS / CUDA) must not cause a device-mismatch
    error against locally-created tensors like ``gt_matched``.

    The matcher does integer counting only — it moves inputs to CPU at entry
    so the lightweight matching loop runs without explicit device threading.
    Regression for the Lightning-driven evaluation path where the model emits
    predictions on the trainer's accelerator (MPS / CUDA) but the metric
    state is registered on CPU via ``add_state``.
    """
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # No accelerator available; skip silently — the CPU path is covered above.
        return

    pred = DetectionOutput(
        boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device),
        scores=torch.tensor([0.9], device=device),
        labels=torch.tensor([1], dtype=torch.int64, device=device),
    )
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
    }
    cm.update([pred], [target])  # must not raise
    out = cm.compute()
    assert int(out["tp"]) == 1 and int(out["fp"]) == 0 and int(out["fn"]) == 0


def test_confusion_reset_clears_state() -> None:
    cm = DetectionConfusionMatrix(iou_threshold=0.5, score_threshold=0.0)
    cm.update(
        [_pred([[0, 0, 10, 10]], [0.9], [1])],
        [_gt([[0, 0, 10, 10]], [1])],
    )
    assert int(cm.compute()["tp"]) == 1
    cm.reset()
    out = cm.compute()
    assert int(out["tp"]) == 0 and int(out["fp"]) == 0 and int(out["fn"]) == 0
