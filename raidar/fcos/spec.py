"""The ONE tiny detector architecture, written once so several frameworks can implement it.

Every other model in raidar wraps somebody's zoo — torchvision's builders, RF-DETR, a Hub
checkpoint. This is the opposite: a small network specified here and implemented natively per
framework. It exists because **the detection zoos do not line up**: torchvision, RF-DETR and
``transformers`` are all torch-only, so a cross-framework comparison built on zoos would compare
different networks and tell you nothing about the frameworks. Implementing ONE architecture
several times makes the comparison mean something.

For real work prefer a zoo model — a tiny net trained from scratch loses to a pretrained
detector every time. This is the control, not the recommendation.

**The architecture** — a single-level, anchor-free FCOS-style detector, deliberately small::

    input (H, W, C)
      backbone: for w in widths:                  # (16, 32, 64)
                    x = conv3x3(w, stride 1) -> BN -> ReLU
                    x = conv3x3(w, stride 2) -> BN -> ReLU     # each stage halves H and W
      trunk:    x = conv3x3(head_width) -> BN -> ReLU          # 64, shared by the three heads
      heads:    cls = conv1x1(num_classes)        # per-cell class logits, NO activation
                box = conv1x1(4)                  # raw values; distances are exp(raw) * stride
                ctr = conv1x1(1)                  # per-cell centerness logit, NO activation

Three stages of stride 2 give ``stride = 8``: a ``(256, 256)`` input yields a ``32 x 32`` grid,
and grid cell ``(i, j)`` owns the image point ``((j + 0.5) * stride, (i + 0.5) * stride)``.
Everything below — assignment, decode, loss — is written against that grid.

**The FCOS rules the implementations share** (the numpy functions in this module are the
reference; per-framework code re-states only what must be differentiable):

* **Assignment** (:func:`assign_targets`): a cell is POSITIVE for a ground-truth box when its
  center point lies inside the box AND within ``center_radius * stride`` of the box's center
  (FCOS center sampling — without it every cell under a large box is positive and the box head
  drowns). A cell claimed by several boxes takes the SMALLEST by area. A positive cell's box
  target is the ``(l, t, r, b)`` pixel distances from the cell's point to the box sides, and its
  centerness target is ``sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))``.
* **Loss** (per-engine, differentiable): sigmoid focal loss (``alpha=0.25``, ``gamma=2``) on the
  class map over ALL cells, ``1 - IoU`` of predicted vs target ``(l, t, r, b)`` on the positive
  cells, binary cross-entropy on the centerness logit on the positive cells — each summed and
  divided by ``max(1, num_positive)``.
* **Decode** (:func:`decode_predictions`): score = ``sqrt(sigmoid(cls) * sigmoid(ctr))`` per cell
  and class; boxes reconstructed from the cell points and ``exp(raw) * stride`` distances,
  clipped to the image; candidates below ``score_threshold`` dropped; greedy per-class NMS.
* **Scoring** (:func:`average_precision_50`): greedy match at IoU >= 0.5 per class, ranked by
  score; AP is the all-point interpolated area under precision/recall, averaged over the classes
  present in the ground truth. Torchmetrics is torch-only, so the non-torch engines report THESE
  numbers — under the same canonical metric names.

**Two choices that exist to keep the implementations comparable, not because they are best:**

* **Downsampling is strided convolution**, not max-pool — one primitive fewer to disagree on,
  and the parameter count then follows from the convolutions alone.
* **The input size is FIXED** (``input_shape``, default ``(256, 256, 3)``): dense per-cell
  targets only stack into regular batches when every image shares one grid. Resize upstream
  (the joint image+boxes resize op); the knob stays ``(H, W, C)`` channels-LAST in every
  implementation so one config line drives any of them, and only ``C`` reaches the parameter
  count.

**The constants below are the single source of truth**, imported by each implementation for its
constructor defaults. The LAYER ORDER is prose — different APIs cannot share code — and is held
together by :func:`trainable_parameter_count`, which DERIVES the count from the architecture;
each implementation is checked against that CLAIM rather than against whichever one was written
first.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "BACKBONE_WIDTHS",
    "CENTER_RADIUS",
    "HEAD_WIDTH",
    "INPUT_SHAPE",
    "KERNEL_SIZE",
    "assign_targets",
    "average_precision_50",
    "decode_predictions",
    "grid_size",
    "stride",
    "trainable_parameter_count",
]

#: Channels per backbone stage, doubling. Three stages — small enough to train in minutes on
#: CPU, deep enough that the stride-8 grid sees real receptive fields.
BACKBONE_WIDTHS: Tuple[int, ...] = (16, 32, 64)

#: Channels of the shared head trunk between the backbone and the three 1x1 heads.
HEAD_WIDTH: int = 64

#: Square conv kernel, ``padding="same"`` so only the strided convs change the spatial size.
KERNEL_SIZE: int = 3

#: Default input, matching the other model wrappers' ``(H, W, C)`` — channels LAST.
INPUT_SHAPE: Tuple[int, int, int] = (256, 256, 3)

#: FCOS center-sampling radius, in strides: a cell is positive only within this distance of the
#: box center. 1.5 is the FCOS paper's value.
CENTER_RADIUS: float = 1.5


def stride(widths: Tuple[int, ...] = BACKBONE_WIDTHS) -> int:
    """The backbone's total downsampling factor — one halving per stage.

    Stated as a function rather than a constant because it follows from ``widths``, and an
    implementation that exposes ``widths`` as a knob must be able to recompute it.
    """
    return int(2 ** len(widths))


def grid_size(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE, widths: Tuple[int, ...] = BACKBONE_WIDTHS
) -> Tuple[int, int]:
    """The ``(rows, cols)`` of the prediction grid for an input, validating divisibility.

    ``H`` and ``W`` must each divide by :func:`stride` so every implementation's strided convs
    land on the same grid — checked here so the error names the input rule rather than a shape
    mismatch three layers down.
    """
    s = stride(widths)
    height, width = int(input_shape[0]), int(input_shape[1])
    if height % s or width % s:
        raise ValueError(
            f"the FCOS spec needs H and W divisible by {s} (one halving per stage), got H={height}, W={width}"
        )
    return height // s, width // s


def trainable_parameter_count(
    num_classes: int,
    widths: Tuple[int, ...] = BACKBONE_WIDTHS,
    head_width: int = HEAD_WIDTH,
    in_channels: int = 3,
    kernel_size: int = KERNEL_SIZE,
) -> int:
    """The parameter count the spec implies — what every implementation must report.

    Per conv: ``k*k*cin*cout`` weights + ``cout`` biases. Per BatchNorm: ``2*cout`` (scale and
    shift; running mean/variance are STATE, excluded the way each framework requires). Derived
    from the architecture rather than measured from one framework, so it is a CLAIM the
    implementations are each checked against instead of one of them being the reference.

    With the defaults and 5 classes this is 110,234.
    """
    k = kernel_size * kernel_size

    def conv_bn(cin: int, cout: int) -> int:
        return k * cin * cout + cout + 2 * cout

    total = 0
    channels = in_channels
    for width in widths:
        total += conv_bn(channels, width) + conv_bn(width, width)  # stride-1 conv, then stride-2
        channels = width

    total += conv_bn(channels, head_width)
    # The three 1x1 heads: class logits, the 4 raw box values, the centerness logit.
    total += head_width * num_classes + num_classes
    total += head_width * 4 + 4
    total += head_width * 1 + 1
    return total


# --------------------------------------------------------------------------- #
# The numpy reference: target assignment
# --------------------------------------------------------------------------- #
def _check_boxes_are_in_frame(boxes: np.ndarray, input_shape: Tuple[int, int, int]) -> None:
    """Reject boxes that cannot be in this frame — the ONE silent failure this spec has.

    Assignment is the only place the boxes' coordinate system is used, and a mismatch produces
    no error of its own: every box falls outside the grid, no cell is positive, and training
    proceeds against pure background forever (measured — ``loss_box`` and ``loss_ctr`` pinned at
    ``-0.0`` while ``loss_cls`` climbs). Nothing downstream notices, because the SHAPES are all
    valid; only the numbers are meaningless.

    The cause in practice is an image-only resize: ``ConvertToImage`` and friends move pixels
    without moving boxes, so the image ends up matching the model's ``input_shape`` (passing
    ``TorchFcos``'s own size check) while the boxes stay in the source frame. The coupled
    ``ResizeDetection`` moves both, which is why it is the op the spec's configs use.

    Checked HERE rather than in a caller because both callers reach the spec through this one
    function — the op (:class:`raidar.fcos.targets.FcosTargets`) and the torch model
    (:class:`raidar.fcos.torch_fcos.TorchFcos`) — so one implementation covers every engine.

    The slack is deliberate: a box may legitimately sit a pixel or two past an edge (annotations
    are not always clipped), so only a box well outside the frame is an error.
    """
    if not boxes.shape[0]:
        return
    height, width = float(input_shape[0]), float(input_shape[1])
    slack_x, slack_y = max(2.0, 0.01 * width), max(2.0, 0.01 * height)
    max_x, max_y = float(boxes[:, [0, 2]].max()), float(boxes[:, [1, 3]].max())
    if max_x > width + slack_x or max_y > height + slack_y:
        raise ValueError(
            f"the boxes reach ({max_x:.1f}, {max_y:.1f}) but the frame is {width:.0f}x{height:.0f} "
            f"(W x H) — they are not in this frame, so every box would fall outside the grid and "
            f"the model would train on pure background with no error. An IMAGE-ONLY resize "
            f"(ConvertToImage and friends) moves pixels without moving boxes; use "
            f"recordstream's ResizeDetection, which moves both."
        )


def assign_targets(
    boxes: np.ndarray,
    labels: np.ndarray,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 0,
    widths: Tuple[int, ...] = BACKBONE_WIDTHS,
    center_radius: float = CENTER_RADIUS,
) -> Dict[str, np.ndarray]:
    """One image's ground truth as the dense per-cell maps the heads train against.

    This is what makes the non-torch engines tractable: encoded through this function, every
    record's variable-length boxes become FIXED-SHAPE arrays, so keras/tf/jax/mlx batches stack
    regularly with no ragged tensors.

    Args:
        boxes: ``[N, 4]`` ground-truth boxes, xyxy, in input-image pixels.
        labels: ``[N]`` integer class ids indexing the class map's channels directly.
        input_shape: The fixed ``(H, W, C)`` the image was resized to.
        num_classes: Width of the class map (the one required argument).
        widths: Backbone stage widths — only their COUNT matters here, via :func:`stride`.
        center_radius: FCOS center-sampling radius, in strides.

    Returns:
        ``cls``: ``[rows, cols, num_classes]`` float32 one-hot map (all-zero where negative);
        ``box``: ``[rows, cols, 4]`` float32 ``(l, t, r, b)`` pixel distances (zero where negative);
        ``centerness``: ``[rows, cols]`` float32 targets in ``[0, 1]`` (zero where negative);
        ``foreground``: ``[rows, cols]`` bool positive-cell mask.
    """
    if not num_classes:
        raise ValueError("assign_targets needs num_classes")
    rows, cols = grid_size(input_shape, widths)
    _check_boxes_are_in_frame(np.asarray(boxes, dtype=np.float32).reshape(-1, 4), input_shape)
    s = float(stride(widths))
    cls_map = np.zeros((rows, cols, num_classes), dtype=np.float32)
    box_map = np.zeros((rows, cols, 4), dtype=np.float32)
    ctr_map = np.zeros((rows, cols), dtype=np.float32)
    fg = np.zeros((rows, cols), dtype=bool)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    if boxes.shape[0] == 0:
        return {"cls": cls_map, "box": box_map, "centerness": ctr_map, "foreground": fg}

    # Every cell's owned image point, [rows, cols].
    cx = (np.arange(cols, dtype=np.float32) + 0.5) * s
    cy = (np.arange(rows, dtype=np.float32) + 0.5) * s
    px, py = np.meshgrid(cx, cy)

    # Distances from every point to every box side, [rows, cols, N].
    left = px[..., None] - boxes[None, None, :, 0]
    top = py[..., None] - boxes[None, None, :, 1]
    right = boxes[None, None, :, 2] - px[..., None]
    bottom = boxes[None, None, :, 3] - py[..., None]
    inside = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)

    # FCOS center sampling: also within `radius * stride` of the box center, per axis.
    radius = center_radius * s
    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    near = (np.abs(px[..., None] - box_cx[None, None, :]) <= radius) & (
        np.abs(py[..., None] - box_cy[None, None, :]) <= radius
    )
    candidate = inside & near

    # A cell claimed by several boxes takes the SMALLEST by area — the FCOS ambiguity rule.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_grid = np.where(candidate, areas[None, None, :], np.inf)
    chosen = np.argmin(area_grid, axis=-1)
    fg = candidate.any(axis=-1)

    rr, cc = np.nonzero(fg)
    picked = chosen[rr, cc]
    l, t = left[rr, cc, picked], top[rr, cc, picked]
    r, b = right[rr, cc, picked], bottom[rr, cc, picked]
    box_map[rr, cc, 0], box_map[rr, cc, 1] = l, t
    box_map[rr, cc, 2], box_map[rr, cc, 3] = r, b
    ctr_map[rr, cc] = np.sqrt((np.minimum(l, r) / np.maximum(l, r)) * (np.minimum(t, b) / np.maximum(t, b)))
    cls_map[rr, cc, labels[picked].astype(int)] = 1.0
    return {"cls": cls_map, "box": box_map, "centerness": ctr_map, "foreground": fg}


# --------------------------------------------------------------------------- #
# The numpy reference: dense decode + NMS
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Greedy NMS over one class's candidates; returns kept indices, best-first."""
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _pairwise_iou(boxes[best : best + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return keep


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix ``[len(a), len(b)]`` for xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def decode_predictions(
    cls_logits: np.ndarray,
    box_raw: np.ndarray,
    ctr_logit: np.ndarray,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    widths: Tuple[int, ...] = BACKBONE_WIDTHS,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> Dict[str, np.ndarray]:
    """One image's RAW head maps into scored, NMS-filtered xyxy boxes.

    Takes the raw maps rather than transformed ones so the sigmoid/exp rules live in exactly one
    place — every implementation's eval path is this function over its head outputs.

    Args:
        cls_logits: ``[rows, cols, num_classes]`` raw class logits.
        box_raw: ``[rows, cols, 4]`` raw box values; distances are ``exp(raw) * stride`` pixels.
        ctr_logit: ``[rows, cols]`` (or ``[rows, cols, 1]``) raw centerness logit.
        input_shape: The fixed ``(H, W, C)`` the network consumed — boxes are clipped to it.
        widths: Backbone stage widths, via :func:`stride`.
        score_threshold: Candidates below ``sqrt(cls_prob * ctr_prob)`` are dropped.
        iou_threshold: Greedy per-class NMS overlap threshold.
        max_detections: Keep at most this many detections, best-first.

    Returns:
        ``boxes`` ``[K, 4]`` float32 xyxy, ``scores`` ``[K]`` float32 descending within class
        groups, ``labels`` ``[K]`` int64 — the channel indices of ``cls_logits``.
    """
    rows, cols = grid_size(input_shape, widths)
    s = float(stride(widths))
    cls_prob = _sigmoid(np.asarray(cls_logits, dtype=np.float32).reshape(rows, cols, -1))
    ctr_prob = _sigmoid(np.asarray(ctr_logit, dtype=np.float32).reshape(rows, cols))
    distances = np.exp(np.asarray(box_raw, dtype=np.float32).reshape(rows, cols, 4)) * s

    cx = (np.arange(cols, dtype=np.float32) + 0.5) * s
    cy = (np.arange(rows, dtype=np.float32) + 0.5) * s
    px, py = np.meshgrid(cx, cy)
    corners = np.stack(
        [px - distances[..., 0], py - distances[..., 1], px + distances[..., 2], py + distances[..., 3]],
        axis=-1,
    )
    corners[..., 0::2] = np.clip(corners[..., 0::2], 0.0, float(input_shape[1]))
    corners[..., 1::2] = np.clip(corners[..., 1::2], 0.0, float(input_shape[0]))

    scores = np.sqrt(cls_prob * ctr_prob[..., None])  # [rows, cols, C]
    boxes_out: List[np.ndarray] = []
    scores_out: List[np.ndarray] = []
    labels_out: List[np.ndarray] = []
    for c in range(scores.shape[-1]):
        mask = scores[..., c] >= score_threshold
        if not mask.any():
            continue
        class_boxes = corners[mask]
        class_scores = scores[..., c][mask]
        keep = _nms(class_boxes, class_scores, iou_threshold)
        boxes_out.append(class_boxes[keep])
        scores_out.append(class_scores[keep])
        labels_out.append(np.full(len(keep), c, dtype=np.int64))

    if not boxes_out:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }
    boxes_all = np.concatenate(boxes_out).astype(np.float32)
    scores_all = np.concatenate(scores_out).astype(np.float32)
    labels_all = np.concatenate(labels_out)
    order = np.argsort(-scores_all)[:max_detections]
    return {"boxes": boxes_all[order], "scores": scores_all[order], "labels": labels_all[order]}


# --------------------------------------------------------------------------- #
# The numpy reference: AP50 / precision / recall
# --------------------------------------------------------------------------- #
def average_precision_50(
    predictions: Sequence[Dict[str, np.ndarray]],
    targets: Sequence[Dict[str, np.ndarray]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Detection quality over a split: mAP@50 plus dataset precision/recall.

    Torchmetrics is torch-only, so the non-torch engines score with THIS — greedy matching per
    class at ``iou_threshold``, predictions ranked by score across the whole split, all-point
    interpolated AP averaged over the classes that appear in the ground truth.

    Args:
        predictions: Per image, ``{"boxes": [K, 4] xyxy, "scores": [K], "labels": [K]}``.
        targets: Per image, ``{"boxes": [N, 4] xyxy, "labels": [N]}``.
        iou_threshold: Match threshold — 0.5 is AP50.

    Returns:
        ``{"map_50": ..., "precision": ..., "recall": ...}`` — precision/recall are over ALL
        returned predictions (decode already applied its score threshold), micro-averaged.
    """
    if len(predictions) != len(targets):
        raise ValueError(f"got {len(predictions)} predictions for {len(targets)} targets")

    gt_classes = sorted({int(label) for target in targets for label in np.asarray(target["labels"]).reshape(-1)})
    per_class_ap: List[float] = []
    # Precision denominates over ALL predictions — including a class no target carries, whose
    # rows the per-class loop below never visits (it walks ground-truth classes only).
    n_predictions = sum(len(np.asarray(p["labels"]).reshape(-1)) for p in predictions)
    total_tp = total_gt = 0
    for c in gt_classes:
        # Collect this class's predictions across the split, remembering the source image.
        rows: List[Tuple[float, int, np.ndarray]] = []
        for image_index, prediction in enumerate(predictions):
            labels = np.asarray(prediction["labels"]).reshape(-1)
            for k in np.nonzero(labels == c)[0]:
                rows.append(
                    (
                        float(np.asarray(prediction["scores"]).reshape(-1)[k]),
                        image_index,
                        np.asarray(prediction["boxes"], dtype=np.float32).reshape(-1, 4)[k],
                    )
                )
        rows.sort(key=lambda row: -row[0])

        gt_boxes: Dict[int, np.ndarray] = {}
        gt_used: Dict[int, np.ndarray] = {}
        n_gt = 0
        for image_index, target in enumerate(targets):
            labels = np.asarray(target["labels"]).reshape(-1)
            boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)[labels == c]
            gt_boxes[image_index] = boxes
            gt_used[image_index] = np.zeros(len(boxes), dtype=bool)
            n_gt += len(boxes)
        total_gt += n_gt

        tp = np.zeros(len(rows))
        fp = np.zeros(len(rows))
        for rank, (_, image_index, box) in enumerate(rows):
            candidates = gt_boxes[image_index]
            if len(candidates) == 0:
                fp[rank] = 1
                continue
            ious = _pairwise_iou(box[None, :], candidates)[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and not gt_used[image_index][best]:
                gt_used[image_index][best] = True
                tp[rank] = 1
            else:
                fp[rank] = 1
        total_tp += int(tp.sum())

        if n_gt == 0:
            continue
        if len(rows) == 0:
            per_class_ap.append(0.0)
            continue
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        # All-point interpolation: precision envelope, summed over recall increments.
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        recall_steps = np.concatenate([[0.0], recall])
        per_class_ap.append(float(np.sum((recall_steps[1:] - recall_steps[:-1]) * envelope)))

    return {
        "map_50": float(np.mean(per_class_ap)) if per_class_ap else 0.0,
        "precision": total_tp / n_predictions if n_predictions else 0.0,
        "recall": total_tp / total_gt if total_gt else 0.0,
    }
