"""``FcosTargets`` — the op that makes detection batches REGULAR for the non-torch engines.

``collate_records`` keeps a variable-N ``Regions`` as per-record COLUMNS, which is exactly what
the torch-family backends want (``batch_regions`` reads it back as per-image target dicts). The
keras/tf/jax/mlx loops need the opposite: their models take DENSE arrays, a ragged ``[N, 4]`` box
column is not one, and it MUST NOT be padded (the collate mandate). The FCOS spec's answer is to
encode each record's boxes through the spec's numpy assignment into FIXED-SHAPE dense maps
BEFORE collation, so every batch stacks with no ragged tensors and no padding.

This op does that encoding, per record, in the ops chain (after the joint image+boxes resize —
the assignment is written against the resized frame):

It writes the three dense float32 maps the heads train against — ``cls_key``
``[rows, cols, C]``, ``box_key`` ``[rows, cols, 4]``, ``ctr_key`` ``[rows, cols]`` — and
**destroys nothing**: the :class:`~recordstream.Regions` stays under ``target_key`` for the eval
pass to score against, read back by :func:`~recordstream.batch_regions`.

It briefly did replace that ``Regions`` with a plain ``{"boxes", "labels"}`` dict, on the belief
that a variable-N region column could not survive collation. That was WRONG and measured so
(2026-08-05): a ``Regions``' ``boxes``/``labels`` are declared item ATTRS, which the collate
lists per record rather than stacking, so the item batches correctly with no help — while the
dense maps beside it are plain arrays the engine's ``_to_arrays`` stacks. The degrade bought
nothing and cost the item's ``canvas``/``extras``.

It lives in raidar, not recordstream, because the dense-grid encoding IS the FCOS spec's
assignment rule — there is nothing modality-neutral about it.
"""

from typing import Optional, Tuple

import numpy as np
from confluid import configurable
from recordstream import Record, Regions, Transform

from raidar.fcos.spec import BACKBONE_WIDTHS, CENTER_RADIUS, INPUT_SHAPE, assign_targets

__all__ = ["FcosTargets"]


@configurable(category="op", group="fcos")
class FcosTargets(Transform):
    """Encode a record's ``Regions`` detection target as the FCOS spec's dense per-cell maps.

    Args:
        num_classes: Width of the class map; required at use (validated lazily, ``0`` = unset).
        input_shape: The ``(H, W, C)`` the images were resized to — the assignment's frame.
        widths: The spec's backbone stage widths (only their count matters, via the stride).
        center_radius: FCOS center-sampling radius, in strides.
        target_key: Record key carrying the ``Regions`` target to encode (default ``"target"``).
        cls_key: Output key for the ``[rows, cols, num_classes]`` one-hot class map.
        box_key: Output key for the ``[rows, cols, 4]`` ``(l, t, r, b)`` distance map.
        ctr_key: Output key for the ``[rows, cols]`` centerness map.
    """

    # It READS the Regions and writes three plain array keys — it consumes no item and
    # produces no item, which is why both tuples are empty.
    handles = (Regions,)
    consumes: Tuple[type, ...] = ()
    produces: Tuple[type, ...] = ()

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = BACKBONE_WIDTHS,
        center_radius: float = CENTER_RADIUS,
        target_key: str = "target",
        cls_key: str = "fcos_cls",
        box_key: str = "fcos_box",
        ctr_key: str = "fcos_ctr",
    ) -> None:
        super().__init__()
        # Lazy / zero-arg: store config only; num_classes is validated in __call__.
        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.center_radius = float(center_radius)
        self.target_key = str(target_key)
        self.cls_key = str(cls_key)
        self.box_key = str(box_key)
        self.ctr_key = str(ctr_key)

    def _check_frame(self, target: Regions) -> None:
        """The boxes' own frame must be the frame this op assigns against.

        THREE config knobs have to agree — ``ResizeDetection``'s ``width``/``height``, this op's
        ``input_shape``, and the model's — and the op can only see the first two. Since
        2026-08-05 it can see the first one at all: ``ResizeDetection`` records the resized frame
        as the ``Regions``' ``canvas``, and that item now survives to this point (it used to be
        replaced by a plain dict, which threw the evidence away).

        This is an EARLIER, better-aimed error rather than a correctness fix: a real mismatch
        also dies downstream, where Keras reports *"expected shape=(None, 128, 128, 3), found
        shape=(2, 256, 256, 3)"* — true, but about tensors, several frames from the config lines
        that caused it. Here it names the knobs. It also catches the ordering trap specifically:
        ``ResizeDetection`` takes ``width``/``height`` while ``input_shape`` is ``(H, W, C)``.

        A ``canvas`` of ``None`` skips the check. That used to be every chain without a resize;
        since recordstream records the frame at target CREATION (2026-08-06) it is only a record
        with no image to read the frame from, so this fires far earlier than it once did.
        """
        canvas = target.canvas
        if canvas is None:
            return
        frame = (int(canvas[0]), int(canvas[1]))
        expected = (int(self.input_shape[0]), int(self.input_shape[1]))
        if frame != expected:
            raise ValueError(
                f"FcosTargets: the target's boxes are in a {frame} (H, W) frame but this op "
                f"assigns against {expected}. These must agree: ResizeDetection's "
                f"width/height, this op's input_shape, and the model's input_shape — note the "
                f"ordering, ResizeDetection takes (width, height) while input_shape is (H, W, C)."
            )

    def __call__(self, record: Record) -> Optional[Record]:
        if not self.num_classes:
            raise ValueError("FcosTargets.num_classes is not set — it must match the model's.")
        target = record.get(self.target_key)
        if not isinstance(target, Regions):
            raise ValueError(
                f"FcosTargets: expected a Regions item under {self.target_key!r} "
                f"(got {type(target).__name__}) — run it after the detection-target op."
            )
        self._check_frame(target)
        # np.asarray reads a CPU torch tensor as readily as a list — the Regions may carry either.
        boxes = np.asarray(target.boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(target.labels if target.labels is not None else [], dtype=np.int64).reshape(-1)
        assigned = assign_targets(
            boxes,
            labels,
            input_shape=self.input_shape,  # type: ignore[arg-type]
            num_classes=self.num_classes,
            widths=self.widths,
            center_radius=self.center_radius,
        )
        merged = dict(record)
        merged[self.cls_key] = assigned["cls"]
        merged[self.box_key] = assigned["box"]
        # float32 like its siblings: these stack into a batch the loss consumes directly.
        merged[self.ctr_key] = assigned["centerness"]
        # `target_key` is left ALONE — the Regions is the eval pass's ground truth, and it
        # batches fine on its own (see the module docstring).
        return merged
