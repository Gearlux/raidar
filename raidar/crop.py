"""``RandomCropDetection`` — a fixed-size piece of a picture, cut together with its boxes and mask outlines.

A detector whose training loop resizes every picture itself (RF-DETR) can only be handed pictures of one shape once a
seg model's masks are batched: its batch pads the pictures to one size but joins the masks as they come, so two
pictures of different shapes in a batch stop the run (``Sizes of tensors must match except in dimension 0``). A
picture of a whole capture — 504 columns of band, as many rows as the capture has time — is any height from 100 to
7 000 rows. This step cuts every picture to ONE size BEFORE the loop sees it, in the picture's own pixels: a piece
``height`` x ``width`` at a random place (training) or the middle one (validation, the same every epoch); a picture
shorter or narrower than the piece is padded with its median, never stretched (the tiled pictures' rule). Its boxes
move with the piece and are clipped to it, a box left smaller than ``min_size_px`` either way dropped with its label;
its mask outlines (``Boxes.extras[POLYGONS]``, :mod:`raidar.yolo_sink`'s contract) are clipped exactly to the piece.

It lives here, not in recordstream: it is an augmentation, and the outline contract is raidar's.
"""

import dataclasses
import random
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np
from confluid import configurable
from recordstream import Boxes, Image, Record, Transform, item_data
from recordstream._compat import is_torch_tensor
from recordstream.items import NDArrayItem, with_data

from raidar.yolo_sink import POLYGONS

#: Where the piece is cut: anywhere (training), or the middle (validation — the same piece every epoch).
Placement = Literal["random", "centre"]

Ring = List[List[float]]

__all__ = ["Placement", "RandomCropDetection"]


def _clip_ring(ring: Sequence[Sequence[float]], width: float, height: float) -> Ring:
    """``ring`` clipped to the rectangle ``[0, width] x [0, height]`` (Sutherland–Hodgman: exact for a convex window,
    so a concave outline keeps its shape inside the piece)."""
    points: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in ring]
    edges = (
        (lambda p: p[0] >= 0.0, lambda a, b: _cross_x(a, b, 0.0)),
        (lambda p: p[0] <= width, lambda a, b: _cross_x(a, b, width)),
        (lambda p: p[1] >= 0.0, lambda a, b: _cross_y(a, b, 0.0)),
        (lambda p: p[1] <= height, lambda a, b: _cross_y(a, b, height)),
    )
    for inside, cross in edges:
        if not points:
            break
        kept: List[Tuple[float, float]] = []
        previous = points[-1]
        for current in points:
            if inside(current):
                if not inside(previous):
                    kept.append(cross(previous, current))
                kept.append(current)
            elif inside(previous):
                kept.append(cross(previous, current))
            previous = current
        points = kept
    return [[x, y] for x, y in points]


def _cross_x(a: Tuple[float, float], b: Tuple[float, float], x: float) -> Tuple[float, float]:
    t = (x - a[0]) / (b[0] - a[0])
    return x, a[1] + t * (b[1] - a[1])


def _cross_y(a: Tuple[float, float], b: Tuple[float, float], y: float) -> Tuple[float, float]:
    t = (y - a[1]) / (b[1] - a[1])
    return a[0] + t * (b[0] - a[0]), y


def _area(ring: Ring) -> float:
    xs, ys = np.asarray(ring, dtype=float).T
    return float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2)


@configurable(category="op", group="detection")
class RandomCropDetection(Transform):
    """Cut a picture to a ``height`` x ``width`` piece with its boxes and mask outlines (module docstring).

    ``random`` draws the piece's place from Python's ``random`` module, which a data loader seeds per worker and per
    epoch — each epoch cuts new pieces; ``random.seed`` makes a run repeat.

    Args:
        height: The piece's height in pixels (time rows of a waterfall); required at use (``0`` = unset).
        width: The piece's width in pixels (frequency columns); required at use (``0`` = unset).
        placement: ``random`` — anywhere the piece fits, ends included; ``centre`` — the middle piece, every call.
        min_size_px: A box the cut leaves narrower or lower than this many pixels is dropped with its label.
        input_key: Record key carrying the picture (default ``"image"``).
        target_key: Record key carrying the ``Boxes``; a record without it has its picture cut alone.
    """

    consumes = (Boxes,)
    produces = (Boxes,)

    def __init__(
        self,
        height: int = 0,
        width: int = 0,
        placement: Placement = "random",
        min_size_px: float = 3.0,
        input_key: str = "image",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        # Store only (zero-arg construction); the size is checked at use — the placement by confluid, when built.
        self.height = int(height)
        self.width = int(width)
        self.placement: Placement = placement
        self.min_size_px = float(min_size_px)
        self.input_key = str(input_key)
        self.target_key = str(target_key)

    def _start(self, size: int, piece: int) -> int:
        """Where the piece starts along an axis of ``size`` pixels (0 when the picture is no larger)."""
        room = size - piece
        if room <= 0:
            return 0
        return random.randint(0, room) if self.placement == "random" else room // 2

    def _cut(self, pixels: np.ndarray, top: int, left: int) -> np.ndarray:
        """The piece of ``pixels`` at ``(top, left)``, padded with the picture's median where it falls short."""
        piece = pixels[top : top + self.height, left : left + self.width]
        if piece.shape[0] == self.height and piece.shape[1] == self.width:
            return np.ascontiguousarray(piece)
        fill = np.median(pixels.reshape(pixels.shape[0], pixels.shape[1], -1), axis=(0, 1)).astype(pixels.dtype)
        out = np.empty((self.height, self.width) + pixels.shape[2:], dtype=pixels.dtype)
        out[...] = fill.reshape(pixels.shape[2:]) if pixels.ndim == 3 else fill[0]
        out[: piece.shape[0], : piece.shape[1]] = piece
        return out

    def _cut_boxes(self, target: Boxes, top: int, left: int) -> Boxes:
        boxes = np.asarray(target.boxes.detach().cpu() if is_torch_tensor(target.boxes) else target.boxes, dtype=float)
        boxes = boxes.reshape(-1, 4) - np.array([left, top, left, top], dtype=float)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0.0, self.width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0.0, self.height)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= self.min_size_px) & ((boxes[:, 3] - boxes[:, 1]) >= self.min_size_px)
        count = len(boxes)
        extras = dict(target.extras or {})
        for key, value in list(extras.items()):
            if key == POLYGONS and value is not None:
                extras[key] = [self._cut_rings(rings, top, left) for rings, k in zip(value, keep) if k]
            elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == count:
                extras[key] = [v for v, k in zip(value, keep) if k]  # a per-box column follows its boxes
        return dataclasses.replace(
            target,
            boxes=self._like(target.boxes, boxes[keep]),
            labels=self._like(target.labels, np.asarray(self._numpy(target.labels)).reshape(-1)[keep]),
            canvas=(self.height, self.width),
            extras=extras,
        )

    def _cut_rings(self, rings: Optional[Sequence[Ring]], top: int, left: int) -> List[Ring]:
        """An instance's outlines in the piece's pixels; one cut away entirely is dropped, so an instance whose
        outlines all lie outside keeps none — its box is then its mask."""
        kept: List[Ring] = []
        for ring in rings or ():
            moved = [[float(x) - left, float(y) - top] for x, y in ring]
            clipped = _clip_ring(moved, float(self.width), float(self.height))
            if len(clipped) >= 3 and _area(clipped) > 0.0:  # a ring clipped to a sliver of an edge is no outline
                kept.append(clipped)
        return kept

    @staticmethod
    def _numpy(value: Any) -> Any:
        return value.detach().cpu().numpy() if is_torch_tensor(value) else value

    @staticmethod
    def _like(original: Any, values: np.ndarray) -> Any:
        """``values`` as the framework ``original`` was (torch stays torch, numpy numpy), its dtype kept."""
        if is_torch_tensor(original):
            import torch

            return torch.as_tensor(values, dtype=original.dtype)
        return np.asarray(values, dtype=np.asarray(original).dtype)

    def __call__(self, record: Record) -> Record:
        if self.height < 1 or self.width < 1:
            raise ValueError(f"RandomCropDetection: height and width must be set (got {self.height} x {self.width})")
        if self.input_key not in record:
            raise ValueError(f"RandomCropDetection: no {self.input_key!r} in the record (keys: {list(record)})")
        item = record[self.input_key]
        pixels = np.asarray(item_data(item))
        if isinstance(item, Image) and item.layout == "CHW":
            raise ValueError("RandomCropDetection: a CHW picture — cut it before it is tensorized (HWC or 2-D)")
        top, left = self._start(pixels.shape[0], self.height), self._start(pixels.shape[1], self.width)
        out = dict(record)
        piece = self._cut(pixels, top, left)
        out[self.input_key] = with_data(item, piece) if isinstance(item, NDArrayItem) else piece
        target = record.get(self.target_key)
        if isinstance(target, Boxes):
            out[self.target_key] = self._cut_boxes(target, top, left)
        return out
