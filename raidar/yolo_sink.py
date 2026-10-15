"""YoloSink — write a detection Stream to a native Ultralytics dataset dir.

A RecordStream ``DataSink`` (``open`` / ``write(record)`` / ``flush`` / ``close``):
``recordstream run`` streams a detection Stream into it via
``DatasetProcessor(stream=…, sink=YoloSink(…))``, exactly like any other
source → ops → sink pipeline — there is no bespoke processor. Each
record dict is materialised to the layout
``YOLO.train(data=<output_dir>/data.yaml)`` natively consumes:

::

    <output_dir>/                      # layout: ultralytics (the default)
      data.yaml                        # YOLO descriptor (merged across splits)
      metadata.jsonl                   # per-image provenance sidecar
      annotations/<split>.json         # COCO sidecar (optional)
      images/<split>/<idx>.{png,jpg}
      labels/<split>/<idx>.txt         # one line per box: class cx cy w h
                                       #   (or per instance, class x1 y1 x2 y2 ...: see "Instance masks")

    <output_dir>/                      # layout: rfdetr — what RF-DETR's own reader asks for
      data.yaml
      metadata.jsonl
      <split>/images/<idx>.{png,jpg}
      <split>/labels/<idx>.txt
      <split>/_annotations.coco.json   # COCO sidecar (optional), where its COCO reader looks

Measured with RF-DETR 1.9.3's ``is_valid_yolo_dataset`` on three real windows: the default
layout answers ``False`` (it wants ``train/`` and ``val/`` folders holding ``images`` and
``labels``), the ``rfdetr`` layout ``True``. Ultralytics reads the default one.

**One sink writes ONE split.** To build a train+val(+test) dataset, run
``recordstream run`` once per split into the SAME ``output_dir`` (vary ``split``
and the source); each run rewrites only its own ``images/<split>`` /
``labels/<split>`` / ``annotations/<split>.json`` and **merges** ``data.yaml``
(and appends to ``metadata.jsonl``), so the splits accumulate without clobbering
each other.

Compared to the in-memory ``raidar.ultralytics.stream_yolo_dataset.StreamYOLODataset``
(which recomputes spectrograms every epoch from the IQ source), the on-disk
dataset trades startup cost for faster epochs and lets third-party tools
(FiftyOne, pycocotools, ultralytics' own CLI) consume the same data.

Generic over any detection Stream whose input-key value is an image
(``[3,H,W]`` / ``[H,W,3]`` / ``[H,W]``, float ``[0,1]`` or ``uint8``, numpy or
torch) and whose target-key value is ``{"boxes": [N,4] xyxy-pixel float32, "labels": [N] int64}``
— i.e. exactly what :class:`waivefront.targets.RegionsToDetectionBoxes`
produces. (No ``ultralytics`` import — it only writes the on-disk layout.)

**Instance masks — the** ``polygons`` **contract.** A producer MAY put each instance's polygons in
``target.extras["polygons"]`` (the key is :data:`POLYGONS`): a list of length N whose element ``i``
is a LIST of polygons for box ``i``, each polygon a list of ``[x, y]`` pixel points (at least 3, a
closed ring, the first point not repeated). The polygons of one instance do not overlap (they may
touch). An element that is ``[]`` or ``None`` means the instance has no mask of its own: its mask is
its box rectangle. Without the key (or with a key that holds no polygon at all) the sink writes
exactly what it always wrote. With polygons:

* ``labels/<split>/<idx>.txt`` holds the YOLO-seg form, ``class x1 y1 x2 y2 ...`` (coordinates
  normalised by the picture's width and height, six decimals like a box line), ONE ring per instance
  — YOLO-seg has room for one polygon a line. Several polygons of an instance are joined into one ring
  by zero-width bridges (:func:`_bridged_ring`): the ring walks the first polygon round from the
  vertex nearest the next polygon, steps across to that polygon's nearest vertex, walks it round, and
  so on, then retraces the same bridge points back to the start. Every bridge is walked twice, once
  each way, so it adds no area under even-odd or non-zero fill. An instance without polygons is its
  box's four corners as a ring.
* The COCO sidecar's ``segmentation`` is the instance's polygons, one flat ``[x1, y1, x2, y2, ...]``
  list each, in pixels, and ``area`` their summed shoelace area; an instance without polygons has its
  box rectangle as one polygon (and its box's area).
* A split holding ANY ring holds no box line. Measured with Ultralytics 8.4.126: it refuses a label
  file that mixes the two forms (``labels mix segment and detection rows``), and a segment model
  refuses a dataset whose segments are fewer than its boxes (``Segment dataset requires equal
  numbers of boxes and segments``). The sink therefore makes every line of such a split a ring (and
  every COCO annotation carry a polygon) when it flushes, however the records arrived — in this run,
  or an earlier one an ``append`` continues. Nothing is recorded in ``data.yaml``: RF-DETR 1.9.3's
  YOLO reader tells a box line (5 fields) from a polygon line (7 or more) line by line and builds
  masks when the model asks for them, and Ultralytics decides per label file.

**Grey pictures.** ``image_mode: grey`` writes a picture whose three channels are equal (a spectrogram's
grey scale) as ONE channel — a PNG mode ``L``, a third of the pixels to compress (measured on 60
spectrogram pictures of 1024 x 1024: 27.6 ms and 0.80 MB a picture as RGB, 12.5 ms and 0.52 MB grey). Every
reader of the folder turns it back into the same three channels: RF-DETR's YOLO reader and this
package's source call ``convert("RGB")``, Ultralytics reads with OpenCV's colour flag — so ``data.yaml``
keeps ``channels: 3``, what the model sees. A picture whose channels differ is refused, never reduced.

**Writer threads.** The pictures are encoded and written on ``image_threads`` threads — PIL's encoder runs
outside Python's lock (60 PNGs: 1.74 s on one thread, 0.23 s on eight) — each under the index its record
was given when it arrived, so the files are the ones a single thread writes. At most twice as many
pictures as threads wait at once; a write that fails stops the run at the next record or at ``flush``.
"""

from __future__ import annotations

import json
import shutil
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Deque, Dict, List, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
import yaml
from annotated_types import Interval
from confluid import configurable
from loggair import get_logger
from PIL import Image
from recordstream import Boxes, Label, Record, item_data
from recordstream.storage.base import Storage

logger = get_logger(__name__)

#: Where a split's images and labels go: Ultralytics' ``images/<split>`` + ``labels/<split>``, or
#: RF-DETR's ``<split>/images`` + ``<split>/labels``.
YoloLayout = Literal["ultralytics", "rfdetr"]
#: What a run does to a split that already exists: ``replace`` wipes it first (a rerun is idempotent),
#: ``append`` continues it after its last record (a second source joins the folder).
YoloMode = Literal["replace", "append"]
#: How a picture's pixels are stored: ``rgb`` three channels, ``grey`` one (a picture whose channels are equal).
ImageMode = Literal["rgb", "grey"]
#: How many threads encode and write the pictures.
ImageThreads = Annotated[int, Interval(ge=1)]
#: The key of ``Boxes.extras`` (or of a dict target) under which a producer hands the sink each
#: instance's polygons — the contract is in the module docstring. A producer imports this name rather
#: than spelling the string.
POLYGONS = "polygons"
#: Every instance's polygons as ``(k, 2)`` float arrays in pixels (an empty list: no mask of its own).
PolygonSets = List[List[np.ndarray]]


def _to_uint8_hwc(arr_in: Any) -> np.ndarray:
    """Normalise an image-shaped record input into ``[H, W, 3] uint8`` for PIL.

    Accepts ``[3,H,W]`` / ``[H,W,3]`` / ``[H,W]`` numpy or torch tensors, float
    in ``[0,1]`` or ``uint8``. Greyscale is broadcast to 3 channels (Ultralytics
    expects RGB).
    """
    if hasattr(arr_in, "detach"):  # torch tensor
        arr_in = arr_in.detach().cpu().numpy()
    arr = np.asarray(arr_in)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)  # H, W → H, W, 3
    elif arr.ndim == 3:
        # Detect CHW vs HWC: a 3-channel CHW tensor has shape[0] in (1, 3);
        # an HWC tensor has shape[-1] in (1, 3). When both are ambiguous (e.g.
        # a 3x3 toy fixture) prefer HWC since that's PIL-native.
        if arr.shape[-1] in (1, 3):
            pass  # HWC
        elif arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))  # CHW → HWC
        else:
            raise ValueError(f"_to_uint8_hwc: 3-D input has shape {arr.shape}; expected a channel dim of size 1 or 3.")
        if arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
    else:
        raise ValueError(f"_to_uint8_hwc: input must be 2-D or 3-D; got shape {arr.shape}.")

    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _record_image(record: Record, input_key: str) -> Any:
    """The image payload (tensor / ndarray / PIL) under ``input_key`` in a record."""
    return item_data(record[input_key])


def _record_target_dict(record: Record, target_key: str) -> Dict[str, Any]:
    """The ``{boxes, labels}`` detection target under ``target_key`` — a :class:`~recordstream.Boxes`
    item (the ``CocoToTorchVisionDetection`` output) unwrapped, with its ``extras["polygons"]`` carried
    along when it has them, a dict-valued ``Label`` / raw dict passed through, or ``{}`` when the key
    is absent."""
    if target_key not in record:
        return {}
    item = record[target_key]
    if isinstance(item, Boxes):
        target: Dict[str, Any] = {"boxes": item.boxes, "labels": item.labels}
        if POLYGONS in item.extras:
            target[POLYGONS] = item.extras[POLYGONS]
        return target
    data = item.value if isinstance(item, Label) else item_data(item)
    return data if isinstance(data, dict) else {}


def _record_metadata(record: Record, input_key: str, target_key: str) -> Dict[str, Any]:
    """Gather a record's remaining keys into a plain metadata dict (a ``Label``'s ``.value``,
    else the item payload) — the per-record provenance the YOLO/COCO sidecar records."""
    return {
        key: (item.value if isinstance(item, Label) else item_data(item))
        for key, item in record.items()
        if key not in (input_key, target_key)
    }


def _boxes_and_labels(target: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A target's ``(boxes [N,4] float64, labels [N] int64)`` as numpy; ``None`` without either."""
    boxes = target.get("boxes") if isinstance(target, dict) else None
    labels = target.get("labels") if isinstance(target, dict) else None
    if boxes is None or labels is None:
        return None
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray(boxes, dtype=np.float64), np.asarray(labels).astype(np.int64).reshape(-1)


def _target_polygons(target: Dict[str, Any], where: str) -> Optional[PolygonSets]:
    """The instances' polygons from ``target["polygons"]`` as ``(k, 2)`` arrays, validated.

    ``None`` when the target carries no polygon at all — no key, or a key whose every instance is
    ``[]`` / ``None`` — which is the box-only record the sink has always written. A malformed list
    (one entry per box is the contract; a polygon is at least three ``[x, y]`` points) is refused,
    naming ``where`` (the record) and the instance and polygon at fault.
    """
    raw = target.get(POLYGONS) if isinstance(target, dict) else None
    arrays = _boxes_and_labels(target) if raw is not None else None
    if raw is None or arrays is None:
        return None
    count = arrays[0].shape[0]
    if len(raw) != count:
        raise ValueError(
            f"YoloSink: {where}: {POLYGONS!r} lists {len(raw)} entries for {count} boxes — "
            "one list of polygons per box, [] for a box with no mask of its own"
        )
    sets: PolygonSets = []
    for i, instance in enumerate(raw):
        rings: List[np.ndarray] = []
        for j, polygon in enumerate(instance if instance is not None else []):
            points = np.asarray(polygon, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
                raise ValueError(
                    f"YoloSink: {where}: instance {i} polygon {j} is shaped {points.shape} — "
                    "a polygon is a list of at least 3 [x, y] points"
                )
            rings.append(points)
        sets.append(rings)
    return sets if any(sets) else None


def _box_ring(box: np.ndarray) -> np.ndarray:
    """An ``[x1, y1, x2, y2]`` box as its four-corner ring (top-left, top-right, bottom-right, bottom-left)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _polygon_area(polygon: np.ndarray) -> float:
    """A polygon's area by the shoelace formula."""
    x, y = polygon[:, 0], polygon[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _nearest_vertices(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """The vertex of ``a`` and the vertex of ``b`` that lie nearest each other, and that distance."""
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return int(i), int(j), float(distance[i, j])


def _bridged_ring(polygons: Sequence[np.ndarray]) -> np.ndarray:
    """One closed ring that fills the same pixels as ``polygons`` (non-overlapping, they may touch).

    YOLO-seg holds one polygon per label line, so an instance of several is joined by zero-width
    bridges. The first polygon is walked round from the vertex nearest its closest neighbour and
    back to that vertex, a bridge crosses to the neighbour's nearest vertex, that polygon is walked
    round and back, and so on — each next polygon is the one with a vertex nearest the vertex the
    walk stands on (greedy, so every bridge is short; on a tie the polygon listed first, entered at its
    first such vertex). The ring then retraces the bridge vertices
    in reverse to its first point, which the closing edge of the ring reaches: every bridge is walked
    once each way and so encloses no area, under even-odd and non-zero fill alike. The first point is
    not repeated at the end. A lone polygon is its own ring.

    Each step measures the distance from the vertex the walk stands on to EVERY vertex at once and
    takes each polygon's nearest by ``np.minimum.reduceat`` — the same distances, element for element, as
    measuring polygon by polygon (an instance of hundreds of bursts took 7.6 ms a picture that way, 0.7 ms
    this way, every ring identical;
    ``tests/test_yolo_sink.py::test_the_ring_is_the_walk_its_description_gives_ties_included``).

    Written for this sink from the shape of the problem; it copies no library's code.
    """
    if len(polygons) == 1:
        return np.asarray(polygons[0], dtype=np.float64)
    shapes = [np.asarray(polygon, dtype=np.float64) for polygon in polygons]
    counts = np.array([len(shape) for shape in shapes])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    vertices = np.concatenate(shapes)
    remaining = list(range(1, len(shapes)))
    # the first bridge: polygon 0's vertices against every vertex, each polygon's nearest pair
    from_first = np.linalg.norm(shapes[0][:, None, :] - vertices[None, :, :], axis=2).min(axis=0)
    first = remaining[int(np.argmin(np.minimum.reduceat(from_first, starts)[remaining]))]
    start, entry, _ = _nearest_vertices(shapes[0], shapes[first])
    order, entries = [0, first], [start, entry]
    remaining.remove(first)
    while remaining:
        here = shapes[order[-1]][entries[-1]]
        distance = np.linalg.norm(here[None, None, :] - vertices[None, :, :], axis=2)[0]
        following = remaining[int(np.argmin(np.minimum.reduceat(distance, starts)[remaining]))]
        order.append(following)
        entries.append(int(np.argmin(distance[starts[following] : starts[following] + counts[following]])))
        remaining.remove(following)
    ring: List[np.ndarray] = []
    for k, vertex in zip(order, entries):
        walk = np.roll(shapes[k], -vertex, axis=0)  # the polygon from its bridge vertex ...
        ring.extend(walk)
        ring.append(walk[0])  # ... round, and back to that vertex
    for k, vertex in reversed(list(zip(order, entries))[1:-1]):
        ring.append(shapes[k][vertex])  # the bridge vertices again, last polygon to first
    return np.asarray(ring, dtype=np.float64)


def _grey(image: np.ndarray, record: int) -> np.ndarray:
    """The one channel of a ``[H, W, 3]`` picture whose channels are equal; a picture whose channels differ is
    refused, naming its record."""
    if not (np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 0], image[..., 2])):
        raise ValueError(
            f"YoloSink: image_mode 'grey' writes one channel, and the picture of record {record} has three that "
            "differ — write this folder with image_mode: rgb"
        )
    return np.ascontiguousarray(image[..., 0])


def _save_picture(pixels: np.ndarray, path: Path, save_kwargs: Dict[str, Any]) -> None:
    """Encode and write one picture (run on a writer thread)."""
    Image.fromarray(pixels).save(path, **save_kwargs)


def _holds_a_ring(path: Path) -> bool:
    """Whether a label file has a YOLO-seg line (more numbers than a box line's four)."""
    return any(len(line.split()) > 5 for line in path.read_text().splitlines())


def _box_line_as_ring(line: str) -> str:
    """A box line ``class cx cy w h`` as the ``class x1 y1 x2 y2 ...`` line of its four corners."""
    label, cx, cy, w, h = line.split()
    x1, y1 = float(cx) - float(w) / 2, float(cy) - float(h) / 2
    x2, y2 = float(cx) + float(w) / 2, float(cy) + float(h) / 2
    return f"{label} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"


def _ring_line(label: int, ring: np.ndarray, height: int, width: int) -> str:
    """One YOLO-seg label line: ``class x1 y1 x2 y2 ...``, normalised by the picture, six decimals."""
    return f"{label} " + " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in ring.tolist())


def _yolo_lines(target: Dict[str, Any], height: int, width: int, polygons: Optional[PolygonSets] = None) -> List[str]:
    """Convert ``{boxes: [N,4] xyxy-pixel, labels: [N] int}`` to YOLO label lines.

    One line per box: ``class_id cx_norm cy_norm w_norm h_norm`` (coords
    normalised to ``[0,1]``). An empty box list yields ``[]`` (the file is still
    written — the YOLO convention for a negative-example image).

    Given the instances' ``polygons`` (from :func:`_target_polygons`), one YOLO-seg line per
    instance instead — ``class_id x1 y1 x2 y2 ...`` over its bridged ring, or over its box's four
    corners when it has no polygon.
    """
    arrays = _boxes_and_labels(target)
    if arrays is None:
        return []
    boxes, labels = arrays
    if boxes.shape[0] == 0:
        return []
    if polygons is not None:
        rings = [_bridged_ring(own) if own else _box_ring(boxes[i]) for i, own in enumerate(polygons)]
        return [_ring_line(int(labels[i]), ring, height, width) for i, ring in enumerate(rings)]
    cx = (boxes[:, 0] + boxes[:, 2]) / (2.0 * width)
    cy = (boxes[:, 1] + boxes[:, 3]) / (2.0 * height)
    bw = (boxes[:, 2] - boxes[:, 0]) / width
    bh = (boxes[:, 3] - boxes[:, 1]) / height
    return [f"{int(labels[i])} {cx[i]:.6f} {cy[i]:.6f} {bw[i]:.6f} {bh[i]:.6f}" for i in range(boxes.shape[0])]


def _coco_annotation_records(
    target: Dict[str, Any],
    image_id: int,
    next_ann_id: int,
    polygons: Optional[PolygonSets] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build COCO annotation entries for one image's target dict.

    COCO ``bbox`` is ``[x, y, w, h]`` top-left pixel coords; ``area`` is ``w*h``. Given the
    instances' ``polygons``, ``segmentation`` is the instance's polygons (one flat ``[x1, y1, ...]``
    list each, in pixels) and ``area`` their summed shoelace area; an instance without polygons has
    its box rectangle as the one polygon. Without ``polygons`` ``segmentation`` is ``[]``.
    """
    arrays = _boxes_and_labels(target)
    if arrays is None:
        return [], next_ann_id
    boxes, labels = arrays
    out: List[Dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        segmentation: List[List[float]] = []
        area = float(w * h)
        if polygons is not None:
            own = polygons[i]
            segmentation = [polygon.reshape(-1).tolist() for polygon in (own or [_box_ring(boxes[i])])]
            if own:
                area = float(sum(_polygon_area(polygon) for polygon in own))
        out.append(
            {
                "id": next_ann_id + i,
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        )
    return out, next_ann_id + boxes.shape[0]


_MAX_META_ARRAY_ELEMS = 256


def _jsonable_metadata(metadata: Any, *, max_array_elems: int = _MAX_META_ARRAY_ELEMS) -> Any:
    """Best-effort JSON-friendly conversion of a record's metadata payload — NEVER raises.

    Source metadata can carry values ``json.dumps`` cannot encode: NumPy/torch arrays and **complex**
    IQ samples — a step upstream may keep a whole complex signal in the metadata. So this:
      * converts a ``complex`` scalar to a string (JSON has no complex type);
      * summarises a LARGE array (> ``max_array_elems``) as ``<array shape=… dtype=…>`` — a 1.2M-element
        complex IQ array must not be dumped verbatim into every per-image label JSON (bloat AND the
        un-encodable complex values that crashed the sink);
      * lists a SMALL array element-wise and RECURSES, so nested complex / NumPy scalars from
        ``.tolist()`` are sanitised too (the bug: the old code returned ``.tolist()`` un-recursed, so
        a complex array became a list of Python ``complex`` that ``json.dumps`` still rejected).
    """
    if metadata is None or isinstance(metadata, (str, int, float, bool)):
        return metadata
    if isinstance(metadata, complex):
        return str(metadata)
    if isinstance(metadata, dict):
        return {str(k): _jsonable_metadata(v, max_array_elems=max_array_elems) for k, v in metadata.items()}
    if isinstance(metadata, (list, tuple)):
        return [_jsonable_metadata(v, max_array_elems=max_array_elems) for v in metadata]
    # NumPy/torch 0-d scalar → its Python value (recurse to sanitise a complex scalar).
    if hasattr(metadata, "item") and getattr(metadata, "shape", None) == ():
        try:
            return _jsonable_metadata(metadata.item(), max_array_elems=max_array_elems)
        except Exception:
            return str(metadata)
    shape = getattr(metadata, "shape", None)
    if shape is not None and hasattr(metadata, "tolist"):
        try:
            n = int(np.prod(shape)) if len(shape) else 1
        except Exception:
            n = max_array_elems + 1  # unknown size → treat as large
        if n > max_array_elems:
            return f"<array shape={tuple(shape)} dtype={getattr(metadata, 'dtype', '?')}>"
        try:
            return _jsonable_metadata(metadata.tolist(), max_array_elems=max_array_elems)
        except Exception:
            return str(metadata)
    if hasattr(metadata, "tolist"):
        try:
            return _jsonable_metadata(metadata.tolist(), max_array_elems=max_array_elems)
        except Exception:
            return str(metadata)
    return str(metadata)


# category="sink": listed among the sinks of a graph editor's catalogue, and wired into DatasetProcessor.sink.
@configurable(category="sink")
class YoloSink(Storage):
    """DataSink that materialises one split of a detection Stream to a YOLO dataset dir.

    Wired as the ``sink:`` of a ``recordstream.processing.DatasetProcessor`` and run
    via ``recordstream run``. Writes ``images/<split>`` + ``labels/<split>``
    (+ optional ``annotations/<split>.json``), appends per-image provenance to
    ``metadata.jsonl``, and merges ``data.yaml`` so repeated runs (one per
    split, same ``output_dir``) accumulate. Does no work in ``__init__``;
    validates + touches the filesystem lazily in ``open()``.

    A record whose target carries ``extras["polygons"]`` (:data:`POLYGONS`, the contract is in the
    module docstring) is written as instance masks: YOLO-seg label lines, one ring per instance,
    and a COCO ``segmentation`` per annotation. A split holding any ring holds no box line — the
    box-only records beside it are written as their four-corner rings when the split is flushed.
    Records without polygons, in a split without any, are written exactly as before.

    Args:
        output_dir: Destination dataset root (created if missing). Repeated runs into the same dir merge.
        class_names: Mapping ``class_id → display name`` (e.g. ``{0: "background", 1: "drone"}``); ``nc = max(id)+1``.
        split: Which split this run writes — ``"train"`` / ``"val"`` / ``"test"``.
        layout: ``"ultralytics"`` (default) — ``images/<split>`` + ``labels/<split>``, what
            ``YOLO.train`` reads; ``"rfdetr"`` — ``<split>/images`` + ``<split>/labels`` with the COCO
            sidecar at ``<split>/_annotations.coco.json``, what RF-DETR's own reader asks for.
        mode: ``"replace"`` (default) wipes this split before writing, so a rerun is idempotent;
            ``"append"`` continues the split after its last record — the index, ``metadata.jsonl``
            and the COCO sidecar grow, and ``data.yaml`` must already name every class the same
            way (a different name for an id is refused).
        facts: Record entries whose DISTINCT values are listed in ``data.yaml`` under ``facts``
            (``[scale]``: the picture scales a folder holds — a merge of mixed scales is allowed and
            recorded, never refused). Every record's entries ride ``metadata.jsonl`` regardless.
        image_format: ``"png"`` (lossless, default — spectrogram detail quantises poorly under JPEG) or ``"jpg"``.
        jpeg_quality: PIL JPEG quality (1-95); ignored for PNG.
        index_padding: Zero-padding width for the per-split sequential filenames (``6`` → ``000000.png``).
        write_coco: Also write ``annotations/<split>.json`` in COCO format alongside the YOLO-txt labels.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the detection target ``Boxes`` (default ``"target"``).
        image_mode: ``"rgb"`` (default) three channels a picture, or ``"grey"`` one — for pictures whose three
            channels are equal; every reader turns it back into three.
        image_threads: How many threads encode and write the pictures (default 4); the files are the same at any count.
    """

    def __init__(
        self,
        output_dir: str = "./ultralytics_dataset",
        class_names: Optional[Dict[int, str]] = None,
        split: Literal["train", "val", "test"] = "train",
        image_format: Literal["png", "jpg"] = "png",
        jpeg_quality: int = 95,
        index_padding: int = 6,
        write_coco: bool = True,
        input_key: str = "image",
        target_key: str = "target",
        layout: YoloLayout = "ultralytics",
        mode: YoloMode = "replace",
        facts: Optional[List[str]] = None,
        image_mode: ImageMode = "rgb",
        image_threads: ImageThreads = 4,
    ) -> None:
        # Partial / zero-arg: store config only; validate + create dirs (and the writer threads) in open().
        self.output_dir = output_dir
        self.layout: YoloLayout = layout
        self.mode: YoloMode = mode
        self.facts: List[str] = list(facts) if facts else []
        self._fact_values: Dict[str, List[Any]] = {}
        self.class_names = class_names
        self.split = split
        self.image_format = image_format
        self.jpeg_quality = int(jpeg_quality)
        self.index_padding = int(index_padding)
        self.write_coco = bool(write_coco)
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.image_mode: ImageMode = image_mode
        self.image_threads = image_threads
        # Per-run state (populated in open()).
        self._writer: Optional[ThreadPoolExecutor] = None
        self._writing: Deque["Future[None]"] = deque()
        self._root: Optional[Path] = None
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._counter = 0
        self._coco_images: List[Dict[str, Any]] = []
        self._coco_anns: List[Dict[str, Any]] = []
        self._next_ann_id = 0
        self._kept_meta: List[str] = []
        self._meta_lines: List[str] = []
        self._segmentation = False  # a record of this run carried polygons

    def open(self) -> "YoloSink":
        if not self.class_names:
            raise ValueError("YoloSink: class_names must be a non-empty Dict[int, str].")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError(f"YoloSink: jpeg_quality must be in [1, 95]; got {self.jpeg_quality}.")
        if self.index_padding < 1:
            raise ValueError(f"YoloSink: index_padding must be >= 1; got {self.index_padding}.")

        if self.layout not in get_args(YoloLayout):
            raise ValueError(f"YoloSink: layout must be one of {list(get_args(YoloLayout))}; got {self.layout!r}.")
        if self.mode not in get_args(YoloMode):
            raise ValueError(f"YoloSink: mode must be one of {list(get_args(YoloMode))}; got {self.mode!r}.")
        if self.image_mode not in get_args(ImageMode):
            raise ValueError(
                f"YoloSink: image_mode must be one of {list(get_args(ImageMode))}; got {self.image_mode!r}."
            )
        # confluid refuses a count under 1 when the sink is BUILT; this catches one set later.
        if int(self.image_threads) < 1:
            raise ValueError(f"YoloSink: image_threads must be >= 1; got {self.image_threads}.")

        self._root = Path(self.output_dir).expanduser()
        self._images_dir = self._root / self._images_rel()
        self._labels_dir = self._root / self._labels_rel()
        if self.mode == "replace":
            # Replace THIS split's image/label dirs so a re-run is idempotent; leave
            # the other splits (and the merged data.yaml) untouched.
            for d in (self._images_dir, self._labels_dir):
                if d.exists():
                    shutil.rmtree(d)
        for d in (self._images_dir, self._labels_dir):
            d.mkdir(parents=True, exist_ok=True)
        # An append refuses a vocabulary that disagrees BEFORE anything is written.
        if self.mode == "append":
            self._check_vocabulary()

        # Preserve metadata.jsonl lines from OTHER splits (replace) or from every split (append).
        meta_path = self._root / "metadata.jsonl"
        self._kept_meta = []
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    keep = self.mode == "append" or json.loads(line).get("split") != self.split
                except json.JSONDecodeError:
                    keep = True
                if keep:
                    self._kept_meta.append(line)

        self._counter = self._next_index() if self.mode == "append" else 0
        self._coco_images = []
        self._coco_anns = []
        self._next_ann_id = 0
        if self.mode == "append" and self.write_coco and self._coco_sidecar().is_file():
            existing = json.loads(self._coco_sidecar().read_text())
            self._coco_images = list(existing.get("images", []))
            self._coco_anns = list(existing.get("annotations", []))
            self._next_ann_id = 1 + max((int(a["id"]) for a in self._coco_anns), default=-1)
        self._meta_lines = []
        self._segmentation = False
        self._fact_values = {key: [] for key in self.facts}
        self._close_writer()
        self._writer = ThreadPoolExecutor(max_workers=int(self.image_threads), thread_name_prefix="YoloSink")
        self._writing = deque()
        return self

    def _next_index(self) -> int:
        """The index after the last picture of this split — 0 on an empty split."""
        assert self._images_dir is not None
        stems = [p.stem for p in self._images_dir.iterdir() if p.is_file() and p.stem.isdigit()]
        return 1 + max((int(stem) for stem in stems), default=-1)

    def _check_vocabulary(self) -> None:
        """An append must name every class the folder already names the same way."""
        assert self._root is not None
        data_path = self._root / "data.yaml"
        if not data_path.exists():
            return
        loaded = yaml.safe_load(data_path.read_text())
        existing = loaded.get("names") if isinstance(loaded, dict) else None
        if not isinstance(existing, dict):
            return
        mine = self._class_names_int()
        clashes = [
            (int(cid), str(name), mine[int(cid)])
            for cid, name in existing.items()
            if int(cid) in mine and mine[int(cid)] != str(name)
        ]
        if clashes:
            spelled = ", ".join(f"class {cid} is {old!r} in the folder and {new!r} here" for cid, old, new in clashes)
            raise ValueError(f"YoloSink: mode 'append' into {self._root} with a different vocabulary — {spelled}.")

    def write(self, record: Record) -> None:
        assert self._images_dir is not None and self._labels_dir is not None, "YoloSink.write before open()"
        assert self._writer is not None, "YoloSink.write before open()"
        idx = self._counter
        image = _to_uint8_hwc(_record_image(record, self.input_key))
        h, w = int(image.shape[0]), int(image.shape[1])
        pixels = _grey(image, idx) if self.image_mode == "grey" else image
        stem = f"{idx:0{self.index_padding}d}"
        image_filename = f"{stem}.{self.image_format}"

        # The target is read (and a malformed polygon list refused) before anything is written.
        target = _record_target_dict(record, self.target_key)
        polygons = _target_polygons(target, f"record {idx}")
        self._segmentation = self._segmentation or polygons is not None

        save_kwargs: Dict[str, Any] = {"quality": self.jpeg_quality} if self.image_format == "jpg" else {}
        self._writing.append(self._writer.submit(_save_picture, pixels, self._images_dir / image_filename, save_kwargs))
        while len(self._writing) > 2 * int(self.image_threads):  # a bounded number of pictures held at once
            self._writing.popleft().result()

        lines = _yolo_lines(target, h, w, polygons)
        (self._labels_dir / f"{stem}.txt").write_text("\n".join(lines))

        if self.write_coco:
            self._coco_images.append({"id": idx, "file_name": image_filename, "height": h, "width": w})
            new_anns, self._next_ann_id = _coco_annotation_records(target, idx, self._next_ann_id, polygons)
            self._coco_anns.extend(new_anns)

        source_metadata = _jsonable_metadata(_record_metadata(record, self.input_key, self.target_key))
        for key in self.facts:
            if isinstance(source_metadata, dict) and key in source_metadata:
                self._fact_values[key].append(source_metadata[key])
        self._meta_lines.append(
            json.dumps(
                {
                    "index": idx,
                    "split": self.split,
                    "image": f"{self._images_rel()}/{image_filename}",
                    "label": f"{self._labels_rel()}/{stem}.txt",
                    "source_metadata": source_metadata,
                }
            )
        )
        self._counter += 1

    def flush(self) -> None:
        assert self._root is not None, "YoloSink.flush before open()"
        while self._writing:  # every picture on disk (a failed write raises here) before the sidecars name them
            self._writing.popleft().result()
        self._write_one_form_per_split()
        if self.write_coco:
            sidecar = self._coco_sidecar()
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "info": {"description": "YoloSink COCO sidecar", "split": self.split},
                "images": self._coco_images,
                "annotations": self._coco_anns,
                "categories": [{"id": cid, "name": name} for cid, name in sorted(self._class_names_int().items())],
            }
            sidecar.write_text(json.dumps(payload, indent=2))

        meta_path = self._root / "metadata.jsonl"
        meta_path.write_text("\n".join(self._kept_meta + self._meta_lines) + ("\n" if self._meta_lines else ""))

        self._merge_data_yaml()
        logger.info(f"[{self.split}] wrote {self._counter} images to {self._images_dir}")

    def close(self) -> None:
        # flush() does the finalisation; close() releases the writer threads (a run stopped by an error cancels
        # the pictures not started yet).
        self._close_writer()

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.shutdown(wait=True, cancel_futures=True)
            self._writer = None
        self._writing = deque()

    # ------------------------------------------------------------------

    def _write_one_form_per_split(self) -> None:
        """A split holding any ring holds no box line: turn the box lines (and boxes in COCO) into rings.

        Ultralytics refuses a label file that mixes the two forms and a segment model refuses a dataset
        with fewer segments than boxes (see the module docstring), so a split of masks must not mix
        them. A box-only record written before a record with polygons came, or by an earlier run an
        ``append`` continues, is rewritten as its four-corner ring (the coordinates the box line
        holds, to its six decimals). A split with no ring anywhere is not touched, and a ``replace``
        run that saw no polygons does not even read its label files.
        """
        assert self._labels_dir is not None
        paths = sorted(self._labels_dir.glob("*.txt"))
        # An append's earlier records may be rings though this run's are not: only the files say.
        if self._segmentation or (self.mode == "append" and any(_holds_a_ring(path) for path in paths)):
            for path in paths:
                lines = path.read_text().splitlines()
                if any(len(line.split()) == 5 for line in lines):
                    path.write_text(
                        "\n".join(_box_line_as_ring(line) if len(line.split()) == 5 else line for line in lines)
                    )
        if any(annotation.get("segmentation") for annotation in self._coco_anns):
            for annotation in self._coco_anns:
                if not annotation.get("segmentation"):
                    x, y, w, h = annotation["bbox"]
                    annotation["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]

    def _images_rel(self) -> str:
        """This split's image folder, relative to the root — the layout's spelling."""
        return f"images/{self.split}" if self.layout == "ultralytics" else f"{self.split}/images"

    def _labels_rel(self) -> str:
        return f"labels/{self.split}" if self.layout == "ultralytics" else f"{self.split}/labels"

    def _coco_sidecar(self) -> Path:
        """The COCO sidecar: ``annotations/<split>.json``, or RF-DETR's ``<split>/_annotations.coco.json``."""
        assert self._root is not None
        if self.layout == "ultralytics":
            return self._root / "annotations" / f"{self.split}.json"
        return self._root / self.split / "_annotations.coco.json"

    def _class_names_int(self) -> Dict[int, str]:
        assert self.class_names is not None
        return {int(k): str(v) for k, v in self.class_names.items()}

    def _merge_data_yaml(self) -> None:
        """Write data.yaml, preserving any split entries from prior runs."""
        assert self._root is not None
        names = self._class_names_int()
        data_path = self._root / "data.yaml"
        payload: Dict[str, Any] = {}
        if data_path.exists():
            loaded = yaml.safe_load(data_path.read_text())
            if isinstance(loaded, dict):
                payload = loaded
        payload["path"] = str(self._root.resolve())
        payload["names"] = {int(k): str(v) for k, v in sorted(names.items())}
        payload["nc"] = int(max(names.keys()) + 1)
        payload["channels"] = 3
        payload[self.split] = self._images_rel()
        if self.facts:
            existing_facts = payload.get("facts")
            facts: Dict[str, Any] = dict(existing_facts) if isinstance(existing_facts, dict) else {}
            for key in self.facts:
                seen: Dict[str, Any] = {}
                for value in list(facts.get(key) or []) + self._fact_values.get(key, []):
                    seen.setdefault(json.dumps(value, sort_keys=True), value)
                facts[key] = list(seen.values())
            payload["facts"] = facts
        with data_path.open("w") as fp:
            yaml.safe_dump(payload, fp, sort_keys=False)


__all__ = ["POLYGONS", "YoloLayout", "YoloMode", "YoloSink"]
