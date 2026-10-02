"""YoloSink — write a detection Stream to a native Ultralytics dataset dir.

A RecordStream ``DataSink`` (``open`` / ``write(record)`` / ``flush`` / ``close``):
``recordstream run`` streams a detection Stream into it via
``DatasetProcessor(stream=…, sink=YoloSink(…))``, exactly like any other
source → ops → sink pipeline — there is no bespoke processor. Each
record dict is materialised to the layout
``YOLO.train(data=<output_dir>/data.yaml)`` natively consumes:

::

    <output_dir>/
      data.yaml                        # YOLO descriptor (merged across splits)
      metadata.jsonl                   # per-image provenance sidecar
      annotations/<split>.json         # COCO sidecar (optional)
      images/<split>/<idx>.{png,jpg}
      labels/<split>/<idx>.txt         # one line per box: class cx cy w h

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
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import yaml
from confluid import configurable
from loggair import get_logger
from PIL import Image
from recordstream import Boxes, Label, Record, item_data
from recordstream.storage.base import Storage

logger = get_logger(__name__)


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
    item (the ``CocoToTorchVisionDetection`` output) unwrapped, a dict-valued ``Label`` / raw dict
    passed through, or ``{}`` when the key is absent."""
    if target_key not in record:
        return {}
    item = record[target_key]
    if isinstance(item, Boxes):
        return {"boxes": item.boxes, "labels": item.labels}
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


def _yolo_lines(target: Dict[str, Any], height: int, width: int) -> List[str]:
    """Convert ``{boxes: [N,4] xyxy-pixel, labels: [N] int}`` to YOLO label lines.

    One line per box: ``class_id cx_norm cy_norm w_norm h_norm`` (coords
    normalised to ``[0,1]``). An empty box list yields ``[]`` (the file is still
    written — the YOLO convention for a negative-example image).
    """
    boxes = target.get("boxes") if isinstance(target, dict) else None
    labels = target.get("labels") if isinstance(target, dict) else None
    if boxes is None or labels is None:
        return []
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if boxes.shape[0] == 0:
        return []
    cx = (boxes[:, 0] + boxes[:, 2]) / (2.0 * width)
    cy = (boxes[:, 1] + boxes[:, 3]) / (2.0 * height)
    bw = (boxes[:, 2] - boxes[:, 0]) / width
    bh = (boxes[:, 3] - boxes[:, 1]) / height
    return [f"{int(labels[i])} {cx[i]:.6f} {cy[i]:.6f} {bw[i]:.6f} {bh[i]:.6f}" for i in range(boxes.shape[0])]


def _coco_annotation_records(
    target: Dict[str, Any],
    image_id: int,
    next_ann_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build COCO annotation entries for one image's target dict.

    COCO ``bbox`` is ``[x, y, w, h]`` top-left pixel coords; ``area`` is ``w*h``.
    """
    boxes = target.get("boxes") if isinstance(target, dict) else None
    labels = target.get("labels") if isinstance(target, dict) else None
    if boxes is None or labels is None:
        return [], next_ann_id
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    out: List[Dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        out.append(
            {
                "id": next_ann_id + i,
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
            }
        )
    return out, next_ann_id + boxes.shape[0]


_MAX_META_ARRAY_ELEMS = 256


def _jsonable_metadata(metadata: Any, *, max_array_elems: int = _MAX_META_ARRAY_ELEMS) -> Any:
    """Best-effort JSON-friendly conversion of a record's metadata payload — NEVER raises.

    Source metadata can carry values ``json.dumps`` cannot encode: NumPy/torch arrays, **complex**
    IQ samples, and StreamStudio's internal ``__taidal_stash_*`` snapshots (a stashed complex signal
    from a DAG→sequential fan-out). So this:
      * converts a ``complex`` scalar to a string (JSON has no complex type);
      * summarises a LARGE array (> ``max_array_elems``) as ``<array shape=… dtype=…>`` — a 1.2M-element
        complex IQ stash must not be dumped verbatim into every per-image label JSON (bloat AND the
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


# category="sink": surfaced as a StreamStudio sink node (RECORDSTREAM_OBJECT:sink → DatasetProcessor.sink).
@configurable(category="sink")
class YoloSink(Storage):
    """DataSink that materialises one split of a detection Stream to a YOLO dataset dir.

    Wired as the ``sink:`` of a ``recordstream.processing.DatasetProcessor`` and run
    via ``recordstream run``. Writes ``images/<split>`` + ``labels/<split>``
    (+ optional ``annotations/<split>.json``), appends per-image provenance to
    ``metadata.jsonl``, and merges ``data.yaml`` so repeated runs (one per
    split, same ``output_dir``) accumulate. Does no work in ``__init__``;
    validates + touches the filesystem lazily in ``open()``.

    Args:
        output_dir: Destination dataset root (created if missing). Repeated runs into the same dir merge.
        class_names: Mapping ``class_id → display name`` (e.g. ``{0: "background", 1: "drone"}``); ``nc = max(id)+1``.
        split: Which split this run writes — ``"train"`` / ``"val"`` / ``"test"``.
        image_format: ``"png"`` (lossless, default — spectrogram detail quantises poorly under JPEG) or ``"jpg"``.
        jpeg_quality: PIL JPEG quality (1-95); ignored for PNG.
        index_padding: Zero-padding width for the per-split sequential filenames (``6`` → ``000000.png``).
        write_coco: Also write ``annotations/<split>.json`` in COCO format alongside the YOLO-txt labels.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the detection target ``Boxes`` (default ``"target"``).
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
    ) -> None:
        # Partial / zero-arg: store config only; validate + create dirs in open().
        self.output_dir = output_dir
        self.class_names = class_names
        self.split = split
        self.image_format = image_format
        self.jpeg_quality = int(jpeg_quality)
        self.index_padding = int(index_padding)
        self.write_coco = bool(write_coco)
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        # Per-run state (populated in open()).
        self._root: Optional[Path] = None
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._counter = 0
        self._coco_images: List[Dict[str, Any]] = []
        self._coco_anns: List[Dict[str, Any]] = []
        self._next_ann_id = 0
        self._kept_meta: List[str] = []
        self._meta_lines: List[str] = []

    def open(self) -> "YoloSink":
        if not self.class_names:
            raise ValueError("YoloSink: class_names must be a non-empty Dict[int, str].")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError(f"YoloSink: jpeg_quality must be in [1, 95]; got {self.jpeg_quality}.")
        if self.index_padding < 1:
            raise ValueError(f"YoloSink: index_padding must be >= 1; got {self.index_padding}.")

        self._root = Path(self.output_dir).expanduser()
        self._images_dir = self._root / "images" / self.split
        self._labels_dir = self._root / "labels" / self.split
        # Replace THIS split's image/label dirs so a re-run is idempotent; leave
        # the other splits (and the merged data.yaml) untouched.
        for d in (self._images_dir, self._labels_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # Preserve metadata.jsonl lines from OTHER splits; this split's are rewritten.
        meta_path = self._root / "metadata.jsonl"
        self._kept_meta = []
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    keep = json.loads(line).get("split") != self.split
                except json.JSONDecodeError:
                    keep = True
                if keep:
                    self._kept_meta.append(line)

        self._counter = 0
        self._coco_images = []
        self._coco_anns = []
        self._next_ann_id = 0
        self._meta_lines = []
        return self

    def write(self, record: Record) -> None:
        assert self._images_dir is not None and self._labels_dir is not None, "YoloSink.write before open()"
        idx = self._counter
        image = _to_uint8_hwc(_record_image(record, self.input_key))
        h, w = int(image.shape[0]), int(image.shape[1])
        stem = f"{idx:0{self.index_padding}d}"
        image_filename = f"{stem}.{self.image_format}"

        save_kwargs: Dict[str, Any] = {"quality": self.jpeg_quality} if self.image_format == "jpg" else {}
        Image.fromarray(image).save(self._images_dir / image_filename, **save_kwargs)

        target = _record_target_dict(record, self.target_key)
        lines = _yolo_lines(target, h, w)
        (self._labels_dir / f"{stem}.txt").write_text("\n".join(lines))

        if self.write_coco:
            self._coco_images.append({"id": idx, "file_name": image_filename, "height": h, "width": w})
            new_anns, self._next_ann_id = _coco_annotation_records(target, idx, self._next_ann_id)
            self._coco_anns.extend(new_anns)

        self._meta_lines.append(
            json.dumps(
                {
                    "index": idx,
                    "split": self.split,
                    "image": f"images/{self.split}/{image_filename}",
                    "label": f"labels/{self.split}/{stem}.txt",
                    "source_metadata": _jsonable_metadata(_record_metadata(record, self.input_key, self.target_key)),
                }
            )
        )
        self._counter += 1

    def flush(self) -> None:
        assert self._root is not None, "YoloSink.flush before open()"
        if self.write_coco:
            ann_dir = self._root / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "info": {"description": "YoloSink COCO sidecar", "split": self.split},
                "images": self._coco_images,
                "annotations": self._coco_anns,
                "categories": [{"id": cid, "name": name} for cid, name in sorted(self._class_names_int().items())],
            }
            (ann_dir / f"{self.split}.json").write_text(json.dumps(payload, indent=2))

        meta_path = self._root / "metadata.jsonl"
        meta_path.write_text("\n".join(self._kept_meta + self._meta_lines) + ("\n" if self._meta_lines else ""))

        self._merge_data_yaml()
        logger.info(f"[{self.split}] wrote {self._counter} images to {self._images_dir}")

    def close(self) -> None:
        # flush() does the finalisation; nothing extra to release.
        pass

    # ------------------------------------------------------------------

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
        payload[self.split] = f"images/{self.split}"
        with data_path.open("w") as fp:
            yaml.safe_dump(payload, fp, sort_keys=False)


__all__ = ["YoloSink"]
