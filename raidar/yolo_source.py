"""``YoloSource`` — a YOLO dataset folder read back as detection records, the sink's mirror.

Merging dataset folders is reading them and writing them again: a ``ConcatSource`` of readers
feeds ``YoloSink(mode="append")``. The reader takes either layout the sink writes
(``images/<split>`` + ``labels/<split>``, or ``<split>/images`` + ``<split>/labels``), reads the
class names from ``data.yaml``, scales each label line back up by its picture, re-attaches the
record's ``metadata.jsonl`` row (its ``source_metadata`` entries become plain record entries, so a
picture's scale or its provenance survives a merge) and, given a target vocabulary, remaps class
ids by NAME — a name the vocabulary lacks is refused rather than renumbered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import yaml
from confluid import configurable
from PIL import Image as PILImage
from recordstream import Boxes, Image, Record

#: The split folders the sink writes.
YoloSplit = Literal["train", "val", "test"]
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_DATA_FILES = ("data.yaml", "data.yml")


def _names_of(data: Dict[str, Any], path: Path) -> Dict[int, str]:
    """``names`` as ``{id: name}`` — the sink writes a mapping, Ultralytics also accepts a list."""
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"YoloSource: {path} carries no usable 'names' (a mapping id -> name, or a list)")


def _parse_label_line(line: str, width: int, height: int) -> Optional[Tuple[int, List[float]]]:
    """One YOLO label line → ``(class_id, [x0, y0, x1, y1] px)``; a polygon (≥ 3 points) → its box."""
    values = line.split()
    if not values:
        return None
    class_id = int(float(values[0]))
    numbers = [float(v) for v in values[1:]]
    if len(numbers) == 4:
        cx, cy, w, h = numbers
        return class_id, [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
    if len(numbers) >= 6 and len(numbers) % 2 == 0:
        xs, ys = numbers[0::2], numbers[1::2]
        return class_id, [min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height]
    raise ValueError(
        f"YoloSource: a label line holds {len(numbers)} numbers; 4 (a box) or an even count ≥ 6 (a polygon)"
    )


@configurable(category="source")
class YoloSource:
    """Read one split of a YOLO dataset folder as records of ``image`` + ``target``.

    Each record carries the picture (``Image``, HWC uint8), its boxes (``Boxes`` in absolute
    pixels with the class ids and the folder's class names as ``classes``), every entry of its
    ``metadata.jsonl`` row's ``source_metadata`` (a picture ``scale``, a provenance), and
    ``yolo_source`` — the folder, the split and the picture's path. Indexable, so
    ``ConcatSource`` / ``DatasetSplit`` / ``RangeSource`` wrap it. Construction touches no disk.

    Args:
        root: The dataset folder — the one holding ``data.yaml``.
        split: The split read (``train`` / ``val`` / ``test``).
        classes: A target vocabulary ``{id: name}``; given, every class id is remapped by name to it
            and a name it lacks is refused. ``None`` keeps the folder's own ids.
        image_key: Record key the picture is written to (default ``"image"``).
        target_key: Record key the boxes are written to (default ``"target"``).
    """

    def __init__(
        self,
        root: str = "",
        split: YoloSplit = "train",
        classes: Optional[Dict[int, str]] = None,
        image_key: str = "image",
        target_key: str = "target",
    ) -> None:
        # Partial / zero-arg: store config only; the folder is read on first use.
        self.root = root
        self.split: YoloSplit = split
        self.classes = None if classes is None else {int(k): str(v) for k, v in classes.items()}
        self.image_key = image_key
        self.target_key = target_key
        self._resolved: Optional[Tuple[Path, Path, Path, List[Path], Dict[int, str], Dict[str, Dict[str, Any]]]] = None

    # -- the folder ----------------------------------------------------------------------------

    def _resolve(self) -> Tuple[Path, Path, Path, List[Path], Dict[int, str], Dict[str, Dict[str, Any]]]:
        """``(root, images_dir, labels_dir, image files, names, metadata rows by picture path)`` — read once."""
        if self._resolved is not None:
            return self._resolved
        if not self.root:
            raise ValueError("YoloSource: root is required — the dataset folder holding data.yaml")
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"YoloSource: {root} is not a folder")
        data_file = next((root / name for name in _DATA_FILES if (root / name).is_file()), None)
        if data_file is None:
            raise FileNotFoundError(f"YoloSource: {root} holds no data.yaml")
        names = _names_of(yaml.safe_load(data_file.read_text()) or {}, data_file)
        layouts = (
            (root / "images" / self.split, root / "labels" / self.split),
            (root / self.split / "images", root / self.split / "labels"),
        )
        found = next(((i, l) for i, l in layouts if i.is_dir()), None)
        if found is None:
            raise FileNotFoundError(
                f"YoloSource: {root} has no split {self.split!r} (neither images/{self.split} nor {self.split}/images)"
            )
        images_dir, labels_dir = found
        files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        rows: Dict[str, Dict[str, Any]] = {}
        meta_path = root / "metadata.jsonl"
        if meta_path.is_file():
            for line in meta_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("split") == self.split and isinstance(row.get("image"), str):
                        rows[row["image"]] = row
        self._resolved = (root, images_dir, labels_dir, files, names, rows)
        return self._resolved

    @property
    def names(self) -> Dict[int, str]:
        """The folder's class names, ``{id: name}``."""
        return self._resolve()[4]

    def _remap(self) -> Optional[Dict[int, int]]:
        """Folder id → vocabulary id, by name; ``None`` without a vocabulary."""
        if self.classes is None:
            return None
        by_name = {name: new for new, name in self.classes.items()}
        mapping: Dict[int, int] = {}
        for old, name in self.names.items():
            if name not in by_name:
                raise ValueError(
                    f"YoloSource: the folder's class {name!r} (id {old}) is not in the vocabulary "
                    f"{sorted(self.classes.values())}"
                )
            mapping[old] = by_name[name]
        return mapping

    def _vocabulary(self) -> List[str]:
        names = self.classes if self.classes is not None else self.names
        return [names.get(i, "") for i in range(max(names) + 1)] if names else []

    # -- the records ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._resolve()[3])

    def __getitem__(self, index: int) -> Record:
        root, _images_dir, labels_dir, files, _names, rows = self._resolve()
        path = files[index]
        pixels = np.asarray(PILImage.open(path).convert("RGB"))
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        remap = self._remap()
        boxes: List[List[float]] = []
        labels: List[int] = []
        label_path = labels_dir / f"{path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parsed = _parse_label_line(line, width, height)
                if parsed is None:
                    continue
                class_id, box = parsed
                if remap is not None:
                    if class_id not in remap:
                        raise ValueError(f"YoloSource: {label_path} names class {class_id}, which data.yaml does not")
                    class_id = remap[class_id]
                boxes.append(box)
                labels.append(class_id)
        rel = str(path.relative_to(root))
        record: Record = {}
        row = rows.get(rel)
        if row is not None and isinstance(row.get("source_metadata"), dict):
            record.update(
                {k: v for k, v in row["source_metadata"].items() if k not in (self.image_key, self.target_key)}
            )
        record[self.image_key] = Image(pixels, layout="HWC")
        record[self.target_key] = Boxes(
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.asarray(labels, dtype=np.int64),
            canvas=(height, width),
            classes=self._vocabulary(),
        )
        record["yolo_source"] = {"root": str(root), "split": self.split, "image": rel}
        return record

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]


__all__ = ["YoloSource", "YoloSplit"]
