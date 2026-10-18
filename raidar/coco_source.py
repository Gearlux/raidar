"""``CocoSource`` — one split of a COCO dataset folder read as detection records, the COCO twin of ``YoloSource``.

A COCO file sits in one of three places, and the reader looks in each, in this order:

* ``<root>/<split>/_annotations.coco.json`` — RF-DETR's (and Roboflow's) layout, what the sink's ``rfdetr`` layout
  writes; the pictures in ``<split>/images`` or beside the file;
* ``<root>/annotations/<split>.json`` — the sink's default layout; the pictures in ``images/<split>``;
* ``<root>/annotations/instances_<split>*.json`` — the official dataset's (``instances_val2017.json``); the
  pictures in ``val2017/`` or ``images/val2017/``.

The classes are numbered by the categories' ORDER (sorted by id), not by the ids themselves: the official
categories run 1 … 90 with gaps, and a label that indexes the class list must find a name, never a blank. A
polygon ``segmentation`` comes back as the instance's polygons, as ``YoloSource`` gives a YOLO-seg line's ring;
a run-length one has no ring to give and leaves the instance's entry empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from confluid import configurable
from PIL import Image as PILImage
from recordstream import Boxes, Image, Record

from raidar.yolo_sink import POLYGONS
from raidar.yolo_source import YoloSplit

#: The file RF-DETR's reader asks for in each split folder.
RFDETR_FILE = "_annotations.coco.json"


def _rings(segmentation: Any) -> List[List[List[float]]]:
    """A COCO ``segmentation`` as rings ``[[x, y], ...]`` — a polygon list's rings, ``[]`` for a run-length one."""
    if not isinstance(segmentation, list):
        return []
    return [
        [[float(x), float(y)] for x, y in zip(flat[0::2], flat[1::2])]
        for flat in segmentation
        if isinstance(flat, list) and len(flat) >= 6
    ]


@configurable(category="source")
class CocoSource:
    """Read one split of a COCO dataset folder as records of ``image`` + ``target``.

    Each record carries the picture (``Image``, HWC uint8), its boxes (``Boxes`` in absolute pixels, labelled by
    the categories' order, with their names as ``classes``) and ``coco_source`` — the COCO file, the split and the
    picture's ``file_name``. Indexable, so ``ConcatSource`` / ``DatasetSplit`` / ``RangeSource`` wrap it. Construction
    touches no disk; the COCO file is read once, on first use.

    Args:
        root: The dataset folder — the one holding the split folders or ``annotations/``.
        split: The split read (``train`` / ``val`` / ``test``).
        image_key: Record key the picture is written to (default ``"image"``).
        target_key: Record key the boxes are written to (default ``"target"``).
    """

    def __init__(
        self,
        root: str = "",
        split: YoloSplit = "train",
        image_key: str = "image",
        target_key: str = "target",
    ) -> None:
        # Partial / zero-arg: store config only; the COCO file is read on first use.
        self.root = root
        self.split: YoloSplit = split
        self.image_key = image_key
        self.target_key = target_key
        self._resolved: Optional[
            Tuple[Path, Path, List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, int], List[str]]
        ] = None

    # -- the folder ----------------------------------------------------------------------------

    def annotation_file(self) -> Optional[Path]:
        """The split's COCO file — the first of the three places that holds one — or ``None``. Two official files
        for one split (``instances_val2014.json`` and ``instances_val2017.json``) are refused: which is meant is
        the person's to say, by pointing at a folder holding one."""
        root = self._root()
        for candidate in (root / self.split / RFDETR_FILE, root / "annotations" / f"{self.split}.json"):
            if candidate.is_file():
                return candidate
        official = sorted((root / "annotations").glob(f"instances_{self.split}*.json"))
        if len(official) > 1:
            raise ValueError(
                f"CocoSource: {root / 'annotations'} holds {len(official)} COCO files for split {self.split!r} "
                f"({', '.join(path.name for path in official)}) — keep one"
            )
        return official[0] if official else None

    def _root(self) -> Path:
        if not self.root:
            raise ValueError("CocoSource: root is required — the dataset folder holding the COCO file")
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CocoSource: {root} is not a folder")
        return root

    def _picture_folders(self, root: Path, file: Path) -> List[Path]:
        """Where the pictures the COCO file names may be, in the order they are looked for."""
        if file.name == RFDETR_FILE:
            return [file.parent / "images", file.parent]
        tag = file.stem[len("instances_") :] if file.stem.startswith("instances_") else self.split
        return [root / "images" / tag, root / tag, root / tag / "images", root / "images" / self.split, root]

    def _resolve(
        self,
    ) -> Tuple[Path, Path, List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, int], List[str]]:
        """``(root, file, images, annotations by image id, category id → label, class names)`` — read once."""
        if self._resolved is not None:
            return self._resolved
        root = self._root()
        file = self.annotation_file()
        if file is None:
            raise FileNotFoundError(
                f"CocoSource: {root} has no COCO file for split {self.split!r} (looked for "
                f"{self.split}/{RFDETR_FILE}, annotations/{self.split}.json, annotations/instances_{self.split}*.json)"
            )
        document = json.loads(file.read_text())
        categories = sorted(document.get("categories") or [], key=lambda category: int(category["id"]))
        labels = {int(category["id"]): index for index, category in enumerate(categories)}
        names = [str(category.get("name", category["id"])) for category in categories]
        by_image: Dict[int, List[Dict[str, Any]]] = {}
        for annotation in document.get("annotations") or []:
            by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self._resolved = (root, file, list(document.get("images") or []), by_image, labels, names)
        return self._resolved

    @property
    def class_names(self) -> List[str]:
        """The classes in label order — the categories' names, sorted by their ids."""
        return list(self._resolve()[5])

    # -- the records ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._resolve()[2])

    def _picture(self, root: Path, file: Path, name: str) -> Path:
        for folder in self._picture_folders(root, file):
            if (folder / name).is_file():
                return folder / name
        raise FileNotFoundError(
            f"CocoSource: {file} names {name}, which is in none of "
            f"{', '.join(str(folder) for folder in self._picture_folders(root, file))}"
        )

    def __getitem__(self, index: int) -> Record:
        root, file, images, by_image, labels, names = self._resolve()
        entry = images[index]
        name = str(entry["file_name"])
        pixels = np.asarray(PILImage.open(self._picture(root, file, name)).convert("RGB"))
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        boxes: List[List[float]] = []
        classes: List[int] = []
        polygons: List[List[List[List[float]]]] = []
        for annotation in by_image.get(int(entry["id"]), []):
            category = int(annotation["category_id"])
            if category not in labels:
                raise ValueError(
                    f"CocoSource: {file} annotation {annotation.get('id')} names category {category}, "
                    "which its categories do not list"
                )
            x, y, w, h = (float(value) for value in annotation["bbox"])
            boxes.append([x, y, x + w, y + h])
            classes.append(labels[category])
            polygons.append(_rings(annotation.get("segmentation")))
        record: Record = {
            self.image_key: Image(pixels, layout="HWC"),
            self.target_key: Boxes(
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(classes, dtype=np.int64),
                canvas=(height, width),
                extras={POLYGONS: polygons} if any(polygons) else {},
                classes=list(names),
            ),
            "coco_source": {"annotations": str(file), "split": self.split, "image": name},
        }
        return record

    def __iter__(self) -> Iterator[Record]:
        for index in range(len(self)):
            yield self[index]


__all__ = ["CocoSource", "RFDETR_FILE"]
