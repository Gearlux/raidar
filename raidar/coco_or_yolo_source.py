"""``CocoOrYoloSource`` — a detection dataset folder read whichever way it is labelled.

The folder a person points at may hold YOLO labels (``data.yaml`` beside the split folders), a COCO file for the
split (any of :class:`~raidar.coco_source.CocoSource`'s three places), or both — a folder the sink wrote with its
COCO sidecar holds both for the same pictures. This source reads it without the person saying which: both present,
``labels`` picks (YOLO by default — the labels a YOLO trainer reads); one present, that one is read whatever
``labels`` says; neither, the folder is refused naming what was looked for.

It exists so a dataset can be loaded by its SETTINGS alone — a folder, a split, which labels — from a settings form
that offers a folder browser (the parameter's name, ``folder``, earns it), with no graph to draw. Every setting is
read on each use, so a changed folder or split is the next read's; the reader it builds is cached for the settings
it was built from.
"""

import os
import re
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple, Union

from confluid import configurable
from recordstream import Record

from raidar.coco_source import CocoSource
from raidar.yolo_source import _DATA_FILES, YoloSource, YoloSplit

#: Which labels are read when the folder holds both.
DatasetLabels = Literal["yolo", "coco"]
#: Where the folder starts until one is chosen — the field shows it and a settings form's folder browser opens there.
DATA_ROOT = "$DATA_ROOT"
#: A ``$VAR`` / ``${VAR}`` that :func:`os.path.expandvars` left as written — a variable the machine does not have.
_UNSET_VARIABLE = re.compile(r"\$(?:\{(\w+)\}|(\w+))")

_Reader = Union[YoloSource, CocoSource]


class DatasetFolderError(ValueError):
    """A folder setting that names no dataset this source can read — a missing folder, a split it lacks, labels of
    neither kind. A ``ValueError``: the kind a host reports to the person as a wrong setting, never a crash."""


@configurable(category="source")
class CocoOrYoloSource:
    """Read one split of a COCO or YOLO dataset folder as records of ``image`` + ``target``.

    The records are :class:`~raidar.yolo_source.YoloSource`'s or :class:`~raidar.coco_source.CocoSource`'s — the
    picture, its boxes with the class names, and ``yolo_source`` / ``coco_source`` saying where they came from —
    and :attr:`class_names` is the dataset's class list. Construction touches no disk.

    Args:
        folder: The dataset folder — the one holding ``data.yaml`` or the COCO file's folder; ``~`` and ``$VAR``
            are expanded (``$DATA_ROOT/signal_detection/pictures/d``). Until one is chosen it is ``$DATA_ROOT``.
        split: The split read (``train`` / ``val`` / ``test``).
        labels: Which labels are read when the folder holds both — ``yolo`` (default) or ``coco``.
        image_key: Record key the picture is written to (default ``"image"``).
        target_key: Record key the boxes are written to (default ``"target"``).
    """

    def __init__(
        self,
        folder: str = DATA_ROOT,
        split: YoloSplit = "train",
        labels: DatasetLabels = "yolo",
        image_key: str = "image",
        target_key: str = "target",
    ) -> None:
        # Values only — the folder is looked at on first use, and again whenever a setting changed.
        self.folder = folder
        self.split: YoloSplit = split
        self.labels: DatasetLabels = labels
        self.image_key = image_key
        self.target_key = target_key
        self._reader_cache: Optional[Tuple[Tuple[str, ...], _Reader]] = None

    @property
    def directory(self) -> Path:
        """The folder the settings name, expanded and resolved — refused (:class:`DatasetFolderError`) when it is
        empty, names a variable the machine does not have, is missing, or is a file."""
        if not self.folder:
            raise DatasetFolderError("CocoOrYoloSource: choose the dataset folder in the settings first")
        expanded = os.path.expandvars(self.folder)
        unset = _UNSET_VARIABLE.search(expanded)
        if unset:
            raise DatasetFolderError(
                f"CocoOrYoloSource: {unset.group(1) or unset.group(2)} is not set — choose the dataset folder"
            )
        path = Path(expanded).expanduser().resolve()
        if path.is_file():
            raise DatasetFolderError(f"CocoOrYoloSource: {path} is a file, not a folder — choose its dataset folder")
        if not path.is_dir():
            raise DatasetFolderError(f"CocoOrYoloSource: no such folder: {path}")
        return path

    @property
    def reader(self) -> _Reader:
        """The source the folder is read through, for the settings as they are now."""
        key = (self.folder, self.split, self.labels, self.image_key, self.target_key)
        if self._reader_cache is not None and self._reader_cache[0] == key:
            return self._reader_cache[1]
        reader = self._choose(self.directory)
        self._reader_cache = (key, reader)
        return reader

    def _choose(self, root: Path) -> _Reader:
        yolo = YoloSource(root=str(root), split=self.split, image_key=self.image_key, target_key=self.target_key)
        coco = CocoSource(root=str(root), split=self.split, image_key=self.image_key, target_key=self.target_key)
        has_yolo = any((root / name).is_file() for name in _DATA_FILES)
        has_coco = coco.annotation_file() is not None
        if has_yolo and has_coco:
            return coco if self.labels == "coco" else yolo
        if has_coco:
            return coco
        if has_yolo:
            if not ((root / "images" / self.split).is_dir() or (root / self.split / "images").is_dir()):
                raise DatasetFolderError(
                    f"CocoOrYoloSource: {root} has no split {self.split!r} "
                    f"(neither images/{self.split} nor {self.split}/images)"
                )
            return yolo
        raise DatasetFolderError(
            f"CocoOrYoloSource: {root} holds neither a YOLO dataset (data.yaml) nor a COCO file for split "
            f"{self.split!r} — choose the folder holding them"
        )

    @property
    def class_names(self) -> List[str]:
        """The dataset's classes in label order — what each record's boxes carry."""
        return self.reader.class_names

    @property
    def dataset_uri(self) -> str:
        """``<folder>?split=<split>`` — a consumer names the records after the folder and the split (``d-val``)."""
        return f"{self.directory}?split={self.split}"

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> Record:
        return self.reader[index]

    def __iter__(self) -> Iterator[Record]:
        reader = self.reader
        for index in range(len(reader)):
            yield reader[index]

    def __repr__(self) -> str:
        return f"CocoOrYoloSource(folder={self.folder!r}, split={self.split!r}, labels={self.labels!r})"


__all__ = ["CocoOrYoloSource", "DATA_ROOT", "DatasetFolderError", "DatasetLabels"]
