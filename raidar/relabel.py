"""``RelabelLabels`` — a dataset folder's class ids remapped BY NAME, always from the labels it was first written with.

A YOLO or COCO folder is labelled once, in the finest vocabulary its writer knows; a coarser one a trainer wants
(one class, a family per standard) is a remap. The first remap keeps the original beside the labels and never
touches it again, and every later remap starts from it, so no remap loses a name:

==============================  ===========================================================================
``labels/``                     → ``labels.orig/`` (renamed once), then written anew from it by the map
``data.yaml``                   → ``data.orig.yaml`` (copied once); ``data.yaml`` gets the new ``names`` /
                                ``nc`` plus the ``label_map`` it was made with and ``remapped_from``
``<split>/_annotations.coco.json``, ``annotations/*.json``
                                → ``<stem>.orig.json`` (copied once); each annotation of the new file names
                                its original class in ``original_label``
==============================  ===========================================================================

Each YOLO line keeps every character after its class id, and every line keeps its place, so a file that numbers
instances by line (an instance mask) still matches. A map that leaves a name of the original vocabulary out is
refused (``"*"`` names the class every other name goes to); so is a name the new vocabulary does not hold.
``mode: restore`` puts the original back, byte for byte, and leaves no ``.orig`` behind.

Never run it while a trainer reads the folder.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import yaml
from confluid import configurable
from loggair import get_logger

from raidar.coco_or_yolo_source import DatasetFolderError

logger = get_logger(__name__)

#: What a run does: remap from the original, or put the original back.
RelabelMode = Literal["remap", "restore"]

LABELS = "labels"
ORIGINAL_LABELS = "labels.orig"
DATA_YAML = "data.yaml"
ORIGINAL_DATA_YAML = "data.orig.yaml"
#: The suffix the original of a COCO file is kept under (``train.json`` → ``train.orig.json``).
ORIGINAL_COCO_SUFFIX = ".orig.json"
#: The key of a map naming the class every name it does not list goes to.
WILDCARD = "*"
#: The entry of a remapped COCO annotation naming its original class.
ORIGINAL_LABEL = "original_label"
_COCO_RFDETR = "_annotations.coco.json"
#: The folder written beside ``labels/`` before it replaces it, so an interrupted run never leaves half a tree.
_NEW_LABELS = "labels.new"


class RelabelError(DatasetFolderError):
    """A folder or a map the remap refuses: the message names the file (and the line) or the setting."""


def _names_of(document: Any, path: Path) -> Dict[int, str]:
    """``names`` of a ``data.yaml`` as ``{id: name}`` — a mapping or a list, as YOLO writers write either."""
    names = document.get("names") if isinstance(document, dict) else None
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    raise RelabelError(f"{path}: no 'names' (a list, or a mapping of ids to names)")


def label_trees(root: Path) -> List[Path]:
    """The folders holding a YOLO label tree: ``<root>/labels`` and every ``<root>/<split>/labels`` (RF-DETR's
    layout) — or, once remapped, the same folders' ``labels.orig``. Each is named by its ``labels`` path."""
    found: List[Path] = []
    for parent in [root] + sorted(
        child for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")
    ):
        if (parent / LABELS).is_dir() or (parent / ORIGINAL_LABELS).is_dir():
            found.append(parent / LABELS)
    return found


def coco_files(root: Path) -> List[Path]:
    """The COCO files of the folder — ``<split>/_annotations.coco.json`` and ``annotations/*.json`` — each named by
    its current path, whether or not it has been remapped yet (never an ``.orig.json`` itself)."""
    found: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and (child / _COCO_RFDETR).is_file():
            found.append(child / _COCO_RFDETR)
    annotations = root / "annotations"
    if annotations.is_dir():
        found.extend(
            path for path in sorted(annotations.glob("*.json")) if not path.name.endswith(ORIGINAL_COCO_SUFFIX)
        )
    return found


def original_of(path: Path) -> Path:
    """Where a label tree's, a ``data.yaml``'s or a COCO file's original is kept."""
    if path.name == LABELS:
        return path.with_name(ORIGINAL_LABELS)
    if path.name == DATA_YAML:
        return path.with_name(ORIGINAL_DATA_YAML)
    return path.with_name(path.name[: -len(".json")] + ORIGINAL_COCO_SUFFIX)


def _label_files(tree: Path) -> Iterator[Path]:
    for here, subfolders, names in os.walk(tree):
        subfolders[:] = sorted(name for name in subfolders if not name.startswith("."))
        for name in sorted(names):
            if name.endswith(".txt"):
                yield Path(here) / name


def _resolve_map(
    original: Dict[int, str], label_map: Dict[str, str], classes: Optional[Dict[int, str]]
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """``(original id → new id, the new vocabulary)``, refusing a name left out or a class not in ``classes``."""
    missing = [name for name in original.values() if name not in label_map and WILDCARD not in label_map]
    if missing:
        raise RelabelError(f"RelabelLabels: label_map does not say where {', '.join(map(repr, missing))} go")
    targets = {oid: str(label_map.get(name, label_map.get(WILDCARD))) for oid, name in sorted(original.items())}
    if classes is None:
        vocabulary: Dict[int, str] = {}
        for name in targets.values():
            if name not in vocabulary.values():
                vocabulary[len(vocabulary)] = name
    else:
        vocabulary = {int(key): str(value) for key, value in classes.items()}
    ids = {name: nid for nid, name in vocabulary.items()}
    if len(ids) != len(vocabulary):
        raise RelabelError(f"RelabelLabels: classes names a class twice: {sorted(vocabulary.values())}")
    unknown = sorted({name for name in targets.values() if name not in ids})
    if unknown:
        raise RelabelError(f"RelabelLabels: label_map sends names to {', '.join(map(repr, unknown))}, not in classes")
    return {oid: ids[name] for oid, name in targets.items()}, vocabulary


def _remap_line(line: str, ids: Dict[int, int], path: Path, number: int) -> str:
    """The line with its class id replaced and every other character kept; a blank line as it is."""
    stripped = line.lstrip()
    if not stripped.strip():
        return line
    token = stripped.split(maxsplit=1)[0]
    try:
        old = int(float(token))
    except ValueError:
        raise RelabelError(f"{path}:{number}: the class id {token!r} is not a number") from None
    if old not in ids:
        raise RelabelError(f"{path}:{number}: class {old} is not one of the original vocabulary's ids")
    lead = line[: len(line) - len(stripped)]
    return f"{lead}{ids[old]}{stripped[len(token):]}"


def _remap_tree(original: Path, target: Path, ids: Dict[int, int]) -> int:
    """Write ``target`` from ``original`` line by line; returns the number of files. Built beside it first."""
    building = target.with_name(_NEW_LABELS)
    if building.exists():
        shutil.rmtree(building)
    count = 0
    for source in _label_files(original):
        out = building / source.relative_to(original)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        lines = text.split("\n")
        out.write_text(
            "\n".join(_remap_line(line, ids, source, number) for number, line in enumerate(lines, start=1)),
            encoding="utf-8",
        )
        count += 1
    if target.exists():
        shutil.rmtree(target)  # a tree derived from the original, which stays
    building.rename(target)
    return count


def _coco_original_names(document: Dict[str, Any], path: Path) -> Dict[int, str]:
    categories = document.get("categories")
    if not isinstance(categories, list):
        raise RelabelError(f"{path}: no 'categories' list")
    return {int(category["id"]): str(category["name"]) for category in categories}


@configurable
class RelabelLabels:
    """Remap a YOLO or COCO dataset folder's classes by name, from the original labels it keeps (module docstring).

    Args:
        root: The dataset folder — the one holding ``data.yaml``, ``labels/`` or the COCO files; ``~`` and ``$VAR``
            are expanded.
        label_map: Each name of the ORIGINAL vocabulary → its new class, many to one; ``"*"`` names the class every
            other name goes to. Every original name must be covered.
        classes: The new vocabulary, id → name; None numbers the map's classes in the order the original
            vocabulary first reaches them.
        mode: remap (from the original, kept on the first run) or restore (the original put back, nothing kept).
    """

    def __init__(
        self,
        root: str = "",
        label_map: Optional[Dict[str, str]] = None,
        classes: Optional[Dict[int, str]] = None,
        mode: RelabelMode = "remap",
    ) -> None:
        self.root = root
        self.label_map = label_map
        self.classes = classes
        self.mode = mode

    @property
    def folder(self) -> Path:
        """The folder the settings name — refused unless set and present."""
        if not str(self.root).strip():
            raise RelabelError("RelabelLabels: 'root' (the dataset folder) is required")
        folder = Path(os.path.expandvars(os.path.expanduser(str(self.root))))
        if "$" in str(folder):
            raise RelabelError(f"RelabelLabels: root {self.root!r} names an unset environment variable")
        if not folder.is_dir():
            raise RelabelError(f"RelabelLabels: {folder} is not a folder")
        return folder

    def run(self) -> Dict[str, Any]:
        """Remap or restore; returns what changed: ``{"root", "mode", "classes", "label_files", "coco_files"}``."""
        root = self.folder
        trees = label_trees(root)
        cocos = coco_files(root)
        if not trees and not cocos:
            raise RelabelError(f"{root}: no labels/ folder, <split>/labels/ folder or COCO file to remap")
        if self.mode == "restore":
            return self._restore(root, trees, cocos)
        if not self.label_map:
            raise RelabelError("RelabelLabels: label_map is required to remap")
        label_map = {str(key): str(value) for key, value in self.label_map.items()}

        vocabulary: Dict[int, str] = {}
        files = 0
        data_path = root / DATA_YAML
        if trees:
            if not data_path.is_file() and not original_of(data_path).is_file():
                raise RelabelError(f"{root}: a YOLO label tree but no {DATA_YAML} naming its classes")
            if not original_of(data_path).is_file():
                shutil.copy2(data_path, original_of(data_path))
            original_document = yaml.safe_load(original_of(data_path).read_text(encoding="utf-8")) or {}
            ids, vocabulary = _resolve_map(
                _names_of(original_document, original_of(data_path)), label_map, self.classes
            )
            for tree in trees:
                if not original_of(tree).is_dir():
                    tree.rename(original_of(tree))
                files += _remap_tree(original_of(tree), tree, ids)
            self._write_data_yaml(data_path, original_document, vocabulary, label_map)
        for path in cocos:
            if not original_of(path).is_file():
                shutil.copy2(path, original_of(path))
            vocabulary = self._remap_coco(path, label_map)
        logger.info(f"relabelled {root}: {files} label file(s), {len(cocos)} COCO file(s), {len(vocabulary)} classes")
        return {
            "root": str(root),
            "mode": "remap",
            "classes": {int(key): name for key, name in vocabulary.items()},
            "label_files": files,
            "coco_files": len(cocos),
        }

    def _write_data_yaml(
        self, path: Path, original: Dict[str, Any], vocabulary: Dict[int, str], label_map: Dict[str, str]
    ) -> None:
        document = dict(original)
        names_were_a_list = isinstance(original.get("names"), list)
        ordered = [vocabulary[key] for key in sorted(vocabulary)]
        document["names"] = ordered if names_were_a_list else {key: vocabulary[key] for key in sorted(vocabulary)}
        document["nc"] = max(vocabulary) + 1
        document["label_map"] = label_map
        document["remapped_from"] = ORIGINAL_DATA_YAML
        path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _remap_coco(self, path: Path, label_map: Dict[str, str]) -> Dict[int, str]:
        original_path = original_of(path)
        document = json.loads(original_path.read_text(encoding="utf-8"))
        original_names = _coco_original_names(document, original_path)
        ids, vocabulary = _resolve_map(original_names, label_map, self.classes)
        for number, annotation in enumerate(document.get("annotations") or []):
            old = int(annotation["category_id"])
            if old not in ids:
                raise RelabelError(f"{original_path}: annotation {number} has category {old}, not in its categories")
            annotation[ORIGINAL_LABEL] = original_names[old]
            annotation["category_id"] = ids[old]
        document["categories"] = [{"id": key, "name": vocabulary[key]} for key in sorted(vocabulary)]
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return vocabulary

    def _restore(self, root: Path, trees: List[Path], cocos: List[Path]) -> Dict[str, Any]:
        files = 0
        for tree in trees:
            original = original_of(tree)
            if not original.is_dir():
                continue  # never remapped
            if tree.exists():
                shutil.rmtree(tree)
            original.rename(tree)
            files += sum(1 for _ in _label_files(tree))
        data_path = root / DATA_YAML
        if original_of(data_path).is_file():
            original_of(data_path).replace(data_path)
        restored = 0
        for path in cocos:
            if original_of(path).is_file():
                original_of(path).replace(path)
                restored += 1
        logger.info(f"restored {root}: {files} label file(s), {restored} COCO file(s)")
        return {"root": str(root), "mode": "restore", "classes": {}, "label_files": files, "coco_files": restored}

    def __repr__(self) -> str:
        return f"RelabelLabels(root={self.root!r}, mode={self.mode!r})"


__all__ = [
    "ORIGINAL_LABEL",
    "RelabelError",
    "RelabelLabels",
    "RelabelMode",
    "WILDCARD",
    "coco_files",
    "label_trees",
    "original_of",
]
