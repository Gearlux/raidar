"""The graphs raidar ships for an annotation workspace's ``detection`` type.

An annotation workspace scans the ``annotaide.templates`` entry-point group for ``templates()`` —
``{type: {graph slot: YAML document}}`` — and runs each document from where it is installed. raidar ships ONE: the
detection type's ``source``, a COCO or YOLO dataset folder read by :class:`~raidar.coco_or_yolo_source.
CocoOrYoloSource`, whose settings (folder, split, labels) are the whole of loading a dataset. Its ``app:`` block names
the words the workspace's tools menu lists it by (:data:`TOOLS_WORDS`), so a person loads a dataset from that menu
without drawing a graph.

This module imports nothing of the workspace: the document names its classes, and the workspace resolves them.
"""

from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent

#: The detection type's source graph.
SOURCE_GRAPH = _HERE / "coco_or_yolo_dataset_source.yaml"
#: The words the tools menu lists the source graph by (its ``app: {tools: …}``).
TOOLS_WORDS = "load a COCO or YOLO dataset"


def templates() -> Dict[str, Dict[str, Path]]:
    """``{type: {graph slot: template path}}`` — the detection type's source graph."""
    return {"detection": {"source": SOURCE_GRAPH}}


__all__ = ["SOURCE_GRAPH", "TOOLS_WORDS", "templates"]
