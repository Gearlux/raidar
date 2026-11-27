"""The detection workspace's SOURCE graph raidar ships — a COCO or YOLO dataset loaded by its settings (2026-10-16).

``raidar.annotation.templates()`` hands the annotation workspace's ``detection`` type one graph, ``source``: a
:class:`~raidar.coco_or_yolo_source.CocoOrYoloSource` into the records stream, its class names the graph's class
list. Its ``app:`` block asks for the source's folder, split and labels, and names the words ``tools ▾`` lists it
by — so a person loads a dataset from that menu, its settings form only, without drawing a graph.

The checks that need the workspace itself run where it is installed (raidar never depends on it) and skip
elsewhere; the real signal detection pictures are read in the slow run when ``$DATA_ROOT`` holds them.
"""

import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
import torch
from confluid import load
from recordstream import Boxes
from recordstream import Image as SFImage
from recordstream import Record
from recordstream.ops.contract import GraphContract

from raidar.annotation import SOURCE_GRAPH, TOOLS_WORDS, templates
from raidar.coco_or_yolo_source import CocoOrYoloSource
from raidar.yolo_sink import YoloSink


def _dataset(root: Path, split: str = "val", count: int = 2) -> Path:
    records: List[Record] = [
        {
            "image": SFImage((torch.rand(3, 32, 48) * 255).to(torch.uint8).numpy(), layout="CHW"),
            "target": Boxes(
                boxes=torch.as_tensor([[4.0, 4.0, 20.0, 16.0]], dtype=torch.float32),
                labels=torch.as_tensor([1], dtype=torch.int64),
            ),
        }
        for _ in range(count)
    ]
    names = {0: "5GNR", 1: "LTE"}
    sink = YoloSink(output_dir=str(root), class_names=names, layout="rfdetr", split=split)  # type: ignore[arg-type]
    with sink:
        for record in records:
            sink.write(record)
        sink.flush()
    return root


class TestTheShippedGraph:
    def test_the_detection_type_gets_one_source_graph(self) -> None:
        assert templates() == {"detection": {"source": SOURCE_GRAPH}}
        assert SOURCE_GRAPH.is_file() and SOURCE_GRAPH.suffix == ".yaml"

    def test_its_settings_are_the_sources_folder_split_and_labels_and_tools_lists_it(self) -> None:
        app = load(SOURCE_GRAPH.read_text())["app"]
        assert app["inputs"] == [
            ["CocoOrYoloSource", "folder"],
            ["CocoOrYoloSource", "split"],
            ["CocoOrYoloSource", "labels"],
        ]
        assert app["tools"] == TOOLS_WORDS == "load a COCO or YOLO dataset"

    def test_it_delivers_the_datasets_records_and_class_names(self, tmp_path: Path) -> None:
        document: Any = load(SOURCE_GRAPH.read_text())
        dataset = document["dataset"]
        assert isinstance(dataset, CocoOrYoloSource) and dataset.image_key == "input"
        dataset.folder, dataset.split = str(_dataset(tmp_path / "d")), "val"
        contract = document["contract"]
        assert isinstance(contract, GraphContract)
        contract.check()
        record = next(iter(document["stream"]))
        assert isinstance(record["input"], SFImage) and record["target"].classes == ["5GNR", "LTE"]

    def test_CON_left_at_data_root_it_is_refused_naming_the_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        document: Any = load(SOURCE_GRAPH.read_text())
        with pytest.raises(ValueError, match=f"{tmp_path.resolve()} holds neither"):
            document["contract"].check()


class TestInTheAnnotationWorkspace:
    """A detection workspace of the installed type: the dataset's settings written, the list shows the dataset."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Any:
        pytest.importorskip("annotaide")
        from annotaide.cli import open_type_workspace

        workspace = open_type_workspace("detection")
        workspace.set_path(str(tmp_path / "ws" / "w.workspace.yaml"))
        return workspace

    def test_the_type_runs_raidars_source_and_tools_lists_it(self, workspace: Any) -> None:
        assert workspace.graph_file("source") == SOURCE_GRAPH
        assert {"name": "source", "label": TOOLS_WORDS} in workspace.info()["settings_tools"]

    def test_its_settings_load_the_dataset_with_its_boxes_and_classes(self, workspace: Any, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d")
        workspace.settings_file("source").write_text(f"dataset.folder: {folder}\ndataset.split: val\n")
        report = workspace.sync_source_graph()
        assert report["status"] == "applied", report
        assert report["message"] == "detection source: 2 recording(s), 2 class(es)"
        ids = [row["id"] for row in workspace.listing(limit=10)["recordings"]]
        assert ids == ["d-val#0", "d-val#1"]
        (overlay,) = workspace.box_overlays("d-val#0")
        assert overlay["key"] == "target" and [box["name"] for box in overlay["boxes"]] == ["LTE"]

    def test_CON_a_folder_holding_no_dataset_is_reported_and_lists_nothing(
        self, workspace: Any, tmp_path: Path
    ) -> None:
        workspace.settings_file("source").write_text(f"dataset.folder: {tmp_path}\n")
        report = workspace.sync_source_graph()
        assert report["status"] == "error" and "holds neither a YOLO dataset" in report["message"]
        assert workspace.listing(limit=10)["total"] == 0

    def test_CON_a_split_the_dataset_lacks_is_reported_naming_it(self, workspace: Any, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d")
        workspace.settings_file("source").write_text(f"dataset.folder: {folder}\ndataset.split: test\n")
        report = workspace.sync_source_graph()
        assert report["status"] == "error" and "has no split 'test'" in report["message"]


@pytest.mark.slow
def test_the_signal_detection_pictures_load(tmp_path: Path) -> None:
    """``$DATA_ROOT/signal_detection/pictures/d`` (YOLO and COCO side by side): the val split, both ways."""
    folder = Path(os.environ.get("DATA_ROOT", "")) / "signal_detection" / "pictures" / "d"
    if not os.environ.get("DATA_ROOT") or not (folder / "data.yaml").is_file():
        pytest.skip(f"no signal detection pictures at {folder}")
    yolo = CocoOrYoloSource(folder=str(folder), split="val")
    coco = CocoOrYoloSource(folder=str(folder), split="val", labels="coco")
    assert len(yolo) == len(coco) > 0 and yolo.class_names == coco.class_names
    first_yolo, first_coco = yolo[0]["target"], coco[0]["target"]
    np.testing.assert_allclose(np.asarray(first_yolo.boxes), np.asarray(first_coco.boxes), atol=0.5)
    assert np.asarray(first_yolo.labels).tolist() == np.asarray(first_coco.labels).tolist()
