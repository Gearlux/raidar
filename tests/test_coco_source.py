"""``CocoSource`` — one split of a COCO dataset folder read as detection records (2026-10-16).

The reader takes the three places a COCO file sits — RF-DETR's ``<split>/_annotations.coco.json`` (the sink's
``rfdetr`` layout), the sink's ``annotations/<split>.json`` and the official ``annotations/instances_<split>*.json``
— finds each picture beside it, and numbers the classes by the categories' order, so a category list with gaps
(the official 1 … 90) gives labels 0 … n-1 and a class list without blanks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from recordstream import Boxes
from recordstream import Image as SFImage
from recordstream import Record

from raidar.coco_source import CocoSource
from raidar.yolo_sink import POLYGONS, YoloSink


def _record(boxes: Any, labels: Any, height: int = 64, width: int = 96) -> Record:
    return {
        "image": SFImage((torch.rand(3, height, width) * 255).to(torch.uint8).numpy(), layout="CHW"),
        "target": Boxes(
            boxes=torch.as_tensor(boxes, dtype=torch.float32), labels=torch.as_tensor(labels, dtype=torch.int64)
        ),
    }


def _write(out: Path, records: List[Record], layout: str, split: str = "val") -> None:
    sink = YoloSink(
        output_dir=str(out),
        class_names={0: "signal", 1: "ignore"},
        layout=layout,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
    )
    with sink:
        for record in records:
            sink.write(record)
        sink.flush()


def _picture(path: Path, width: int = 40, height: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (width, height), (10, 20, 30)).save(path)


def _official(root: Path, annotations: List[Dict[str, Any]], name: str = "instances_val2017.json") -> Path:
    """An official COCO folder: ``annotations/instances_val2017.json`` naming pictures in ``val2017/``, two
    categories with a GAP in their ids (1 and 3), as the official 1 … 90 have."""
    _picture(root / "val2017" / "a.jpg")
    _picture(root / "val2017" / "b.jpg")
    document = {
        "images": [
            {"id": 7, "file_name": "a.jpg", "width": 40, "height": 30},
            {"id": 9, "file_name": "b.jpg", "width": 40, "height": 30},
        ],
        "annotations": annotations,
        "categories": [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}],
    }
    path = root / "annotations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


class TestTheThreeLayouts:
    def test_rfdetrs_file_in_the_split_folder(self, tmp_path: Path) -> None:
        out = tmp_path / "d"
        _write(out, [_record([[8.0, 16.0, 40.0, 48.0], [0.0, 0.0, 96.0, 64.0]], [0, 1]), _record([], [])], "rfdetr")
        source = CocoSource(root=str(out), split="val")
        assert len(source) == 2
        first = source[0]
        image = np.asarray(first["image"])
        assert image.shape == (64, 96, 3) and image.dtype == np.uint8 and first["image"].layout == "HWC"
        boxes = first["target"]
        np.testing.assert_allclose(
            np.asarray(boxes.boxes), [[8.0, 16.0, 40.0, 48.0], [0.0, 0.0, 96.0, 64.0]], atol=1e-3
        )
        assert np.asarray(boxes.labels).tolist() == [0, 1] and boxes.canvas == (64, 96)
        assert boxes.classes == ["signal", "ignore"] and source.class_names == ["signal", "ignore"]
        assert first["coco_source"] == {
            "annotations": str((out / "val" / "_annotations.coco.json").resolve()),
            "split": "val",
            "image": "000000.png",
        }

    def test_CON_a_picture_with_no_annotation_has_no_box(self, tmp_path: Path) -> None:
        out = tmp_path / "d"
        _write(out, [_record([[8.0, 16.0, 40.0, 48.0]], [0]), _record([], [])], "rfdetr")
        empty = CocoSource(root=str(out), split="val")[1]["target"]
        assert np.asarray(empty.boxes).shape == (0, 4) and np.asarray(empty.labels).tolist() == []
        assert empty.classes == ["signal", "ignore"]

    def test_the_sinks_annotations_folder(self, tmp_path: Path) -> None:
        out = tmp_path / "u"
        _write(out, [_record([[0.0, 0.0, 48.0, 32.0]], [1])], "ultralytics", split="train")
        source = CocoSource(root=str(out), split="train")
        assert len(source) == 1
        np.testing.assert_allclose(np.asarray(source[0]["target"].boxes), [[0.0, 0.0, 48.0, 32.0]], atol=1e-3)
        assert source[0]["coco_source"]["annotations"] == str((out / "annotations" / "train.json").resolve())

    def test_the_official_layout_numbers_the_classes_by_the_categories_order(self, tmp_path: Path) -> None:
        _official(
            tmp_path,
            [
                {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1.0, 2.0, 10.0, 5.0]},
                {"id": 2, "image_id": 7, "category_id": 1, "bbox": [0.0, 0.0, 4.0, 4.0]},
            ],
        )
        source = CocoSource(root=str(tmp_path), split="val")
        assert source.class_names == ["person", "car"]  # ids 1 and 3, sorted by id, numbered 0 and 1
        boxes = source[0]["target"]
        np.testing.assert_allclose(np.asarray(boxes.boxes), [[1.0, 2.0, 11.0, 7.0], [0.0, 0.0, 4.0, 4.0]])
        assert np.asarray(boxes.labels).tolist() == [1, 0] and boxes.canvas == (30, 40)
        assert source[1]["coco_source"]["image"] == "b.jpg"


class TestPolygons:
    def test_a_polygon_segmentation_is_the_instances_ring(self, tmp_path: Path) -> None:
        _official(
            tmp_path,
            [
                {"id": 1, "image_id": 7, "category_id": 1, "bbox": [0, 0, 4, 4], "segmentation": [[0, 0, 4, 0, 4, 4]]},
                {"id": 2, "image_id": 7, "category_id": 1, "bbox": [1, 1, 2, 2], "segmentation": []},
            ],
        )
        boxes = CocoSource(root=str(tmp_path), split="val")[0]["target"]
        assert boxes.extras[POLYGONS] == [[[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]]], []]

    def test_CON_a_record_with_no_polygon_carries_no_polygons_entry(self, tmp_path: Path) -> None:
        _official(
            tmp_path,
            [
                {
                    "id": 1,
                    "image_id": 7,
                    "category_id": 1,
                    "bbox": [0, 0, 4, 4],
                    "segmentation": {"counts": "abc", "size": [30, 40]},  # run-length encoded: no ring to keep
                }
            ],
        )
        assert POLYGONS not in CocoSource(root=str(tmp_path), split="val")[0]["target"].extras


class TestRefusals:
    def test_a_folder_without_a_coco_file_for_the_split_names_where_it_looked(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"no COCO file for split 'test'.*test/_annotations\.coco\.json"):
            len(CocoSource(root=str(tmp_path), split="test"))

    def test_two_official_files_for_one_split_are_refused_naming_both(self, tmp_path: Path) -> None:
        _official(tmp_path, [])
        _official(tmp_path, [], name="instances_val2014.json")
        with pytest.raises(ValueError, match="instances_val2014.json, .*instances_val2017.json"):
            len(CocoSource(root=str(tmp_path), split="val"))

    def test_a_picture_the_file_names_but_the_disk_lacks_is_refused_by_name(self, tmp_path: Path) -> None:
        _official(tmp_path, [])
        (tmp_path / "val2017" / "b.jpg").unlink()
        source = CocoSource(root=str(tmp_path), split="val")
        assert len(source) == 2  # the listing reads the file, not the pictures
        with pytest.raises(FileNotFoundError, match=r"instances_val2017\.json names b\.jpg"):
            source[1]

    def test_an_annotation_of_an_unknown_category_is_refused(self, tmp_path: Path) -> None:
        _official(tmp_path, [{"id": 1, "image_id": 7, "category_id": 5, "bbox": [0, 0, 1, 1]}])
        with pytest.raises(ValueError, match="category 5, which its categories do not list"):
            CocoSource(root=str(tmp_path), split="val")[0]

    def test_construction_touches_no_disk(self) -> None:
        source = CocoSource(root="/nowhere/at/all")
        assert source.root == "/nowhere/at/all"
        with pytest.raises(FileNotFoundError, match="nowhere"):
            len(source)
        with pytest.raises(ValueError, match="root"):
            len(CocoSource())


class TestIteration:
    def test_iterating_yields_every_picture_in_the_files_order(self, tmp_path: Path) -> None:
        _official(tmp_path, [])
        assert [record["coco_source"]["image"] for record in CocoSource(root=str(tmp_path), split="val")] == [
            "a.jpg",
            "b.jpg",
        ]

    def test_the_keys_are_settable(self, tmp_path: Path) -> None:
        _official(tmp_path, [])
        record = CocoSource(root=str(tmp_path), split="val", image_key="input", target_key="truth")[0]
        assert isinstance(record["input"], SFImage) and isinstance(record["truth"], Boxes)
