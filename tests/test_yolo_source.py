"""``YoloSource`` — a YOLO dataset folder read back as records, the sink's mirror (2026-10-06).

Merging folders is reading them and appending: ``ConcatSource`` of readers → ``YoloSink(mode:
append)``. The reader takes both layouts, scales the labels back by the picture, re-attaches the
``metadata.jsonl`` row (so a record's picture scale survives a merge) and remaps class ids by
NAME to a target vocabulary.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import yaml
from recordstream import Boxes
from recordstream import Image as SFImage
from recordstream import Record
from recordstream.core import Stream
from recordstream.sources.concat import ConcatSource

from raidar.yolo_sink import YoloSink
from raidar.yolo_source import YoloSource


def _record(boxes: Any, labels: Any, height: int = 64, width: int = 96, **extra: Any) -> Record:
    record: Record = {
        "image": SFImage((torch.rand(3, height, width) * 255).to(torch.uint8).numpy(), layout="CHW"),
        "target": Boxes(
            boxes=torch.as_tensor(boxes, dtype=torch.float32), labels=torch.as_tensor(labels, dtype=torch.int64)
        ),
    }
    record.update(extra)
    return record


def _write(out: Path, records: Any, *, class_names: Dict[int, str], layout: str = "rfdetr", **kwargs: Any) -> None:
    sink = YoloSink(output_dir=str(out), class_names=class_names, layout=layout, **kwargs)  # type: ignore[arg-type]
    with sink:
        for record in records:
            sink.write(record)
        sink.flush()


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    out = tmp_path / "a"
    _write(
        out,
        [
            _record([[8.0, 16.0, 40.0, 48.0], [0.0, 0.0, 96.0, 64.0]], [0, 1], scale={"khz_per_px": 78.1}, pack="P1"),
            _record([], [], scale={"khz_per_px": 78.1}, pack="P2"),
        ],
        class_names={0: "signal", 1: "ignore"},
        facts=["scale"],
    )
    return out


class TestReading:
    def test_the_records_come_back(self, folder: Path) -> None:
        source = YoloSource(root=str(folder), split="train")
        assert len(source) == 2
        first = source[0]
        image = np.asarray(first["image"])
        assert image.shape == (64, 96, 3) and image.dtype == np.uint8 and first["image"].layout == "HWC"
        boxes = first["target"]
        np.testing.assert_allclose(
            np.asarray(boxes.boxes), [[8.0, 16.0, 40.0, 48.0], [0.0, 0.0, 96.0, 64.0]], atol=1e-3
        )
        assert np.asarray(boxes.labels).tolist() == [0, 1] and boxes.canvas == (64, 96)
        assert boxes.classes == ["signal", "ignore"]
        assert first["scale"] == {"khz_per_px": 78.1} and first["pack"] == "P1"
        assert first["yolo_source"] == {
            "root": str(folder.resolve()),
            "split": "train",
            "image": "train/images/000000.png",
        }
        assert np.asarray(source[1]["target"].boxes).shape == (0, 4) and source[1]["pack"] == "P2"
        assert [r["pack"] for r in source] == ["P1", "P2"]

    def test_the_ultralytics_layout_reads_too(self, tmp_path: Path) -> None:
        out = tmp_path / "u"
        _write(out, [_record([[0.0, 0.0, 48.0, 32.0]], [0])], class_names={0: "signal"}, layout="ultralytics")
        source = YoloSource(root=str(out))
        assert len(source) == 1
        np.testing.assert_allclose(np.asarray(source[0]["target"].boxes), [[0.0, 0.0, 48.0, 32.0]], atol=1e-3)
        assert source[0]["yolo_source"]["image"] == "images/train/000000.png"

    def test_a_polygon_label_is_read_as_its_box(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        _write(out, [_record([[0.0, 0.0, 48.0, 32.0]], [0])], class_names={0: "signal"})
        (out / "train" / "labels" / "000000.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
        boxes = np.asarray(YoloSource(root=str(out))[0]["target"].boxes)
        np.testing.assert_allclose(boxes, [[24.0, 16.0, 72.0, 48.0]], atol=1e-3)

    def test_a_folder_without_the_split_is_refused(self, folder: Path) -> None:
        with pytest.raises(FileNotFoundError, match="val"):
            len(YoloSource(root=str(folder), split="val"))

    def test_construction_touches_no_disk(self) -> None:
        source = YoloSource(root="/nowhere/at/all")
        assert source.root == "/nowhere/at/all"
        with pytest.raises(FileNotFoundError, match="nowhere"):
            len(source)
        with pytest.raises(ValueError, match="root"):
            len(YoloSource())


class TestTheVocabulary:
    def test_ids_are_remapped_by_name(self, folder: Path) -> None:
        source = YoloSource(root=str(folder), classes={0: "ignore", 1: "signal", 2: "other"})
        boxes = source[0]["target"]
        assert np.asarray(boxes.labels).tolist() == [1, 0] and boxes.classes == ["ignore", "signal", "other"]

    def test_an_unknown_name_is_refused(self, folder: Path) -> None:
        with pytest.raises(ValueError, match="'ignore'.*signal"):
            YoloSource(root=str(folder), classes={0: "signal"})[0]


class TestTheRoundTrip:
    def test_sink_source_sink_keeps_the_labels(self, folder: Path, tmp_path: Path) -> None:
        out = tmp_path / "b"
        _write(out, YoloSource(root=str(folder)), class_names={0: "signal", 1: "ignore"}, facts=["scale"])
        for name in ("000000.txt", "000001.txt"):
            a = (folder / "train" / "labels" / name).read_text().split()
            b = (out / "train" / "labels" / name).read_text().split()
            assert [float(x) for x in a] == pytest.approx([float(x) for x in b], abs=1e-5)
        assert yaml.safe_load((out / "data.yaml").read_text())["facts"] == {"scale": [{"khz_per_px": 78.1}]}
        meta = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
        assert meta[0]["source_metadata"]["pack"] == "P1" and meta[0]["source_metadata"]["yolo_source"][
            "image"
        ].endswith("000000.png")

    def test_two_folders_merge_by_appending(self, folder: Path, tmp_path: Path) -> None:
        other = tmp_path / "c"
        _write(
            other,
            [_record([[0.0, 0.0, 10.0, 10.0]], [0], scale={"khz_per_px": 39.1})],
            class_names={0: "signal"},
            facts=["scale"],
        )
        merged = tmp_path / "m"
        readers = ConcatSource(
            sources=[YoloSource(root=str(folder)), YoloSource(root=str(other), classes={0: "signal", 1: "ignore"})]
        )
        _write(merged, Stream(source=readers), class_names={0: "signal", 1: "ignore"}, mode="append", facts=["scale"])
        assert sorted(p.name for p in (merged / "train" / "images").iterdir()) == [
            "000000.png",
            "000001.png",
            "000002.png",
        ]
        data = yaml.safe_load((merged / "data.yaml").read_text())
        assert data["facts"] == {"scale": [{"khz_per_px": 78.1}, {"khz_per_px": 39.1}]}
