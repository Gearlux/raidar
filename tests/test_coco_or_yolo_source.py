"""``CocoOrYoloSource`` — a detection dataset folder read whichever way it is labelled (2026-10-16).

The folder a person points at holds YOLO labels (``data.yaml``), a COCO file for the split, or both — the signal
detection pictures hold both. Both present, ``labels`` picks; one present, that one is read whatever ``labels``
says; neither, the folder is refused naming what was looked for. The settings are read on every use, so a
changed folder or split is the next read's.
"""

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
import torch
from recordstream import Boxes
from recordstream import Image as SFImage
from recordstream import Record

from raidar.coco_or_yolo_source import CocoOrYoloSource, DatasetFolderError
from raidar.yolo_sink import YoloSink


def _record(boxes: Any, labels: Any) -> Record:
    return {
        "image": SFImage((torch.rand(3, 32, 48) * 255).to(torch.uint8).numpy(), layout="CHW"),
        "target": Boxes(
            boxes=torch.as_tensor(boxes, dtype=torch.float32), labels=torch.as_tensor(labels, dtype=torch.int64)
        ),
    }


def _dataset(root: Path, split: str = "val", count: int = 2, write_coco: bool = True) -> Path:
    """A folder in the signal detection pictures' layout: ``<split>/images``, ``<split>/labels``, ``data.yaml`` and,
    with ``write_coco``, ``<split>/_annotations.coco.json``."""
    records: List[Record] = [_record([[4.0, 4.0, 20.0, 16.0]], [1]) for _ in range(count)]
    sink = YoloSink(
        output_dir=str(root),
        class_names={0: "5GNR", 1: "LTE"},
        layout="rfdetr",
        split=split,  # type: ignore[arg-type]
        write_coco=write_coco,
    )
    with sink:
        for record in records:
            sink.write(record)
        sink.flush()
    return root


class TestWhichLabelsAreRead:
    def test_both_present_yolo_is_read_by_default(self, tmp_path: Path) -> None:
        source = CocoOrYoloSource(folder=str(_dataset(tmp_path / "d")), split="val")
        record = source[0]
        assert "yolo_source" in record and "coco_source" not in record
        assert len(source) == 2 and source.class_names == ["5GNR", "LTE"]
        np.testing.assert_allclose(np.asarray(record["target"].boxes), [[4.0, 4.0, 20.0, 16.0]], atol=1e-3)

    def test_both_present_coco_when_asked(self, tmp_path: Path) -> None:
        record = CocoOrYoloSource(folder=str(_dataset(tmp_path / "d")), split="val", labels="coco")[0]
        assert "coco_source" in record and "yolo_source" not in record
        np.testing.assert_allclose(np.asarray(record["target"].boxes), [[4.0, 4.0, 20.0, 16.0]], atol=1e-3)
        assert record["target"].classes == ["5GNR", "LTE"]

    def test_CON_yolo_alone_is_read_even_when_coco_is_asked(self, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d", write_coco=False)
        assert "yolo_source" in CocoOrYoloSource(folder=str(folder), split="val", labels="coco")[0]

    def test_CON_coco_alone_is_read_even_when_yolo_is_asked(self, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d")
        (folder / "data.yaml").unlink()
        source = CocoOrYoloSource(folder=str(folder), split="val", labels="yolo")
        assert "coco_source" in source[0] and source.class_names == ["5GNR", "LTE"]

    def test_the_image_and_target_keys_reach_the_reader(self, tmp_path: Path) -> None:
        record = CocoOrYoloSource(folder=str(_dataset(tmp_path / "d")), split="val", image_key="input")[0]
        assert isinstance(record["input"], SFImage) and isinstance(record["target"], Boxes)

    def test_iterating_yields_every_picture(self, tmp_path: Path) -> None:
        assert len(list(CocoOrYoloSource(folder=str(_dataset(tmp_path / "d", count=3)), split="val"))) == 3


class TestTheSettingsAreReadOnEveryUse:
    def test_a_changed_split_is_the_next_reads(self, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d", split="val", count=2)
        _dataset(folder, split="train", count=3)
        source = CocoOrYoloSource(folder=str(folder), split="val")
        assert len(source) == 2
        source.split = "train"
        assert len(source) == 3
        source.labels = "coco"
        assert "coco_source" in source[0]

    def test_its_name_is_the_folder_and_the_split(self, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d")
        assert CocoOrYoloSource(folder=str(folder), split="val").dataset_uri == f"{folder.resolve()}?split=val"


class TestTheFolder:
    def test_home_and_variables_are_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _dataset(tmp_path / "pictures" / "d")
        monkeypatch.setenv("PICTURES", str(tmp_path / "pictures"))
        assert len(CocoOrYoloSource(folder="$PICTURES/d", split="val")) == 2
        monkeypatch.setenv("HOME", str(tmp_path))
        assert len(CocoOrYoloSource(folder="~/pictures/d", split="val")) == 2

    def test_a_folder_holding_neither_is_refused_naming_what_was_looked_for(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFolderError, match=r"holds neither a YOLO dataset \(data\.yaml\) nor a COCO file"):
            len(CocoOrYoloSource(folder=str(tmp_path), split="val"))

    def test_CON_a_yolo_folder_without_the_split_says_which_split(self, tmp_path: Path) -> None:
        folder = _dataset(tmp_path / "d", write_coco=False)
        with pytest.raises(DatasetFolderError, match=r"has no split 'test' \(neither images/test nor test/images\)"):
            len(CocoOrYoloSource(folder=str(folder), split="test"))

    def test_an_empty_folder_setting_is_refused(self) -> None:
        with pytest.raises(DatasetFolderError, match="choose the dataset folder"):
            len(CocoOrYoloSource(folder=""))

    def test_a_variable_the_server_lacks_is_refused_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        with pytest.raises(DatasetFolderError, match="NOT_SET_ANYWHERE is not set"):
            len(CocoOrYoloSource(folder="$NOT_SET_ANYWHERE/d"))

    def test_a_missing_folder_and_a_file_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFolderError, match="no such folder"):
            len(CocoOrYoloSource(folder=str(tmp_path / "nope")))
        (tmp_path / "a.txt").write_text("x")
        with pytest.raises(DatasetFolderError, match="is a file, not a folder"):
            len(CocoOrYoloSource(folder=str(tmp_path / "a.txt")))

    def test_a_refused_folder_is_a_value_error_the_kind_a_host_reports_as_a_wrong_setting(self) -> None:
        assert issubclass(DatasetFolderError, ValueError)

    def test_construction_touches_no_disk(self) -> None:
        source = CocoOrYoloSource(folder="/nowhere/at/all")
        assert source.folder == "/nowhere/at/all" and source.split == "train" and source.labels == "yolo"
        # Until a folder is chosen it is $DATA_ROOT, where the settings form's folder browser opens.
        assert CocoOrYoloSource().folder == "$DATA_ROOT"
