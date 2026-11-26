"""``YoloSink.planned_changes()`` — what opening the sink for a run would delete, said before the run.

In ``replace`` mode a run wipes its split's picture and label folders when the sink opens. A plan of the run
(``recordstream run --dry_run``, an editor's "run all") asks the sink and lists the answer, so the folders a run
deletes are named before anything is deleted. Asking changes nothing on disk.
"""

from pathlib import Path
from typing import Any, List

from recordstream.core import Stream
from recordstream.processing import DatasetProcessor

from raidar.yolo_sink import YoloSink
from tests.test_yolo_sink import _convert, _train_stream

NAMES = {0: "background", 1: "drone-signal"}


def _sink(out: Path, **settings: Any) -> YoloSink:
    return YoloSink(output_dir=str(out), class_names=NAMES, **settings)


def _files(folder: Path) -> List[str]:
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file())


class TestWhatARunWouldDelete:
    def test_a_folder_that_is_not_there_yet_loses_nothing(self, tmp_path: Path) -> None:
        assert _sink(tmp_path / "out").planned_changes() == []
        assert not (tmp_path / "out").exists()  # asking made nothing

    def test_a_replace_names_the_splits_two_folders_with_their_files_and_size(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(_train_stream(), out, split="train")  # 2 pictures
        before = _files(out)
        lines = _sink(out).planned_changes()
        assert len(lines) == 2
        assert lines[0].startswith(f"delete {out / 'images' / 'train'} (2 files, ")
        assert lines[1].startswith(f"delete {out / 'labels' / 'train'} (2 files, ")
        assert _files(out) == before  # asking deleted nothing

    def test_another_split_is_not_named(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(_train_stream(), out, split="train")
        assert _sink(out, split="val").planned_changes() == []

    def test_the_rfdetr_layout_names_its_own_folders(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(_train_stream(), out, split="train", layout="rfdetr")
        lines = _sink(out, layout="rfdetr").planned_changes()
        assert [line.split(" (")[0] for line in lines] == [
            f"delete {out / 'train' / 'images'}",
            f"delete {out / 'train' / 'labels'}",
        ]

    def test_an_append_deletes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(_train_stream(), out, split="train")
        assert _sink(out, mode="append").planned_changes() == []

    def test_one_file_is_one_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(Stream(source=list(_train_stream())[:1]), out, split="train")
        assert " (1 file, " in _sink(out).planned_changes()[0]

    def test_the_processors_plan_lists_them(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(_train_stream(), out, split="train")
        plan = DatasetProcessor(stream=_train_stream(), sink=_sink(out)).plan()
        assert [action.split(" (")[0] for action in plan.actions] == [
            f"delete {out / 'images' / 'train'}",
            f"delete {out / 'labels' / 'train'}",
        ]
