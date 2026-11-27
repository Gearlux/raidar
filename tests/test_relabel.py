"""``RelabelLabels``: a folder's classes remapped by name, always from the original it keeps on the first run."""

import json
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from raidar.relabel import ORIGINAL_LABEL, RelabelError, RelabelLabels

FINE = ["APCO25", "DMR", "802.11ax HE", "802.11n HT"]
FAMILY = {"APCO25": "LMR", "DMR": "LMR", "802.11ax HE": "WLAN", "802.11n HT": "WLAN"}


def _yolo_folder(root: Path, names: object, files: Dict[str, str], tree: str = "labels") -> Path:
    """A folder of label files under ``<root>/<tree>`` (paths relative to it) and a data.yaml naming ``names``."""
    for relative, text in files.items():
        path = root / tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "names": names}))
    return root


def _snapshot(root: Path) -> Dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """A store-shaped folder: ``labels/<split>/<group>/sample.txt`` with a sixth column and a mask beside it."""
    root = _yolo_folder(
        tmp_path / "store",
        FINE,
        {
            "train/lmr/sample_000000.txt": "0 0.5 0.5 0.1 0.2 6.213\n1 0.25 0.5 0.05 1.0 33.255\n",
            "train/wlan/sample_000000.txt": "2 0.4 0.1 0.2 0.05 12.000\n3 0.6 0.9 0.1 0.05 4.500\n",
            "val/wlan/sample_000000.txt": "",
        },
    )
    (root / "masks" / "train" / "lmr").mkdir(parents=True)
    (root / "masks" / "train" / "lmr" / "sample_000000.png").write_bytes(b"mask")
    return root


def _lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


class TestARemap:
    def test_each_line_gets_its_new_id_and_keeps_every_other_character_and_its_place(self, store: Path) -> None:
        RelabelLabels(root=str(store), label_map=FAMILY, classes={0: "WLAN", 1: "LMR"}).run()

        assert _lines(store / "labels/train/lmr/sample_000000.txt") == [
            "1 0.5 0.5 0.1 0.2 6.213",
            "1 0.25 0.5 0.05 1.0 33.255",
        ]
        assert _lines(store / "labels/train/wlan/sample_000000.txt") == [
            "0 0.4 0.1 0.2 0.05 12.000",
            "0 0.6 0.9 0.1 0.05 4.500",
        ]
        assert (store / "labels/val/wlan/sample_000000.txt").read_text() == ""

    def test_the_first_run_keeps_the_original_labels_and_data_yaml(self, store: Path) -> None:
        before = _snapshot(store / "labels")
        data_before = (store / "data.yaml").read_bytes()

        RelabelLabels(root=str(store), label_map=FAMILY).run()

        assert _snapshot(store / "labels.orig") == before
        assert (store / "data.orig.yaml").read_bytes() == data_before

    def test_data_yaml_names_the_new_classes_the_map_and_the_original_and_keeps_its_other_keys(
        self, store: Path
    ) -> None:
        RelabelLabels(root=str(store), label_map=FAMILY, classes={0: "WLAN", 1: "LMR"}).run()

        data = yaml.safe_load((store / "data.yaml").read_text())
        assert data["names"] == ["WLAN", "LMR"]  # a list stays a list
        assert data["nc"] == 2
        assert data["label_map"] == FAMILY
        assert data["remapped_from"] == "data.orig.yaml"
        assert data["train"] == "images/train"

    def test_a_mapping_of_names_stays_a_mapping(self, tmp_path: Path) -> None:
        root = _yolo_folder(tmp_path, {0: "APCO25", 1: "DMR"}, {"train/a.txt": "1 0.5 0.5 0.1 0.1\n"})

        RelabelLabels(root=str(root), label_map={"*": "signal"}).run()

        assert yaml.safe_load((root / "data.yaml").read_text())["names"] == {0: "signal"}
        assert _lines(root / "labels/train/a.txt") == ["0 0.5 0.5 0.1 0.1"]

    def test_without_classes_the_new_ids_follow_the_order_the_original_vocabulary_reaches_them(
        self, store: Path
    ) -> None:
        result = RelabelLabels(root=str(store), label_map=FAMILY).run()

        assert result["classes"] == {0: "LMR", 1: "WLAN"}
        assert _lines(store / "labels/train/wlan/sample_000000.txt")[0].startswith("1 ")

    def test_a_second_remap_starts_from_the_original_never_from_the_first(self, store: Path) -> None:
        RelabelLabels(root=str(store), label_map={"*": "signal"}).run()

        # Every name is still there to map: the one-class labels were never the source.
        RelabelLabels(root=str(store), label_map={name: name for name in FINE}, classes=dict(enumerate(FINE))).run()

        assert _snapshot(store / "labels") == _snapshot(store / "labels.orig")

    def test_masks_and_other_trees_are_never_touched(self, store: Path) -> None:
        RelabelLabels(root=str(store), label_map=FAMILY).run()

        assert (store / "masks/train/lmr/sample_000000.png").read_bytes() == b"mask"

    def test_an_interrupted_remap_is_finished_by_the_next_run(self, store: Path) -> None:
        RelabelLabels(root=str(store), label_map=FAMILY).run()
        (store / "labels.new" / "stale").mkdir(parents=True)  # what a run killed while writing leaves

        RelabelLabels(root=str(store), label_map={"*": "signal"}).run()

        assert not (store / "labels.new").exists()
        assert _lines(store / "labels/train/lmr/sample_000000.txt")[0].startswith("0 ")

    def test_the_rfdetr_layout_is_remapped_split_by_split(self, tmp_path: Path) -> None:
        root = tmp_path / "pictures"
        for split in ("train", "val"):
            (root / split / "labels").mkdir(parents=True)
            (root / split / "labels" / "000000.txt").write_text("1 0.5 0.5 0.1 0.1\n")
        (root / "data.yaml").write_text(yaml.safe_dump({"names": {0: "APCO25", 1: "DMR"}}))

        result = RelabelLabels(root=str(root), label_map={"*": "signal"}).run()

        assert result["label_files"] == 2
        for split in ("train", "val"):
            assert _lines(root / split / "labels" / "000000.txt") == ["0 0.5 0.5 0.1 0.1"]
            assert _lines(root / split / "labels.orig" / "000000.txt") == ["1 0.5 0.5 0.1 0.1"]


def _coco(path: Path) -> Dict[str, object]:
    document: Dict[str, object] = {
        "images": [{"id": 0, "file_name": "000000.png", "width": 10, "height": 10}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 0, "bbox": [1, 1, 2, 2]},
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [3, 3, 2, 2]},
        ],
        "categories": [{"id": 0, "name": "APCO25"}, {"id": 1, "name": "DMR"}, {"id": 2, "name": "802.11ax HE"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return document


class TestCoco:
    def test_each_annotation_gets_its_new_class_and_names_its_original(self, tmp_path: Path) -> None:
        path = tmp_path / "train" / "_annotations.coco.json"
        _coco(path)
        original = path.read_bytes()

        RelabelLabels(root=str(tmp_path), label_map=FAMILY, classes={0: "WLAN", 1: "LMR"}).run()

        document = json.loads(path.read_text())
        assert [(a["category_id"], a[ORIGINAL_LABEL]) for a in document["annotations"]] == [
            (1, "APCO25"),
            (0, "802.11ax HE"),
        ]
        assert document["categories"] == [{"id": 0, "name": "WLAN"}, {"id": 1, "name": "LMR"}]
        assert (tmp_path / "train" / "_annotations.coco.orig.json").read_bytes() == original

    def test_annotations_folder_files_are_remapped_and_their_originals_never_count_as_files(
        self, tmp_path: Path
    ) -> None:
        _coco(tmp_path / "annotations" / "train.json")

        RelabelLabels(root=str(tmp_path), label_map={"*": "signal"}).run()
        result = RelabelLabels(root=str(tmp_path), label_map={"*": "signal"}).run()

        assert result["coco_files"] == 1
        document = json.loads((tmp_path / "annotations" / "train.json").read_text())
        assert {a["category_id"] for a in document["annotations"]} == {0}


class TestRestore:
    def test_restore_puts_the_folder_back_byte_for_byte_and_keeps_no_original(self, store: Path) -> None:
        _coco(store / "annotations" / "train.json")
        before = _snapshot(store)

        RelabelLabels(root=str(store), label_map=FAMILY).run()
        RelabelLabels(root=str(store), mode="restore").run()

        assert _snapshot(store) == before

    def test_restore_of_a_folder_never_remapped_changes_nothing(self, store: Path) -> None:
        before = _snapshot(store)

        RelabelLabels(root=str(store), mode="restore").run()

        assert _snapshot(store) == before


class TestRefusals:
    def test_a_map_that_leaves_an_original_name_out_is_refused_naming_it(self, store: Path) -> None:
        with pytest.raises(RelabelError, match="'802.11n HT'"):
            RelabelLabels(root=str(store), label_map={k: v for k, v in FAMILY.items() if k != "802.11n HT"}).run()

    def test_a_class_not_in_classes_is_refused(self, store: Path) -> None:
        with pytest.raises(RelabelError, match="'LMR', not in classes"):
            RelabelLabels(root=str(store), label_map=FAMILY, classes={0: "WLAN"}).run()

    def test_classes_naming_a_class_twice_is_refused(self, store: Path) -> None:
        with pytest.raises(RelabelError, match="twice"):
            RelabelLabels(root=str(store), label_map={"*": "x"}, classes={0: "x", 1: "x"}).run()

    def test_an_id_outside_the_original_vocabulary_is_refused_naming_the_file_and_line(self, tmp_path: Path) -> None:
        root = _yolo_folder(tmp_path, ["A"], {"train/a.txt": "0 0.5 0.5 0.1 0.1\n7 0.5 0.5 0.1 0.1\n"})

        with pytest.raises(RelabelError, match=r"a\.txt:2: class 7"):
            RelabelLabels(root=str(root), label_map={"A": "A"}).run()

    def test_a_class_id_that_is_no_number_is_refused(self, tmp_path: Path) -> None:
        root = _yolo_folder(tmp_path, ["A"], {"train/a.txt": "x 0.5 0.5 0.1 0.1\n"})

        with pytest.raises(RelabelError, match=r"a\.txt:1: the class id 'x'"):
            RelabelLabels(root=str(root), label_map={"A": "A"}).run()

    def test_a_label_tree_without_data_yaml_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "labels" / "train").mkdir(parents=True)

        with pytest.raises(RelabelError, match="no data.yaml"):
            RelabelLabels(root=str(tmp_path), label_map={"*": "x"}).run()

    def test_a_data_yaml_without_names_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "labels").mkdir()
        (tmp_path / "data.yaml").write_text("nc: 2\n")

        with pytest.raises(RelabelError, match="no 'names'"):
            RelabelLabels(root=str(tmp_path), label_map={"*": "x"}).run()

    def test_a_coco_annotation_outside_its_categories_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations" / "train.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"annotations": [{"category_id": 5}], "categories": [{"id": 0, "name": "A"}]}))

        with pytest.raises(RelabelError, match="category 5"):
            RelabelLabels(root=str(tmp_path), label_map={"A": "A"}).run()

    def test_a_coco_file_without_categories_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations" / "train.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"annotations": []}))

        with pytest.raises(RelabelError, match="no 'categories'"):
            RelabelLabels(root=str(tmp_path), label_map={"A": "A"}).run()

    def test_a_remap_without_a_map_is_refused(self, store: Path) -> None:
        with pytest.raises(RelabelError, match="label_map is required"):
            RelabelLabels(root=str(store)).run()

    def test_a_folder_without_labels_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RelabelError, match="no labels/ folder"):
            RelabelLabels(root=str(tmp_path), label_map={"*": "x"}).run()

    @pytest.mark.parametrize("root", ["", "  "])
    def test_no_root_is_refused(self, root: str) -> None:
        with pytest.raises(RelabelError, match="'root'"):
            RelabelLabels(root=root, label_map={"*": "x"}).run()

    def test_a_missing_folder_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RelabelError, match="is not a folder"):
            RelabelLabels(root=str(tmp_path / "absent"), label_map={"*": "x"}).run()

    def test_an_unset_variable_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAIDAR_RELABEL_UNSET", raising=False)

        with pytest.raises(RelabelError, match="unset environment variable"):
            RelabelLabels(root="$RAIDAR_RELABEL_UNSET/x", label_map={"*": "x"}).run()

    def test_the_refusal_is_a_value_error(self) -> None:
        assert issubclass(RelabelError, ValueError)

    def test_construction_touches_no_disk_and_repr_names_the_settings(self) -> None:
        relabel = RelabelLabels(root="/nowhere", mode="restore")

        assert repr(relabel) == "RelabelLabels(root='/nowhere', mode='restore')"


def test_a_variable_in_root_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _yolo_folder(tmp_path / "data", ["A"], {"train/a.txt": "0 0.5 0.5 0.1 0.1\n"})
    monkeypatch.setenv("RAIDAR_RELABEL_ROOT", str(tmp_path))

    result = RelabelLabels(root="$RAIDAR_RELABEL_ROOT/data", label_map={"A": "B"}).run()

    assert result["root"] == str(root)
    assert yaml.safe_load((root / "data.yaml").read_text())["names"] == ["B"]


EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "relabel.yaml"


def _run_example(folder: Path, scopes: List[str]) -> None:
    """The shipped example, its folder pointed at ``folder`` (the only line changed), run under ``scopes``."""
    import confluid
    from confluid import flow
    from confluid.fluid import Fluid

    text = EXAMPLE.read_text().replace("folder: ./runs/dataset_merged", f"folder: {folder}")
    assert str(folder) in text
    runnable = confluid.load(text, scopes=scopes)["runnable"]
    (flow(runnable) if isinstance(runnable, Fluid) else runnable).run()


class TestTheExample:
    def test_it_maps_every_name_to_its_family(self, store: Path) -> None:
        _run_example(store, [])

        assert yaml.safe_load((store / "data.yaml").read_text())["names"] == ["WLAN", "LMR"]
        assert _lines(store / "labels/train/lmr/sample_000000.txt")[0].startswith("1 ")

    def test_one_class_maps_every_name_to_signal(self, store: Path) -> None:
        _run_example(store, ["one_class"])

        assert yaml.safe_load((store / "data.yaml").read_text())["names"] == ["signal"]

    def test_original_puts_the_folder_back(self, store: Path) -> None:
        before = _snapshot(store)
        _run_example(store, [])

        _run_example(store, ["original"])

        assert _snapshot(store) == before
