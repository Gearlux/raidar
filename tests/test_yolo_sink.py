"""Tests for :class:`raidar.yolo_sink.YoloSink`.

The sink is a RecordStream DataSink (open/write/flush/close) written one split at a
time; repeated runs into the same output_dir merge data.yaml + metadata.jsonl.
``recordstream run`` drives it via DatasetProcessor(stream, sink); the tests
exercise it directly (and through a Stream) since that's the same write path.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pytest
import torch
import yaml
from PIL import Image, ImageDraw
from recordstream import Boxes
from recordstream import Image as SFImage
from recordstream import Record
from recordstream.core import Stream

from raidar.yolo_sink import POLYGONS, YoloSink, _bridged_ring, _jsonable_metadata, _to_uint8_hwc, _yolo_lines


def _make_record(
    height: int = 400, width: int = 400, boxes_xyxy: Any = None, labels: Any = None, metadata: Any = None
) -> Record:
    """Detection record matching CocoToTorchVisionDetection: Image [3,H,W] under "image"; Boxes
    (boxes [N,4] xyxy-pixel + labels [N] int64) under "target"; metadata entries as plain keys."""
    boxes: Any = (
        torch.zeros((0, 4), dtype=torch.float32)
        if boxes_xyxy is None
        else torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    )
    lbls: Any = torch.zeros((0,), dtype=torch.int64) if labels is None else torch.as_tensor(labels, dtype=torch.int64)
    record: Record = {
        "image": SFImage(torch.rand(3, height, width).numpy(), layout="CHW"),
        "target": Boxes(boxes=boxes, labels=lbls),
    }
    record.update(metadata or {})
    return record


def _convert(
    stream: Any, out: Path, *, split: Literal["train", "val", "test"] = "train", class_names: Any = None, **kwargs: Any
) -> None:
    """Stream a stream into a YoloSink exactly as DatasetProcessor would."""
    sink = YoloSink(
        output_dir=str(out),
        class_names=class_names or {0: "background", 1: "drone-signal"},
        split=split,
        **kwargs,
    )
    with sink:  # open() on enter, close() on exit
        for record in stream:
            sink.write(record)
        sink.flush()


# --- unit helpers -----------------------------------------------------------


def test_to_uint8_hwc_from_chw_float() -> None:
    arr = torch.zeros(3, 8, 8, dtype=torch.float32)
    arr[0] = 1.0
    out = _to_uint8_hwc(arr)
    assert out.shape == (8, 8, 3) and out.dtype.name == "uint8"
    assert out[0, 0].tolist() == [255, 0, 0]


def test_yolo_lines_known_box_normalises_correctly() -> None:
    target = {"boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]), "labels": torch.tensor([1])}
    assert _yolo_lines(target, height=400, width=400) == ["1 0.375000 0.375000 0.250000 0.250000"]


def test_yolo_lines_empty_target_returns_empty() -> None:
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
    assert _yolo_lines(target, 400, 400) == []


# --- single-split end-to-end ------------------------------------------------


def _train_stream() -> Stream:
    return Stream(
        source=[
            _make_record(boxes_xyxy=[[100, 100, 200, 200]], labels=[1], metadata={"pack_id": "PACK_A"}),
            _make_record(boxes_xyxy=[[50, 50, 150, 150]], labels=[1], metadata={"pack_id": "PACK_B"}),
        ]
    )


def test_writes_data_yaml_structure(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert Path(data["path"]).is_absolute()
    assert data["train"] == "images/train"
    assert data["nc"] == 2
    assert data["channels"] == 3
    assert data["names"] == {0: "background", 1: "drone-signal"}


def test_writes_yolo_labels_and_coco(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out)
    assert (out / "labels" / "train" / "000000.txt").read_text().strip() == "1 0.375000 0.375000 0.250000 0.250000"
    coco = json.loads((out / "annotations" / "train.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 2
    first = coco["annotations"][0]
    assert first["bbox"] == [100.0, 100.0, 100.0, 100.0] and first["area"] == 10000.0
    assert {c["id"]: c["name"] for c in coco["categories"]} == {0: "background", 1: "drone-signal"}


def test_empty_target_writes_empty_label_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(Stream(source=[_make_record()]), out, class_names={1: "drone-signal"})
    label = out / "labels" / "train" / "000000.txt"
    assert label.exists() and label.read_text() == ""


def test_image_pixel_dims_match_input(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(Stream(source=[_make_record(height=128, width=256, boxes_xyxy=[[0, 0, 64, 64]], labels=[1])]), out)
    assert Image.open(out / "images" / "train" / "000000.png").size == (256, 128)  # PIL (w, h)


def test_jpg_quality_knob(tmp_path: Path) -> None:
    record = _make_record(boxes_xyxy=[[0, 0, 10, 10]], labels=[1])
    _convert(Stream(source=[record]), tmp_path / "hi", image_format="jpg", jpeg_quality=95, write_coco=False)
    _convert(Stream(source=[record]), tmp_path / "lo", image_format="jpg", jpeg_quality=10, write_coco=False)
    hi = (tmp_path / "hi" / "images" / "train" / "000000.jpg").stat().st_size
    lo = (tmp_path / "lo" / "images" / "train" / "000000.jpg").stat().st_size
    assert lo < hi


def test_metadata_jsonl_provenance(tmp_path: Path) -> None:
    out = tmp_path / "out"
    stream = Stream(
        source=[_make_record(metadata={"pack_id": "RFUAV:PACK_X", "iq_file": "x.iq", "window_start_sample": 1234})]
    )
    _convert(stream, out, class_names={1: "drone-signal"})
    lines = (out / "metadata.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["index"] == 0 and entry["split"] == "train"
    assert entry["image"] == "images/train/000000.png"
    assert entry["source_metadata"]["pack_id"] == "RFUAV:PACK_X"
    assert entry["source_metadata"]["window_start_sample"] == 1234


def test_jsonable_metadata_sanitizes_complex_and_large_arrays() -> None:
    # Live crash: Record.metadata carried StreamStudio's internal `__taidal_stash_*` snapshot — a
    # complex IQ array — so `json.dumps(...)` raised "Object of type complex is not JSON serializable".
    meta = {
        "pack_id": "X",  # plain provenance kept as-is
        "snr_db": np.float32(12.5),  # numpy scalar -> python float
        "center_freq": 1 + 2j,  # complex scalar -> string
        "boxes": np.array([[0, 0, 4, 4]]),  # small array -> listed element-wise
        "__taidal_stash_436:424_input": np.arange(2048, dtype=np.complex64),  # large complex -> summary
    }
    out = _jsonable_metadata(meta)
    json.dumps(out)  # MUST be fully encodable — this is exactly what YoloSink.write does
    assert out["pack_id"] == "X"
    assert out["snr_db"] == 12.5
    assert isinstance(out["center_freq"], str)
    assert out["boxes"] == [[0, 0, 4, 4]]
    assert out["__taidal_stash_436:424_input"].startswith("<array shape=(2048,)")


def test_jsonable_metadata_small_complex_array_becomes_strings() -> None:
    # A SMALL complex array is listed element-wise and each complex is stringified (NOT left as a
    # Python complex, which json.dumps rejects — the un-recursed `.tolist()` bug).
    out = _jsonable_metadata({"iq": np.array([1 + 1j, 2 - 2j], dtype=np.complex64)})
    json.dumps(out)
    assert all(isinstance(x, str) for x in out["iq"])


def test_metadata_jsonl_survives_complex_stash_array(tmp_path: Path) -> None:
    # End-to-end regression: a record carrying a complex `__taidal_stash_*` snapshot must write
    # without crashing the sink, with the big array summarised (not dumped) into the label JSON.
    out = tmp_path / "out"
    stream = Stream(source=[_make_record(metadata={"__taidal_stash_x_input": np.arange(4096, dtype=np.complex64)})])
    _convert(stream, out, class_names={1: "drone-signal"})
    entry = json.loads((out / "metadata.jsonl").read_text().strip())
    assert entry["source_metadata"]["__taidal_stash_x_input"].startswith("<array shape=(4096,)")


# --- multi-split merge / append (the key new behaviour) ---------------------


def test_second_split_run_merges_data_yaml_and_preserves_first(tmp_path: Path) -> None:
    """Running val into the SAME dir after train must keep train + add val."""
    out = tmp_path / "out"
    _convert(_train_stream(), out, split="train")
    _convert(Stream(source=[_make_record(boxes_xyxy=[[0, 0, 40, 40]], labels=[1])]), out, split="val")

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["train"] == "images/train" and data["val"] == "images/val"  # both present
    # First split's files survive the second run.
    assert (out / "images" / "train" / "000001.png").exists()
    assert (out / "images" / "val" / "000000.png").exists()
    # metadata.jsonl carries both splits (2 train + 1 val).
    splits = [json.loads(line)["split"] for line in (out / "metadata.jsonl").read_text().strip().splitlines()]
    assert sorted(splits) == ["train", "train", "val"]


def test_rerunning_same_split_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out, split="train")  # 2 images
    _convert(Stream(source=[_make_record()]), out, split="train")  # re-run: 1 image
    # The split dir is replaced, not appended — only the new single image remains.
    pngs = sorted(p.name for p in (out / "images" / "train").glob("*.png"))
    assert pngs == ["000000.png"]
    splits = [json.loads(line)["split"] for line in (out / "metadata.jsonl").read_text().strip().splitlines()]
    assert splits == ["train"]  # the stale 2-image train metadata was pruned


def test_three_splits_accumulate(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for split in ("train", "val", "test"):
        _convert(Stream(source=[_make_record(boxes_xyxy=[[1, 2, 3, 4]], labels=[1])]), out, split=split)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert {"train", "val", "test"} <= set(data)
    assert json.loads((out / "annotations" / "test.json").read_text())["images"]


# --- construction / validation ----------------------------------------------


def test_zero_arg_construction_then_requires_class_names() -> None:
    sink = YoloSink()  # zero-arg construction must work (lazy convention)
    with pytest.raises(ValueError, match="class_names"):
        sink.open()


def test_rejects_invalid_jpeg_quality_on_open(tmp_path: Path) -> None:
    sink = YoloSink(output_dir=str(tmp_path), class_names={1: "x"}, jpeg_quality=120)
    with pytest.raises(ValueError, match="jpeg_quality"):
        sink.open()


# --- the rfdetr layout (2026-10-06) ---------------------------------------
#
# Measured before the knob existed, with RF-DETR 1.9.3's own reader on three real windows written
# by this sink: ``is_valid_yolo_dataset(images/<split> layout) -> False``,
# ``is_valid_yolo_dataset(<split>/images layout) -> True``. The default layout is what
# Ultralytics reads and stays as it was; ``layout: rfdetr`` writes the second one.


def test_rfdetr_layout_puts_each_split_in_its_own_folder(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out, layout="rfdetr")
    assert (out / "train" / "images" / "000000.png").exists()
    assert (out / "train" / "labels" / "000000.txt").read_text().strip() == "1 0.375000 0.375000 0.250000 0.250000"
    coco = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 2
    assert not (out / "images").exists() and not (out / "labels").exists() and not (out / "annotations").exists()
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["train"] == "train/images" and data["names"] == {0: "background", 1: "drone-signal"}
    meta = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
    assert meta[0]["image"] == "train/images/000000.png" and meta[0]["label"] == "train/labels/000000.txt"


def test_rfdetr_layout_second_split_merges_and_a_rerun_is_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out, layout="rfdetr")
    _convert(Stream(source=[_make_record(boxes_xyxy=[[0, 0, 40, 40]], labels=[1])]), out, split="val", layout="rfdetr")
    _convert(_train_stream(), out, layout="rfdetr")
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["train"] == "train/images" and data["val"] == "val/images"
    assert sorted(p.name for p in (out / "train" / "images").iterdir()) == ["000000.png", "000001.png"]
    assert sorted(p.name for p in (out / "val" / "images").iterdir()) == ["000000.png"]
    meta = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
    assert [m["split"] for m in meta] == ["val", "train", "train"]


def test_the_rfdetr_layout_is_what_rfdetr_reads_and_the_default_is_not(tmp_path: Path) -> None:
    yolo = pytest.importorskip("rfdetr.datasets.yolo")
    for layout in ("ultralytics", "rfdetr"):
        out = tmp_path / layout
        _convert(_train_stream(), out, layout=layout)
        _convert(
            Stream(source=[_make_record(boxes_xyxy=[[0, 0, 40, 40]], labels=[1])]), out, split="val", layout=layout
        )
    assert yolo.is_valid_yolo_dataset(str(tmp_path / "rfdetr")) is True
    assert yolo.is_valid_yolo_dataset(str(tmp_path / "ultralytics")) is False


def test_an_unknown_layout_is_refused(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="ultralytics"):  # the Literal, checked at construction by confluid
        YoloSink(output_dir=str(tmp_path / "out"), class_names={0: "x"}, layout="coco")  # type: ignore[arg-type]
    sink = YoloSink(output_dir=str(tmp_path / "out"), class_names={0: "x"})
    sink.layout = "coco"  # type: ignore[assignment]  # set after construction: open() refuses it too
    with pytest.raises(ValueError, match="layout"):
        sink.open()


# --- append mode and facts (2026-10-06) ------------------------------------
#
# Today a rerun of a split wipes it and restarts the index at 0 (the `replace` mode, kept as the
# default). `append` continues a folder: the next source lands after the last stem, the metadata
# and the COCO sidecar grow, and `facts` lists the distinct values of chosen entries (a record's
# picture scale) in data.yaml, so a folder of mixed scales says which it holds.


def _scaled_stream(khz: float, n: int = 2) -> Stream:
    return Stream(
        source=[
            _make_record(
                boxes_xyxy=[[10, 10, 50, 50]], labels=[1], metadata={"scale": {"khz_per_px": khz, "us_per_px": 19.5}}
            )
            for _ in range(n)
        ]
    )


def test_append_continues_the_index_and_grows_the_sidecars(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out, layout="rfdetr")
    _convert(_scaled_stream(78.1), out, layout="rfdetr", mode="append", facts=["scale"])
    names = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert names == ["000000.png", "000001.png", "000002.png", "000003.png"]
    meta = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
    assert [m["index"] for m in meta] == [0, 1, 2, 3]
    coco = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert [img["id"] for img in coco["images"]] == [0, 1, 2, 3]
    assert [ann["id"] for ann in coco["annotations"]] == [0, 1, 2, 3] and coco["annotations"][3]["image_id"] == 3


def test_facts_list_the_distinct_values_across_appends(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_scaled_stream(78.1), out, layout="rfdetr", facts=["scale"])
    _convert(_scaled_stream(78.1), out, layout="rfdetr", mode="append", facts=["scale"])
    _convert(_scaled_stream(39.1), out, layout="rfdetr", mode="append", facts=["scale"])
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["facts"] == {
        "scale": [{"khz_per_px": 78.1, "us_per_px": 19.5}, {"khz_per_px": 39.1, "us_per_px": 19.5}]
    }
    assert sorted(p.name for p in (out / "train" / "images").iterdir()) == [f"{i:06d}.png" for i in range(6)]


def test_a_record_without_the_fact_adds_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out, facts=["scale"])
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["facts"] == {"scale": []}


def test_append_refuses_a_different_vocabulary(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out)
    with pytest.raises(ValueError, match="drone-signal.*other"):
        _convert(_train_stream(), out, mode="append", class_names={0: "background", 1: "other"})
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["000000.png", "000001.png"]


def test_append_on_an_empty_folder_is_a_plain_write(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out, mode="append")
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["000000.png", "000001.png"]


def test_replace_still_wipes_the_split(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_stream(), out)
    _convert(Stream(source=[_make_record(boxes_xyxy=[[0, 0, 40, 40]], labels=[1])]), out)
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["000000.png"]


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="replace"):
        YoloSink(output_dir=str(tmp_path / "out"), class_names={0: "x"}, mode="merge")  # type: ignore[arg-type]


# --- instance masks: polygons in `Boxes.extras["polygons"]` (2026-10-07) -----------------------------
#
# A producer may put each instance's polygons in the target's extras. The sink then writes the
# YOLO-seg form (`class x1 y1 x2 y2 ...`, ONE ring per instance — several polygons are joined into
# one by zero-width bridges) and the COCO sidecar's `segmentation`. A split holding any ring holds no
# box line: Ultralytics reads a label file as segments as soon as one line is longer than a box, so
# a box line beside a ring is mis-read there. The canvas is 64 x 32 so every pixel coordinate
# normalises to an exact 6-decimal number (PIL, which RF-DETR rasterises with, truncates floats).

W, H = 64, 32
TRIANGLE = [[8, 4], [40, 4], [24, 28]]
R1 = [[2, 2], [10, 2], [10, 10], [2, 10]]
R2 = [[20, 3], [30, 3], [30, 12], [20, 12]]
R3 = [[40, 16], [56, 16], [56, 28], [40, 28]]
THREE_RECTS_RING = [
    [10, 2], [10, 10], [2, 10], [2, 2], [10, 2],
    [20, 3], [30, 3], [30, 12], [20, 12], [20, 3],
    [40, 16], [56, 16], [56, 28], [40, 28], [40, 16],
    [20, 3],
]  # fmt: skip
BRIDGES = [([10, 2], [20, 3]), ([20, 3], [40, 16])]


def _seg_record(boxes: Any, labels: Any, polygons: Any = None, **extras: Any) -> Record:
    """A 64 x 32 record whose target carries `polygons` (when given) in its extras."""
    record = _make_record(height=H, width=W, boxes_xyxy=boxes, labels=labels)
    target = record["target"]
    assert isinstance(target, Boxes)
    if polygons is not None:
        target.extras[POLYGONS] = polygons
    target.extras.update(extras)
    return record


def _seg_records() -> List[Record]:
    """Instance 0 one triangle, instance 1 three disjoint rectangles, instance 2 no mask of its own."""
    return [
        _seg_record(
            [[8, 4, 40, 28], [2, 2, 56, 28], [44, 2, 60, 10]],
            [0, 1, 1],
            [[TRIANGLE], [R1, R2, R3], []],
        )
    ]


def _raster(polygon: Any) -> np.ndarray:
    """A polygon filled the way RF-DETR fills one (`ImageDraw.polygon(..., fill=1)`)."""
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in np.asarray(polygon).tolist()], fill=1)
    return np.array(mask, dtype=bool)


def _segment_distance(point: Any, start: Any, stop: Any) -> float:
    p, a, b = (np.asarray(v, dtype=float) for v in (point, start, stop))
    t = float(np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * (b - a))))


def _shoelace(polygon: Any) -> float:
    xy = np.asarray(polygon, dtype=float)
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def _label_lines(out: Path, stem: str = "000000", layout: str = "rfdetr") -> List[List[str]]:
    folder = out / "train" / "labels" if layout == "rfdetr" else out / "labels" / "train"
    return [line.split() for line in (folder / f"{stem}.txt").read_text().splitlines()]


def _coco(out: Path, layout: str = "rfdetr") -> Dict[str, Any]:
    path = out / "train" / "_annotations.coco.json" if layout == "rfdetr" else out / "annotations" / "train.json"
    sidecar: Dict[str, Any] = json.loads(path.read_text())
    return sidecar


def test_the_key_producers_import_is_exported_from_the_package() -> None:
    import raidar

    assert raidar.POLYGONS == POLYGONS == "polygons"


class TestBoxOnlyRecordsAreUnchanged:
    def test_a_box_only_record_is_written_exactly_as_before(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        records = [
            _seg_record([[8, 4, 40, 20], [0, 0, 64, 32]], [0, 1], other=[1, 2]),  # other extras ride along
            _seg_record([], []),
        ]
        _convert(Stream(source=records), out, class_names={0: "a", 1: "b"}, layout="rfdetr")
        # the text the sink wrote before the polygons existed, byte for byte
        assert (out / "train" / "labels" / "000000.txt").read_text() == (
            "0 0.375000 0.375000 0.500000 0.500000\n1 0.500000 0.500000 1.000000 1.000000"
        )
        assert (out / "train" / "labels" / "000001.txt").read_text() == ""
        first, second = _coco(out)["annotations"]
        assert first == {
            "id": 0, "image_id": 0, "category_id": 0, "bbox": [8.0, 4.0, 32.0, 16.0],
            "area": 512.0, "iscrowd": 0, "segmentation": [],
        }  # fmt: skip
        assert second["area"] == 2048.0 and second["segmentation"] == []

    def test_a_polygons_entry_without_a_polygon_is_a_box_record(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        record = _seg_record([[8, 4, 40, 20]], [0], [[]])
        _convert(Stream(source=[record]), out, class_names={0: "a"}, layout="rfdetr")
        assert (out / "train" / "labels" / "000000.txt").read_text() == "0 0.375000 0.375000 0.500000 0.500000"
        assert _coco(out)["annotations"][0]["segmentation"] == []


class TestOnePolygon:
    def test_one_polygon_is_one_seg_line_and_a_box_without_a_mask_is_its_ring(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        record = _seg_record([[8, 4, 40, 28], [44, 2, 60, 10]], [0, 1], [[TRIANGLE], None])
        _convert(Stream(source=[record]), out, class_names={0: "a", 1: "b"}, layout="rfdetr")
        assert (out / "train" / "labels" / "000000.txt").read_text().splitlines() == [
            "0 0.125000 0.125000 0.625000 0.125000 0.375000 0.875000",
            "1 0.687500 0.062500 0.937500 0.062500 0.937500 0.312500 0.687500 0.312500",
        ]

    def test_the_ultralytics_layout_writes_the_same_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(Stream(source=_seg_records()), out, class_names={0: "a", 1: "b"})
        lines = _label_lines(out, layout="ultralytics")
        assert [len(line) for line in lines] == [1 + 6, 1 + 2 * 16, 1 + 8]
        assert lines[0][0] == "0" and lines[2][0] == "1"

    def test_the_coco_sidecar_carries_the_polygons_and_their_area(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(Stream(source=_seg_records()), out, class_names={0: "a", 1: "b"}, layout="rfdetr")
        triangle, rects, bare = _coco(out)["annotations"]
        assert triangle["segmentation"] == [[8.0, 4.0, 40.0, 4.0, 24.0, 28.0]]
        assert triangle["area"] == 384.0 and triangle["bbox"] == [8.0, 4.0, 32.0, 24.0]
        assert rects["segmentation"] == [[2.0, 2.0, 10.0, 2.0, 10.0, 10.0, 2.0, 10.0], sum(R2, []), sum(R3, [])]
        assert rects["area"] == 8 * 8 + 10 * 9 + 16 * 12  # the polygons' shoelace areas, summed
        assert bare["segmentation"] == [[44.0, 2.0, 60.0, 2.0, 60.0, 10.0, 44.0, 10.0]] and bare["area"] == 128.0


class TestSeveralPolygonsOfOneInstance:
    def test_the_ring_walks_each_polygon_and_back_over_its_bridges(self) -> None:
        ring = _bridged_ring([np.asarray(p, dtype=float) for p in (R1, R2, R3)])
        assert ring.tolist() == [[float(x), float(y)] for x, y in THREE_RECTS_RING]

    def test_each_bridge_is_walked_twice_and_adds_no_area(self) -> None:
        ring = _bridged_ring([np.asarray(p, dtype=float) for p in (R1, R2, R3)])
        edges = [(tuple(a), tuple(b)) for a, b in zip(ring.tolist(), np.roll(ring, -1, axis=0).tolist())]
        both_ways = {(a, b) for a, b in edges if (b, a) in edges and a != b}
        assert both_ways == {(tuple(map(float, a)), tuple(map(float, b))) for a, b in BRIDGES} | {
            (tuple(map(float, b)), tuple(map(float, a))) for a, b in BRIDGES
        }
        assert _shoelace(ring) == _shoelace(R1) + _shoelace(R2) + _shoelace(R3)

    def test_the_ring_fills_the_union_and_the_rest_lies_on_the_bridges(self) -> None:
        ring = _bridged_ring([np.asarray(p, dtype=float) for p in (R1, R2, R3)])
        union = _raster(R1) | _raster(R2) | _raster(R3)
        filled = _raster(ring)
        assert not (union & ~filled).any()  # no pixel of any rectangle is lost
        extra = np.argwhere(filled & ~union)[:, ::-1]  # (x, y) of every pixel the bridges add
        assert 0 < len(extra) <= sum(np.ceil(np.linalg.norm(np.subtract(b, a))) + 1 for a, b in BRIDGES)
        for point in extra:
            assert min(_segment_distance(point, a, b) for a, b in BRIDGES) <= 1.0, point.tolist()

    def test_the_polygons_are_ordered_so_each_bridge_is_short(self) -> None:
        def square(x: float) -> List[List[float]]:
            return [[x, 2.0], [x + 4, 2.0], [x + 4, 6.0], [x, 6.0]]

        a, b, c, d = square(2.0), square(16.0), square(30.0), square(48.0)  # left to right
        ring = _bridged_ring([np.asarray(p, dtype=float) for p in (a, d, c, b)])  # listed far end first
        points = [tuple(p) for p in ring.tolist()]
        first = [min(points.index((x, y)) for x, y in square_) for square_ in (a, b, c, d)]
        assert first == sorted(first)  # walked left to right: each bridge reaches the nearest polygon
        assert len(ring) == 4 * 5 + 2  # four walks, and the two middle bridge vertices back

    def test_a_lone_polygon_is_its_own_ring(self) -> None:
        assert _bridged_ring([np.asarray(TRIANGLE, dtype=float)]).tolist() == [[8.0, 4.0], [40.0, 4.0], [24.0, 28.0]]

    def test_the_sink_writes_that_ring_as_one_line(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        record = _seg_record([[2, 2, 56, 28]], [1], [[R1, R2, R3]])
        _convert(Stream(source=[record]), out, class_names={0: "a", 1: "b"}, layout="rfdetr")
        (line,) = _label_lines(out)
        numbers = np.asarray(line[1:], dtype=float).reshape(-1, 2) * [W, H]
        assert line[0] == "1" and numbers.tolist() == [[float(x), float(y)] for x, y in THREE_RECTS_RING]
        (annotation,) = _coco(out)["annotations"]
        assert len(annotation["segmentation"]) == 3  # COCO keeps the polygons apart


class TestOneFormPerSplit:
    def _lines(self, out: Path, layout: str = "rfdetr") -> List[List[str]]:
        folder = out / "train" / "labels" if layout == "rfdetr" else out / "labels" / "train"
        return [line.split() for path in sorted(folder.glob("*.txt")) for line in path.read_text().splitlines()]

    @pytest.mark.parametrize("polygons_first", [True, False])
    @pytest.mark.parametrize("layout", ["rfdetr", "ultralytics"])
    def test_one_record_with_polygons_makes_every_line_a_ring(
        self, tmp_path: Path, polygons_first: bool, layout: str
    ) -> None:
        out = tmp_path / "out"
        with_polygons = _seg_record([[8, 4, 40, 28]], [0], [[TRIANGLE]])
        without = _seg_record([[0, 0, 64, 32], [8, 4, 40, 20]], [1, 0])
        records = [with_polygons, without] if polygons_first else [without, with_polygons]
        _convert(Stream(source=records), out, class_names={0: "a", 1: "b"}, layout=layout)
        lines = self._lines(out, layout)
        assert len(lines) == 3 and all(len(line) > 5 for line in lines)  # no 5-field box line left
        rings = {tuple(line) for line in lines}
        assert (
            "1",
            "0.000000",
            "0.000000",
            "1.000000",
            "0.000000",
            "1.000000",
            "1.000000",
            "0.000000",
            "1.000000",
        ) in rings
        assert (
            "0",
            "0.125000",
            "0.125000",
            "0.625000",
            "0.125000",
            "0.625000",
            "0.625000",
            "0.125000",
            "0.625000",
        ) in rings
        annotations = _coco(out, layout)["annotations"]
        assert all(a["segmentation"] for a in annotations)
        full = next(a for a in annotations if a["bbox"] == [0.0, 0.0, 64.0, 32.0])
        assert full["segmentation"] == [[0.0, 0.0, 64.0, 0.0, 64.0, 32.0, 0.0, 32.0]] and full["area"] == 2048.0

    def test_an_append_of_polygons_turns_the_earlier_box_lines_into_rings(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(Stream(source=[_seg_record([[8, 4, 40, 20]], [0])]), out, class_names={0: "a"}, layout="rfdetr")
        assert [len(line) for line in self._lines(out)] == [5]
        _convert(
            Stream(source=[_seg_record([[8, 4, 40, 28]], [0], [[TRIANGLE]])]),
            out, class_names={0: "a"}, layout="rfdetr", mode="append",
        )  # fmt: skip
        assert [len(line) for line in self._lines(out)] == [9, 7]
        assert all(a["segmentation"] for a in _coco(out)["annotations"])

    def test_an_append_of_boxes_to_a_split_of_rings_writes_rings(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(
            Stream(source=[_seg_record([[8, 4, 40, 28]], [0], [[TRIANGLE]])]),
            out,
            class_names={0: "a"},
            layout="rfdetr",
        )
        _convert(
            Stream(source=[_seg_record([[8, 4, 40, 20]], [0])]),
            out, class_names={0: "a"}, layout="rfdetr", mode="append",
        )  # fmt: skip
        assert [len(line) for line in self._lines(out)] == [7, 9]
        assert all(a["segmentation"] for a in _coco(out)["annotations"])

    def test_an_append_of_boxes_to_boxes_stays_boxes(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(Stream(source=[_seg_record([[8, 4, 40, 20]], [0])]), out, class_names={0: "a"}, layout="rfdetr")
        _convert(
            Stream(source=[_seg_record([[0, 0, 64, 32]], [0])]),
            out, class_names={0: "a"}, layout="rfdetr", mode="append",
        )  # fmt: skip
        assert [len(line) for line in self._lines(out)] == [5, 5]
        assert all(a["segmentation"] == [] for a in _coco(out)["annotations"])

    def test_the_coco_sidecar_is_left_alone_without_coco(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _convert(Stream(source=_seg_records()), out, class_names={0: "a", 1: "b"}, layout="rfdetr", write_coco=False)
        assert not (out / "train" / "_annotations.coco.json").exists()
        assert len(self._lines(out)) == 3


class TestWhatTheTargetMayCarry:
    def test_a_dict_target_carries_polygons_too(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        target = {
            "boxes": np.asarray([[8, 4, 40, 28]], dtype=np.float32),
            "labels": np.asarray([0]),
            POLYGONS: [[TRIANGLE]],
        }
        record: Record = {
            "image": SFImage(np.zeros((3, H, W), dtype=np.float32), layout="CHW"),
            "target": target,  # type: ignore[dict-item]
        }
        _convert(Stream(source=[record]), out, class_names={0: "a"}, layout="rfdetr")
        assert (
            out / "train" / "labels" / "000000.txt"
        ).read_text() == "0 0.125000 0.125000 0.625000 0.125000 0.375000 0.875000"

    def test_polygons_may_be_arrays(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        polygons = [[np.asarray(TRIANGLE, dtype=np.float32)]]
        _convert(
            Stream(source=[_seg_record([[8, 4, 40, 28]], [0], polygons)]), out, class_names={0: "a"}, layout="rfdetr"
        )
        assert (out / "train" / "labels" / "000000.txt").read_text().startswith("0 0.125000 0.125000")

    def test_a_list_of_the_wrong_length_is_refused(self, tmp_path: Path) -> None:
        record = _seg_record([[8, 4, 40, 28], [0, 0, 4, 4]], [0, 0], [[TRIANGLE]])
        with pytest.raises(ValueError, match="record 0.*1 entries.*2 boxes"):
            _convert(Stream(source=[record]), tmp_path / "out", class_names={0: "a"})

    def test_a_polygon_needs_three_points(self, tmp_path: Path) -> None:
        record = _seg_record([[8, 4, 40, 28]], [0], [[[[8, 4], [40, 4]]]])
        with pytest.raises(ValueError, match="record 0.*instance 0.*polygon 0.*at least 3"):
            _convert(Stream(source=[record]), tmp_path / "out", class_names={0: "a"})

    def test_a_polygon_is_a_list_of_x_y_points(self, tmp_path: Path) -> None:
        record = _seg_record([[8, 4, 40, 28]], [0], [[[8, 4, 40, 4, 24, 28]]])
        with pytest.raises(ValueError, match="record 0.*instance 0.*polygon 0"):
            _convert(Stream(source=[record]), tmp_path / "out", class_names={0: "a"})

    def test_a_record_after_the_first_names_its_own_index_in_the_refusal(self, tmp_path: Path) -> None:
        records = [_seg_record([[8, 4, 40, 28]], [0]), _seg_record([[8, 4, 40, 28]], [0], [[[[1, 1], [2, 2]]]])]
        with pytest.raises(ValueError, match="record 1"):
            _convert(Stream(source=records), tmp_path / "out", class_names={0: "a"})


class TestRfdetrReadsTheFolderAsSegmentation:
    """RF-DETR 1.9.3's own YOLO and COCO readers, run over what the sink wrote."""

    def _folder(self, tmp_path: Path) -> Path:
        out = tmp_path / "out"
        _convert(Stream(source=_seg_records()), out, class_names={0: "a", 1: "b"}, layout="rfdetr")
        return out

    def test_the_yolo_reader_yields_masks_that_match_the_polygons(self, tmp_path: Path) -> None:
        yolo = pytest.importorskip("rfdetr.datasets.yolo")
        out = self._folder(tmp_path)
        dataset = yolo.YoloDetection(
            img_folder=str(out / "train" / "images"),
            lb_folder=str(out / "train" / "labels"),
            data_file=str(out / "data.yaml"),
            include_masks=True,
        )
        _image, target = dataset[0]
        masks = target["masks"].numpy()
        assert masks.shape == (3, H, W) and target["labels"].tolist() == [0, 1, 1]
        assert target["boxes"].tolist() == [[8.0, 4.0, 40.0, 28.0], [2.0, 2.0, 56.0, 28.0], [44.0, 2.0, 60.0, 10.0]]
        assert (masks[0] == _raster(TRIANGLE)).all()  # one polygon: the polygon, pixel for pixel
        union = _raster(R1) | _raster(R2) | _raster(R3)
        assert not (union & ~masks[1]).any()  # three polygons: the union ...
        extra = np.argwhere(masks[1] & ~union)[:, ::-1]
        assert all(min(_segment_distance(p, a, b) for a, b in BRIDGES) <= 1.0 for p in extra)  # ... and the bridges
        assert (masks[2] == _raster([[44, 2], [60, 2], [60, 10], [44, 10]])).all()  # no polygon: the box

    def test_a_box_only_folder_has_the_box_masks_it_always_had(self, tmp_path: Path) -> None:
        yolo = pytest.importorskip("rfdetr.datasets.yolo")
        out = tmp_path / "out"
        _convert(Stream(source=[_seg_record([[8, 4, 40, 20]], [0])]), out, class_names={0: "a"}, layout="rfdetr")
        dataset = yolo.YoloDetection(
            img_folder=str(out / "train" / "images"),
            lb_folder=str(out / "train" / "labels"),
            data_file=str(out / "data.yaml"),
            include_masks=True,
        )
        assert (dataset[0][1]["masks"].numpy()[0] == _raster([[8, 4], [40, 4], [40, 20], [8, 20]])).all()

    def test_the_coco_reader_decodes_the_segmentation(self, tmp_path: Path) -> None:
        coco = pytest.importorskip("rfdetr.datasets.coco")
        out = self._folder(tmp_path)
        annotations = _coco(out)["annotations"]
        image = Image.new("RGB", (W, H))
        _image, target = coco.ConvertCoco(include_masks=True)(image, {"image_id": 0, "annotations": annotations})
        masks = target["masks"].numpy()
        assert masks.shape == (3, H, W)
        assert int(masks[1].sum()) == 8 * 8 + 10 * 9 + 16 * 12 == annotations[1]["area"]  # the union of the three
        assert int(masks[2].sum()) == 16 * 8 == annotations[2]["area"]  # the box without a mask
        assert abs(int(masks[0].sum()) - annotations[0]["area"]) <= 40  # the triangle, up to its edge pixels
