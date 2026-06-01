"""Tests for :class:`raidar.ultralytics_sink.UltralyticsSink`.

The sink is a DataFlux DataSink (open/write/flush/close) written one split at a
time; repeated runs into the same output_dir merge data.yaml + metadata.jsonl.
``marainer convert`` drives it via DatasetProcessor(flux, sink); the tests
exercise it directly (and through a Flux) since that's the same write path.
"""

import json
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
import yaml
from dataflux.core import Flux
from dataflux.sample import Sample
from PIL import Image

from raidar.ultralytics_sink import UltralyticsSink, _to_uint8_hwc, _yolo_lines


def _make_sample(
    height: int = 400, width: int = 400, boxes_xyxy: Any = None, labels: Any = None, metadata: Any = None
) -> Sample:
    """Sample matching RegionsToDetectionBoxesOp: input [3,H,W] float32 [0,1];
    target {"boxes": [N,4] xyxy-pixel, "labels": [N] int64}."""
    boxes = (
        torch.zeros((0, 4), dtype=torch.float32)
        if boxes_xyxy is None
        else torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    )
    lbls = torch.zeros((0,), dtype=torch.int64) if labels is None else torch.as_tensor(labels, dtype=torch.int64)
    return Sample(input=torch.rand(3, height, width), target={"boxes": boxes, "labels": lbls}, metadata=metadata or {})


def _convert(
    flux: Any, out: Path, *, split: Literal["train", "val", "test"] = "train", class_names: Any = None, **kwargs: Any
) -> None:
    """Stream a flux into an UltralyticsSink exactly as DatasetProcessor would."""
    sink = UltralyticsSink(
        output_dir=str(out),
        class_names=class_names or {0: "background", 1: "drone-signal"},
        split=split,
        **kwargs,
    )
    with sink:  # open() on enter, close() on exit
        for sample in flux:
            sink.write(sample)
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


def _train_flux() -> Flux:
    return Flux(
        source=[
            _make_sample(boxes_xyxy=[[100, 100, 200, 200]], labels=[1], metadata={"pack_id": "PACK_A"}),
            _make_sample(boxes_xyxy=[[50, 50, 150, 150]], labels=[1], metadata={"pack_id": "PACK_B"}),
        ]
    )


def test_writes_data_yaml_structure(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_flux(), out)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert Path(data["path"]).is_absolute()
    assert data["train"] == "images/train"
    assert data["nc"] == 2
    assert data["channels"] == 3
    assert data["names"] == {0: "background", 1: "drone-signal"}


def test_writes_yolo_labels_and_coco(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(_train_flux(), out)
    assert (out / "labels" / "train" / "000000.txt").read_text().strip() == "1 0.375000 0.375000 0.250000 0.250000"
    coco = json.loads((out / "annotations" / "train.json").read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 2
    first = coco["annotations"][0]
    assert first["bbox"] == [100.0, 100.0, 100.0, 100.0] and first["area"] == 10000.0
    assert {c["id"]: c["name"] for c in coco["categories"]} == {0: "background", 1: "drone-signal"}


def test_empty_target_writes_empty_label_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(Flux(source=[_make_sample()]), out, class_names={1: "drone-signal"})
    label = out / "labels" / "train" / "000000.txt"
    assert label.exists() and label.read_text() == ""


def test_image_pixel_dims_match_input(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _convert(Flux(source=[_make_sample(height=128, width=256, boxes_xyxy=[[0, 0, 64, 64]], labels=[1])]), out)
    assert Image.open(out / "images" / "train" / "000000.png").size == (256, 128)  # PIL (w, h)


def test_jpg_quality_knob(tmp_path: Path) -> None:
    sample = _make_sample(boxes_xyxy=[[0, 0, 10, 10]], labels=[1])
    _convert(Flux(source=[sample]), tmp_path / "hi", image_format="jpg", jpeg_quality=95, write_coco=False)
    _convert(Flux(source=[sample]), tmp_path / "lo", image_format="jpg", jpeg_quality=10, write_coco=False)
    hi = (tmp_path / "hi" / "images" / "train" / "000000.jpg").stat().st_size
    lo = (tmp_path / "lo" / "images" / "train" / "000000.jpg").stat().st_size
    assert lo < hi


def test_metadata_jsonl_provenance(tmp_path: Path) -> None:
    out = tmp_path / "out"
    flux = Flux(
        source=[_make_sample(metadata={"pack_id": "RFUAV:PACK_X", "iq_file": "x.iq", "window_start_sample": 1234})]
    )
    _convert(flux, out, class_names={1: "drone-signal"})
    lines = (out / "metadata.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["index"] == 0 and entry["split"] == "train"
    assert entry["image"] == "images/train/000000.png"
    assert entry["source_metadata"]["pack_id"] == "RFUAV:PACK_X"
    assert entry["source_metadata"]["window_start_sample"] == 1234


# --- multi-split merge / append (the key new behaviour) ---------------------


def test_second_split_run_merges_data_yaml_and_preserves_first(tmp_path: Path) -> None:
    """Running val into the SAME dir after train must keep train + add val."""
    out = tmp_path / "out"
    _convert(_train_flux(), out, split="train")
    _convert(Flux(source=[_make_sample(boxes_xyxy=[[0, 0, 40, 40]], labels=[1])]), out, split="val")

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
    _convert(_train_flux(), out, split="train")  # 2 images
    _convert(Flux(source=[_make_sample()]), out, split="train")  # re-run: 1 image
    # The split dir is replaced, not appended — only the new single image remains.
    pngs = sorted(p.name for p in (out / "images" / "train").glob("*.png"))
    assert pngs == ["000000.png"]
    splits = [json.loads(line)["split"] for line in (out / "metadata.jsonl").read_text().strip().splitlines()]
    assert splits == ["train"]  # the stale 2-image train metadata was pruned


def test_three_splits_accumulate(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for split in ("train", "val", "test"):
        _convert(Flux(source=[_make_sample(boxes_xyxy=[[1, 2, 3, 4]], labels=[1])]), out, split=split)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert {"train", "val", "test"} <= set(data)
    assert json.loads((out / "annotations" / "test.json").read_text())["images"]


# --- construction / validation ----------------------------------------------


def test_zero_arg_construction_then_requires_class_names() -> None:
    sink = UltralyticsSink()  # zero-arg construction must work (lazy convention)
    with pytest.raises(ValueError, match="class_names"):
        sink.open()


def test_rejects_invalid_jpeg_quality_on_open(tmp_path: Path) -> None:
    sink = UltralyticsSink(output_dir=str(tmp_path), class_names={1: "x"}, jpeg_quality=120)
    with pytest.raises(ValueError, match="jpeg_quality"):
        sink.open()
