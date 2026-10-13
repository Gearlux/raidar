"""``RandomCropDetection`` — a fixed-size piece of a picture with its boxes and mask outlines, cut before a model's
own resize: every piece comes out exactly ``height`` x ``width`` (a short picture padded with its median, never
stretched), its boxes moved and clipped, a box cut below ``min_size_px`` dropped, its outlines clipped exactly."""

import random
from typing import Optional, Sequence

import confluid
import numpy as np
import pytest
import torch
from recordstream import Boxes, Image, Record
from recordstream.core import Stream

from raidar.crop import RandomCropDetection
from raidar.yolo_sink import POLYGONS


def _picture(height: int, width: int) -> np.ndarray:
    """Rows numbered by their index (mod 256) in every channel, so a piece's offset reads off its first row."""
    rows = (np.arange(height) % 256).astype(np.uint8)
    return np.repeat(np.repeat(rows[:, None, None], width, axis=1), 3, axis=2)


def _record(
    height: int,
    width: int,
    boxes: Sequence[Sequence[float]],
    polygons: Optional[Sequence[Sequence[Sequence[Sequence[float]]]]] = None,
) -> Record:
    extras = {POLYGONS: polygons} if polygons is not None else {}
    return {
        "image": Image(_picture(height, width), layout="HWC"),
        "target": Boxes(
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.arange(len(boxes), dtype=np.int64),
            canvas=(height, width),
            extras=extras,
            classes=["a", "b", "c", "d"],
        ),
    }


def _pixels(record: Record) -> np.ndarray:
    return np.asarray(record["image"].data)


def _area(ring: Sequence[Sequence[float]]) -> float:
    xs, ys = np.asarray(ring, dtype=float).T
    return float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2)


CENTRE = RandomCropDetection(height=40, width=40, placement="centre")


def test_a_tall_picture_becomes_exactly_the_piece() -> None:
    out = CENTRE(_record(200, 40, []))
    assert _pixels(out).shape == (40, 40, 3)
    assert out["target"].canvas == (40, 40)


def test_boxes_move_with_the_piece_are_clipped_and_a_box_cut_below_three_pixels_is_dropped() -> None:
    """The centre piece of a 200-row picture starts at row 80: a box over rows 70-100 keeps rows 80-100, one at
    150-160 lies outside, one at 78-82 keeps 2 rows (under 3) — both dropped, their labels with them."""
    out = CENTRE(_record(200, 40, [[5, 70, 30, 100], [5, 150, 30, 160], [5, 78, 30, 82]]))
    assert int(_pixels(out)[0, 0, 0]) == 80
    target = out["target"]
    np.testing.assert_allclose(target.boxes, [[5, 0, 30, 20]])
    assert target.labels.tolist() == [0]
    assert target.classes == ["a", "b", "c", "d"]


def test_mask_outlines_are_clipped_to_the_piece_and_an_instance_without_one_keeps_none() -> None:
    ring = [[5, 70], [30, 70], [30, 100], [5, 100]]
    out = CENTRE(_record(200, 40, [[5, 70, 30, 100], [2, 90, 10, 95]], polygons=[[ring], []]))
    rings = out["target"].extras[POLYGONS]
    assert len(rings) == 2 and rings[1] == []
    (clipped,) = rings[0]
    assert _area(clipped) == pytest.approx(25 * 20)
    assert np.asarray(clipped)[:, 1].min() == pytest.approx(0) and np.asarray(clipped)[:, 1].max() == pytest.approx(20)


def test_an_outline_the_piece_cuts_away_entirely_leaves_its_kept_box_maskless() -> None:
    """A box kept by the cut whose outline lies wholly outside the piece (an outline need not fill its box) — its
    instance keeps no ring, so its box is its mask."""
    outside = [[5, 0], [30, 0], [30, 10], [5, 10]]
    out = CENTRE(_record(200, 40, [[5, 0, 30, 100]], polygons=[[outside]]))
    assert out["target"].extras[POLYGONS] == [[]]


def test_a_short_picture_is_padded_with_its_median_and_its_boxes_stay_where_they_were() -> None:
    record = _record(20, 40, [[5, 2, 30, 12]])
    out = CENTRE(record)
    pixels = _pixels(out)
    assert pixels.shape == (40, 40, 3)
    np.testing.assert_array_equal(pixels[:20], _pixels(record))
    assert (pixels[20:] == np.median(_pixels(record), axis=(0, 1)).astype(np.uint8)).all()
    np.testing.assert_allclose(out["target"].boxes, [[5, 2, 30, 12]])


def test_a_piece_as_wide_as_the_picture_never_cuts_frequency() -> None:
    crop = RandomCropDetection(height=40, width=40)
    random.seed(3)
    for _ in range(50):
        boxes = crop(_record(200, 40, [[5, 0, 30, 200]]))["target"].boxes
        np.testing.assert_allclose(boxes[:, [0, 2]], [[5, 30]])


def test_a_wider_picture_is_cut_in_frequency_too() -> None:
    out = CENTRE(_record(40, 100, [[40, 5, 60, 20]]))
    np.testing.assert_allclose(out["target"].boxes, [[10, 5, 30, 20]])  # the centre piece starts at column 30


def test_centre_is_the_same_piece_every_call() -> None:
    first = [int(_pixels(CENTRE(_record(200, 40, [])))[0, 0, 0]) for _ in range(5)]
    assert first == [80] * 5


def test_random_reaches_both_ends_of_a_tall_picture_and_draws_from_pythons_random() -> None:
    """``random`` places the piece anywhere, ends included, drawing from the ``random`` module — which a data loader
    seeds per worker and per epoch, so each epoch cuts new pieces."""
    crop = RandomCropDetection(height=40, width=40)
    random.seed(0)
    tops = {int(_pixels(crop(_record(100, 40, [])))[0, 0, 0]) for _ in range(400)}
    assert min(tops) == 0 and max(tops) == 60
    random.seed(7)
    again = [int(_pixels(crop(_record(100, 40, [])))[0, 0, 0]) for _ in range(10)]
    random.seed(7)
    assert again == [int(_pixels(crop(_record(100, 40, [])))[0, 0, 0]) for _ in range(10)]


def test_a_record_without_a_target_has_its_picture_cut_alone() -> None:
    record = _record(200, 40, [])
    del record["target"]
    out = CENTRE(record)
    assert _pixels(out).shape == (40, 40, 3) and "target" not in out


def test_torch_boxes_and_labels_stay_torch() -> None:
    record = _record(200, 40, [[5, 70, 30, 100]])
    target = record["target"]
    record["target"] = Boxes(
        boxes=torch.as_tensor(target.boxes), labels=torch.as_tensor(target.labels), canvas=(200, 40)
    )
    out = CENTRE(record)["target"]
    assert isinstance(out.boxes, torch.Tensor) and isinstance(out.labels, torch.Tensor)
    assert out.boxes.tolist() == [[5.0, 0.0, 30.0, 20.0]]


def test_it_builds_with_no_arguments_and_refuses_a_missing_size_when_used() -> None:
    crop = RandomCropDetection()
    with pytest.raises(ValueError, match=r"RandomCropDetection: height and width must be set \(got 0 x 0\)"):
        crop(_record(20, 20, []))


def test_a_record_without_its_picture_is_refused_naming_the_key() -> None:
    with pytest.raises(ValueError, match=r"RandomCropDetection: no 'image' in the record \(keys: \['target'\]\)"):
        CENTRE({"target": _record(20, 20, [])["target"]})


def test_an_unknown_placement_is_refused_when_built() -> None:
    with pytest.raises(Exception, match="placement"):
        confluid.load("!class:raidar.crop.RandomCropDetection {height: 4, width: 4, placement: middle}")
    with pytest.raises(Exception, match="placement"):
        RandomCropDetection(height=4, width=4, placement="middle")  # type: ignore[arg-type]


def test_it_loads_from_a_config_and_runs_in_a_stream() -> None:
    crop = confluid.load("!class:raidar.crop.RandomCropDetection {height: 40, width: 40, placement: centre}")
    stream = Stream(source=[_record(200, 40, [[5, 70, 30, 100]])], ops=[crop])
    (out,) = list(stream)
    assert _pixels(out).shape == (40, 40, 3)
    np.testing.assert_allclose(out["target"].boxes, [[5, 0, 30, 20]])


def test_a_tensorized_picture_is_refused() -> None:
    record = {"image": Image(np.zeros((3, 20, 20), dtype=np.uint8), layout="CHW")}
    with pytest.raises(ValueError, match="a CHW picture — cut it before it is tensorized"):
        CENTRE(record)


def test_a_per_box_extra_column_follows_its_boxes() -> None:
    record = _record(200, 40, [[5, 70, 30, 100], [5, 150, 30, 160]])
    record["target"] = Boxes(
        boxes=record["target"].boxes, labels=record["target"].labels, canvas=(200, 40), extras={"score": [0.9, 0.4]}
    )
    assert CENTRE(record)["target"].extras["score"] == [0.9]


def test_an_extra_that_is_not_per_box_passes_unchanged() -> None:
    record = _record(200, 40, [[5, 70, 30, 100]])
    record["target"] = Boxes(boxes=record["target"].boxes, labels=record["target"].labels, extras={"source": "set a"})
    assert CENTRE(record)["target"].extras["source"] == "set a"


def test_an_outline_crossing_the_left_and_right_edges_is_clipped_to_them() -> None:
    """A 100-column picture's centre piece spans columns 30-70: a diamond from column 20 to 80 keeps its middle."""
    diamond = [[20, 20], [50, 5], [80, 20], [50, 35]]
    out = CENTRE(_record(40, 100, [[20, 5, 80, 35]], polygons=[[diamond]]))
    (clipped,) = out["target"].extras[POLYGONS][0]
    xs = np.asarray(clipped)[:, 0]
    assert xs.min() == pytest.approx(0) and xs.max() == pytest.approx(40)
    # 900 px² less its two tips: each 10 px long, 10 px tall where the edge cuts it — 50 px² apiece
    assert _area(diamond) == pytest.approx(900) and _area(clipped) == pytest.approx(800)


def test_a_grey_picture_is_cut_and_padded_too() -> None:
    record = {"image": Image(np.full((20, 40), 7, dtype=np.uint8), layout="HW")}
    pixels = _pixels(CENTRE(record))
    assert pixels.shape == (40, 40) and (pixels == 7).all()
